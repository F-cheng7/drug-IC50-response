# ============================================================
# train.py
# Training / test / optional few-shot adaptation
# - keeps the original train/test protocol unchanged
# - keeps the few-shot interface unchanged
# - stabilizes optimizer/checkpoint implementation without changing model flow
# ============================================================

import copy
import csv
import gc
import inspect
import os
from collections import OrderedDict
from typing import Dict, Optional

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import JointDataset, collate_fn_joint


class ComboLabelTransform:
    """
    仅用于 combo(Synergy_Score) 的标签变换：
        y' = sign(y) * log(1 + |y| / c)
    反变换：
        y  = sign(y') * (exp(|y'|) - 1) * c
    """
    def __init__(self, mode: str = "none", scale: float = 5.0):
        self.mode = mode
        self.scale = float(scale)

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def transform_np(self, y):
        y = np.asarray(y, dtype=np.float32)
        if not self.enabled:
            return y.astype(np.float32)
        if self.mode == "signed_log1p":
            return (np.sign(y) * np.log1p(np.abs(y) / self.scale)).astype(np.float32)
        raise ValueError(f"Unsupported combo label transform: {self.mode}")

    def inverse_np(self, y):
        y = np.asarray(y, dtype=np.float32)
        if not self.enabled:
            return y.astype(np.float32)
        if self.mode == "signed_log1p":
            return (np.sign(y) * np.expm1(np.abs(y)) * self.scale).astype(np.float32)
        raise ValueError(f"Unsupported combo label transform: {self.mode}")

    def transform_tensor(self, y: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return y
        if self.mode == "signed_log1p":
            return torch.sign(y) * torch.log1p(torch.abs(y) / self.scale)
        raise ValueError(f"Unsupported combo label transform: {self.mode}")

    def inverse_tensor(self, y: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return y
        if self.mode == "signed_log1p":
            return torch.sign(y) * torch.expm1(torch.abs(y)) * self.scale
        raise ValueError(f"Unsupported combo label transform: {self.mode}")


def _batch_with_new_label(batch, new_label):
    new_batch = dict(batch)
    new_batch["label"] = new_label
    return new_batch


def _combo_inverse_transform_for_mode(dataset_mode: str, combo_label_tf: ComboLabelTransform):
    if dataset_mode == "combo" and combo_label_tf is not None and combo_label_tf.enabled:
        return combo_label_tf
    return None


def _snapshot_model_state(model):
    state = OrderedDict()
    for k, v in model.state_dict().items():
        if torch.is_tensor(v):
            state[k] = v.detach().cpu().clone()
        else:
            state[k] = copy.deepcopy(v)
    return state


def _snapshot_optimizer_state(optimizer):
    state = optimizer.state_dict()

    def _to_cpu(obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu().clone()
        if isinstance(obj, dict):
            return {k: _to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_cpu(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_to_cpu(v) for v in obj)
        return copy.deepcopy(obj)

    return _to_cpu(state)


def _build_adamw(params, lr, weight_decay):
    kwargs = {"lr": lr, "weight_decay": weight_decay}
    try:
        sig = inspect.signature(torch.optim.AdamW)
        if "foreach" in sig.parameters:
            kwargs["foreach"] = False
        if "fused" in sig.parameters:
            kwargs["fused"] = False
        if "capturable" in sig.parameters:
            kwargs["capturable"] = False
    except Exception:
        pass
    return torch.optim.AdamW(params, **kwargs)


def _sanitize_gradients(params, clamp_val: float = 5.0) -> bool:
    has_bad = False
    for p in params:
        g = p.grad
        if g is None:
            continue
        if not g.is_contiguous():
            p.grad = g.contiguous()
            g = p.grad
        if not torch.isfinite(g).all():
            has_bad = True
            torch.nan_to_num_(g, nan=0.0, posinf=clamp_val, neginf=-clamp_val)
        g.clamp_(-clamp_val, clamp_val)
    return not has_bad


def _clip_grad_norm_stable(parameters, max_norm: float, norm_type: float = 2.0, eps: float = 1e-6):
    params = [p for p in parameters if p.grad is not None]
    if len(params) == 0:
        return torch.tensor(0.0)

    device = params[0].grad.device
    norm_type = float(norm_type)

    if norm_type == float("inf"):
        total_norm = torch.zeros((), device=device, dtype=torch.float32)
        for p in params:
            g = p.grad.detach()
            if g.numel() == 0:
                continue
            gmax = g.abs().max().to(dtype=torch.float32)
            total_norm = torch.maximum(total_norm, gmax)
    else:
        total = torch.zeros((), device=device, dtype=torch.float32)
        for p in params:
            g = p.grad.detach()
            if g.numel() == 0:
                continue
            gn = torch.linalg.vector_norm(g.to(dtype=torch.float32), ord=norm_type)
            total = total + gn.pow(norm_type)
        total_norm = total.pow(1.0 / norm_type)

    if torch.isfinite(total_norm):
        clip_coef = (float(max_norm) / (total_norm + eps)).clamp(max=1.0)
        if clip_coef < 1.0:
            for p in params:
                p.grad.mul_(clip_coef.to(dtype=p.grad.dtype))

    return total_norm



def _optimizer_barrier(device):
    if isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _format_meta(meta_batch):
    parts = []
    for k in ["meta_sample_index", "meta_sample_id", "meta_cell_line", "meta_smiles", "meta_smiles_1", "meta_smiles_2"]:
        if k in meta_batch and len(meta_batch[k]) > 0:
            parts.append(f"{k}={meta_batch[k][0]}")
    return " | ".join(parts) if parts else "no-meta"


def _format_joint_meta(single_meta, combo_meta):
    parts = []
    if single_meta:
        parts.append(f"single[{_format_meta(single_meta)}]")
    if combo_meta:
        parts.append(f"combo[{_format_meta(combo_meta)}]")
    return " | ".join(parts) if parts else "no-meta"


def _clear_cuda():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except RuntimeError:
        pass
    gc.collect()


def move_batch_to_device(batch, device):
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def _validate_batch_tensors(batch) -> bool:
    for _, v in batch.items():
        if not torch.is_tensor(v):
            continue
        if v.dtype.is_floating_point and (not torch.isfinite(v).all()):
            return False
    return True


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if mask.sum() < 2:
        return {"pearson": float("nan"), "spearman": float("nan"), "r2": float("nan"), "mae": float("nan"), "rmse": float("nan"), "mse": float("nan")}

    yt = y_true[mask]
    yp = y_pred[mask]

    try:
        pearson = pearsonr(yt, yp)[0]
    except Exception:
        pearson = float("nan")
    try:
        spearman = spearmanr(yt, yp)[0]
    except Exception:
        spearman = float("nan")

    mse = mean_squared_error(yt, yp)
    rmse = float(np.sqrt(mse))

    return {
        "pearson": float(pearson),
        "spearman": float(spearman),
        "r2": float(r2_score(yt, yp)),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(rmse),
        "mse": float(mse),
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    ordered = ["pearson", "spearman", "r2", "mae", "rmse", "mse"]
    return " | ".join([f"{k.upper()}: {metrics[k]:.4f}" for k in ordered])


def _is_better_checkpoint(metrics: Dict[str, float], best_pearson: float, best_rmse: float, eps: float = 1e-8) -> bool:
    pearson = float(metrics.get("pearson", float("nan")))
    rmse = float(metrics.get("rmse", float("inf")))

    pearson_finite = np.isfinite(pearson)
    best_pearson_finite = np.isfinite(best_pearson)

    if pearson_finite and not best_pearson_finite:
        return True
    if not pearson_finite and best_pearson_finite:
        return False

    if pearson_finite and best_pearson_finite:
        if pearson > best_pearson + eps:
            return True
        if pearson < best_pearson - eps:
            return False

    rmse_finite = np.isfinite(rmse)
    best_rmse_finite = np.isfinite(best_rmse)

    if rmse_finite and not best_rmse_finite:
        return True
    if not rmse_finite and best_rmse_finite:
        return False

    if rmse_finite and best_rmse_finite:
        if rmse < best_rmse - eps:
            return True
        if rmse > best_rmse + eps:
            return False

    return False


def _gather_meta_from_batch(batch):
    meta = {}
    for k, v in batch.items():
        if isinstance(k, str) and k.startswith("meta_"):
            if torch.is_tensor(v):
                meta[k] = v.detach().cpu().numpy().reshape(-1).tolist()
            elif isinstance(v, (list, tuple)):
                meta[k] = list(v)
            else:
                meta[k] = [v]
    return meta


def _ensure_parent_dir(path: Optional[str]):
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _ordered_meta_keys(meta_accumulator):
    return [
        k for k in ["meta_sample_index", "meta_sample_id", "meta_cell_line", "meta_smiles", "meta_smiles_1", "meta_smiles_2"]
        if k in meta_accumulator
    ]


def _save_prediction_csv(save_path: str, meta_accumulator, y_true, y_pred, true_col: str = "y_true", pred_col: str = "y_pred"):
    _ensure_parent_dir(save_path)
    ordered_meta_keys = _ordered_meta_keys(meta_accumulator)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(ordered_meta_keys + [true_col, pred_col])
        for i in range(len(y_true)):
            row = [meta_accumulator[k][i] if i < len(meta_accumulator[k]) else "" for k in ordered_meta_keys]
            row += [float(y_true[i]), float(y_pred[i])]
            writer.writerow(row)


def _predict_with_metadata(
    model,
    loader,
    device,
    split_name="Test",
    label_context: Optional[torch.Tensor] = None,
    prediction_inverse_transform=None,
):
    model.eval()
    y_true, y_pred = [], []
    meta_accumulator = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[{split_name}]", ncols=100):
            meta_batch = _gather_meta_from_batch(batch)
            batch = move_batch_to_device(batch, device)
            current_context = None
            if label_context is not None:
                current_context = label_context.expand(batch["label"].size(0), 1)
            pred = model.predict(batch, label_context=current_context)
            y_true.append(batch["label"].detach().cpu().numpy())
            y_pred.append(pred.detach().cpu().numpy())

            for k, values in meta_batch.items():
                meta_accumulator.setdefault(k, []).extend(values)

    y_true = np.concatenate(y_true, axis=0).reshape(-1)
    y_pred = np.concatenate(y_pred, axis=0).reshape(-1)

    if prediction_inverse_transform is not None and prediction_inverse_transform.enabled:
        y_pred = prediction_inverse_transform.inverse_np(y_pred)

    metrics = compute_metrics(y_true, y_pred)
    return metrics, y_true, y_pred, meta_accumulator


def evaluate(
    model,
    loader,
    device,
    split_name="Test",
    label_context: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    prediction_inverse_transform=None,
):
    metrics, y_true, y_pred, meta_accumulator = _predict_with_metadata(
        model=model,
        loader=loader,
        device=device,
        split_name=split_name,
        label_context=label_context,
        prediction_inverse_transform=prediction_inverse_transform,
    )

    if save_path is not None:
        _save_prediction_csv(save_path, meta_accumulator, y_true, y_pred)

    return metrics, y_true, y_pred


def _create_train_test_split(dataset, train_ratio: float, seed: int):
    total_len = len(dataset)
    train_len = int(total_len * train_ratio)
    train_len = min(max(train_len, 1), total_len - 1)
    test_len = total_len - train_len
    if test_len <= 0:
        test_len = 1
        train_len = total_len - 1

    return random_split(dataset, [train_len, test_len], generator=torch.Generator().manual_seed(seed))


def _get_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def _support_mean_label(dataset, device, batch_size=64, label_transform=None):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    labels = []
    for batch in loader:
        y = batch["label"]
        if label_transform is not None and label_transform.enabled:
            y = label_transform.transform_tensor(y)
        labels.append(y)
    label_mean = torch.cat(labels, dim=0).mean().view(1, 1).to(device)
    return label_mean


def _task_value_names(dataset_mode: str):
    if dataset_mode == "single":
        return "original_IC50", "pred_IC50_before_adapt", "pred_IC50_after_adapt"
    if dataset_mode == "combo":
        return "original_Synergy_Score", "pred_Synergy_before_adapt", "pred_Synergy_after_adapt"
    return "original_value", "pred_before_adapt", "pred_after_adapt"


def _save_few_shot_comparison_csv(save_path: str, dataset_mode: str, meta_accumulator, y_true, y_pred_before, y_pred_after):
    true_col, before_col, after_col = _task_value_names(dataset_mode)
    _ensure_parent_dir(save_path)
    ordered_meta_keys = _ordered_meta_keys(meta_accumulator)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(ordered_meta_keys + [true_col, before_col, after_col])
        for i in range(len(y_true)):
            row = [meta_accumulator[k][i] if i < len(meta_accumulator[k]) else "" for k in ordered_meta_keys]
            row += [float(y_true[i]), float(y_pred_before[i]), float(y_pred_after[i])]
            writer.writerow(row)


def _prefixed_name(prefix: Optional[str], filename: str) -> str:
    if prefix is None or str(prefix).strip() == "":
        return filename
    return f"{prefix}_{filename}"


def _safe_weighted_average(values, weights):
    finite = [(float(v), float(w)) for v, w in zip(values, weights) if np.isfinite(v) and np.isfinite(w)]
    if len(finite) == 0:
        return float("nan")
    total_w = sum(w for _, w in finite)
    if total_w <= 0:
        return float("nan")
    return float(sum(v * w for v, w in finite) / total_w)


def _inverse_count_weights(single_count: int, combo_count: int):
    single_w = 1.0 / max(1, int(single_count))
    combo_w = 1.0 / max(1, int(combo_count))
    total = single_w + combo_w
    if total <= 0:
        return 0.5, 0.5
    return single_w / total, combo_w / total


def _combined_joint_monitor(single_metrics: Dict[str, float], combo_metrics: Dict[str, float], single_w: float, combo_w: float):
    return {
        "pearson": _safe_weighted_average([single_metrics.get("pearson", float("nan")), combo_metrics.get("pearson", float("nan"))], [single_w, combo_w]),
        "rmse": _safe_weighted_average([single_metrics.get("rmse", float("nan")), combo_metrics.get("rmse", float("nan"))], [single_w, combo_w]),
    }


# ------------------------------
# Few-shot representation adaptation losses
# ------------------------------
def _pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor, min_delta: float = 1e-4) -> torch.Tensor:
    """
    连续标签的排序约束：
    让预测差值的符号与真实标签差值尽可能一致，避免 few-shot 仅学会“回归到均值”。
    """
    pred = pred.view(-1)
    target = target.view(-1)
    n = pred.numel()
    if n < 2:
        return pred.new_zeros(())

    pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)
    target_diff = target.unsqueeze(0) - target.unsqueeze(1)

    valid = torch.abs(target_diff) > float(min_delta)
    if not torch.any(valid):
        return pred.new_zeros(())

    sign = torch.sign(target_diff[valid])
    pair_margin = pred_diff[valid]
    return torch.nn.functional.softplus(-sign * pair_margin).mean()



def _distribution_matching_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    在 support 集上对齐预测分布与真实分布的均值/方差，
    让 few-shot 更像“重建新分布”，而不是仅做点状校准。
    """
    pred = pred.view(-1)
    target = target.view(-1)

    mean_loss = torch.abs(pred.mean() - target.mean())
    if pred.numel() < 2:
        return mean_loss

    pred_std = pred.std(unbiased=False)
    target_std = target.std(unbiased=False)
    std_loss = torch.abs(pred_std - target_std)
    return mean_loss + std_loss


def few_shot_adapt(model, dataset, device, batch_size=8, lr=5e-4, num_epochs=30, support_ratio=0.8, num_workers=0, ckpt_dir="./checkpoints", tune_bert=False, output_prefix: Optional[str] = None, combo_label_transform: str = "none", combo_label_scale: float = 5.0):
    print("\n▶ 开始 few-shot 表征适配阶段...")
    dataset_mode = getattr(dataset, "mode", "single")
    combo_label_tf = ComboLabelTransform(combo_label_transform, combo_label_scale)
    prediction_inverse_transform = _combo_inverse_transform_for_mode(dataset_mode, combo_label_tf)
    support_len = max(1, int(len(dataset) * support_ratio))
    query_len = len(dataset) - support_len
    if query_len <= 0:
        query_len = 1
        support_len = len(dataset) - 1

    support_set, query_set = random_split(dataset, [support_len, query_len], generator=torch.Generator().manual_seed(42))
    print(f"[Few-shot Split] support: {len(support_set)}, query: {len(query_set)}")

    support_loader = DataLoader(support_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    query_loader = DataLoader(query_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 适配前评估：保持与 few-shot 前后对比实验兼容
    pre_metrics, pre_true, pre_pred, pre_meta = _predict_with_metadata(
        model,
        query_loader,
        device,
        split_name="Few-shot Query Before Adapt",
        label_context=None,
        prediction_inverse_transform=prediction_inverse_transform,
    )
    print(f"[Few-shot Before Adapt] {format_metrics(pre_metrics)}")

    # 进入“表征适配”模式：不再使用 support 标签均值作为共享 label_context，
    # 避免所有 query 被强行拉向同一个中心值。
    model.freeze_for_few_shot(tune_bert=tune_bert)
    if tune_bert and hasattr(model, "text_encoder"):
        try:
            model.text_encoder.to(device)
        except Exception:
            pass
    elif getattr(dataset, "cache_text_embeddings", False) and hasattr(model, "text_encoder"):
        try:
            model.text_encoder.to("cpu")
        except Exception:
            pass

    optimizer = _build_adamw(_get_trainable_params(model), lr=lr, weight_decay=1e-5)

    best_state = _snapshot_model_state(model)
    best_pearson = float("-inf")
    best_rmse = float("inf")

    rank_loss_weight = 0.25
    dist_loss_weight = 0.15

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_reg_loss = 0.0
        epoch_rank_loss = 0.0
        epoch_dist_loss = 0.0
        n = 0

        for batch in tqdm(support_loader, desc=f"[Few-shot Train] {epoch}/{num_epochs}", ncols=100):
            batch = move_batch_to_device(batch, device)

            if not _validate_batch_tensors(batch):
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            loss_batch = batch
            if prediction_inverse_transform is not None and prediction_inverse_transform.enabled:
                transformed_label = prediction_inverse_transform.transform_tensor(batch["label"])
                loss_batch = _batch_with_new_label(batch, transformed_label)

            pred = model.forward(loss_batch, label_context=None)
            target = loss_batch["label"]

            reg_loss = model.loss_fn(pred, target)
            rank_loss = _pairwise_rank_loss(pred, target)
            dist_loss = _distribution_matching_loss(pred, target)
            loss = reg_loss + rank_loss_weight * rank_loss + dist_loss_weight * dist_loss

            loss.backward()
            params = _get_trainable_params(model)
            grads_ok = _sanitize_gradients(params, clamp_val=5.0)
            if not grads_ok:
                optimizer.zero_grad(set_to_none=True)
                _clear_cuda()
                continue
            _clip_grad_norm_stable(params, 2.0)
            _optimizer_barrier(device)
            optimizer.step()

            bsz = batch["label"].size(0)
            epoch_loss += loss.item() * bsz
            epoch_reg_loss += reg_loss.item() * bsz
            epoch_rank_loss += float(rank_loss.item()) * bsz
            epoch_dist_loss += float(dist_loss.item()) * bsz
            n += bsz

        metrics, _, _ = evaluate(
            model,
            query_loader,
            device,
            split_name="Few-shot Query",
            label_context=None,
            prediction_inverse_transform=prediction_inverse_transform,
        )
        print(
            f"[Few-shot Epoch {epoch}] "
            f"Loss: {epoch_loss / max(1, n):.4f} | "
            f"Reg: {epoch_reg_loss / max(1, n):.4f} | "
            f"Rank: {epoch_rank_loss / max(1, n):.4f} | "
            f"Dist: {epoch_dist_loss / max(1, n):.4f} | "
            f"{format_metrics(metrics)}"
        )

        if _is_better_checkpoint(metrics, best_pearson=best_pearson, best_rmse=best_rmse):
            best_pearson = float(metrics.get("pearson", float("nan")))
            best_rmse = float(metrics.get("rmse", float("inf")))
            best_state = _snapshot_model_state(model)

    model.load_state_dict(best_state)
    save_path = os.path.join(ckpt_dir, _prefixed_name(output_prefix, "few_shot_best.pt"))
    torch.save({"model_state": _snapshot_model_state(model)}, save_path)
    print(f"✔ Few-shot best checkpoint saved: {save_path}")

    final_metrics, final_true, final_pred, final_meta = _predict_with_metadata(
        model,
        query_loader,
        device,
        split_name="Few-shot Final",
        label_context=None,
        prediction_inverse_transform=prediction_inverse_transform,
    )
    print(f"[Few-shot Final] {format_metrics(final_metrics)}")

    compare_save_path = os.path.join(ckpt_dir, _prefixed_name(output_prefix, "few_shot_predictions_compare.csv"))
    meta_for_save = final_meta if len(final_meta) > 0 else pre_meta
    true_for_save = final_true if len(final_true) == len(pre_true) else pre_true
    _save_few_shot_comparison_csv(
        save_path=compare_save_path,
        dataset_mode=getattr(dataset, "mode", "single"),
        meta_accumulator=meta_for_save,
        y_true=true_for_save,
        y_pred_before=pre_pred,
        y_pred_after=final_pred,
    )
    print(f"✔ Few-shot comparison predictions saved: {compare_save_path}")
    return final_metrics

def train(model, dataset, train_ratio=0.8, batch_size=16, lr=1e-4, weight_decay=1e-4, num_epochs=30, num_workers=4, device="cuda", ckpt_dir="./checkpoints", evaluate_every=1, use_wandb=False, seed=42, resume_from=None, patience=10, few_shot_dataset=None, few_shot_epochs=30, few_shot_lr=5e-4, few_shot_batch_size=8, few_shot_support_ratio=0.8, few_shot_tune_bert=False, output_prefix: Optional[str] = None, combo_label_transform: str = "none", combo_label_scale: float = 5.0):
    del use_wandb

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if getattr(dataset, "cache_text_embeddings", False) and hasattr(model, "text_encoder") and (not getattr(model.text_encoder, "trainable", False)) and (not few_shot_tune_bert):
        try:
            model.text_encoder.to("cpu")
            print("✅ Frozen BERT forward will be bypassed by cached text features; text encoder kept on CPU.")
        except Exception:
            pass
    os.makedirs(ckpt_dir, exist_ok=True)

    dataset_mode = getattr(dataset, "mode", "single")
    combo_label_tf = ComboLabelTransform(combo_label_transform, combo_label_scale)
    prediction_inverse_transform = _combo_inverse_transform_for_mode(dataset_mode, combo_label_tf)

    train_set, test_set = _create_train_test_split(dataset, train_ratio=train_ratio, seed=seed)
    print(f"[Split] train: {len(train_set)}, test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    optimizer = _build_adamw(_get_trainable_params(model), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))

    start_epoch = 1
    if resume_from is not None:
        print(f"▶ Resuming from checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1

    best_state = _snapshot_model_state(model)
    best_epoch = 0
    best_pearson = float("-inf")
    best_rmse = float("inf")
    wait = 0

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        sample_count = 0

        pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{num_epochs}", ncols=100)
        for step, batch in enumerate(pbar, start=1):
            meta_batch = _gather_meta_from_batch(batch)
            batch = move_batch_to_device(batch, device)

            try:
                if not _validate_batch_tensors(batch):
                    print(f"❌ 检测到非有限输入张量，跳过该 batch | {_format_meta(meta_batch)}")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss_batch = batch
                if prediction_inverse_transform is not None and prediction_inverse_transform.enabled:
                    transformed_label = prediction_inverse_transform.transform_tensor(batch["label"])
                    loss_batch = _batch_with_new_label(batch, transformed_label)

                loss_dict = model.compute_loss(loss_batch, label_context=None)
                loss = loss_dict["loss"]

                if not torch.isfinite(loss):
                    print(f"❌ 检测到非有限 loss，跳过该 batch | {_format_meta(meta_batch)}")
                    _clear_cuda()
                    continue

                loss.backward()
                params = _get_trainable_params(model)
                grads_ok = _sanitize_gradients(params, clamp_val=5.0)
                if not grads_ok:
                    print(f"❌ 检测到非有限梯度分量，跳过该 batch | {_format_meta(meta_batch)}")
                    optimizer.zero_grad(set_to_none=True)
                    _clear_cuda()
                    continue

                grad_norm = _clip_grad_norm_stable(params, 1.0)

                if not torch.isfinite(grad_norm):
                    print(f"❌ 检测到非有限梯度，跳过该 batch | {_format_meta(meta_batch)}")
                    optimizer.zero_grad(set_to_none=True)
                    _clear_cuda()
                    continue

                _optimizer_barrier(device)
                optimizer.step()

            except RuntimeError as e:
                msg = str(e)
                print(f"❌ backward/step 失败: {msg} | {_format_meta(meta_batch)}")
                if "illegal memory access" not in msg.lower():
                    _clear_cuda()
                raise

            bsz = batch["label"].size(0)
            epoch_loss += loss.item() * bsz
            sample_count += bsz
            pbar.set_postfix(loss=f"{epoch_loss / max(1, sample_count):.4f}")

        scheduler.step()
        avg_train_loss = epoch_loss / max(1, sample_count)
        print(f"[Epoch {epoch}] Train loss: {avg_train_loss:.4f}")
        _clear_cuda()

        test_metrics = None
        if epoch % evaluate_every == 0:
            test_metrics, _, _ = evaluate(
                model,
                test_loader,
                device,
                split_name=f"Test@Epoch{epoch}",
                label_context=None,
                save_path=os.path.join(ckpt_dir, _prefixed_name(output_prefix, f"test_predictions_epoch_{epoch}.csv")),
                prediction_inverse_transform=prediction_inverse_transform,
            )
            print(f"[Epoch {epoch}] Test monitor | {format_metrics(test_metrics)}")
            _clear_cuda()

        if test_metrics is not None and _is_better_checkpoint(test_metrics, best_pearson=best_pearson, best_rmse=best_rmse):
            best_pearson = float(test_metrics.get("pearson", float("nan")))
            best_rmse = float(test_metrics.get("rmse", float("inf")))
            best_epoch = epoch
            wait = 0
            best_state = _snapshot_model_state(model)
            best_path = os.path.join(ckpt_dir, _prefixed_name(output_prefix, "best_model.pt"))
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": _snapshot_model_state(model),
                    "optimizer_state": _snapshot_optimizer_state(optimizer),
                    "best_pearson": best_pearson,
                    "best_rmse": best_rmse,
                },
                best_path,
            )
            print(f"✔ Saved best checkpoint by Pearson-first/RMSE-second: {best_path}")
        else:
            wait += 1
            if wait >= patience:
                print(f"⏹ Early stopping triggered at epoch {epoch}. Best epoch = {best_epoch}")
                break

        epoch_path = os.path.join(ckpt_dir, _prefixed_name(output_prefix, f"epoch_{epoch}.pt"))
        torch.save(
            {
                "epoch": epoch,
                "model_state": _snapshot_model_state(model),
                "optimizer_state": _snapshot_optimizer_state(optimizer),
            },
            epoch_path,
        )

    model.load_state_dict(best_state)

    print(f"\n▶ 使用 Pearson 优先、RMSE 次优 的最佳模型进行最终测试（best epoch = {best_epoch}）...")
    test_metrics, y_true, y_pred = evaluate(
        model,
        test_loader,
        device,
        split_name="Final Test",
        label_context=None,
        save_path=os.path.join(ckpt_dir, _prefixed_name(output_prefix, "test_predictions.csv")),
        prediction_inverse_transform=prediction_inverse_transform,
    )
    print(f"[Final Test] {format_metrics(test_metrics)}")

    result = {"best_epoch": best_epoch, "best_pearson": best_pearson, "best_rmse": best_rmse, "test_metrics": test_metrics, "y_true": y_true, "y_pred": y_pred}

    if few_shot_dataset is not None and len(few_shot_dataset) >= 2:
        few_shot_prefix = output_prefix if output_prefix is not None else getattr(few_shot_dataset, "mode", "fewshot")
        result["few_shot_metrics"] = few_shot_adapt(
            model=model,
            dataset=few_shot_dataset,
            device=device,
            batch_size=few_shot_batch_size,
            lr=few_shot_lr,
            num_epochs=few_shot_epochs,
            support_ratio=few_shot_support_ratio,
            num_workers=num_workers,
            ckpt_dir=ckpt_dir,
            tune_bert=few_shot_tune_bert,
            output_prefix=few_shot_prefix,
            combo_label_transform=combo_label_transform,
            combo_label_scale=combo_label_scale,
        )

    print("🎉 Training finished.")
    return result


def train_joint(model, single_dataset, combo_dataset, train_ratio=0.8, batch_size=16, lr=1e-4, weight_decay=1e-4, num_epochs=30, num_workers=4, device="cuda", ckpt_dir="./checkpoints", evaluate_every=1, use_wandb=False, seed=42, resume_from=None, patience=10, few_shot_dataset=None, few_shot_epochs=30, few_shot_lr=5e-4, few_shot_batch_size=8, few_shot_support_ratio=0.8, few_shot_tune_bert=False, output_prefix: Optional[str] = "joint", combo_label_transform: str = "none", combo_label_scale: float = 5.0):
    del use_wandb

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if getattr(single_dataset, "cache_text_embeddings", False) and getattr(combo_dataset, "cache_text_embeddings", False) and hasattr(model, "text_encoder") and (not getattr(model.text_encoder, "trainable", False)) and (not few_shot_tune_bert):
        try:
            model.text_encoder.to("cpu")
            print("✅ Frozen BERT forward will be bypassed by cached text features; text encoder kept on CPU.")
        except Exception:
            pass
    os.makedirs(ckpt_dir, exist_ok=True)

    combo_label_tf = ComboLabelTransform(combo_label_transform, combo_label_scale)
    combo_prediction_inverse_transform = _combo_inverse_transform_for_mode("combo", combo_label_tf)

    single_train_set, single_test_set = _create_train_test_split(single_dataset, train_ratio=train_ratio, seed=seed)
    combo_train_set, combo_test_set = _create_train_test_split(combo_dataset, train_ratio=train_ratio, seed=seed)
    print(f"[Joint Split] single train/test: {len(single_train_set)}/{len(single_test_set)} | combo train/test: {len(combo_train_set)}/{len(combo_test_set)}")

    single_loss_weight, combo_loss_weight = _inverse_count_weights(len(single_train_set), len(combo_train_set))
    print(f"[Joint Loss Weight] single={single_loss_weight:.6f} | combo={combo_loss_weight:.6f}")

    joint_train_dataset = JointDataset(single_train_set, combo_train_set)
    joint_train_loader = DataLoader(
        joint_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn_joint,
    )
    single_test_loader = DataLoader(single_test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    combo_test_loader = DataLoader(combo_test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    optimizer = _build_adamw(_get_trainable_params(model), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))

    start_epoch = 1
    if resume_from is not None:
        print(f"▶ Resuming from checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1

    best_state = _snapshot_model_state(model)
    best_epoch = 0
    best_pearson = float("-inf")
    best_rmse = float("inf")
    wait = 0

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_single_loss = 0.0
        epoch_combo_loss = 0.0
        total_steps = 0
        single_sample_count = 0
        combo_sample_count = 0

        pbar = tqdm(joint_train_loader, desc=f"[Joint Train] Epoch {epoch}/{num_epochs}", ncols=120)
        for step, joint_batch in enumerate(pbar, start=1):
            single_meta = _gather_meta_from_batch(joint_batch["single_batch"]) if joint_batch.get("single_batch") is not None else {}
            combo_meta = _gather_meta_from_batch(joint_batch["combo_batch"]) if joint_batch.get("combo_batch") is not None else {}
            optimizer.zero_grad(set_to_none=True)
            total_loss = None
            step_single_loss = None
            step_combo_loss = None

            try:
                if joint_batch.get("single_batch") is not None:
                    single_batch = move_batch_to_device(joint_batch["single_batch"], device)
                    if not _validate_batch_tensors(single_batch):
                        print(f"❌ 检测到非有限单药输入张量，跳过该子 batch | {_format_meta(single_meta)}")
                        single_batch = None
                    else:
                        single_loss_dict = model.compute_loss(single_batch, label_context=None)
                        step_single_loss = single_loss_dict["loss"]
                        if torch.isfinite(step_single_loss):
                            total_loss = single_loss_weight * step_single_loss if total_loss is None else total_loss + single_loss_weight * step_single_loss
                        else:
                            print(f"❌ 检测到非有限单药 loss，跳过该子 batch | {_format_meta(single_meta)}")
                            step_single_loss = None
                            single_batch = None
                else:
                    single_batch = None

                if joint_batch.get("combo_batch") is not None:
                    combo_batch = move_batch_to_device(joint_batch["combo_batch"], device)
                    if not _validate_batch_tensors(combo_batch):
                        print(f"❌ 检测到非有限多药输入张量，跳过该子 batch | {_format_meta(combo_meta)}")
                        combo_batch = None
                    else:
                        combo_loss_batch = combo_batch
                        if combo_prediction_inverse_transform is not None and combo_prediction_inverse_transform.enabled:
                            transformed_label = combo_prediction_inverse_transform.transform_tensor(combo_batch["label"])
                            combo_loss_batch = _batch_with_new_label(combo_batch, transformed_label)

                        combo_loss_dict = model.compute_loss(combo_loss_batch, label_context=None)
                        step_combo_loss = combo_loss_dict["loss"]
                        if torch.isfinite(step_combo_loss):
                            total_loss = combo_loss_weight * step_combo_loss if total_loss is None else total_loss + combo_loss_weight * step_combo_loss
                        else:
                            print(f"❌ 检测到非有限多药 loss，跳过该子 batch | {_format_meta(combo_meta)}")
                            step_combo_loss = None
                            combo_batch = None
                else:
                    combo_batch = None

                if total_loss is None or (not torch.isfinite(total_loss)):
                    print(f"❌ 当前 joint batch 无有效 loss，跳过 | {_format_joint_meta(single_meta, combo_meta)}")
                    _clear_cuda()
                    continue

                total_loss.backward()
                params = _get_trainable_params(model)
                grads_ok = _sanitize_gradients(params, clamp_val=5.0)
                if not grads_ok:
                    print(f"❌ 检测到非有限梯度分量，跳过该 joint batch | {_format_joint_meta(single_meta, combo_meta)}")
                    optimizer.zero_grad(set_to_none=True)
                    _clear_cuda()
                    continue

                grad_norm = _clip_grad_norm_stable(params, 1.0)
                if not torch.isfinite(grad_norm):
                    print(f"❌ 检测到非有限梯度，跳过该 joint batch | {_format_joint_meta(single_meta, combo_meta)}")
                    optimizer.zero_grad(set_to_none=True)
                    _clear_cuda()
                    continue

                _optimizer_barrier(device)
                optimizer.step()

            except RuntimeError as e:
                msg = str(e)
                print(f"❌ joint backward/step 失败: {msg} | {_format_joint_meta(single_meta, combo_meta)}")
                if "illegal memory access" not in msg.lower():
                    _clear_cuda()
                raise

            total_steps += 1
            epoch_loss += float(total_loss.item())
            if step_single_loss is not None and single_batch is not None:
                bsz = int(single_batch["label"].size(0))
                epoch_single_loss += float(step_single_loss.item()) * bsz
                single_sample_count += bsz
            if step_combo_loss is not None and combo_batch is not None:
                bsz = int(combo_batch["label"].size(0))
                epoch_combo_loss += float(step_combo_loss.item()) * bsz
                combo_sample_count += bsz

            current_single_loss = epoch_single_loss / max(1, single_sample_count)
            current_combo_loss = epoch_combo_loss / max(1, combo_sample_count)
            current_total_loss = epoch_loss / max(1, total_steps)
            pbar.set_postfix(total=f"{current_total_loss:.4f}", single=f"{current_single_loss:.4f}", combo=f"{current_combo_loss:.4f}")

        scheduler.step()
        avg_total_loss = epoch_loss / max(1, total_steps)
        avg_single_loss = epoch_single_loss / max(1, single_sample_count)
        avg_combo_loss = epoch_combo_loss / max(1, combo_sample_count)
        print(f"[Joint Epoch {epoch}] total_loss={avg_total_loss:.4f} | single_loss={avg_single_loss:.4f} | combo_loss={avg_combo_loss:.4f}")
        _clear_cuda()

        joint_monitor = None
        single_test_metrics = None
        combo_test_metrics = None
        if epoch % evaluate_every == 0:
            single_test_metrics, _, _ = evaluate(
                model,
                single_test_loader,
                device,
                split_name=f"Joint-Single-Test@Epoch{epoch}",
                label_context=None,
                save_path=os.path.join(ckpt_dir, _prefixed_name(output_prefix, f"single_test_predictions_epoch_{epoch}.csv")),
                prediction_inverse_transform=None,
            )
            combo_test_metrics, _, _ = evaluate(
                model,
                combo_test_loader,
                device,
                split_name=f"Joint-Combo-Test@Epoch{epoch}",
                label_context=None,
                save_path=os.path.join(ckpt_dir, _prefixed_name(output_prefix, f"combo_test_predictions_epoch_{epoch}.csv")),
                prediction_inverse_transform=combo_prediction_inverse_transform,
            )
            joint_monitor = _combined_joint_monitor(single_test_metrics, combo_test_metrics, single_loss_weight, combo_loss_weight)
            print(f"[Joint Epoch {epoch}] Single monitor | {format_metrics(single_test_metrics)}")
            print(f"[Joint Epoch {epoch}] Combo  monitor | {format_metrics(combo_test_metrics)}")
            print(f"[Joint Epoch {epoch}] Combined monitor | Pearson={joint_monitor['pearson']:.4f} | RMSE={joint_monitor['rmse']:.4f}")
            _clear_cuda()

        if joint_monitor is not None and _is_better_checkpoint(joint_monitor, best_pearson=best_pearson, best_rmse=best_rmse):
            best_pearson = float(joint_monitor.get("pearson", float("nan")))
            best_rmse = float(joint_monitor.get("rmse", float("inf")))
            best_epoch = epoch
            wait = 0
            best_state = _snapshot_model_state(model)
            best_path = os.path.join(ckpt_dir, _prefixed_name(output_prefix, "best_model.pt"))
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": _snapshot_model_state(model),
                    "optimizer_state": _snapshot_optimizer_state(optimizer),
                    "best_pearson": best_pearson,
                    "best_rmse": best_rmse,
                    "single_metrics": single_test_metrics,
                    "combo_metrics": combo_test_metrics,
                },
                best_path,
            )
            print(f"✔ Saved joint best checkpoint: {best_path}")
        else:
            wait += 1
            if wait >= patience:
                print(f"⏹ Joint early stopping triggered at epoch {epoch}. Best epoch = {best_epoch}")
                break

        epoch_path = os.path.join(ckpt_dir, _prefixed_name(output_prefix, f"epoch_{epoch}.pt"))
        torch.save(
            {
                "epoch": epoch,
                "model_state": _snapshot_model_state(model),
                "optimizer_state": _snapshot_optimizer_state(optimizer),
            },
            epoch_path,
        )

    model.load_state_dict(best_state)

    print(f"\n▶ 使用联合训练最佳模型进行最终测试（best epoch = {best_epoch}）...")
    final_single_metrics, final_single_true, final_single_pred = evaluate(
        model,
        single_test_loader,
        device,
        split_name="Joint Final Single Test",
        label_context=None,
        save_path=os.path.join(ckpt_dir, _prefixed_name(output_prefix, "single_test_predictions.csv")),
        prediction_inverse_transform=None,
    )
    final_combo_metrics, final_combo_true, final_combo_pred = evaluate(
        model,
        combo_test_loader,
        device,
        split_name="Joint Final Combo Test",
        label_context=None,
        save_path=os.path.join(ckpt_dir, _prefixed_name(output_prefix, "combo_test_predictions.csv")),
        prediction_inverse_transform=combo_prediction_inverse_transform,
    )
    print(f"[Joint Final Single Test] {format_metrics(final_single_metrics)}")
    print(f"[Joint Final Combo Test] {format_metrics(final_combo_metrics)}")

    result = {
        "best_epoch": best_epoch,
        "best_pearson": best_pearson,
        "best_rmse": best_rmse,
        "single_test_metrics": final_single_metrics,
        "combo_test_metrics": final_combo_metrics,
        "single_y_true": final_single_true,
        "single_y_pred": final_single_pred,
        "combo_y_true": final_combo_true,
        "combo_y_pred": final_combo_pred,
    }

    if few_shot_dataset is not None and len(few_shot_dataset) >= 2:
        few_shot_prefix = f"{output_prefix}_{getattr(few_shot_dataset, 'mode', 'fewshot')}"
        result["few_shot_metrics"] = few_shot_adapt(
            model=model,
            dataset=few_shot_dataset,
            device=device,
            batch_size=few_shot_batch_size,
            lr=few_shot_lr,
            num_epochs=few_shot_epochs,
            support_ratio=few_shot_support_ratio,
            num_workers=num_workers,
            ckpt_dir=ckpt_dir,
            tune_bert=few_shot_tune_bert,
            output_prefix=few_shot_prefix,
            combo_label_transform=combo_label_transform,
            combo_label_scale=combo_label_scale,
        )

    print("🎉 Joint training finished.")
    return result
