# ============================================================
# main.py
# Entry point for unified multimodal drug response prediction (stable version)
# - keeps original framework / single-combo unified flow / few-shot interface unchanged
# - removes overly aggressive default CUDA runtime constraints that can destabilize GEMM on Windows
# - debug / strict-determinism flags are now opt-in via environment variables
# ============================================================

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "y", "on"}


if _env_flag("FJC_DEBUG_CUDA"):
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import numpy as np
import torch

from dataset import DEFAULT_ENST_END, DEFAULT_ENST_START, EnhancedMolDataset
from model import DrugResponsePredictor
from train import train, train_joint


def _configure_torch_runtime():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    if _env_flag("FJC_STRICT_DETERMINISM"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False

    if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


_configure_torch_runtime()


def parse_args():
    parser = argparse.ArgumentParser(description="Train unified multimodal drug response predictor")
    parser.add_argument("--csv_path", type=str, default=None, help="大规模训练集 CSV（single/combo 模式使用）")
    parser.add_argument("--joint_single_csv_path", type=str, default=None, help="联合训练时的单药训练集 CSV")
    parser.add_argument("--joint_combo_csv_path", type=str, default=None, help="联合训练时的多药训练集 CSV")
    parser.add_argument("--few_shot_csv_path", type=str, default=None, help="少量真实样本 CSV（可选）")
    parser.add_argument("--preprocessed_dir", type=str, default="./cache", help="预处理缓存目录")
    parser.add_argument("--reload", action="store_true", help="强制重新预处理")
    parser.add_argument("--mode", type=str, default="auto", choices=["single", "combo", "joint", "auto"], help="单药/多药/联合模式")
    parser.add_argument("--few_shot_mode", type=str, default="single", choices=["single", "combo", "auto"], help="few-shot 数据模式（joint 训练时使用）")
    parser.add_argument("--enst_start_col", type=str, default=DEFAULT_ENST_START, help="954维转录本起始列名")
    parser.add_argument("--enst_end_col", type=str, default=DEFAULT_ENST_END, help="954维转录本结束列名")
    parser.add_argument("--label_col", type=str, default=None, help="显式指定标签列名；单药默认 IC50，多药默认 Synergy_Score")
    parser.add_argument("--joint_single_label_col", type=str, default=None, help="联合训练时单药标签列名（可选）")
    parser.add_argument("--joint_combo_label_col", type=str, default=None, help="联合训练时多药标签列名（可选）")
    parser.add_argument("--text_model_name", type=str, default="bert-base-uncased", help="BERT 模型名称")
    parser.add_argument("--max_atoms", type=int, default=100)
    parser.add_argument("--max_text_len", type=int, default=128)
    parser.add_argument("--text_cache_batch_size", type=int, default=32, help="冻结 BERT 时预缓存文本特征的 CPU batch size")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="train/test 划分比例中的 train 比例")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--evaluate_every", type=int, default=1, help="每多少个 epoch 监控一次 test 指标")
    parser.add_argument("--patience", type=int, default=10, help="基于 train loss 的 early stopping")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--egnn_layers", type=int, default=3)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--feature_dropout", type=float, default=0.1)
    parser.add_argument("--bert_trainable", action="store_true", help="是否端到端训练 BERT")
    parser.add_argument("--few_shot_epochs", type=int, default=30)
    parser.add_argument("--few_shot_lr", type=float, default=5e-4)
    parser.add_argument("--few_shot_batch_size", type=int, default=8)
    parser.add_argument("--few_shot_support_ratio", type=float, default=0.8)
    parser.add_argument("--few_shot_tune_bert", action="store_true", help="few-shot 阶段是否打开 BERT 最后两层")
    parser.add_argument(
        "--combo_label_transform",
        type=str,
        default="none",
        choices=["none", "signed_log1p"],
        help="仅对多药任务标签做变换；single 不受影响"
    )
    parser.add_argument(
        "--combo_label_scale",
        type=float,
        default=5.0,
        help="signed_log1p 变换中的尺度 c"
    )
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(args, csv_path: str, reload: bool = False, mode_override: str = None, label_col_override: str = None):
    dataset_mode = mode_override if mode_override is not None else args.mode
    label_col = label_col_override if label_col_override is not None else args.label_col
    return EnhancedMolDataset(
        csv_path=csv_path,
        enst_start_col=args.enst_start_col,
        enst_end_col=args.enst_end_col,
        mode=dataset_mode,
        preprocessed_dir=args.preprocessed_dir,
        reload=reload,
        text_model_name=args.text_model_name,
        max_atoms=args.max_atoms,
        max_text_len=args.max_text_len,
        label_col=label_col,
        cache_text_embeddings=(not args.bert_trainable),
        text_cache_batch_size=args.text_cache_batch_size,
    )


def _validate_single_combo_compatibility(single_dataset, combo_dataset):
    single_sample = single_dataset[0]
    combo_sample = combo_dataset[0]
    single_cell_dim = int(single_sample["cell_feat"].shape[0])
    combo_cell_dim = int(combo_sample["cell_feat"].shape[0])
    if single_cell_dim != combo_cell_dim:
        raise ValueError(f"❌ joint 训练要求单药/多药数据的 cell_dim 一致，当前为 {single_cell_dim} vs {combo_cell_dim}")
    if int(single_dataset.atom_feature_dim) != int(combo_dataset.atom_feature_dim):
        raise ValueError(
            f"❌ joint 训练要求单药/多药数据的 atom_feature_dim 一致，当前为 {single_dataset.atom_feature_dim} vs {combo_dataset.atom_feature_dim}"
        )
    return single_cell_dim



def main():
    args = parse_args()
    set_seed(args.seed)

    if args.mode == "joint":
        if not args.joint_single_csv_path or not args.joint_combo_csv_path:
            raise ValueError("❌ mode=joint 时必须同时提供 --joint_single_csv_path 和 --joint_combo_csv_path")

        print("▶ Loading joint single-task dataset...")
        single_dataset = build_dataset(
            args,
            args.joint_single_csv_path,
            reload=args.reload,
            mode_override="single",
            label_col_override=args.joint_single_label_col,
        )
        print(f"▶ Joint single mode: {single_dataset.mode} | label: {single_dataset.label_col}")

        print("▶ Loading joint combo-task dataset...")
        combo_dataset = build_dataset(
            args,
            args.joint_combo_csv_path,
            reload=args.reload,
            mode_override="combo",
            label_col_override=args.joint_combo_label_col,
        )
        print(f"▶ Joint combo mode: {combo_dataset.mode} | label: {combo_dataset.label_col}")

        cell_dim = _validate_single_combo_compatibility(single_dataset, combo_dataset)
        print(
            f"▶ Inferred dims | atom_feature_dim={single_dataset.atom_feature_dim}, cell_dim={cell_dim}"
        )

        print("▶ Building model...")
        model = DrugResponsePredictor(
            cell_dim=cell_dim,
            atom_feature_dim=single_dataset.atom_feature_dim,
            hidden_dim=args.hidden_dim,
            text_model_name=args.text_model_name,
            egnn_layers=args.egnn_layers,
            num_layers=args.num_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
            feature_dropout=args.feature_dropout,
            bert_trainable=args.bert_trainable,
        )
        print("▶ Model initialized.")

        few_shot_dataset = None
        if args.few_shot_csv_path:
            print("▶ Loading few-shot dataset...")
            few_shot_dataset = build_dataset(
                args,
                args.few_shot_csv_path,
                reload=args.reload,
                mode_override=args.few_shot_mode,
                label_col_override=args.label_col,
            )

        print("▶ Start joint training...")
        train_joint(
            model=model,
            single_dataset=single_dataset,
            combo_dataset=combo_dataset,
            train_ratio=args.train_ratio,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_epochs=args.epochs,
            num_workers=args.num_workers,
            device=args.device,
            ckpt_dir=args.ckpt_dir,
            evaluate_every=args.evaluate_every,
            use_wandb=False,
            seed=args.seed,
            resume_from=args.resume_from,
            patience=args.patience,
            few_shot_dataset=few_shot_dataset,
            few_shot_epochs=args.few_shot_epochs,
            few_shot_lr=args.few_shot_lr,
            few_shot_batch_size=args.few_shot_batch_size,
            few_shot_support_ratio=args.few_shot_support_ratio,
            few_shot_tune_bert=args.few_shot_tune_bert,
            output_prefix="joint",
            combo_label_transform=args.combo_label_transform,
            combo_label_scale=args.combo_label_scale,
        )
        return

    if not args.csv_path:
        raise ValueError("❌ 当前模式下必须提供 --csv_path")

    print("▶ Loading training dataset...")
    dataset = build_dataset(args, args.csv_path, reload=args.reload)
    sample = dataset[0]
    cell_dim = sample["cell_feat"].shape[0]

    print(f"▶ Dataset mode: {dataset.mode}")
    print(f"▶ Inferred label column: {dataset.label_col}")
    print(f"▶ Inferred dims | atom_feature_dim={dataset.atom_feature_dim}, cell_dim={cell_dim}")

    print("▶ Building model...")
    model = DrugResponsePredictor(
        cell_dim=cell_dim,
        atom_feature_dim=dataset.atom_feature_dim,
        hidden_dim=args.hidden_dim,
        text_model_name=args.text_model_name,
        egnn_layers=args.egnn_layers,
        num_layers=args.num_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        feature_dropout=args.feature_dropout,
        bert_trainable=args.bert_trainable,
    )
    print("▶ Model initialized.")

    few_shot_dataset = None
    if args.few_shot_csv_path:
        print("▶ Loading few-shot dataset...")
        few_shot_dataset = build_dataset(
            args,
            args.few_shot_csv_path,
            reload=args.reload,
            mode_override=dataset.mode,
            label_col_override=args.label_col,
        )
        if few_shot_dataset.mode != dataset.mode:
            raise ValueError(f"❌ few-shot 数据模式 ({few_shot_dataset.mode}) 与主训练数据模式 ({dataset.mode}) 不一致。")

    print("▶ Start training...")
    train(
        model=model,
        dataset=dataset,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        num_workers=args.num_workers,
        device=args.device,
        ckpt_dir=args.ckpt_dir,
        evaluate_every=args.evaluate_every,
        use_wandb=False,
        seed=args.seed,
        resume_from=args.resume_from,
        patience=args.patience,
        few_shot_dataset=few_shot_dataset,
        few_shot_epochs=args.few_shot_epochs,
        few_shot_lr=args.few_shot_lr,
        few_shot_batch_size=args.few_shot_batch_size,
        few_shot_support_ratio=args.few_shot_support_ratio,
        few_shot_tune_bert=args.few_shot_tune_bert,
        output_prefix=dataset.mode,
        combo_label_transform=args.combo_label_transform,
        combo_label_scale=args.combo_label_scale,
    )


if __name__ == "__main__":
    main()
