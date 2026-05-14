# ============================================================
# model.py
# Unified multimodal drug response model (stable non-diffusion version)
# - keeps the original framework flow unchanged
# - replaces nn.MultiheadAttention with an explicit stable cross-attention path
# - replaces trainable nn.Linear hotpaths with StableLinear to avoid cublasLt-heavy kernels
# ============================================================

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from ic50_transformer import ResponseTransformerBackbone


def _safe_tensor(x: torch.Tensor, clamp_val: float = 100.0) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(x, min=-clamp_val, max=clamp_val)


class StableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _safe_tensor(x, clamp_val=50.0)
        orig_shape = x.shape
        x2 = x.reshape(-1, self.in_features).contiguous().to(dtype=self.weight.dtype)
        out = F.linear(x2, self.weight, self.bias)
        out = out.reshape(*orig_shape[:-1], self.out_features)
        return _safe_tensor(out, clamp_val=50.0)


class EGNNLayer(nn.Module):
    """
    稳定版 EGNN：
    1. 保留 3D 坐标计算 pairwise distance
    2. 不再执行坐标更新，避免 CUDA illegal instruction
    3. 节点聚合使用 masked mean，数值更稳定
    4. 线性层改为 StableLinear，规避剩余的 Lt/GEMM 不稳定路径
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            StableLinear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            StableLinear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            StableLinear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            StableLinear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor, coords: torch.Tensor, mask: Optional[torch.Tensor] = None):
        bsz, n_atoms, _ = h.size()

        h = _safe_tensor(h, clamp_val=50.0)
        coords = _safe_tensor(coords, clamp_val=20.0)

        h_i = h.unsqueeze(2).expand(-1, -1, n_atoms, -1)
        h_j = h.unsqueeze(1).expand(-1, n_atoms, -1, -1)

        coord_diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist2 = (coord_diff.float() ** 2).sum(dim=-1, keepdim=True)
        dist2 = torch.clamp(dist2, min=0.0, max=400.0)
        dist2 = _safe_tensor(dist2, clamp_val=400.0)

        if mask is not None:
            mask = mask.float()
            edge_mask = (mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(-1)
        else:
            edge_mask = None

        edge_input = torch.cat([h_i, h_j, dist2.to(h.dtype)], dim=-1).contiguous()
        edge_input = _safe_tensor(edge_input, clamp_val=50.0)
        if edge_mask is not None:
            edge_input = edge_input * edge_mask

        messages = self.edge_mlp(edge_input)
        messages = _safe_tensor(messages, clamp_val=50.0)

        if edge_mask is not None:
            messages = messages * edge_mask
            neighbor_count = edge_mask.sum(dim=2).clamp(min=1.0)
        else:
            neighbor_count = torch.full(
                (bsz, n_atoms, 1),
                fill_value=max(n_atoms, 1),
                device=h.device,
                dtype=h.dtype,
            )

        m_agg = messages.sum(dim=2) / neighbor_count
        m_agg = _safe_tensor(m_agg, clamp_val=50.0)

        node_input = torch.cat([h, m_agg], dim=-1)
        node_input = _safe_tensor(node_input, clamp_val=50.0)

        h = self.node_mlp(node_input)
        h = _safe_tensor(h, clamp_val=50.0)

        if mask is not None:
            h = h * mask.unsqueeze(-1)
            coords = coords * mask.unsqueeze(-1)

        coords = _safe_tensor(coords, clamp_val=20.0)
        return h, coords


class MoleculeEncoder(nn.Module):
    def __init__(self, atom_feature_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        self.input_proj = StableLinear(atom_feature_dim, hidden_dim)
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(num_layers)])
        self.output_proj = nn.Sequential(
            StableLinear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, atom_feat: torch.Tensor, atom_coords: torch.Tensor, atom_mask: torch.Tensor):
        atom_mask = atom_mask.float()
        h = self.input_proj(atom_feat)
        h = _safe_tensor(h, clamp_val=50.0)
        coords = _safe_tensor(atom_coords, clamp_val=20.0)

        for layer in self.layers:
            h, coords = layer(h, coords, atom_mask)

        mask_sum = atom_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (h * atom_mask.unsqueeze(-1)).sum(dim=1) / mask_sum
        pooled = _safe_tensor(pooled, clamp_val=50.0)

        out = self.output_proj(pooled)
        return _safe_tensor(out, clamp_val=50.0)


class BertTextEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", trainable: bool = False):
        super().__init__()

        try:
            self.config = AutoConfig.from_pretrained(model_name, local_files_only=True)
            print(f"✅ BERT config loaded from local cache: {model_name}")
        except Exception:
            self.config = AutoConfig.from_pretrained(model_name)
            print(f"✅ BERT config loaded with online fallback: {model_name}")

        if hasattr(self.config, "_attn_implementation"):
            self.config._attn_implementation = "eager"

        try:
            try:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    config=self.config,
                    attn_implementation="eager",
                    local_files_only=True,
                )
                print(f"✅ BERT weights loaded from local cache: {model_name}")
            except TypeError:
                self.model = AutoModel.from_pretrained(model_name, config=self.config, local_files_only=True)
                if hasattr(self.model.config, "_attn_implementation"):
                    self.model.config._attn_implementation = "eager"
                print(f"✅ BERT weights loaded from local cache: {model_name}")
        except Exception:
            try:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    config=self.config,
                    attn_implementation="eager",
                )
                print(f"✅ BERT weights loaded with online fallback: {model_name}")
            except TypeError:
                self.model = AutoModel.from_pretrained(model_name, config=self.config)
                if hasattr(self.model.config, "_attn_implementation"):
                    self.model.config._attn_implementation = "eager"
                print(f"✅ BERT weights loaded with online fallback: {model_name}")

        if hasattr(self.model, "set_attn_implementation"):
            try:
                self.model.set_attn_implementation("eager")
            except Exception:
                pass

        self.hidden_size = self.config.hidden_size
        self.trainable = trainable
        self.set_trainable(trainable)

    def set_trainable(self, trainable: bool = False):
        self.trainable = trainable
        for param in self.model.parameters():
            param.requires_grad = trainable

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        input_ids = input_ids.contiguous()
        attention_mask = attention_mask.to(dtype=torch.long).contiguous()

        if self.trainable:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            self.model.eval()
            with torch.inference_mode():
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return _safe_tensor(out.last_hidden_state, clamp_val=50.0)


class StableCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} 必须能被 n_heads={n_heads} 整除。")

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.q_proj = StableLinear(hidden_dim, hidden_dim)
        self.k_proj = StableLinear(hidden_dim, hidden_dim)
        self.v_proj = StableLinear(hidden_dim, hidden_dim)
        self.out_proj = StableLinear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, q_len, _ = q.size()
        kv_len = kv.size(1)

        q = self.q_proj(_safe_tensor(q, clamp_val=50.0))
        k = self.k_proj(_safe_tensor(kv, clamp_val=50.0))
        v = self.v_proj(_safe_tensor(kv, clamp_val=50.0))

        q = q.view(bsz, q_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.view(bsz, kv_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.view(bsz, kv_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        q2 = q.reshape(bsz * self.n_heads, q_len, self.head_dim).float()
        k2 = k.reshape(bsz * self.n_heads, kv_len, self.head_dim).float()
        v2 = v.reshape(bsz * self.n_heads, kv_len, self.head_dim).float()

        scores = torch.bmm(q2, k2.transpose(1, 2)) / math.sqrt(self.head_dim)
        scores = torch.clamp(scores, min=-80.0, max=80.0)

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, :].expand(bsz, self.n_heads, kv_len).reshape(bsz * self.n_heads, 1, kv_len).to(dtype=torch.bool)
            scores = scores.masked_fill(mask, -1e4)

        attn = F.softmax(scores, dim=-1, dtype=torch.float32)
        attn = self.attn_dropout(attn)

        out = torch.bmm(attn, v2).to(q.dtype)
        out = out.reshape(bsz, self.n_heads, q_len, self.head_dim).permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.hidden_dim)
        out = self.out_proj(_safe_tensor(out, clamp_val=50.0))
        out = self.out_dropout(out)
        return _safe_tensor(out, clamp_val=50.0)


class GraphConditionedTextAttention(nn.Module):
    def __init__(self, mol_dim: int, text_dim: int, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mol_proj = StableLinear(mol_dim, hidden_dim)
        self.text_proj = StableLinear(text_dim, hidden_dim)

        self.cross_attn = StableCrossAttention(hidden_dim=hidden_dim, n_heads=n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            StableLinear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            StableLinear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.gate = nn.Sequential(
            StableLinear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            StableLinear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, mol_feat: torch.Tensor, text_seq: torch.Tensor, text_mask: torch.Tensor):
        q = self.mol_proj(mol_feat).unsqueeze(1)
        kv = self.text_proj(text_seq)

        q = _safe_tensor(q, clamp_val=50.0)
        kv = _safe_tensor(kv, clamp_val=50.0)

        key_padding_mask = (text_mask == 0) if text_mask is not None else None
        attn_out = self.cross_attn(q, kv, key_padding_mask=key_padding_mask)
        attn_out = _safe_tensor(attn_out, clamp_val=50.0)

        x = self.norm1(q + attn_out)
        x = _safe_tensor(x, clamp_val=50.0)

        ffn_out = self.ffn(x)
        ffn_out = _safe_tensor(ffn_out, clamp_val=50.0)

        x = self.norm2(x + ffn_out)
        x = x.squeeze(1)
        x = _safe_tensor(x, clamp_val=50.0)

        mol_token = _safe_tensor(q.squeeze(1), clamp_val=50.0)
        gate_in = torch.cat([mol_token, x], dim=-1)
        gate_in = _safe_tensor(gate_in, clamp_val=50.0)

        gate = self.gate(gate_in)
        fused = gate * x + (1.0 - gate) * mol_token
        return _safe_tensor(fused, clamp_val=50.0)


class UnifiedDrugPairEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.null_drug = nn.Parameter(torch.zeros(1, hidden_dim))
        self.mlp = nn.Sequential(
            StableLinear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            StableLinear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, drug1: torch.Tensor, drug2: Optional[torch.Tensor] = None):
        if drug2 is None:
            drug2 = self.null_drug.expand(drug1.size(0), -1)
        pair_feat = torch.cat([drug1, drug2, torch.abs(drug1 - drug2), drug1 * drug2], dim=-1)
        pair_feat = _safe_tensor(pair_feat, clamp_val=50.0)
        out = self.mlp(pair_feat)
        return _safe_tensor(out, clamp_val=50.0)


class ConditionalContextFusion(nn.Module):
    def __init__(self, cell_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.drug_proj = nn.Sequential(
            StableLinear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.cell_proj = nn.Sequential(
            StableLinear(cell_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.task_embed = nn.Embedding(2, hidden_dim)
        self.label_context_proj = nn.Sequential(
            StableLinear(1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.null_label_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.token_dropout = nn.Dropout(dropout)

    def forward(
        self,
        drug_feat: torch.Tensor,
        cell_feat: torch.Tensor,
        task_id: torch.Tensor,
        label_context: Optional[torch.Tensor] = None,
    ):
        drug_token = self.drug_proj(_safe_tensor(drug_feat, clamp_val=50.0))
        cell_token = self.cell_proj(_safe_tensor(cell_feat, clamp_val=50.0))
        task_token = self.task_embed(task_id)

        if label_context is None:
            label_token = self.null_label_token.expand(drug_feat.size(0), -1)
        else:
            label_token = self.label_context_proj(_safe_tensor(label_context, clamp_val=20.0))

        tokens = torch.stack([drug_token, cell_token, task_token, label_token], dim=1)
        tokens = _safe_tensor(tokens, clamp_val=50.0)
        return self.token_dropout(tokens)


class DrugResponsePredictor(nn.Module):
    def __init__(
        self,
        cell_dim: int,
        atom_feature_dim: int,
        hidden_dim: int = 256,
        text_model_name: str = "bert-base-uncased",
        egnn_layers: int = 3,
        num_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        feature_dropout: float = 0.1,
        bert_trainable: bool = False,
    ):
        super().__init__()

        self.molecule_encoder = MoleculeEncoder(
            atom_feature_dim=atom_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=egnn_layers,
        )
        self.text_encoder = BertTextEncoder(
            model_name=text_model_name,
            trainable=bert_trainable,
        )
        self.drug_fusion = GraphConditionedTextAttention(
            mol_dim=hidden_dim,
            text_dim=self.text_encoder.hidden_size,
            hidden_dim=hidden_dim,
            n_heads=max(1, min(4, n_heads)),
            dropout=dropout,
        )
        self.pair_encoder = UnifiedDrugPairEncoder(hidden_dim=hidden_dim, dropout=dropout)
        self.context_fusion = ConditionalContextFusion(cell_dim=cell_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.backbone = ResponseTransformerBackbone(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            n_heads=n_heads,
            dropout=dropout,
            feature_dropout=feature_dropout,
        )
        self.loss_fn = nn.SmoothL1Loss(beta=1.0)

    def _encode_drug(
        self,
        atom_feat: torch.Tensor,
        atom_coords: torch.Tensor,
        atom_mask: torch.Tensor,
        input_ids: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        text_hidden: Optional[torch.Tensor] = None,
    ):
        mol_feat = self.molecule_encoder(atom_feat, atom_coords, atom_mask)
        if text_hidden is not None and not self.text_encoder.trainable:
            text_seq = _safe_tensor(text_hidden, clamp_val=50.0)
        else:
            if input_ids is None:
                raise RuntimeError("❌ 当前需要在线 BERT 前向，但 batch 中缺少 input_ids。")
            text_seq = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.drug_fusion(mol_feat, text_seq, attention_mask)

    def _resolve_mode(self, batch) -> str:
        mode = batch.get("mode", "single")
        if isinstance(mode, (list, tuple)):
            return mode[0]
        return mode

    def _build_tokens(self, batch, label_context: Optional[torch.Tensor] = None):
        mode = self._resolve_mode(batch)
        task_id = batch["task_id"]
        if task_id.dim() == 0:
            task_id = task_id.view(1)
        task_id = task_id.long()

        if mode == "single":
            drug1 = self._encode_drug(
                batch["atom_feat"],
                batch["atom_coords"],
                batch["atom_mask"],
                batch.get("input_ids"),
                batch["attention_mask"],
                batch.get("text_hidden"),
            )
            drug_pair_feat = self.pair_encoder(drug1, None)
        else:
            drug1 = self._encode_drug(
                batch["atom_feat_1"],
                batch["atom_coords_1"],
                batch["atom_mask_1"],
                batch.get("input_ids_1"),
                batch["attention_mask_1"],
                batch.get("text_hidden_1"),
            )
            drug2 = self._encode_drug(
                batch["atom_feat_2"],
                batch["atom_coords_2"],
                batch["atom_mask_2"],
                batch.get("input_ids_2"),
                batch["attention_mask_2"],
                batch.get("text_hidden_2"),
            )
            drug_pair_feat = self.pair_encoder(drug1, drug2)

        if label_context is not None and label_context.dim() == 1:
            label_context = label_context.unsqueeze(-1)

        tokens = self.context_fusion(
            drug_feat=drug_pair_feat,
            cell_feat=batch["cell_feat"],
            task_id=task_id,
            label_context=label_context,
        )
        return tokens

    def forward(self, batch, label_context: Optional[torch.Tensor] = None):
        tokens = self._build_tokens(batch, label_context=label_context)
        tokens = _safe_tensor(tokens, clamp_val=50.0)
        pred = self.backbone(tokens)
        return _safe_tensor(pred, clamp_val=1e3)

    def compute_loss(self, batch, label_context: Optional[torch.Tensor] = None):
        pred = self.forward(batch, label_context=label_context)
        target = batch["label"]

        if not torch.isfinite(pred).all():
            raise RuntimeError("❌ pred 中出现 NaN / Inf")
        if not torch.isfinite(target).all():
            raise RuntimeError("❌ target 中出现 NaN / Inf")

        loss = self.loss_fn(pred, target)
        mse = F.mse_loss(pred, target)
        mae = F.l1_loss(pred, target)
        return {
            "loss": loss,
            "smooth_l1": loss.detach(),
            "mse": mse.detach(),
            "mae": mae.detach(),
        }

    @torch.no_grad()
    def predict(self, batch, label_context: Optional[torch.Tensor] = None):
        return self.forward(batch, label_context=label_context)

    def freeze_for_few_shot(self, tune_bert: bool = False):
        """
        Few-shot 阶段采用“表征适配”而不是仅调最后回归头：
        1. 取消全模型训练，避免少量样本破坏大模型稳定性；
        2. 释放与任务分布重建最相关的模块：
           - 药物跨模态融合层（drug_fusion）
           - 药物对编码层（pair_encoder）
           - 条件融合层（context_fusion）
           - 分子编码器最后一层/输出层
           - Backbone 的输入投影、最后一层编码器、归一化、few-shot adapter 与 head
        3. 默认仍冻结 BERT，仅在 tune_bert=True 时放开最后两层。
        这样 few-shot 能真正改变样本表征，而不只是把输出往均值附近拉。
        """
        for param in self.parameters():
            param.requires_grad = False

        self.text_encoder.trainable = bool(tune_bert)

        modules_to_tune = [
            self.drug_fusion,
            self.pair_encoder,
            self.context_fusion,
            self.molecule_encoder.output_proj,
            self.backbone.input_proj,
            self.backbone.final_norm,
            self.backbone.fewshot_adapter,
            self.backbone.head,
        ]

        if len(self.molecule_encoder.layers) > 0:
            modules_to_tune.append(self.molecule_encoder.layers[-1])
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layers") and len(self.backbone.encoder.layers) > 0:
            modules_to_tune.append(self.backbone.encoder.layers[-1])

        for module in modules_to_tune:
            for param in module.parameters():
                param.requires_grad = True

        self.context_fusion.null_label_token.requires_grad = True

        if tune_bert:
            if hasattr(self.text_encoder.model, "embeddings"):
                for param in self.text_encoder.model.embeddings.parameters():
                    param.requires_grad = False
            if hasattr(self.text_encoder.model, "encoder") and hasattr(self.text_encoder.model.encoder, "layer"):
                for layer in self.text_encoder.model.encoder.layer[:-2]:
                    for param in layer.parameters():
                        param.requires_grad = False
                for layer in self.text_encoder.model.encoder.layer[-2:]:
                    for param in layer.parameters():
                        param.requires_grad = True

    def unfreeze_all(self):
        self.text_encoder.trainable = True
        for param in self.parameters():
            param.requires_grad = True


DrugResponseContinuousDiffusion = DrugResponsePredictor
