# ============================================================
# ic50_transformer.py
# Stable non-diffusion response backbone
# - keeps the original token-level transformer structure
# - replaces nn.TransformerEncoderLayer with an explicit fp32 attention path
# - replaces trainable nn.Linear hotpaths with StableLinear to avoid cublasLt-heavy kernels
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class StableMultiHeadSelfAttention(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.size()
        x = _safe_tensor(x, clamp_val=50.0)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = _safe_tensor(q, clamp_val=50.0).view(bsz, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = _safe_tensor(k, clamp_val=50.0).view(bsz, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = _safe_tensor(v, clamp_val=50.0).view(bsz, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        q2 = q.reshape(bsz * self.n_heads, seq_len, self.head_dim).float()
        k2 = k.reshape(bsz * self.n_heads, seq_len, self.head_dim).float()
        v2 = v.reshape(bsz * self.n_heads, seq_len, self.head_dim).float()

        scores = torch.bmm(q2, k2.transpose(1, 2)) / math.sqrt(self.head_dim)
        scores = torch.clamp(scores, min=-80.0, max=80.0)
        attn = F.softmax(scores, dim=-1, dtype=torch.float32)
        attn = self.attn_dropout(attn)

        out = torch.bmm(attn, v2).to(x.dtype)
        out = out.reshape(bsz, self.n_heads, seq_len, self.head_dim).permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, self.hidden_dim)
        out = self.out_proj(_safe_tensor(out, clamp_val=50.0))
        out = self.out_dropout(out)
        return _safe_tensor(out, clamp_val=50.0)


class StableTransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = StableMultiHeadSelfAttention(hidden_dim=hidden_dim, n_heads=n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            StableLinear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            StableLinear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out = self.self_attn(attn_in)
        x = _safe_tensor(x + attn_out, clamp_val=50.0)

        ffn_in = self.norm2(x)
        ffn_out = self.ffn(ffn_in)
        x = _safe_tensor(x + ffn_out, clamp_val=50.0)
        return x


class StableTransformerEncoder(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            StableTransformerEncoderLayer(hidden_dim=hidden_dim, n_heads=n_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ResponseTransformerBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        feature_dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            StableLinear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(feature_dropout),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 5, hidden_dim))
        self.token_dropout = nn.Dropout(dropout)

        self.encoder = StableTransformerEncoder(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

        bottleneck = max(hidden_dim // 4, 32)
        self.fewshot_adapter = nn.Sequential(
            StableLinear(hidden_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            StableLinear(bottleneck, hidden_dim),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            StableLinear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            StableLinear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for _, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        print("✅ ResponseTransformerBackbone initialized (stable explicit-attention version)")

    def forward(self, tokens: torch.Tensor, return_hidden: bool = False):
        x = self.input_proj(tokens)
        x = _safe_tensor(x, clamp_val=50.0)

        bsz = x.size(0)
        cls = self.cls_token.expand(bsz, -1, -1)

        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.size(1)]
        x = _safe_tensor(x, clamp_val=50.0)

        x = self.token_dropout(x)
        x = self.encoder(x)
        x = self.final_norm(x)
        x = _safe_tensor(x, clamp_val=50.0)

        pooled = x[:, 0]
        pooled = pooled + self.fewshot_adapter(pooled)
        pooled = _safe_tensor(pooled, clamp_val=50.0)

        pred = self.head(pooled)
        pred = _safe_tensor(pred, clamp_val=1e3)

        if return_hidden:
            return pred, pooled
        return pred
