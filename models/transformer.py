"""
models/transformer.py
=======================
Spectral-Spatial Transformer — the project's final proposed model, designed
for the 10%-training low-data regime.

    HSI patch (P x P x B)
           v
    Spectral-spatial patches/tokens
           v
    Linear embedding
           v
    Positional encoding
           v
    Transformer Encoder (multi-head self-attention, residual + LayerNorm)
           v
    CLS token / global pooling
           v
    MLP classifier
           v
    Class

Design choices for the low-data regime
---------------------------------------
* Small model capacity (embed_dim, depth, heads all configurable and kept
  modest by default) to avoid overfitting with only 10% labeled data.
* Dropout on attention, MLP, and embeddings + weight decay (set in
  TrainConfig/AdamW) + stochastic depth (DropPath) all combine to regularize.
* Spectral-spatial tokenization (rather than pure ViT-style flat-patch
  tokens) injects a useful inductive bias: each token already mixes a small
  spatial neighborhood with a slice of the spectral dimension, so the model
  needs fewer samples to learn meaningful groupings.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Spectral-spatial tokenization
# ----------------------------------------------------------------------
class SpectralSpatialTokenizer(nn.Module):
    """Splits a (B, C, P, P) patch (C = spectral bands) into overlapping-free
    spatial sub-patches of size `token_size x token_size`, each covering the
    full spectral depth. Each sub-patch becomes one token, i.e. tokens mix
    spatial neighborhood + spectral signature ("spectral-spatial tokens").
    """

    def __init__(self, num_bands: int, patch_size: int, token_size: int, embed_dim: int):
        super().__init__()
        assert patch_size % token_size == 0, (
            f"patch_size ({patch_size}) must be divisible by token_size "
            f"({token_size}); adjust config.TRANSFORMER.token_size or "
            f"config.DATA.patch_size."
        )
        self.token_size = token_size
        self.grid = patch_size // token_size
        self.num_tokens = self.grid * self.grid

        token_dim = num_bands * token_size * token_size
        self.proj = nn.Linear(token_dim, embed_dim)

    def forward(self, x):
        # x: (B, C, P, P)
        B, C, P, _ = x.shape
        t = self.token_size
        g = self.grid
        # unfold into non-overlapping spatial blocks
        x = x.unfold(2, t, t).unfold(3, t, t)          # (B, C, g, g, t, t)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()     # (B, g, g, C, t, t)
        x = x.view(B, g * g, C * t * t)                  # (B, num_tokens, C*t*t)
        tokens = self.proj(x)                             # (B, num_tokens, embed_dim)
        return tokens


# ----------------------------------------------------------------------
# Positional encoding
# ----------------------------------------------------------------------
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, num_tokens: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        return x + self.pos_embed


def sinusoidal_positional_encoding(num_tokens: int, embed_dim: int) -> torch.Tensor:
    pe = torch.zeros(num_tokens, embed_dim)
    position = torch.arange(0, num_tokens, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, num_tokens, embed_dim)


# ----------------------------------------------------------------------
# Stochastic depth (DropPath) — helps regularize small-data training
# ----------------------------------------------------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ----------------------------------------------------------------------
# Multi-head self-attention
# ----------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float, proj_dropout: float):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                    # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ----------------------------------------------------------------------
# Transformer encoder block: pre-LN, residual, MHSA + MLP
# ----------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout, attn_dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_dropout, dropout)
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


# ----------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------
class HSITransformer(nn.Module):
    def __init__(self, num_bands: int, patch_size: int, num_classes: int, cfg):
        super().__init__()
        self.tokenizer = SpectralSpatialTokenizer(
            num_bands, patch_size, cfg.token_size, cfg.embed_dim
        )
        num_tokens = self.tokenizer.num_tokens

        self.use_cls_token = cfg.use_cls_token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            num_tokens += 1

        if cfg.pos_encoding == "learnable":
            self.pos_encoding = LearnablePositionalEncoding(num_tokens, cfg.embed_dim)
        else:
            pe = sinusoidal_positional_encoding(num_tokens, cfg.embed_dim)
            self.register_buffer("pos_encoding_buf", pe)
            self.pos_encoding = lambda x: x + self.pos_encoding_buf

        self.embed_dropout = nn.Dropout(cfg.dropout)

        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path, cfg.depth)]
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio,
                cfg.dropout, cfg.attn_dropout, dpr[i]
            ) for i in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim)

        self.head = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_dim // 2, num_classes),
        )

    def forward(self, x):
        # x: (B, C, P, P)  where C = num_bands
        tokens = self.tokenizer(x)                          # (B, N, D)

        if self.use_cls_token:
            cls_tok = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls_tok, tokens], dim=1)     # (B, N+1, D)

        tokens = self.pos_encoding(tokens)
        tokens = self.embed_dropout(tokens)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        if self.use_cls_token:
            pooled = tokens[:, 0]                            # CLS token
        else:
            pooled = tokens.mean(dim=1)                      # global average pooling

        return self.head(pooled)
