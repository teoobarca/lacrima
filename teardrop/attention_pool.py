"""Learnable attention-pooling over tile embeddings.

Context: v4 champion pools 9 tile embeddings per scan with a simple mean. Some
tiles show crystallization patterns, others are background. This module learns
to weight tiles, substituting `mean(X_tiles)` with a softmax-weighted sum.

Architecture (kept deliberately small — 240 scans, overfitting risk is real):

    W_h: Linear(D, hidden) with tanh
    W_a: Linear(hidden, 1)
    attention_scores = W_a(tanh(W_h(X)))   # (n_tiles, 1)
    weights = softmax(attention_scores)    # (n_tiles, 1)
    scan_emb = sum_i(weights_i * X_i)      # (D,)

The attention head is followed by a linear classifier (5-way). Trained jointly
with cross-entropy. Supports:
- variable tile counts per scan (masking via -inf before softmax)
- dropout on attention weights
- optional L2-normalization of the pooled output (mirrors v2/v4 recipe)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class TileAttentionPool(nn.Module):
    """Gated attention pooling (Ilse et al. 2018 style) over a bag of tiles.

    Args:
        embed_dim: tile embedding dim D.
        hidden_dim: bottleneck for attention scoring (kept small; 32-64).
        dropout: dropout rate applied to attention logits before softmax.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = float(dropout)
        self.W_h = nn.Linear(embed_dim, hidden_dim)
        self.W_a = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(
        self,
        X: torch.Tensor,              # (B, T, D)
        mask: torch.Tensor | None = None,  # (B, T) bool, True = valid tile
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (pooled (B, D), weights (B, T))."""
        logits = self.W_a(torch.tanh(self.W_h(X))).squeeze(-1)   # (B, T)
        logits = self.dropout(logits)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        w = F.softmax(logits, dim=-1)                             # (B, T)
        pooled = torch.einsum("btd,bt->bd", X, w)
        return pooled, w


@dataclass
class AttentionClassifierConfig:
    embed_dim: int
    n_classes: int = 5
    hidden_dim: int = 64
    dropout: float = 0.3
    l2_normalize_pool: bool = True


class TileAttentionClassifier(nn.Module):
    """Attention pool + linear head (5-way softmax classifier)."""

    def __init__(self, cfg: AttentionClassifierConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pool = TileAttentionPool(
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        )
        self.classifier = nn.Linear(cfg.embed_dim, cfg.n_classes)

    def forward(
        self,
        X: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (logits (B, C), pooled (B, D), weights (B, T))."""
        pooled, w = self.pool(X, mask)
        if self.cfg.l2_normalize_pool:
            pooled = F.normalize(pooled, p=2, dim=-1)
        logits = self.classifier(pooled)
        return logits, pooled, w

    @torch.no_grad()
    def extract_pooled(
        self,
        X: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inference-time scan-level embedding extraction (no classifier)."""
        self.eval()
        pooled, _ = self.pool(X, mask)
        if self.cfg.l2_normalize_pool:
            pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled
