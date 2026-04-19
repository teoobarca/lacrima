"""GIN (Graph Isomorphism Network) for tear-crystal graph classification.

Uses edge features via NNConv-style edge gating in custom blocks.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv, global_add_pool, global_mean_pool, global_max_pool,
)


class GINEBlock(nn.Module):
    """GINE conv with batchnorm + ReLU + dropout."""
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, dropout: float = 0.3):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.conv = GINEConv(mlp, edge_dim=edge_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr)
        x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class CGNN(nn.Module):
    """Crystal Graph Neural Network for tear-AFM disease classification."""
    def __init__(
        self,
        node_in: int = 5,
        edge_in: int = 5,
        hidden: int = 64,
        n_layers: int = 3,
        n_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.node_proj = nn.Linear(node_in, hidden)
        self.edge_proj = nn.Linear(edge_in, hidden)
        self.layers = nn.ModuleList([
            GINEBlock(hidden, hidden, edge_dim=hidden, dropout=dropout)
            for _ in range(n_layers)
        ])
        # Concat 3 pooling strategies
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, data):
        x = self.node_proj(data.x)
        edge_attr = self.edge_proj(data.edge_attr) if data.edge_attr.numel() > 0 else None
        for layer in self.layers:
            if edge_attr is not None and data.edge_index.shape[1] > 0:
                x = layer(x, data.edge_index, edge_attr)
            else:
                # Empty graph fallback — treat as MLP
                x = F.relu(x)
        # Global pool
        b = data.batch
        h_mean = global_mean_pool(x, b)
        h_max = global_max_pool(x, b)
        h_sum = global_add_pool(x, b)
        h = torch.cat([h_mean, h_max, h_sum], dim=-1)
        return self.head(h)
