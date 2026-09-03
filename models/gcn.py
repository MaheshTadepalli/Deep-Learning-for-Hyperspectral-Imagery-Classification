"""
models/gcn.py
==============
Graph Convolutional Network for hyperspectral classification.

- Nodes            = spatial-spectral patches (same patches used by the other
                      models; each node's feature = flattened/pooled spectral
                      vector of its patch center, optionally averaged over a
                      small spatial window for smoother node features).
- Edges             = k-NN graph combining spatial proximity and spectral
                      similarity ("spatial/spectral neighbors").
- Graph Convolution = standard GCN propagation rule
                      H' = sigma( D^-1/2 A_hat D^-1/2 H W )
- Output            = per-node class prediction.

Two implementations are provided:
  1. `GCNLayer` / `HSIGCN` — a pure PyTorch, dependency-free dense GCN. This
     is what runs by default so the project has zero hard dependency on
     torch-geometric.
  2. If `torch_geometric` is installed, `build_pyg_data()` shows how to
     convert the same graph into a PyG `Data` object and swap in
     `torch_geometric.nn.GCNConv` layers for a sparse, more scalable version.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


# ----------------------------------------------------------------------
# Graph construction
# ----------------------------------------------------------------------
def build_knn_graph(node_features: np.ndarray, coords: np.ndarray, k: int,
                     spatial_weight: float = 0.5) -> np.ndarray:
    """Build a k-NN adjacency matrix combining normalized spatial distance and
    spectral (cosine) distance between patch-center pixels.

    Parameters
    ----------
    node_features : (N, D) spectral feature per node (e.g. center-pixel vector)
    coords        : (N, 2) row/col spatial coordinates of each node
    k             : number of neighbors per node
    spatial_weight: blend factor in [0,1] between spatial and spectral distance
                    (0 = pure spectral kNN, 1 = pure spatial kNN)

    Returns
    -------
    adj : (N, N) dense symmetric 0/1 adjacency (no self-loops; added later).
    """
    N = node_features.shape[0]

    # Normalize each distance component to comparable scales.
    spat = coords.astype(np.float32)
    spat = (spat - spat.mean(0)) / (spat.std(0) + 1e-8)

    spec = node_features.astype(np.float32)
    spec = (spec - spec.mean(0)) / (spec.std(0) + 1e-8)

    combined = np.concatenate([
        np.sqrt(spatial_weight) * spat,
        np.sqrt(1 - spatial_weight) * spec
    ], axis=1)

    nbrs = NearestNeighbors(n_neighbors=min(k + 1, N)).fit(combined)
    _, indices = nbrs.kneighbors(combined)

    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in indices[i][1:]:  # skip self (first neighbor is itself)
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    return adj


def normalize_adjacency(adj: np.ndarray) -> torch.Tensor:
    """Symmetric normalization: D^-1/2 (A + I) D^-1/2."""
    N = adj.shape[0]
    adj_hat = adj + np.eye(N, dtype=np.float32)
    deg = adj_hat.sum(axis=1)
    d_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    d_inv_sqrt[deg == 0] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    norm_adj = D_inv_sqrt @ adj_hat @ D_inv_sqrt
    return torch.from_numpy(norm_adj.astype(np.float32))


# ----------------------------------------------------------------------
# Dense GCN (dependency-free)
# ----------------------------------------------------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, norm_adj):
        # norm_adj: (N, N), x: (N, in_dim)
        support = self.lin(x)
        out = norm_adj @ support
        return out


class HSIGCN(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, cfg):
        super().__init__()
        dims = [in_dim] + [cfg.hidden_dim] * cfg.num_layers
        self.layers = nn.ModuleList([
            GCNLayer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(dims[-1], num_classes)

    def forward(self, x, norm_adj):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, norm_adj)
            h = F.relu(h)
            h = self.dropout(h)
        return self.classifier(h)


# ----------------------------------------------------------------------
# Node feature extraction (reuses patches from preprocessing.py)
# ----------------------------------------------------------------------
def extract_node_features(patches: np.ndarray, pooling: str = "mean") -> np.ndarray:
    """Turn (N, P, P, B) patches into (N, B) node features by spatial pooling
    over the patch (mean pooling captures local spatial context, matching
    "connect spatial/spectral neighbors" node-feature intuition)."""
    if pooling == "mean":
        return patches.mean(axis=(1, 2))
    elif pooling == "center":
        P = patches.shape[1]
        c = P // 2
        return patches[:, c, c, :]
    else:
        raise ValueError(f"Unknown pooling: {pooling}")


# ----------------------------------------------------------------------
# Optional PyTorch Geometric integration
# ----------------------------------------------------------------------
def build_pyg_data(node_features: np.ndarray, adj: np.ndarray, labels: np.ndarray,
                    splits: dict):
    """Convert dense adjacency + features into a torch_geometric.data.Data
    object. Only called if torch_geometric is installed; the dense HSIGCN
    above is used otherwise."""
    try:
        from torch_geometric.data import Data
    except ImportError as e:
        raise ImportError(
            "torch_geometric is not installed. Either `pip install "
            "torch-geometric` or use the dense HSIGCN implementation "
            "instead (default)."
        ) from e

    edge_index = np.array(np.nonzero(adj))
    edge_index = torch.from_numpy(edge_index).long()
    x = torch.from_numpy(node_features).float()
    y = torch.from_numpy(labels).long()

    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    val_mask = torch.zeros(len(labels), dtype=torch.bool)
    test_mask = torch.zeros(len(labels), dtype=torch.bool)
    train_mask[splits["train"]] = True
    val_mask[splits["val"]] = True
    test_mask[splits["test"]] = True

    return Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


class PyGGCN(nn.Module):
    """Sparse GCN using torch_geometric.nn.GCNConv — used only if the library
    is available. Mirrors HSIGCN's depth/width via cfg."""

    def __init__(self, in_dim: int, num_classes: int, cfg):
        super().__init__()
        from torch_geometric.nn import GCNConv
        dims = [in_dim] + [cfg.hidden_dim] * cfg.num_layers
        self.convs = nn.ModuleList([
            GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(dims[-1], num_classes)

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = self.dropout(h)
        return self.classifier(h)
