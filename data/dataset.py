"""
data/dataset.py
=================
PyTorch Dataset wrappers around the preprocessed HSI patches produced by
data/preprocessing.py. Shared by 3D-CNN, Autoencoder, and Transformer models.
(GCN builds its own graph-structured batches — see models/gcn.py.
 SVM operates directly on flattened numpy vectors — see models/svm_baseline.py.)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class HSIPatchDataset(Dataset):
    """Wraps (N, P, P, B) patches + (N,) labels for supervised training.

    `channel_first`:
        True  -> returns patches as (B, P, P) for 2D-style conv nets, or
                 (1, B, P, P) when `add_depth_dim=True` for 3D-CNN input.
        False -> returns patches as (P, P, B), used by the Transformer's own
                 tokenizer which handles the reshape internally.
    """

    def __init__(self, patches: np.ndarray, labels: np.ndarray,
                 indices: np.ndarray = None, channel_first: bool = True,
                 add_depth_dim: bool = False, augment: bool = False):
        self.patches = patches
        self.labels = labels
        self.indices = indices if indices is not None else np.arange(len(labels))
        self.channel_first = channel_first
        self.add_depth_dim = add_depth_dim
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def _augment(self, patch: np.ndarray) -> np.ndarray:
        # Simple label-preserving spatial augmentations, useful in the
        # 10%-training low-data regime.
        if np.random.rand() < 0.5:
            patch = np.flip(patch, axis=0)
        if np.random.rand() < 0.5:
            patch = np.flip(patch, axis=1)
        k = np.random.randint(0, 4)
        if k:
            patch = np.rot90(patch, k, axes=(0, 1))
        return np.ascontiguousarray(patch)

    def __getitem__(self, i):
        idx = self.indices[i]
        patch = self.patches[idx]  # (P, P, B)
        label = self.labels[idx]

        if self.augment:
            patch = self._augment(patch)

        if self.channel_first:
            patch = np.transpose(patch, (2, 0, 1))  # (B, P, P)
            if self.add_depth_dim:
                patch = patch[np.newaxis, ...]        # (1, B, P, P) for Conv3d

        patch_t = torch.from_numpy(patch.copy()).float()
        label_t = torch.tensor(label, dtype=torch.long)
        return patch_t, label_t


def make_loaders(hsi_data, splits, train_cfg, channel_first=True,
                  add_depth_dim=False, augment_train=True):
    """Convenience factory returning (train_loader, val_loader, test_loader)."""
    from torch.utils.data import DataLoader

    train_ds = HSIPatchDataset(hsi_data.patches, hsi_data.labels, splits["train"],
                                channel_first, add_depth_dim, augment=augment_train)
    val_ds = HSIPatchDataset(hsi_data.patches, hsi_data.labels, splits["val"],
                              channel_first, add_depth_dim, augment=False)
    test_ds = HSIPatchDataset(hsi_data.patches, hsi_data.labels, splits["test"],
                               channel_first, add_depth_dim, augment=False)

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True,
                               num_workers=train_cfg.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False,
                             num_workers=train_cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=False,
                              num_workers=train_cfg.num_workers)
    return train_loader, val_loader, test_loader
