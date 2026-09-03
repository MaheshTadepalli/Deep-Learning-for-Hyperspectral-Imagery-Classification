"""
models/cnn3d.py
=================
3D-CNN for joint spatial-spectral feature learning.

    11x11xB patch
         v
    3D Conv -> BN/ReLU
         v
    3D Conv -> BN/ReLU
         v
    3D pooling
         v
    Flatten -> FC -> Class

Input tensor shape convention: (batch, 1, B, P, P)  [Conv3d expects
(N, C_in, D, H, W); here D = spectral bands, H = W = patch_size].
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNN3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(3, 3, 3), padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class HSI3DCNN(nn.Module):
    def __init__(self, num_bands: int, patch_size: int, num_classes: int, cfg):
        super().__init__()
        f1, f2 = cfg.num_filters
        k = cfg.kernel_size
        pad = tuple(kk // 2 for kk in k)

        self.block1 = CNN3DBlock(1, f1, kernel_size=k, padding=pad)
        self.block2 = CNN3DBlock(f1, f2, kernel_size=k, padding=pad)
        self.pool = nn.MaxPool3d(kernel_size=cfg.pool_size)

        # Infer flattened feature size with a dummy forward pass.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_bands, patch_size, patch_size)
            feat = self._forward_features(dummy)
            flat_dim = feat.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, cfg.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fc_hidden, num_classes),
        )

    def _forward_features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        return x

    def forward(self, x):
        # x expected shape: (B, 1, Bands, P, P)
        feat = self._forward_features(x)
        return self.classifier(feat)
