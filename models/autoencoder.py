"""
models/autoencoder.py
=======================
Spatial-spectral convolutional Autoencoder.

    Hyperspectral patch
           v
        Encoder
           v
    Low-dimensional spectral-spatial representation (latent_dim)
           v
        Decoder

Training happens in two stages (see train.py: train_autoencoder_pipeline):
  1. Pretrain encoder+decoder with reconstruction (MSE) loss on ALL available
     patches (unsupervised — labels not needed), which helps in the low-data
     (10% labeled) regime by learning good features from unlabeled structure.
  2. Freeze or fine-tune the encoder and attach a classification head trained
     on the 10% labeled training split.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, channels, latent_dim: int, patch_size: int):
        super().__init__()
        c1, c2, c3 = channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, stride=2),   # downsample
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, patch_size, patch_size)
            feat = self.net(dummy)
            self._feat_shape = feat.shape[1:]   # (C, H', W')
            flat_dim = feat.view(1, -1).shape[1]

        self.to_latent = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        feat = self.net(x)
        flat = feat.view(feat.size(0), -1)
        z = self.to_latent(flat)
        return z, feat.shape


class ConvDecoder(nn.Module):
    def __init__(self, out_channels: int, channels, latent_dim: int, feat_shape):
        super().__init__()
        c1, c2, c3 = channels
        self.feat_shape = feat_shape  # (C, H', W') matching encoder's bottleneck
        flat_dim = feat_shape[0] * feat_shape[1] * feat_shape[2]
        self.from_latent = nn.Linear(latent_dim, flat_dim)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c2, c1, kernel_size=3, padding=1, stride=2,
                                output_padding=1),                    # upsample
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),   # bands were min-max normalized to [0,1]
        )

    def forward(self, z):
        flat = self.from_latent(z)
        feat = flat.view(-1, *self.feat_shape)
        out = self.net(feat)
        return out


class HSIAutoencoder(nn.Module):
    """Full autoencoder used for the unsupervised pretraining stage."""

    def __init__(self, num_bands: int, patch_size: int, cfg):
        super().__init__()
        self.encoder = ConvEncoder(num_bands, cfg.encoder_channels, cfg.latent_dim, patch_size)
        # infer bottleneck shape for the decoder
        with torch.no_grad():
            dummy = torch.zeros(1, num_bands, patch_size, patch_size)
            _, feat_shape = self.encoder(dummy)
        self.decoder = ConvDecoder(num_bands, cfg.encoder_channels, cfg.latent_dim, feat_shape[1:])

    def forward(self, x):
        z, _ = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


class AEClassifier(nn.Module):
    """Stage-2 model: pretrained encoder + MLP classification head."""

    def __init__(self, pretrained_encoder: ConvEncoder, latent_dim: int,
                 num_classes: int, dropout: float, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = pretrained_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, num_classes),
        )

    def forward(self, x):
        z, _ = self.encoder(x)
        return self.head(z)
