"""
config.py
=========
Central configuration for the Hyperspectral Imagery (HSI) Classification project.

Every script (preprocessing, model training, benchmarking) imports its
hyperparameters from here so that all five pipelines (SVM, 3D-CNN, GCN,
Autoencoder, Transformer) are evaluated under *exactly* the same
data conditions (10% training split, same patch size, same seeds, etc.).
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    # ------------------------------------------------------------------
    # Dataset location. Point these at .mat files such as Indian Pines
    # (indian_pines_corrected.mat / indian_pines_gt.mat), PaviaU, or Salinas.
    # ------------------------------------------------------------------
    dataset_name: str = "IndianPines"          # IndianPines | PaviaU | Salinas
    data_path: str = "./raw_data/indian_pines_corrected.mat"
    gt_path: str = "./raw_data/indian_pines_gt.mat"
    data_key: str = "indian_pines_corrected"   # variable name inside the .mat file
    gt_key: str = "indian_pines_gt"

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    patch_size: int = 11                       # spatial window: patch_size x patch_size x B
    pca_components: int = 30                    # spectral dimensionality reduction (0 = disabled)
    remove_water_absorption_bands: bool = True
    # Known noisy / water-absorption band index ranges for AVIRIS-type sensors.
    # Adjust per-sensor. These are 0-indexed, inclusive ranges.
    noisy_band_ranges: List[Tuple[int, int]] = field(
        default_factory=lambda: [(0, 3), (103, 111), (148, 166), (219, 223)]
    )

    # ------------------------------------------------------------------
    # Splits — low-data regime
    # ------------------------------------------------------------------
    train_ratio: float = 0.10
    val_ratio: float = 0.05
    test_ratio: float = 0.85
    stratified: bool = True                     # stratify split per class
    ignore_background_class: bool = True         # label 0 usually means "unlabeled"

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 5e-4
    optimizer: str = "adamw"
    lr_scheduler: str = "cosine"                 # cosine | step | plateau
    early_stopping_patience: int = 15
    grad_clip_norm: float = 1.0
    num_workers: int = 4
    device: str = "cuda"                          # falls back to cpu automatically


@dataclass
class CNN3DConfig:
    num_filters: Tuple[int, int] = (8, 16)
    kernel_size: Tuple[int, int, int] = (3, 3, 3)
    pool_size: Tuple[int, int, int] = (2, 2, 2)
    fc_hidden: int = 256
    dropout: float = 0.4


@dataclass
class AutoencoderConfig:
    latent_dim: int = 64
    encoder_channels: Tuple[int, int, int] = (32, 64, 128)
    pretrain_epochs: int = 60
    finetune_epochs: int = 60
    dropout: float = 0.3


@dataclass
class GCNConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    k_neighbors: int = 8                          # spatial-spectral kNN graph
    dropout: float = 0.5
    use_spectral_similarity: bool = True


@dataclass
class TransformerConfig:
    token_size: int = 3                            # sub-patch size used for spectral-spatial tokenization
    embed_dim: int = 128
    depth: int = 4                                  # number of encoder blocks
    num_heads: int = 8
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    attn_dropout: float = 0.1
    drop_path: float = 0.1
    use_cls_token: bool = True
    pos_encoding: str = "learnable"                 # learnable | sinusoidal


@dataclass
class SVMConfig:
    kernel: str = "rbf"                             # rbf | linear
    C_grid: Tuple[float, ...] = (1, 10, 100, 1000)
    gamma_grid: Tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4)
    cv_folds: int = 3


@dataclass
class BenchmarkConfig:
    models: Tuple[str, ...] = ("svm", "3dcnn", "autoencoder", "gcn", "transformer")
    metrics: Tuple[str, ...] = ("OA", "AA", "Kappa", "F1")
    results_dir: str = "./results"
    save_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"


DATA = DataConfig()
TRAIN = TrainConfig()
CNN3D = CNN3DConfig()
AE = AutoencoderConfig()
GCN = GCNConfig()
TRANSFORMER = TransformerConfig()
SVM = SVMConfig()
BENCH = BenchmarkConfig()

os.makedirs(BENCH.results_dir, exist_ok=True)
os.makedirs(BENCH.checkpoint_dir, exist_ok=True)
