"""
train.py
=========
Per-model training entry points. Each `train_<model>` function:
  1. builds the model
  2. trains it (with early stopping, AdamW, LR scheduling as configured)
  3. returns predictions on the shared test split, ready for utils/metrics.py

These functions are called directly for single-model experiments, or in a
loop by benchmark.py for the full 5-model comparison across multiple seeds.
"""

from __future__ import annotations

import numpy as np
import torch

import config as C
from data.dataset import make_loaders
from utils.trainer import (
    train_classifier, predict, pretrain_autoencoder, train_gcn, predict_gcn, get_device
)
from utils.seed import set_seed

from models.svm_baseline import run_svm_pipeline
from models.cnn3d import HSI3DCNN
from models.autoencoder import HSIAutoencoder, AEClassifier
from models.gcn import build_knn_graph, normalize_adjacency, extract_node_features, HSIGCN
from models.transformer import HSITransformer


# ----------------------------------------------------------------------
# SVM
# ----------------------------------------------------------------------
def train_svm(hsi_data, splits, seed: int = 0):
    set_seed(seed)
    result = run_svm_pipeline(hsi_data, splits, C.SVM)
    return result["y_true"], result["y_pred"], {"best_params": result["best_params"]}


# ----------------------------------------------------------------------
# 3D-CNN
# ----------------------------------------------------------------------
def train_3dcnn(hsi_data, splits, seed: int = 0, verbose: bool = True):
    set_seed(seed)
    train_loader, val_loader, test_loader = make_loaders(
        hsi_data, splits, C.TRAIN, channel_first=True, add_depth_dim=True
    )
    model = HSI3DCNN(hsi_data.num_bands, hsi_data.patch_size, hsi_data.num_classes, C.CNN3D)
    history = train_classifier(model, train_loader, val_loader, C.TRAIN, verbose=verbose)
    y_true, y_pred = predict(model, test_loader, C.TRAIN)
    return y_true, y_pred, {"history": history, "model": model}


# ----------------------------------------------------------------------
# Autoencoder (two-stage: pretrain -> fine-tune classifier)
# ----------------------------------------------------------------------
def train_autoencoder_pipeline(hsi_data, splits, seed: int = 0, verbose: bool = True):
    set_seed(seed)

    # Stage 1: unsupervised pretraining uses ALL patches (train+val+test pool),
    # since reconstruction doesn't need labels — this is a key benefit in the
    # 10%-labeled low-data regime.
    all_idx = np.arange(len(hsi_data.labels))
    pretrain_loader, _, _ = make_loaders(
        hsi_data, {"train": all_idx, "val": splits["val"], "test": splits["test"]},
        C.TRAIN, channel_first=True, add_depth_dim=False, augment_train=False
    )

    ae = HSIAutoencoder(hsi_data.num_bands, hsi_data.patch_size, C.AE)
    if verbose:
        print("  Pretraining autoencoder (reconstruction)...")
    pretrain_autoencoder(ae, pretrain_loader, C.TRAIN, C.AE.pretrain_epochs, verbose=verbose)

    # Stage 2: attach classifier head, fine-tune on the 10% labeled train split.
    train_loader, val_loader, test_loader = make_loaders(
        hsi_data, splits, C.TRAIN, channel_first=True, add_depth_dim=False
    )
    clf = AEClassifier(ae.encoder, C.AE.latent_dim, hsi_data.num_classes,
                        C.AE.dropout, freeze_encoder=False)

    # Use a separate, shorter-epoch TrainConfig-like override for fine-tuning.
    finetune_cfg = C.TrainConfig(**{**C.TRAIN.__dict__, "epochs": C.AE.finetune_epochs})
    if verbose:
        print("  Fine-tuning classifier head on 10% labeled data...")
    history = train_classifier(clf, train_loader, val_loader, finetune_cfg, verbose=verbose)
    y_true, y_pred = predict(clf, test_loader, finetune_cfg)
    return y_true, y_pred, {"history": history, "model": clf}


# ----------------------------------------------------------------------
# GCN
# ----------------------------------------------------------------------
def train_gcn_pipeline(hsi_data, splits, seed: int = 0, verbose: bool = True):
    set_seed(seed)

    node_features = extract_node_features(hsi_data.patches, pooling="mean")  # (N, B)
    adj = build_knn_graph(node_features, hsi_data.coords, k=C.GCN.k_neighbors,
                           spatial_weight=0.5)
    norm_adj = normalize_adjacency(adj)

    x = torch.from_numpy(node_features).float()
    y = torch.from_numpy(hsi_data.labels).long()

    model = HSIGCN(in_dim=node_features.shape[1], num_classes=hsi_data.num_classes, cfg=C.GCN)
    history = train_gcn(model, x, norm_adj, y, splits, C.TRAIN, verbose=verbose)
    y_true, y_pred = predict_gcn(model, x, norm_adj, y, splits, C.TRAIN)
    return y_true, y_pred, {"history": history, "model": model}


# ----------------------------------------------------------------------
# Transformer (final proposed model)
# ----------------------------------------------------------------------
def train_transformer(hsi_data, splits, seed: int = 0, verbose: bool = True):
    set_seed(seed)
    train_loader, val_loader, test_loader = make_loaders(
        hsi_data, splits, C.TRAIN, channel_first=True, add_depth_dim=False
    )
    model = HSITransformer(hsi_data.num_bands, hsi_data.patch_size,
                            hsi_data.num_classes, C.TRANSFORMER)
    history = train_classifier(model, train_loader, val_loader, C.TRAIN, verbose=verbose)
    y_true, y_pred = predict(model, test_loader, C.TRAIN)
    return y_true, y_pred, {"history": history, "model": model}


# ----------------------------------------------------------------------
# Dispatch table used by benchmark.py
# ----------------------------------------------------------------------
MODEL_TRAINERS = {
    "svm": train_svm,
    "3dcnn": train_3dcnn,
    "autoencoder": train_autoencoder_pipeline,
    "gcn": train_gcn_pipeline,
    "transformer": train_transformer,
}


if __name__ == "__main__":
    import argparse
    from data.preprocessing import build_dataset, rebuild_splits_for_seed

    parser = argparse.ArgumentParser(description="Train a single HSI classification model.")
    parser.add_argument("--model", choices=list(MODEL_TRAINERS.keys()), required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Building dataset ({C.DATA.dataset_name})...")
    hsi_data = build_dataset(C.DATA)
    splits = rebuild_splits_for_seed(hsi_data, C.DATA, args.seed)

    print(f"Training {args.model} (seed={args.seed})...")
    y_true, y_pred, extra = MODEL_TRAINERS[args.model](hsi_data, splits, seed=args.seed)

    from utils.metrics import evaluate
    metrics = evaluate(y_true, y_pred, hsi_data.num_classes)
    print(f"\nOA={metrics['OA']:.4f}  AA={metrics['AA']:.4f}  "
          f"F1={metrics['F1']:.4f}  Kappa={metrics['Kappa']:.4f}")
