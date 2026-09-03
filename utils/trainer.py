"""
utils/trainer.py
==================
Shared training loop for all deep models (3D-CNN, Autoencoder classifier,
GCN, Transformer): cross-entropy loss, AdamW, LR scheduler, gradient
clipping, and early stopping on validation loss/accuracy.
"""

from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau


class EarlyStopping:
    def __init__(self, patience: int = 15, mode: str = "min"):
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.best_state = None

    def step(self, metric: float, model: nn.Module) -> bool:
        improved = (
            self.best_score is None or
            (self.mode == "min" and metric < self.best_score) or
            (self.mode == "max" and metric > self.best_score)
        )
        if improved:
            self.best_score = metric
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def build_optimizer(model: nn.Module, train_cfg):
    if train_cfg.optimizer.lower() == "adamw":
        return AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    elif train_cfg.optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    else:
        return torch.optim.SGD(model.parameters(), lr=train_cfg.lr,
                                weight_decay=train_cfg.weight_decay, momentum=0.9)


def build_scheduler(optimizer, train_cfg):
    if train_cfg.lr_scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=train_cfg.epochs)
    elif train_cfg.lr_scheduler == "step":
        return StepLR(optimizer, step_size=max(1, train_cfg.epochs // 4), gamma=0.5)
    elif train_cfg.lr_scheduler == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    return None


def get_device(train_cfg) -> torch.device:
    if train_cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_classifier(model: nn.Module, train_loader, val_loader, train_cfg,
                      criterion=None, verbose: bool = True) -> dict:
    """Generic supervised training loop shared by 3D-CNN / AE-classifier /
    Transformer. GCN uses a full-graph variant (see train_gcn below) since it
    operates on the entire node set at once rather than mini-batches.
    """
    device = get_device(train_cfg)
    model.to(device)

    criterion = criterion or nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    early_stopper = EarlyStopping(patience=train_cfg.early_stopping_patience, mode="min")

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(train_cfg.epochs):
        # ---- train ----
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if train_cfg.grad_clip_norm:
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # ---- validate ----
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if verbose and (epoch % 5 == 0 or epoch == train_cfg.epochs - 1):
            print(f"  epoch {epoch:03d} | train_loss {train_loss:.4f} "
                  f"| val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

        improved = early_stopper.step(val_loss, model)
        if early_stopper.should_stop:
            if verbose:
                print(f"  early stopping at epoch {epoch} (best val_loss={early_stopper.best_score:.4f})")
            break

    early_stopper.restore_best(model)
    return history


@torch.no_grad()
def predict(model: nn.Module, loader, train_cfg):
    device = get_device(train_cfg)
    model.to(device).eval()
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(yb.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)


def pretrain_autoencoder(ae_model: nn.Module, loader, train_cfg, epochs: int, verbose=True):
    """Unsupervised reconstruction pretraining stage for the Autoencoder
    model (uses ALL patches, not just the 10% labeled subset)."""
    device = get_device(train_cfg)
    ae_model.to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(ae_model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    for epoch in range(epochs):
        ae_model.train()
        running_loss = 0.0
        for xb, _ in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon, _ = ae_model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        scheduler.step()
        epoch_loss = running_loss / len(loader.dataset)
        history.append(epoch_loss)
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  [AE pretrain] epoch {epoch:03d} | recon_loss {epoch_loss:.5f}")
    return history


def train_gcn(model: nn.Module, x: torch.Tensor, norm_adj: torch.Tensor, y: torch.Tensor,
              splits: dict, train_cfg, verbose: bool = True) -> dict:
    """Full-graph GCN training: forward pass computes logits for ALL nodes;
    loss is masked to the training node indices only (standard transductive
    GCN training protocol)."""
    device = get_device(train_cfg)
    model.to(device)
    x, norm_adj, y = x.to(device), norm_adj.to(device), y.to(device)

    train_idx = torch.from_numpy(splits["train"]).long().to(device)
    val_idx = torch.from_numpy(splits["val"]).long().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    early_stopper = EarlyStopping(patience=train_cfg.early_stopping_patience, mode="min")

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(train_cfg.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x, norm_adj)
        loss = criterion(logits[train_idx], y[train_idx])
        loss.backward()
        if train_cfg.grad_clip_norm:
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, norm_adj)
            val_loss = criterion(logits[val_idx], y[val_idx]).item()
            val_preds = logits[val_idx].argmax(dim=1)
            val_acc = (val_preds == y[val_idx]).float().mean().item()

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if verbose and (epoch % 5 == 0 or epoch == train_cfg.epochs - 1):
            print(f"  epoch {epoch:03d} | train_loss {loss.item():.4f} "
                  f"| val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

        early_stopper.step(val_loss, model)
        if early_stopper.should_stop:
            if verbose:
                print(f"  early stopping at epoch {epoch}")
            break

    early_stopper.restore_best(model)
    return history


@torch.no_grad()
def predict_gcn(model: nn.Module, x: torch.Tensor, norm_adj: torch.Tensor, y: torch.Tensor,
                 splits: dict, train_cfg):
    device = get_device(train_cfg)
    model.to(device).eval()
    x, norm_adj = x.to(device), norm_adj.to(device)
    logits = model(x, norm_adj)
    preds = logits.argmax(dim=1).cpu().numpy()
    test_idx = splits["test"]
    return y.numpy()[test_idx], preds[test_idx]
