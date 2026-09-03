"""
utils/visualization.py
========================
Plotting helpers for qualitative + quantitative analysis:
  - plot_confusion_matrix : per-model confusion matrix heatmap
  - plot_training_curves  : loss/accuracy curves for a deep model's history
  - plot_classification_map : full-scene predicted-vs-ground-truth map
  - plot_metric_comparison  : bar chart comparing OA/AA/F1/Kappa across models

These are optional (matplotlib/seaborn) and not required to run training or
benchmarking — call them separately when generating report figures.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm: np.ndarray, class_names=None, title="Confusion Matrix",
                           save_path: str = None, normalize: bool = True):
    if normalize:
        cm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=False, cmap="viridis", ax=ax,
                xticklabels=class_names or "auto", yticklabels=class_names or "auto")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_training_curves(history: dict, title="Training Curves", save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train_loss")
    axes[0].plot(history["val_loss"], label="val_loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()
    axes[0].set_title("Loss")

    if "val_acc" in history:
        axes[1].plot(history["val_acc"], label="val_acc", color="green")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].legend()
        axes[1].set_title("Validation Accuracy")

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_classification_map(gt: np.ndarray, coords: np.ndarray, preds: np.ndarray,
                              num_classes: int, save_path: str = None):
    """Reconstruct a full-scene predicted class map from per-pixel test
    predictions, side-by-side with the ground truth map."""
    pred_map = np.zeros_like(gt)
    for (r, c), p in zip(coords, preds):
        pred_map[r, c] = p + 1   # +1 to match ground-truth's 1-indexed classes

    cmap = plt.get_cmap("tab20", num_classes + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(gt, cmap=cmap, vmin=0, vmax=num_classes)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    axes[1].imshow(pred_map, cmap=cmap, vmin=0, vmax=num_classes)
    axes[1].set_title("Predicted (test pixels)")
    axes[1].axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_metric_comparison(summaries: dict, save_path: str = None):
    """Bar chart comparing OA/AA/F1/Kappa (mean +/- std) across all benchmarked
    models. `summaries` = output of utils.metrics.summarize_multi_seed per
    model, keyed by model name (see benchmark.py)."""
    models = list(summaries.keys())
    metrics = ["OA", "AA", "F1", "Kappa"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, metric in zip(axes, metrics):
        means = [summaries[m][metric]["mean"] for m in models]
        stds = [summaries[m][metric]["std"] for m in models]
        ax.bar(models, means, yerr=stds, capsize=4,
               color=sns.color_palette("viridis", len(models)))
        ax.set_title(metric)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylim(0, 1.05 if metric != "Kappa" else 1.0)

    fig.suptitle("Model Comparison — Hyperspectral Classification (10% training)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
