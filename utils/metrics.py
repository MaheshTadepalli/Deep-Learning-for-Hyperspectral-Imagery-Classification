"""
utils/metrics.py
==================
Unified evaluation metrics used by every model in the benchmark, so SVM,
3D-CNN, GCN, Autoencoder, and Transformer are all scored identically on the
same held-out test split.

Metrics
-------
OA    : Overall Accuracy
AA    : Average (per-class mean) Accuracy
F1    : Macro-averaged F1 score
Kappa : Cohen's Kappa coefficient
Also returns the full per-class accuracy vector and confusion matrix for
deeper analysis / plots.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, classification_report
)


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    accs = np.zeros(num_classes)
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() == 0:
            accs[c] = np.nan
            continue
        accs[c] = accuracy_score(y_true[mask], y_pred[mask])
    return accs


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    """Compute OA / AA / F1(macro) / Kappa + supporting artifacts."""
    oa = accuracy_score(y_true, y_pred)
    per_class = per_class_accuracy(y_true, y_pred, num_classes)
    aa = np.nanmean(per_class)
    f1 = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    return {
        "OA": float(oa),
        "AA": float(aa),
        "F1": float(f1),
        "Kappa": float(kappa),
        "per_class_accuracy": per_class,
        "confusion_matrix": cm,
        "report": classification_report(y_true, y_pred, zero_division=0),
    }


def summarize_multi_seed(results_per_seed: list) -> dict:
    """Aggregate metrics across multiple random seeds -> mean +/- std, matching
    the training-section requirement: "Repeat with multiple random seeds."""
    keys = ["OA", "AA", "F1", "Kappa"]
    summary = {}
    for k in keys:
        vals = np.array([r[k] for r in results_per_seed])
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return summary


def format_summary_table(model_summaries: dict) -> str:
    """model_summaries: {model_name: summary_dict_from_summarize_multi_seed}
    Returns a Markdown table string, ready to paste into the README/results."""
    header = "| Model | OA (%) | AA (%) | F1 (%) | Kappa |\n"
    header += "|---|---|---|---|---|\n"
    rows = []
    for name, s in model_summaries.items():
        row = (f"| {name} "
               f"| {s['OA']['mean']*100:.2f} ± {s['OA']['std']*100:.2f} "
               f"| {s['AA']['mean']*100:.2f} ± {s['AA']['std']*100:.2f} "
               f"| {s['F1']['mean']*100:.2f} ± {s['F1']['std']*100:.2f} "
               f"| {s['Kappa']['mean']:.4f} ± {s['Kappa']['std']:.4f} |")
        rows.append(row)
    return header + "\n".join(rows)
