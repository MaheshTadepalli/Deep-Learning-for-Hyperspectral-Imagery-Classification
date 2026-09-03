"""
models/svm_baseline.py
========================
Classical baseline: pixel-wise spectral-vector SVM classifier (RBF or linear
kernel), tuned with grid-search cross-validation on the validation split.

Unlike the deep models, the SVM does not use spatial context — each pixel's
raw spectral vector (length B) is the feature vector, per the project spec:
    "Flatten each pixel's spectral vector: B -> feature vector."
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def extract_center_pixel_vectors(patches: np.ndarray) -> np.ndarray:
    """From (N, P, P, B) spatial-spectral patches, pull out the center pixel's
    raw spectral vector -> (N, B). Keeps the SVM pipeline consistent with the
    same patch-extraction step used by the deep models, while still training
    on pure per-pixel spectra (no spatial context), matching classical HSI
    SVM baselines in the literature."""
    P = patches.shape[1]
    center = P // 2
    return patches[:, center, center, :]


class SVMBaseline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.best_model: SVC | None = None
        self.best_params_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray):
        """Grid-search C / gamma on the validation split, refit best model on
        train+val."""
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)

        param_grid = {"C": list(self.cfg.C_grid)}
        if self.cfg.kernel == "rbf":
            param_grid["gamma"] = list(self.cfg.gamma_grid)

        # Use a fixed held-out validation split (not k-fold) to mirror the
        # deep-learning models' train/val protocol under the 10% regime.
        from sklearn.model_selection import PredefinedSplit
        X_combined = np.concatenate([X_train_s, X_val_s], axis=0)
        y_combined = np.concatenate([y_train, y_val], axis=0)
        test_fold = np.concatenate([
            -1 * np.ones(len(X_train_s)),   # -1 => always train
            np.zeros(len(X_val_s))          # 0  => validation fold
        ])
        ps = PredefinedSplit(test_fold)

        base = SVC(kernel=self.cfg.kernel, decision_function_shape="ovr")
        search = GridSearchCV(base, param_grid, cv=ps, scoring="accuracy",
                               n_jobs=-1, refit=False)
        search.fit(X_combined, y_combined)
        self.best_params_ = search.best_params_

        # Refit on train+val with best hyperparameters.
        self.best_model = SVC(kernel=self.cfg.kernel,
                               decision_function_shape="ovr",
                               **self.best_params_)
        self.best_model.fit(X_combined, y_combined)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self.scaler.transform(X)
        return self.best_model.predict(X_s)


def run_svm_pipeline(hsi_data, splits, cfg) -> dict:
    """End-to-end SVM training + prediction on the shared benchmark splits.
    Returns dict with y_true / y_pred for the test set (fed into
    utils/metrics.py for OA/AA/Kappa/F1)."""
    X = extract_center_pixel_vectors(hsi_data.patches)
    y = hsi_data.labels

    X_train, y_train = X[splits["train"]], y[splits["train"]]
    X_val, y_val = X[splits["val"]], y[splits["val"]]
    X_test, y_test = X[splits["test"]], y[splits["test"]]

    model = SVMBaseline(cfg)
    model.fit(X_train, y_train, X_val, y_val)
    y_pred = model.predict(X_test)

    return {
        "y_true": y_test,
        "y_pred": y_pred,
        "best_params": model.best_params_,
        "model": model,
    }
