"""
data/preprocessing.py
======================
Spatial-spectral preprocessing pipeline shared by every model in the benchmark
(SVM, 3D-CNN, GCN, Autoencoder, Transformer).

Pipeline
--------
1. load_hsi_cube          : load (H, W, B) cube + (H, W) ground-truth map from .mat
2. remove_noisy_bands      : drop water-absorption / low-SNR bands
3. normalize_bands         : per-band min-max or z-score normalization
4. apply_pca               : optional spectral dimensionality reduction
5. pad_cube                : mirror-pad borders so every pixel has a full patch
6. extract_patches         : build (N, patch, patch, B) spatial-spectral patches
7. stratified_split        : create 10% train / val / test indices (per class)

All randomness is seed-controlled so the 5 models are compared on identical
splits for a given seed.
"""

from __future__ import annotations

import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import Dict, Tuple, List


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_hsi_cube(data_path: str, gt_path: str, data_key: str, gt_key: str
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Load a hyperspectral cube and its ground-truth label map from .mat files.

    Returns
    -------
    cube : np.ndarray, shape (H, W, B), float32
    gt   : np.ndarray, shape (H, W), int, 0 = unlabeled/background
    """
    data_mat = loadmat(data_path)
    gt_mat = loadmat(gt_path)

    if data_key not in data_mat:
        raise KeyError(f"'{data_key}' not found in {data_path}. "
                        f"Available keys: {[k for k in data_mat if not k.startswith('__')]}")
    if gt_key not in gt_mat:
        raise KeyError(f"'{gt_key}' not found in {gt_path}. "
                        f"Available keys: {[k for k in gt_mat if not k.startswith('__')]}")

    cube = data_mat[data_key].astype(np.float32)
    gt = gt_mat[gt_key].astype(np.int64)
    assert cube.shape[:2] == gt.shape, "Cube and ground-truth spatial dims mismatch."
    return cube, gt


# ----------------------------------------------------------------------
# Band removal
# ----------------------------------------------------------------------
def remove_noisy_bands(cube: np.ndarray, noisy_ranges: List[Tuple[int, int]]) -> np.ndarray:
    """Drop known water-absorption / low-SNR spectral bands.

    Parameters
    ----------
    cube : (H, W, B)
    noisy_ranges : list of inclusive (start, end) band index ranges to drop.
    """
    B = cube.shape[-1]
    drop_mask = np.zeros(B, dtype=bool)
    for start, end in noisy_ranges:
        start = max(0, start)
        end = min(B - 1, end)
        drop_mask[start:end + 1] = True
    keep_idx = np.where(~drop_mask)[0]
    return cube[..., keep_idx]


# ----------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------
def normalize_bands(cube: np.ndarray, mode: str = "minmax") -> np.ndarray:
    """Normalize each spectral band independently across the whole image.

    mode = "minmax" -> scale each band to [0, 1]
    mode = "zscore" -> zero mean, unit variance per band
    """
    H, W, B = cube.shape
    flat = cube.reshape(-1, B)
    out = np.empty_like(flat)

    if mode == "minmax":
        band_min = flat.min(axis=0, keepdims=True)
        band_max = flat.max(axis=0, keepdims=True)
        denom = np.where((band_max - band_min) == 0, 1.0, band_max - band_min)
        out = (flat - band_min) / denom
    elif mode == "zscore":
        band_mean = flat.mean(axis=0, keepdims=True)
        band_std = flat.std(axis=0, keepdims=True)
        band_std = np.where(band_std == 0, 1.0, band_std)
        out = (flat - band_mean) / band_std
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    return out.reshape(H, W, B).astype(np.float32)


# ----------------------------------------------------------------------
# Optional spectral dimensionality reduction
# ----------------------------------------------------------------------
def apply_pca(cube: np.ndarray, n_components: int) -> np.ndarray:
    """Reduce the spectral dimension with PCA (helps 3D-CNN / GCN / Transformer
    training speed without materially harming accuracy)."""
    if n_components is None or n_components <= 0:
        return cube
    H, W, B = cube.shape
    flat = cube.reshape(-1, B)
    n_components = min(n_components, B)
    pca = PCA(n_components=n_components, whiten=True, random_state=0)
    reduced = pca.fit_transform(flat)
    return reduced.reshape(H, W, n_components).astype(np.float32)


# ----------------------------------------------------------------------
# Patch extraction
# ----------------------------------------------------------------------
def pad_cube(cube: np.ndarray, patch_size: int) -> np.ndarray:
    """Mirror-pad the cube so every pixel (including borders) can produce a
    full patch_size x patch_size window."""
    margin = patch_size // 2
    return np.pad(cube, ((margin, margin), (margin, margin), (0, 0)), mode="reflect")


def extract_patches(cube: np.ndarray, gt: np.ndarray, patch_size: int,
                     ignore_background: bool = True
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a spatial-spectral patch centered on every labeled pixel.

    Returns
    -------
    patches  : (N, patch_size, patch_size, B) float32
    labels   : (N,) int64, 0-indexed class labels
    coords   : (N, 2) int, original (row, col) of patch center — useful for
               building spatial adjacency graphs for the GCN model.
    """
    H, W = gt.shape
    margin = patch_size // 2
    padded = pad_cube(cube, patch_size)

    if ignore_background:
        rows, cols = np.where(gt > 0)
    else:
        rows, cols = np.where(gt >= 0)

    N = len(rows)
    B = cube.shape[-1]
    patches = np.empty((N, patch_size, patch_size, B), dtype=np.float32)
    labels = np.empty((N,), dtype=np.int64)
    coords = np.empty((N, 2), dtype=np.int64)

    for i, (r, c) in enumerate(zip(rows, cols)):
        pr, pc = r + margin, c + margin  # coordinate inside padded cube
        patches[i] = padded[pr - margin:pr + margin + 1, pc - margin:pc + margin + 1, :]
        labels[i] = gt[r, c] - (1 if ignore_background else 0)  # 0-index classes
        coords[i] = (r, c)

    return patches, labels, coords


# ----------------------------------------------------------------------
# Stratified low-data splitting (10% train)
# ----------------------------------------------------------------------
def stratified_split(labels: np.ndarray, train_ratio: float, val_ratio: float,
                      test_ratio: float, seed: int = 0
                      ) -> Dict[str, np.ndarray]:
    """Create stratified train/val/test index splits, preserving per-class
    proportions. Designed for the 10% low-data regime described in the
    project: only `train_ratio` of samples (per class) are used for training.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train/val/test ratios must sum to 1.0"

    idx_all = np.arange(len(labels))

    # First split off the test set.
    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=test_ratio, random_state=seed, stratify=labels
    )
    # From the remaining train+val pool, split train vs val.
    relative_val = val_ratio / (train_ratio + val_ratio)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=relative_val, random_state=seed,
        stratify=labels[idx_trainval]
    )
    return {"train": idx_train, "val": idx_val, "test": idx_test}


# ----------------------------------------------------------------------
# High-level convenience wrapper
# ----------------------------------------------------------------------
@dataclass
class HSIData:
    patches: np.ndarray      # (N, P, P, B)
    labels: np.ndarray       # (N,)
    coords: np.ndarray       # (N, 2)
    splits: Dict[str, np.ndarray]
    num_classes: int
    num_bands: int
    patch_size: int


def build_dataset(data_cfg) -> HSIData:
    """Run the full preprocessing pipeline end to end using a DataConfig object
    (see config.py). Returns an HSIData bundle ready to feed every model."""
    cube, gt = load_hsi_cube(data_cfg.data_path, data_cfg.gt_path,
                              data_cfg.data_key, data_cfg.gt_key)

    if data_cfg.remove_water_absorption_bands:
        cube = remove_noisy_bands(cube, data_cfg.noisy_band_ranges)

    cube = normalize_bands(cube, mode="minmax")

    if data_cfg.pca_components and data_cfg.pca_components > 0:
        cube = apply_pca(cube, data_cfg.pca_components)

    patches, labels, coords = extract_patches(
        cube, gt, data_cfg.patch_size, data_cfg.ignore_background_class
    )

    splits = stratified_split(
        labels, data_cfg.train_ratio, data_cfg.val_ratio, data_cfg.test_ratio,
        seed=data_cfg.seeds[0]
    )

    return HSIData(
        patches=patches,
        labels=labels,
        coords=coords,
        splits=splits,
        num_classes=int(labels.max()) + 1,
        num_bands=cube.shape[-1],
        patch_size=data_cfg.patch_size,
    )


def rebuild_splits_for_seed(hsi_data: HSIData, data_cfg, seed: int) -> Dict[str, np.ndarray]:
    """Recompute train/val/test index splits for a different seed while
    reusing already-extracted patches (used for the multi-seed robustness
    runs described in the training section)."""
    return stratified_split(
        hsi_data.labels, data_cfg.train_ratio, data_cfg.val_ratio,
        data_cfg.test_ratio, seed=seed
    )
