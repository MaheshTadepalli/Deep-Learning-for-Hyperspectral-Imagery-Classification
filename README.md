# Deep Learning for Hyperspectral Imagery Classification

**Self Project | Jan '26 – Apr '26**

A benchmark and final-model implementation for hyperspectral image (HSI)
classification in a **low-data regime (10% training data)**. Five model
families are implemented under one shared preprocessing + evaluation
pipeline — **SVM, 3D-CNN, Autoencoder, GCN, and a Spectral-Spatial
Transformer** — with the Transformer as the final proposed model.

```
Dataset
   │
   ├── SVM
   ├── 3D-CNN
   ├── GCN
   ├── Autoencoder
   └── Transformer (final model)
   │
   ▼
Same held-out test set
   │
   ▼
OA / AA / F1 / Kappa
```

---

## 1. Project Structure

```
hsi_classification/
├── config.py                  # all hyperparameters (data, training, per-model)
├── main.py                    # CLI entry point (preprocess / train / benchmark)
├── train.py                   # per-model training functions
├── benchmark.py                # runs all 5 models x N seeds, aggregates results
├── requirements.txt
│
├── data/
│   ├── preprocessing.py        # load, normalize, denoise, patch-extract, split
│   └── dataset.py               # PyTorch Dataset / DataLoader wrappers
│
├── models/
│   ├── svm_baseline.py          # pixel-wise SVM (RBF/linear, grid-search)
│   ├── cnn3d.py                  # 3D-CNN (joint spatial-spectral conv)
│   ├── autoencoder.py            # conv autoencoder (pretrain + finetune)
│   ├── gcn.py                     # kNN graph construction + GCN
│   └── transformer.py             # spectral-spatial Transformer (final model)
│
├── utils/
│   ├── metrics.py                 # OA / AA / F1 / Kappa, multi-seed summaries
│   ├── trainer.py                  # shared training loop, early stopping
│   ├── seed.py                      # reproducibility helper
│   └── visualization.py              # confusion matrix / maps / bar charts
│
└── results/                          # benchmark outputs (json / csv / md)
```

---

## 2. Setup

```bash
pip install -r requirements.txt
```

`torch-geometric` is optional — the GCN model has a dependency-free dense
PyTorch fallback (`models/gcn.py: HSIGCN`) used by default. If
`torch-geometric` is installed, `models/gcn.py: PyGGCN` + `build_pyg_data()`
provide a sparse, more scalable alternative for larger scenes.

### Dataset

Point `config.DataConfig` at any standard HSI benchmark scene distributed as
`.mat` files (e.g. **Indian Pines**, **Pavia University**, **Salinas**):

```python
# config.py
DATA.data_path = "./raw_data/indian_pines_corrected.mat"
DATA.gt_path   = "./raw_data/indian_pines_gt.mat"
DATA.data_key  = "indian_pines_corrected"
DATA.gt_key    = "indian_pines_gt"
```

---

## 3. Pipeline Details

### 3.1 Dataset Preprocessing (`data/preprocessing.py`)

| Step | Function | Description |
|---|---|---|
| Load | `load_hsi_cube` | Reads `(H, W, B)` cube + `(H, W)` ground truth from `.mat` |
| Denoise | `remove_noisy_bands` | Drops configurable water-absorption / low-SNR band ranges |
| Normalize | `normalize_bands` | Per-band min-max (or z-score) normalization |
| Reduce (optional) | `apply_pca` | PCA spectral dimensionality reduction |
| Patch extraction | `extract_patches` | `11×11×B` spatial-spectral patches centered on every labeled pixel |
| Splitting | `stratified_split` | Stratified **10% train / 5% val / 85% test** split, per class |

All five models consume patches produced by this **single shared pipeline**,
so differences in benchmark results reflect the model architectures only —
not differing preprocessing.

### 3.2 SVM Baseline (`models/svm_baseline.py`)

- Uses the **center pixel's raw spectral vector** (no spatial context) —
  the classical HSI-SVM setup.
- RBF or linear kernel; `C`/`gamma` tuned via grid search on the validation
  split (`sklearn.model_selection.GridSearchCV` with a `PredefinedSplit`
  matching the same train/val split used by the deep models).

### 3.3 3D-CNN (`models/cnn3d.py`)

```
11×11×B patch
      │
3D Conv → BN/ReLU
      │
3D Conv → BN/ReLU
      │
3D pooling
      │
Flatten → FC → Class
```

Learns joint spatial-spectral filters directly on the raw patch cube
(`Conv3d` over `(bands, H, W)`).

### 3.4 Autoencoder (`models/autoencoder.py`)

```
Patch → Encoder → latent (z) → Decoder → reconstruction
```

Two-stage training (`train.py: train_autoencoder_pipeline`):
1. **Unsupervised pretraining** — reconstruction (MSE) loss on *all*
   available patches (label-free), which helps exploit unlabeled structure
   under the 10%-label constraint.
2. **Supervised fine-tuning** — an MLP head is attached to the pretrained
   encoder and fine-tuned on the 10% labeled training split.

### 3.5 GCN (`models/gcn.py`)

- **Nodes** = patches (features = spatially-pooled spectral vector).
- **Edges** = k-NN graph blending spatial proximity + spectral similarity
  (`build_knn_graph`).
- **Propagation** = symmetric-normalized `GCNLayer`
  (`H' = σ(D⁻¹ᐟ²ÂD⁻¹ᐟ²HW)`), stacked `num_layers` times.
- Trained **transductively**: one forward pass computes logits for every
  node; loss is masked to the training-node indices (`utils/trainer.py:
  train_gcn`).
- Dependency-free by default; optional sparse `torch_geometric.nn.GCNConv`
  backend (`PyGGCN`) is available for larger graphs.

### 3.6 Spectral-Spatial Transformer — Final Model (`models/transformer.py`)

```
HSI patch (P×P×B)
      │
Spectral-spatial tokenization  (non-overlapping token_size×token_size sub-patches,
      │                          each covering the FULL spectral depth)
Linear embedding
      │
Positional encoding (learnable or sinusoidal)
      │
Transformer Encoder × depth
  (Multi-head self-attention, residual + LayerNorm, MLP, DropPath)
      │
CLS token / global average pooling
      │
MLP classifier
      │
Class
```

**Low-data-regime design choices** (10% training):
- Compact default capacity (`embed_dim=128`, `depth=4`, `heads=8`) to limit
  overfitting.
- Dropout on embeddings, attention, and MLP + `AdamW` weight decay +
  stochastic depth (`DropPath`) for regularization.
- Spectral-spatial tokens (rather than flat ViT-style patches) inject a
  spatial-neighborhood + spectral-signature inductive bias, reducing the
  amount of data needed to learn useful groupings.

### 3.7 Training (`utils/trainer.py`, `train.py`)

- Cross-entropy loss, **AdamW** optimizer.
- Cosine / step / plateau **LR scheduler** (configurable).
- **Early stopping** on validation loss (`utils/trainer.py: EarlyStopping`).
- Gradient clipping.
- Trained using **only the 10% train split** (SVM/3D-CNN/AE-finetune/
  Transformer) or transductively with train-masked loss (GCN).
- **Repeated across multiple random seeds** (`config.DATA.seeds`) for
  robust, low-variance comparison.

### 3.8 Benchmark Framework (`benchmark.py`)

Runs every configured model across every seed on **identical
preprocessing + identical splits**, then reports mean ± std of:

- **OA** — Overall Accuracy
- **AA** — Average (per-class mean) Accuracy
- **F1** — Macro-averaged F1
- **Kappa** — Cohen's Kappa coefficient

Outputs (written to `results/`):
- `benchmark_summary.json` — mean/std per metric per model
- `benchmark_per_seed.csv` — long-format per-seed results
- `benchmark_summary.md` — ready-to-paste Markdown table

---

## 4. Usage

```bash
# 1) Sanity-check preprocessing on the configured dataset
python main.py preprocess

# 2) Train a single model
python main.py train --model transformer --seed 0
python main.py train --model svm
python main.py train --model 3dcnn --seed 1
python main.py train --model autoencoder
python main.py train --model gcn

# 3) Run the full benchmark (all 5 models × all configured seeds)
python main.py benchmark

# Run a subset
python main.py benchmark --models svm 3dcnn transformer --seeds 0 1 2
```

Programmatic use:

```python
from data.preprocessing import build_dataset, rebuild_splits_for_seed
from train import MODEL_TRAINERS
from utils.metrics import evaluate
import config as C

hsi_data = build_dataset(C.DATA)
splits = rebuild_splits_for_seed(hsi_data, C.DATA, seed=0)

y_true, y_pred, extra = MODEL_TRAINERS["transformer"](hsi_data, splits, seed=0)
metrics = evaluate(y_true, y_pred, hsi_data.num_classes)
print(metrics["OA"], metrics["AA"], metrics["F1"], metrics["Kappa"])
```

---

## 5. Results

*(Populate after running `python main.py benchmark` — `results/benchmark_summary.md`
is generated automatically)

---

## 6. Notes / Extensibility

- **Different sensors/scenes**: adjust `config.DataConfig.noisy_band_ranges`
  (band indices are sensor-specific) and `data_key`/`gt_key` for the `.mat`
  variable names.
- **Patch size vs. token size**: `TransformerConfig.token_size` must evenly
  divide `DataConfig.patch_size` (e.g. `patch_size=11` won't divide evenly
  by `token_size=3`; use e.g. `patch_size=12` or `token_size` that divides
  the chosen patch size — an assertion in `SpectralSpatialTokenizer` will
  catch mismatches).
- **Scaling the GCN** to full scenes: swap `HSIGCN` (dense) for `PyGGCN`
  (sparse, requires `torch-geometric`) via `models/gcn.py: build_pyg_data`.
- **Visualization**: `utils/visualization.py` provides confusion-matrix
  heatmaps, training curves, full-scene classification maps, and
  cross-model bar-chart comparisons for the report/write-up.
