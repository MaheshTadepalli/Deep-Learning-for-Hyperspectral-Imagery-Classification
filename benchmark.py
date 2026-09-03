"""
benchmark.py
=============
Unified benchmark pipeline:

    Dataset
       |
   +---+----+-----+-----+-------------+
   SVM  3DCNN  GCN   AE   Transformer
   +---+----+-----+-----+-------------+
                |
          Same test set
                |
        OA / AA / F1 / Kappa

Runs every model in config.BENCH.models across every seed in
config.DATA.seeds on identical preprocessed data, then aggregates results
(mean +/- std) into a Markdown table and a saved JSON/CSV for the README /
report.

Usage
-----
    python benchmark.py                       # run full benchmark, all models, all seeds
    python benchmark.py --models svm 3dcnn     # run a subset
    python benchmark.py --seeds 0 1            # override seeds
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np
import pandas as pd

import config as C
from data.preprocessing import build_dataset, rebuild_splits_for_seed
from utils.metrics import evaluate, summarize_multi_seed, format_summary_table
from train import MODEL_TRAINERS


def run_benchmark(models=None, seeds=None, verbose=True):
    models = models or list(C.BENCH.models)
    seeds = seeds or list(C.DATA.seeds)

    print(f"Loading + preprocessing dataset: {C.DATA.dataset_name}")
    hsi_data = build_dataset(C.DATA)
    print(f"  patches={hsi_data.patches.shape}, classes={hsi_data.num_classes}, "
          f"bands={hsi_data.num_bands}")

    all_results = {m: [] for m in models}          # per-seed metric dicts
    raw_predictions = {m: [] for m in models}       # for confusion matrices etc.

    for seed in seeds:
        print(f"\n{'=' * 60}\nSeed {seed}\n{'=' * 60}")
        splits = rebuild_splits_for_seed(hsi_data, C.DATA, seed)
        print(f"  train={len(splits['train'])}  val={len(splits['val'])}  "
              f"test={len(splits['test'])}")

        for model_name in models:
            print(f"\n--- {model_name} (seed={seed}) ---")
            trainer_fn = MODEL_TRAINERS[model_name]
            y_true, y_pred, extra = trainer_fn(hsi_data, splits, seed=seed, verbose=verbose) \
                if model_name != "svm" else trainer_fn(hsi_data, splits, seed=seed)

            metrics = evaluate(y_true, y_pred, hsi_data.num_classes)
            print(f"  -> OA={metrics['OA']:.4f}  AA={metrics['AA']:.4f}  "
                  f"F1={metrics['F1']:.4f}  Kappa={metrics['Kappa']:.4f}")

            all_results[model_name].append(metrics)
            raw_predictions[model_name].append({"seed": seed, "y_true": y_true, "y_pred": y_pred})

    # ---- aggregate ----
    summaries = {m: summarize_multi_seed(all_results[m]) for m in models}

    # ---- persist ----
    os.makedirs(C.BENCH.results_dir, exist_ok=True)
    _save_results(summaries, all_results, models, seeds)

    print("\n\n===== BENCHMARK SUMMARY =====")
    print(format_summary_table(summaries))

    return summaries, all_results, raw_predictions


def _save_results(summaries, all_results, models, seeds):
    # JSON summary (mean/std per metric per model)
    json_path = os.path.join(C.BENCH.results_dir, "benchmark_summary.json")
    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2)

    # Full per-seed CSV (long format) for later analysis / plotting.
    rows = []
    for m in models:
        for seed_idx, metrics in enumerate(all_results[m]):
            rows.append({
                "model": m,
                "seed": seeds[seed_idx],
                "OA": metrics["OA"],
                "AA": metrics["AA"],
                "F1": metrics["F1"],
                "Kappa": metrics["Kappa"],
            })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(C.BENCH.results_dir, "benchmark_per_seed.csv")
    df.to_csv(csv_path, index=False)

    # Markdown table for direct paste into README.
    md_path = os.path.join(C.BENCH.results_dir, "benchmark_summary.md")
    with open(md_path, "w") as f:
        f.write("# Benchmark Results\n\n")
        f.write(format_summary_table(summaries))
        f.write("\n")

    print(f"\nSaved: {json_path}\n       {csv_path}\n       {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full HSI classification benchmark.")
    parser.add_argument("--models", nargs="+", default=None,
                         choices=list(C.BENCH.models), help="Subset of models to run.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                         help="Override the seed list from config.py.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-epoch logs.")
    args = parser.parse_args()

    run_benchmark(models=args.models, seeds=args.seeds, verbose=not args.quiet)
