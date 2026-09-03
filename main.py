"""
main.py
========
Top-level CLI entry point.

Examples
--------
Run full preprocessing sanity check:
    python main.py preprocess

Train a single model:
    python main.py train --model transformer --seed 0

Run the entire benchmark (all 5 models x all seeds):
    python main.py benchmark

Run benchmark on a subset:
    python main.py benchmark --models svm 3dcnn transformer --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import config as C


def cmd_preprocess(args):
    from data.preprocessing import build_dataset
    hsi_data = build_dataset(C.DATA)
    print(f"Dataset: {C.DATA.dataset_name}")
    print(f"  Patches shape : {hsi_data.patches.shape}")
    print(f"  Num classes   : {hsi_data.num_classes}")
    print(f"  Num bands     : {hsi_data.num_bands}")
    print(f"  Patch size    : {hsi_data.patch_size}")
    for split_name, idx in hsi_data.splits.items():
        print(f"  {split_name:5s}: {len(idx)} samples")


def cmd_train(args):
    from data.preprocessing import build_dataset, rebuild_splits_for_seed
    from train import MODEL_TRAINERS
    from utils.metrics import evaluate

    hsi_data = build_dataset(C.DATA)
    splits = rebuild_splits_for_seed(hsi_data, C.DATA, args.seed)

    trainer_fn = MODEL_TRAINERS[args.model]
    if args.model == "svm":
        y_true, y_pred, extra = trainer_fn(hsi_data, splits, seed=args.seed)
    else:
        y_true, y_pred, extra = trainer_fn(hsi_data, splits, seed=args.seed, verbose=not args.quiet)

    metrics = evaluate(y_true, y_pred, hsi_data.num_classes)
    print(f"\n{args.model} | seed={args.seed}")
    print(f"  OA={metrics['OA']:.4f}  AA={metrics['AA']:.4f}  "
          f"F1={metrics['F1']:.4f}  Kappa={metrics['Kappa']:.4f}")


def cmd_benchmark(args):
    from benchmark import run_benchmark
    run_benchmark(models=args.models, seeds=args.seeds, verbose=not args.quiet)


def main():
    parser = argparse.ArgumentParser(description="Deep Learning for Hyperspectral Imagery Classification")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("preprocess", help="Run preprocessing pipeline and print dataset stats.")
    p_pre.set_defaults(func=cmd_preprocess)

    p_train = sub.add_parser("train", help="Train a single model.")
    p_train.add_argument("--model", required=True, choices=list(C.BENCH.models))
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--quiet", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_bench = sub.add_parser("benchmark", help="Run the full multi-model benchmark.")
    p_bench.add_argument("--models", nargs="+", default=None, choices=list(C.BENCH.models))
    p_bench.add_argument("--seeds", nargs="+", type=int, default=None)
    p_bench.add_argument("--quiet", action="store_true")
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
