"""
Train RF on Adaptive QuadTree CGR features for all classification datasets.

QuadTree CGR performs chi-square driven recursive splitting of the FCGR grid,
producing a fixed-length Z-order (Morton code) feature vector. Cell splitting
stops when chi-square p-value exceeds a threshold or minimum count is reached.

Usage:
    python3 src/scripts/classification/train_quadtree_rf.py
    python3 src/scripts/classification/train_quadtree_rf.py \\
        --max-depths 6 7 8 --p-thresholds 0.01 0.05 --min-counts 0 4
"""

import argparse
import sys
import os
import time
from itertools import product

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.core.fcgr import batch_fcgr
from src.core.quadtree import quadtree_features
from src.records.records import (
    load_records, has_quadtree, write_quadtree, RECORDS_CSV,
)
from src.training.efficiency import log_efficiency
from src.training.rf_pipeline import train_rf, eval_rf


def _tag_float(x: float) -> str:
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    if s == "":
        s = "0"
    return s.replace(".", "p")


def main():
    parser = argparse.ArgumentParser(
        description="Train RF su Adaptive QuadTree CGR per tutti i dataset"
    )
    parser.add_argument("--data-root", default="/data/genomic_bench/dna_foundation_benchmark/")
    parser.add_argument("--max-depths", nargs="+", type=int, default=[6, 7, 8])
    parser.add_argument("--p-thresholds", nargs="+", type=float, default=[0.01, 0.05, 0.1])
    parser.add_argument("--min-counts", nargs="+", type=int, default=[0, 4, 8])
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Filter datasets (exact match)")
    args = parser.parse_args()

    all_datasets = discover_datasets(args.data_root)
    if args.datasets:
        all_datasets = [d for d in all_datasets if d["name"] in args.datasets]
    records = load_records(RECORDS_CSV)

    combos = list(product(args.max_depths, args.p_thresholds, args.min_counts))

    print("\n🌲 Adaptive QuadTree RF")
    print(f"   Depths     : {args.max_depths}")
    print(f"   p-threshold: {args.p_thresholds}")
    print(f"   min-counts : {args.min_counts}")
    print(f"   Dataset    : {len(all_datasets)}")
    print(f"   Risultati  : {RECORDS_CSV}\n")

    pbar = tqdm(all_datasets, unit="ds", ncols=72)

    for ds in pbar:
        name = ds["name"]
        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )
        n_classes = len(set(train_labels))

        # Compute grids once at max depth and downsample for smaller depths
        max_depth_global = max(args.max_depths)
        grid_size_global = 2 ** max_depth_global
        grids_train_global = batch_fcgr(
            train_seqs, grid_size=grid_size_global, n_workers=args.n_workers
        )
        grids_test_global = batch_fcgr(
            test_seqs, grid_size=grid_size_global, n_workers=args.n_workers
        )

        for max_depth in args.max_depths:
            grid_size = 2 ** max_depth
            if max_depth == max_depth_global:
                grids_train = grids_train_global
                grids_test = grids_test_global
            else:
                factor = grid_size_global // grid_size
                grids_train = [
                    g.reshape(grid_size, factor, grid_size, factor).sum(axis=(1, 3))
                    for g in grids_train_global
                ]
                grids_test = [
                    g.reshape(grid_size, factor, grid_size, factor).sum(axis=(1, 3))
                    for g in grids_test_global
                ]

            for p_threshold, min_count in product(args.p_thresholds, args.min_counts):
                if has_quadtree(name, max_depth, p_threshold, min_count, records):
                    continue

                t0 = time.perf_counter()
                X_train = np.array([
                    quadtree_features(g, max_depth=max_depth,
                                      p_threshold=p_threshold, min_count=min_count)
                    for g in grids_train
                ])
                X_test = np.array([
                    quadtree_features(g, max_depth=max_depth,
                                      p_threshold=p_threshold, min_count=min_count)
                    for g in grids_test
                ])
                t1 = time.perf_counter()

                tag = f"qt_d{max_depth}_p{_tag_float(p_threshold)}_m{min_count}"
                log_efficiency(name, tag, X_train.shape[1], t1 - t0)

                rf = train_rf(X_train, train_labels, n_classes)
                metrics = eval_rf(rf, X_test, test_labels, n_classes)
                write_quadtree(name, max_depth, p_threshold, min_count, metrics)
                records = load_records(RECORDS_CSV)

        pbar.set_description(f"✓ {name}")

    print(f"\n✅ Done! Risultati in: {RECORDS_CSV}")


if __name__ == "__main__":
    main()
