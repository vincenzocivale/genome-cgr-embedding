"""
Train RF on weighted multiscale k-mer features for all classification datasets.

Searches for per-dataset log-scale weights across k-mer scales (k=4,5,6) via
cross-validation, then trains RF on the optimally weighted concatenated vectors.

Usage:
    python3 src/scripts/classification/train_weighted_multiscale_rf.py
    python3 src/scripts/classification/train_weighted_multiscale_rf.py \\
        --k-values 4 5 6 --weighting grid
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
from src.records.records import (
    load_records, has_wms, write_wms, RECORDS_CSV,
)
from src.training.efficiency import log_efficiency
from src.features.kmer_features import kmer_from_grids
from src.training.rf_pipeline import train_rf, eval_rf
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


def _concat_weighted(feats_list, weights):
    parts = [f * float(w) for f, w in zip(feats_list, weights)]
    return np.concatenate(parts, axis=1)


def main():
    parser = argparse.ArgumentParser(
        description="Train RF su weighted multiscale k-mer features"
    )
    parser.add_argument("--data-root", default="data/dna_foundation_benchmark/")
    parser.add_argument("--k-values", nargs="+", type=int, default=[4, 5, 6])
    parser.add_argument("--weighting", choices=["uniform", "grid"], default="grid")
    parser.add_argument("--grid-values", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Filter datasets (exact match)")
    args = parser.parse_args()

    k_values = tuple(sorted(args.k_values))
    feat_dim = sum(4 ** k for k in k_values)

    all_datasets = discover_datasets(args.data_root)
    if args.datasets:
        all_datasets = [d for d in all_datasets if d["name"] in args.datasets]
    records = load_records(RECORDS_CSV)

    print("\n⚖️  Weighted Multiscale k-mer RF")
    print(f"   k-values : {list(k_values)}")
    print(f"   weighting: {args.weighting}")
    print(f"   Features : {feat_dim} ({' + '.join(f'4^{k}={4**k}' for k in k_values)})")
    print(f"   Dataset  : {len(all_datasets)}")
    print(f"   Results: {RECORDS_CSV}\n")

    pbar = tqdm(all_datasets, unit="ds", ncols=72)

    for ds in pbar:
        name = ds["name"]
        if has_wms(name, k_values, args.weighting, records):
            pbar.set_description(f"⏭ {name}")
            continue

        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )
        n_classes = len(set(train_labels))

        t0 = time.perf_counter()
        max_k = max(k_values)
        grid_size = max(128, 2 ** max_k)
        grids_train = batch_fcgr(train_seqs, grid_size=grid_size, n_workers=args.n_workers)
        grids_test = batch_fcgr(test_seqs, grid_size=grid_size, n_workers=args.n_workers)

        feats_train = []
        feats_test = []
        for k in k_values:
            X_tr = kmer_from_grids(grids_train, k, normalize="l1")
            X_te = kmer_from_grids(grids_test, k, normalize="l1")
            X_tr = X_tr / np.where(
                np.linalg.norm(X_tr, axis=1, keepdims=True) == 0,
                1.0,
                np.linalg.norm(X_tr, axis=1, keepdims=True),
            )
            X_te = X_te / np.where(
                np.linalg.norm(X_te, axis=1, keepdims=True) == 0,
                1.0,
                np.linalg.norm(X_te, axis=1, keepdims=True),
            )
            feats_train.append(X_tr.astype(np.float32))
            feats_test.append(X_te.astype(np.float32))

        # choose weights
        if args.weighting == "uniform":
            best_weights = [1.0 for _ in k_values]
        else:
            weight_grid = list(product(args.grid_values, repeat=len(k_values)))
            scoring = "roc_auc" if n_classes == 2 else "accuracy"
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            clf = LogisticRegression(max_iter=1000, random_state=42)
            best_score = -1.0
            best_weights = list(weight_grid[0])
            for weights in weight_grid:
                X_cv = _concat_weighted(feats_train, weights)
                scores = cross_val_score(clf, X_cv, train_labels,
                                         cv=cv, scoring=scoring, n_jobs=1)
                score = float(np.mean(scores))
                if score > best_score:
                    best_score = score
                    best_weights = list(weights)

        X_train = _concat_weighted(feats_train, best_weights)
        X_test = _concat_weighted(feats_test, best_weights)
        t1 = time.perf_counter()

        log_efficiency(name, f"wms_k{'_'.join(str(k) for k in k_values)}_{args.weighting}",
                       X_train.shape[1], t1 - t0)

        rf = train_rf(X_train, train_labels, n_classes)
        metrics = eval_rf(rf, X_test, test_labels, n_classes)
        write_wms(name, k_values, args.weighting, metrics, weights=best_weights)
        records = load_records(RECORDS_CSV)

        pbar.set_description(f"✓ {name} MCC={metrics['MCC']:.3f}")

    print(f"\n✅ Done! Risultati in: {RECORDS_CSV}")


if __name__ == "__main__":
    main()
