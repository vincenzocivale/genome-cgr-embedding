"""
Canonical k-mer (reverse-complement aware) on all 57 classification datasets.

Computes FCGR once per dataset, pools to canonical k-mer frequencies, and
trains either RF or linear probe depending on --classifier.

Usage:
    python3 src/scripts/classification/train_canonical_kmer.py --k-values 4 5 6 --classifier rf
    python3 src/scripts/classification/train_canonical_kmer.py --k-values 4 5 6 --classifier linear
    python3 src/scripts/classification/train_canonical_kmer.py --k-values 4 5 6 --classifier rf --n-parallel 16
"""

import argparse
import os
import sys
import time

from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.core.fcgr import batch_fcgr
from src.data.loader import discover_datasets, load_dataset
from src.features.kmer_features import canonicalize_kmer_features, kmer_from_grids
from src.records.records import (
    RECORDS_CANONICAL_CSV,
    has_canon_kmer,
    has_canon_lp_kmer,
    load_canon_records,
    write_canon_kmer,
    write_canon_lp_kmer,
)
from src.training.linear_pipeline import eval_linear_classifier, train_linear_classifier
from src.training.rf_pipeline import eval_rf, train_rf


def _process_dataset(ds, k_values, classifier, n_workers, n_jobs_lr=4):
    """Process one dataset for all k_values. Returns list of (name, k, metrics)."""
    name = ds["name"]
    records = load_canon_records(RECORDS_CANONICAL_CSV)

    if classifier == "rf":
        pending_ks = [k for k in k_values if not has_canon_kmer(name, k, records)]
    else:
        pending_ks = [k for k in k_values if not has_canon_lp_kmer(name, k, records)]

    if not pending_ks:
        return []

    train_seqs, train_labels, test_seqs, test_labels = load_dataset(
        ds["train_path"], ds["test_path"]
    )
    n_classes = len(set(train_labels))

    max_k = max(pending_ks)
    grid_size = max(128, 2 ** max_k)
    grids_train = batch_fcgr(train_seqs, grid_size=grid_size, n_workers=n_workers)
    grids_test = batch_fcgr(test_seqs, grid_size=grid_size, n_workers=n_workers)

    results = []
    for k in k_values:
        records = load_canon_records(RECORDS_CANONICAL_CSV)
        if classifier == "rf" and has_canon_kmer(name, k, records):
            continue
        if classifier == "linear" and has_canon_lp_kmer(name, k, records):
            continue

        # Standard k-mer → canonical
        X_train_std = kmer_from_grids(grids_train, k, normalize="l1")
        X_test_std = kmer_from_grids(grids_test, k, normalize="l1")
        X_train = canonicalize_kmer_features(X_train_std, k)
        X_test = canonicalize_kmer_features(X_test_std, k)

        t0 = time.perf_counter()
        if classifier == "rf":
            model = train_rf(X_train, train_labels, n_classes)
            metrics = eval_rf(model, X_test, test_labels, n_classes)
        else:
            model = train_linear_classifier(X_train, train_labels, n_classes, n_jobs=n_jobs_lr)
            metrics = eval_linear_classifier(model, X_test, test_labels, n_classes)
        elapsed = time.perf_counter() - t0

        results.append((name, k, metrics, elapsed))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Canonical k-mer (RC-aware) features on all classification datasets"
    )
    parser.add_argument("--data-root", default="data/dna_foundation_benchmark/")
    parser.add_argument("--k-values", nargs="+", type=int, default=[4, 5, 6])
    parser.add_argument("--classifier", choices=["rf", "linear"], default="rf")
    parser.add_argument("--n-workers", type=int, default=2,
                        help="Workers for FCGR computation per dataset")
    parser.add_argument("--n-parallel", type=int, default=2,
                        help="Parallel datasets")
    parser.add_argument("--n-jobs-lr", type=int, default=4,
                        help="Threads for LogisticRegressionCV per dataset (linear mode)")
    args = parser.parse_args()

    all_datasets = discover_datasets(args.data_root)
    n_datasets = len(all_datasets)

    print(f"\nCanonical k-mer — {args.classifier.upper()} classifier")
    print(f"Datasets   : {n_datasets}")
    print(f"K-values   : {args.k_values}")
    print(f"n-parallel : {args.n_parallel}")
    print(f"Results    : {RECORDS_CANONICAL_CSV}\n")

    # Quick check on canonical sizes
    from src.features.kmer_features import _build_canonical_map
    for k in args.k_values:
        _, n_canon = _build_canonical_map(k)
        print(f"  k={k}: {4**k} -> {n_canon} canonical features")
    print()

    records = load_canon_records(RECORDS_CANONICAL_CSV)
    if args.classifier == "rf":
        pending = [
            ds for ds in all_datasets
            if any(not has_canon_kmer(ds["name"], k, records) for k in args.k_values)
        ]
    else:
        pending = [
            ds for ds in all_datasets
            if any(not has_canon_lp_kmer(ds["name"], k, records) for k in args.k_values)
        ]

    skipped = n_datasets - len(pending)
    print(f"Pending: {len(pending)} | Skipped (done): {skipped}\n")

    if not pending:
        print("All datasets already processed.")
        return

    completed = [0]
    total = len(pending)

    def _run_and_save(ds):
        results = _process_dataset(
            ds, args.k_values, args.classifier, args.n_workers, args.n_jobs_lr
        )
        for name, k, metrics, elapsed in results:
            if args.classifier == "rf":
                write_canon_kmer(name, k, metrics)
            else:
                write_canon_lp_kmer(name, k, metrics)
            completed[0] += 1
            print(
                f"  [{completed[0]:3d}/{total}] {name} k={k} [{args.classifier}] "
                f"MCC={metrics['MCC']:.3f} AUROC={metrics['AUROC']:.3f} ({elapsed:.0f}s)",
                flush=True,
            )
        return ds["name"]

    Parallel(n_jobs=args.n_parallel, backend="loky")(
        delayed(_run_and_save)(ds) for ds in pending
    )

    print(f"\nDone! Results in {RECORDS_CANONICAL_CSV}")


if __name__ == "__main__":
    main()
