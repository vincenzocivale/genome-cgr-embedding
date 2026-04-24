"""
Extract k-mer features and train a Random Forest on all classification datasets.

Usage:
    python3 src/scripts/train_kmer_rf.py --k-values 4 5 6 7
    python3 src/scripts/train_kmer_rf.py --k-values 6 --n-workers 8
"""

import argparse
import sys
import os

import time
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.features.kmer_features import extract_kmer_features, kmer_from_grids
from src.records.records import load_records, has_kmer, write_kmer, RECORDS_CSV
from src.training.efficiency import log_efficiency
from src.training.rf_pipeline import train_rf, eval_rf
from src.core.fcgr import batch_fcgr


def main():
    parser = argparse.ArgumentParser(
        description="Train RF on k-mer features for all datasets"
    )
    parser.add_argument("--data-root", default="/data/genomic_bench/dna_foundation_benchmark/")
    parser.add_argument("--k-values", nargs="+", type=int, required=True,
                       help="K-mer sizes to evaluate (e.g., 4 5 6)")
    parser.add_argument("--n-workers", type=int, default=2,
                       help="Workers per FCGR computation")
    args = parser.parse_args()

    all_datasets = discover_datasets(args.data_root)
    n_datasets = len(all_datasets)
    n_k_values = len(args.k_values)
    n_total = n_datasets * n_k_values

    print(f"\nK-mer RF Training")
    print(f"Datasets : {n_datasets}")
    print(f"K-values : {args.k_values}")
    print(f"Results  : {RECORDS_CSV}\n")

    records = load_records(RECORDS_CSV)
    completed = 0
    pbar = tqdm(total=n_total, desc="Progress", unit="task", ncols=70)

    for ds in all_datasets:
        name = ds["name"]

        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )
        n_classes = len(set(train_labels))

        # Determine which k-values still need to be computed
        pending_ks = [k for k in args.k_values if not has_kmer(name, k, records)]

        if not pending_ks:
            completed += len(args.k_values)
            pbar.update(len(args.k_values))
            pbar.set_description(f"skip {name}")
            continue

        # Compute FCGR once at the maximum required resolution
        max_k = max(pending_ks)
        grid_size = max(128, 2 ** max_k)
        pbar.set_description(f"[{name}] FCGR grid={grid_size}")
        grids_train = batch_fcgr(train_seqs, grid_size=grid_size, n_workers=args.n_workers)
        grids_test = batch_fcgr(test_seqs, grid_size=grid_size, n_workers=args.n_workers)

        for k in args.k_values:
            completed += 1

            if not has_kmer(name, k, records):
                t0 = time.perf_counter()
                X_train = kmer_from_grids(grids_train, k, normalize="l1")
                X_test = kmer_from_grids(grids_test, k, normalize="l1")
                t1 = time.perf_counter()

                log_efficiency(name, f"kmer_k{k}", X_train.shape[1], t1 - t0)
                rf = train_rf(X_train, train_labels, n_classes)
                metrics = eval_rf(rf, X_test, test_labels, n_classes)
                write_kmer(name, k, metrics)
                records = load_records(RECORDS_CSV)

                pbar.update(1)
                pbar.set_description(f"[{name} k={k}] {completed}/{n_total}")

    pbar.close()
    print(f"\nDone! {n_total}/{n_total} tasks")


if __name__ == "__main__":
    main()
