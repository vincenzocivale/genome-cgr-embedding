"""
Multiscale k-mer RF: combina vettori di frequenze k-mer a diverse scale.

Per ogni sequenza:
  1. Calcola FCGR una volta alla risoluzione massima
  2. Pool a k=4, k=5, k=6 -> vettori di 256, 1024, 4096 features
  3. Normalizza L2 ciascuno
  4. Concatena -> 5376 features totali
  5. Addestra RF con GridSearchCV

Usage:
    python3 src/fm_experiment/train_multiscale_kmer_rf.py --k-values 4 5 6
"""

import argparse
import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.training.efficiency import log_efficiency
from src.features.kmer_features import extract_multiscale_kmer_features
from src.records.records import (
    RECORDS_CSV,
    has_multiscale,
    load_records,
    write_multiscale,
)
from src.training.rf_pipeline import train_rf, eval_rf


def main():
    parser = argparse.ArgumentParser(
        description="Train RF su multiscale k-mer features per tutti i dataset"
    )
    parser.add_argument(
        "--data-root", default="/data/genomic_bench/dna_foundation_benchmark/"
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[4, 5, 6],
        help="K-mer scales da combinare (default: 4 5 6)",
    )
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    k_values = tuple(sorted(args.k_values))
    feat_dim = sum(4 ** k for k in k_values)

    all_datasets = discover_datasets(args.data_root)
    records = load_records(RECORDS_CSV)
    pending = [
        ds for ds in all_datasets if not has_multiscale(ds["name"], k_values, records)
    ]

    print("\nMultiscale k-mer RF")
    print(f"   k-values : {list(k_values)}")
    print(f"   Features : {feat_dim} ({' + '.join(f'4^{k}={4**k}' for k in k_values)})")
    print(
        f"   Dataset  : {len(pending)} da calcolare, "
        f"{len(all_datasets) - len(pending)} gia presenti"
    )
    print(f"   Results: {RECORDS_CSV}\n")

    pbar = tqdm(pending, unit="ds", ncols=72)

    for ds in pbar:
        name = ds["name"]
        pbar.set_description(f"[{name}]")

        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )
        n_classes = len(set(train_labels))

        t0 = time.perf_counter()
        X_train = extract_multiscale_kmer_features(
            train_seqs, k_values=list(k_values), n_workers=args.n_workers
        )
        X_test = extract_multiscale_kmer_features(
            test_seqs, k_values=list(k_values), n_workers=args.n_workers
        )
        t1 = time.perf_counter()
        log_efficiency(
            name,
            f"multiscale_k{'_'.join(str(k) for k in k_values)}",
            X_train.shape[1],
            t1 - t0,
        )

        rf = train_rf(X_train, train_labels, n_classes)
        metrics = eval_rf(rf, X_test, test_labels, n_classes)
        write_multiscale(name, k_values, metrics)

        records = load_records(RECORDS_CSV)
        pbar.set_description(f"ok {name} MCC={metrics['MCC']:.3f}")

    print(f"\nDone! Risultati in: {RECORDS_CSV}")


if __name__ == "__main__":
    main()
