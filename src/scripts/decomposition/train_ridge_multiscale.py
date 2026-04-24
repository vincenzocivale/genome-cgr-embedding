"""
Ridge Regression from multiscale k-mer features to FM embeddings.

Trains RidgeCV with:
  X = multiscale k-mer frequency vector (concatenation of 4^k per each k)
  Y = FM embedding                      (D dimensions)

Writes global R² and MSE to records.csv.

Usage:
    python3 src/scripts/train_ridge_multiscale.py --model InstaDeepAI/NTv3_650M_pre --k-values 4 5 6
    python3 src/scripts/train_ridge_multiscale.py --model LongSafari/hyenadna-medium-160k-seqlen-hf --k-values 4 5 6
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.features.kmer_features import extract_multiscale_kmer_features
from src.training.ridge_mapping import fit_and_evaluate
from src.records.records import (
    load_records, has_ridge_multi, write_ridge_multi, RECORDS_CSV,
)
from src.embedders.embedding_cache import load_cached_embeddings as load_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Ridge Regression multiscale k-mer → FM embeddings"
    )
    parser.add_argument("--data-root", default="/data/genomic_bench/dna_foundation_benchmark/")
    parser.add_argument("--model", required=True, help="FM model HF name")
    parser.add_argument("--k-values", nargs="+", type=int, default=[4, 5, 6],
                        help="K-mer scales da combinare (default: 4 5 6)")
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=0,
                        help="Subsample dataset con più di N campioni (0=nessun limite)")
    args = parser.parse_args()

    k_values = tuple(sorted(args.k_values))
    feat_dim = sum(4 ** k for k in k_values)

    all_datasets = discover_datasets(args.data_root)
    records = load_records(RECORDS_CSV)

    pending = [ds for ds in all_datasets
               if not has_ridge_multi(ds["name"], k_values, args.model, records)]

    print(f"\n📐 Ridge Regression: multiscale k-mer → FM embeddings")
    print(f"   FM model : {args.model}")
    print(f"   k-values : {list(k_values)}")
    print(f"   Features : {feat_dim} ({' + '.join(f'4^{k}={4**k}' for k in k_values)})")
    print(f"   Dataset  : {len(pending)} da calcolare, "
          f"{len(all_datasets) - len(pending)} già presenti")
    print(f"   Results: {RECORDS_CSV}\n")

    pbar = tqdm(pending, unit="ds", ncols=72)

    for ds in pbar:
        name = ds["name"]
        pbar.set_description(f"[{name}]")

        # Load FM embeddings
        Y_train = load_embeddings(args.model, name, "train")
        Y_test = load_embeddings(args.model, name, "test")

        if Y_train is None or Y_test is None:
            pbar.set_description(f"⚠️  {name} (no FM cache)")
            continue

        # Load sequences
        train_seqs, _, test_seqs, _ = load_dataset(ds["train_path"], ds["test_path"])

        if args.max_train_samples > 0 and Y_train.shape[0] > args.max_train_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(Y_train.shape[0], args.max_train_samples, replace=False)
            idx.sort()
            Y_train = Y_train[idx]
            train_seqs = [train_seqs[i] for i in idx]
            pbar.set_description(f"⚠️  {name} subsampled {args.max_train_samples}")

        # Extract multiscale features
        X_train = extract_multiscale_kmer_features(
            train_seqs, k_values=list(k_values), n_workers=args.n_workers
        )
        X_test = extract_multiscale_kmer_features(
            test_seqs, k_values=list(k_values), n_workers=args.n_workers
        )

        # Ridge regression
        result = fit_and_evaluate(X_train, Y_train, X_test, Y_test)
        write_ridge_multi(name, k_values, args.model,
                          {"R2": result["r2_global"], "MSE": result["mse_global"]})
        records = load_records(RECORDS_CSV)

        pbar.set_description(f"✓ {name} R²={result['r2_global']:.3f}")

    pbar.close()
    print(f"\n✅ Done! Risultati in: {RECORDS_CSV}")


if __name__ == "__main__":
    main()
