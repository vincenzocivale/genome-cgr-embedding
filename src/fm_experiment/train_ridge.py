"""
Script: Ridge Regression k-mer → FM embeddings per tutti i dataset.

Per ogni (dataset, k) addestra RidgeCV con:
  X = vettore frequenze k-mer  (4^k features)
  Y = embedding FM             (D features)

Salva R² e MSE globali in records.csv.
Salta le configurazioni già presenti.

Usage:
    python3 src/fm_experiment/train_ridge.py --model InstaDeepAI/NTv3_650M_pre --k-values 4 6 8
    python3 src/fm_experiment/train_ridge.py --model LongSafari/hyenadna-medium-160k-seqlen-hf --k-values 6
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.fm_experiment.kmer_features import extract_kmer_features
from src.fm_experiment.ridge_mapping import fit_and_evaluate
from src.fm_experiment.records import (
    load_records, has_ridge, write_ridge, RECORDS_CSV,
)


def load_embeddings(model_name: str, dataset_name: str, split: str) -> np.ndarray | None:
    """Carica embeddings dalla cache .npz se presenti, altrimenti None.

    Gestisce due strutture di cache:
      - fm_embedder  : cache/fm_embeddings/{model_tag}/{dataset}/{split}.npz
      - hyena_embedder: cache/hyena_embeddings/{dataset}/{split}.npz  (no model subdir)
    """
    safe_model = model_name.replace("/", "__")
    safe_ds = dataset_name.replace("/", "__").replace("\\", "__")

    candidates = [
        os.path.join("cache/fm_embeddings", safe_model, safe_ds, f"{split}.npz"),
    ]
    if "hyenadna" in model_name.lower():
        candidates.append(
            os.path.join("cache/hyena_embeddings", safe_ds, f"{split}.npz"),
        )
    for path in candidates:
        if os.path.exists(path):
            return np.load(path)["embeddings"].astype(np.float32)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Ridge Regression k-mer → FM embeddings su tutti i dataset"
    )
    parser.add_argument("--data-root", default="/data/genomic_bench/dna_foundation_benchmark/")
    parser.add_argument("--model", required=True, help="FM model HF name")
    parser.add_argument("--k-values", nargs="+", type=int, required=True,
                        help="K-mer sizes (es. 4 5 6)")
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=50000,
                        help="Skip dataset con più di N campioni di training (default 50000)")
    args = parser.parse_args()

    all_datasets = discover_datasets(args.data_root)
    n_total = len(all_datasets) * len(args.k_values)

    print(f"\n📐 Ridge Regression: k-mer → FM embeddings")
    print(f"   FM model : {args.model}")
    print(f"   k-values : {args.k_values}")
    print(f"   Dataset  : {len(all_datasets)}")
    print(f"   Task     : {n_total}\n")

    records = load_records(RECORDS_CSV)
    pbar = tqdm(total=n_total, unit="task", ncols=72)

    for ds in all_datasets:
        name = ds["name"]

        # Controlla se tutti i k-values sono già presenti → skip caricamento sequenze
        pending_ks = [k for k in args.k_values if not has_ridge(name, k, args.model, records)]

        if not pending_ks:
            pbar.update(len(args.k_values))
            pbar.set_description(f"⏭  {name}")
            continue

        # Carica embeddings FM dalla cache
        Y_train = load_embeddings(args.model, name, "train")
        Y_test = load_embeddings(args.model, name, "test")

        if Y_train is None or Y_test is None:
            pbar.update(len(args.k_values))
            pbar.set_description(f"⚠️  {name} (no FM cache)")
            continue

        if Y_train.shape[0] > args.max_train_samples:
            pbar.update(len(args.k_values))
            pbar.set_description(f"⚠️  {name} (n={Y_train.shape[0]} > {args.max_train_samples})")
            continue

        # Carica sequenze solo per i k-values mancanti
        train_seqs, _, test_seqs, _ = load_dataset(ds["train_path"], ds["test_path"])

        for k in args.k_values:
            if not has_ridge(name, k, args.model, records):
                grid_size = max(128, 2 ** k)
                X_train = extract_kmer_features(train_seqs, k=k, grid_size=grid_size,
                                                n_workers=args.n_workers)
                X_test = extract_kmer_features(test_seqs, k=k, grid_size=grid_size,
                                               n_workers=args.n_workers)

                result = fit_and_evaluate(X_train, Y_train, X_test, Y_test)
                write_ridge(name, k, args.model,
                            {"R2": result["r2_global"], "MSE": result["mse_global"]})
                records = load_records(RECORDS_CSV)
                pbar.set_description(f"[{name}] k={k} R²={result['r2_global']:.3f}")
            else:
                pbar.set_description(f"⏭  [{name}] k={k}")

            pbar.update(1)

    pbar.close()
    print(f"\n✅ Completato! Risultati in: {RECORDS_CSV}")


if __name__ == "__main__":
    main()
