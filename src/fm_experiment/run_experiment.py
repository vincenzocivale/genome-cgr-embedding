"""
Main orchestrator: for each dataset, extract k-mer features and FM embeddings,
fit Ridge Regression (k-mer → FM embedding), evaluate R² and MSE.

Usage:
    python3 src/fm_experiment/run_experiment.py --n-workers 8 --k-values 6
    python3 src/fm_experiment/run_experiment.py --datasets mouse --k-values 4 6
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.fm_experiment.fm_embedder import FMEmbedder
from src.fm_experiment.kmer_features import extract_kmer_features
from src.fm_experiment.ridge_mapping import fit_and_evaluate


RESULTS_DIR = "results/fm_approximation"
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")
RESULTS_PARQUET = os.path.join(RESULTS_DIR, "results.parquet")
PER_DIM_DIR = os.path.join(RESULTS_DIR, "per_dim")


# ------------------------------------------------------------------
# Results persistence
# ------------------------------------------------------------------
def load_existing() -> pd.DataFrame:
    if os.path.exists(RESULTS_CSV):
        return pd.read_csv(RESULTS_CSV)
    return pd.DataFrame()


def is_done(dataset: str, k: int, df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    return ((df["dataset"] == dataset) & (df["k"] == k)).any()


def save_result(record: dict, df: pd.DataFrame) -> pd.DataFrame:
    new = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    new.to_csv(RESULTS_CSV, index=False)
    new.to_parquet(RESULTS_PARQUET, index=False)
    return new


def save_per_dim(dataset: str, k: int, r2: np.ndarray, mse: np.ndarray):
    safe = dataset.replace("/", "__").replace("\\", "__")
    d = os.path.join(PER_DIM_DIR, safe)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, f"r2_k{k}.npy"), r2)
    np.save(os.path.join(d, f"mse_k{k}.npy"), mse)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="k-mer → FM embedding Ridge mapping")
    parser.add_argument(
        "--data-root",
        default="/data/genomic_bench/dna_foundation_benchmark/",
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=[6])
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--fm-batch-size", type=int, default=4)
    parser.add_argument(
        "--model", default="InstaDeepAI/NTv3_650M_pre",
        help="HuggingFace model ID for the foundation model",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Filter datasets (substring match on name)",
    )
    parser.add_argument("--list", action="store_true", help="List datasets and exit")
    args = parser.parse_args()

    datasets = discover_datasets(args.data_root)

    if args.list:
        for d in datasets:
            print(d["name"])
        return

    if args.datasets:
        datasets = [
            d for d in datasets
            if d["name"] in args.datasets
        ]

    print(f"Datasets: {len(datasets)} | k-values: {args.k_values} | model: {args.model}")

    existing = load_existing()
    fm = FMEmbedder(model_name=args.model)

    for ds in datasets:
        name = ds["name"]
        train_seqs, _, test_seqs, _ = load_dataset(ds["train_path"], ds["test_path"])

        # --- FM embeddings (cached) ---
        Y_train = fm.embed_sequences(train_seqs, name, "train", args.fm_batch_size)
        Y_test = fm.embed_sequences(test_seqs, name, "test", args.fm_batch_size)

        for k in args.k_values:
            if is_done(name, k, existing):
                print(f"  [{name} k={k}] skip (already done)")
                continue

            # --- k-mer features ---
            X_train = extract_kmer_features(
                train_seqs, k=k, grid_size=128, n_workers=args.n_workers
            )
            X_test = extract_kmer_features(
                test_seqs, k=k, grid_size=128, n_workers=args.n_workers
            )

            # --- Ridge mapping (cast to float32: sklearn doesn't support float16) ---
            metrics = fit_and_evaluate(
                X_train,
                Y_train.astype(np.float32),
                X_test,
                Y_test.astype(np.float32),
            )

            # --- Save ---
            save_per_dim(name, k, metrics["r2_per_dim"], metrics["mse_per_dim"])

            record = {
                "dataset": name,
                "k": k,
                "r2_global": metrics["r2_global"],
                "mse_global": metrics["mse_global"],
                "r2_median_dim": metrics["r2_median_dim"],
                "best_alpha": metrics["best_alpha"],
                "n_train": metrics["n_train"],
                "n_test": metrics["n_test"],
                "n_kmer_features": metrics["n_kmer_features"],
                "n_embed_dims": metrics["n_embed_dims"],
                "model": args.model,
                "timestamp": datetime.now().isoformat(),
            }
            existing = save_result(record, existing)
            print(
                f"  [{name} k={k}] R²={metrics['r2_global']:.4f}  "
                f"MSE={metrics['mse_global']:.6f}  α={metrics['best_alpha']}"
            )

    print(f"\nDone. Results in {RESULTS_CSV}")


if __name__ == "__main__":
    main()
