"""
Train RF con one-hot encoding della finestra centrale su tutti i dataset di classificazione.

Segue lo stesso pattern di `train_kmer_rf.py` e scrive risultati in
`results/classification/records_rf.csv`
con colonne onehot_{W}bp_{metric}.

Usage:
    python3 src/scripts/classification/train_onehot_rf.py --windows 512 1024 2048
    python3 src/scripts/classification/train_onehot_rf.py --windows 512 --n-workers 8
"""

import argparse
import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.records.records import (
    RF_METRICS,
    RECORDS_CSV,
    _ensure_row,
    load_records,
    _save,
)
from src.training.rf_pipeline import train_rf, eval_rf

_NUC_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def onehot_encode(sequences: np.ndarray, window: int) -> np.ndarray:
    """
    One-hot encode the central `window` bp of each sequence.
    Returns (N, 4*window) float32. Unknown nucleotides → all-zeros.
    """
    N = len(sequences)
    X = np.zeros((N, 4 * window), dtype=np.float32)
    for i, seq in enumerate(sequences):
        seq = seq.upper()
        mid = len(seq) // 2
        half = window // 2
        fragment = seq[mid - half: mid - half + window]
        for j, nuc in enumerate(fragment):
            idx = _NUC_IDX.get(nuc)
            if idx is not None:
                X[i, j * 4 + idx] = 1.0
    return X


# ── records helpers ─────────────────────────────────────────────────────────────

def onehot_col(window: int, metric: str) -> str:
    return f"onehot_{window}bp_{metric}"


def has_onehot(dataset: str, window: int, df) -> bool:
    if df.empty or dataset not in df.index:
        return False
    col = onehot_col(window, RF_METRICS[0])
    return col in df.columns and not __import__("pandas").isna(df.loc[dataset, col])


def write_onehot(dataset: str, window: int, rf_metrics: dict,
                 path: str = RECORDS_CSV):
    df = _ensure_row(load_records(path), dataset)
    new_cols = {onehot_col(window, m): rf_metrics[m] for m in RF_METRICS}
    df = df.assign(**new_cols)
    df = df.copy()
    _save(df, path)


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train RF con one-hot della finestra centrale su tutti i dataset"
    )
    parser.add_argument("--data-root", default="data/dna_foundation_benchmark/")
    parser.add_argument("--windows", nargs="+", type=int, default=[512, 1024, 2048],
                        metavar="W", help="Finestre centrali in bp (default: 512 1024 2048)")
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    all_datasets = discover_datasets(args.data_root)
    n_total = len(all_datasets) * len(args.windows)

    print(f"\nOne-hot RF Training")
    print(f"Dataset: {len(all_datasets)}")
    print(f"Finestre: {args.windows} bp")
    print(f"Results: {RECORDS_CSV}\n")

    records = load_records(RECORDS_CSV)
    pbar = tqdm(total=n_total, desc="Progress", unit="task", ncols=70)

    for ds in all_datasets:
        name = ds["name"]
        pending = [W for W in args.windows if not has_onehot(name, W, records)]

        if not pending:
            pbar.update(len(args.windows))
            pbar.set_description(f"skip {name}")
            continue

        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )
        n_classes = len(set(train_labels))

        for W in args.windows:
            pbar.set_description(f"[{name}] onehot_{W}bp")
            if has_onehot(name, W, records):
                pbar.update(1)
                continue

            X_train = onehot_encode(train_seqs, W)
            X_test  = onehot_encode(test_seqs,  W)

            rf = train_rf(X_train, train_labels, n_classes)
            metrics = eval_rf(rf, X_test, test_labels, n_classes)
            write_onehot(name, W, metrics)
            records = load_records(RECORDS_CSV)
            pbar.update(1)

    pbar.close()
    print(f"\nDone! Risultati in {RECORDS_CSV}")


if __name__ == "__main__":
    main()
