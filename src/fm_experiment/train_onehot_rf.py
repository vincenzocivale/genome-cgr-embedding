"""
Train RF con one-hot encoding della finestra centrale su tutti i dataset di classificazione.

Segue lo stesso pattern di train_kmer_rf.py e scrive risultati in results/records.csv
con colonne onehot_{W}bp_{metric}.

Usage:
    python3 src/fm_experiment/train_onehot_rf.py --windows 512 1024 2048
    python3 src/fm_experiment/train_onehot_rf.py --windows 512 --n-workers 8
"""

import argparse
import os
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.fm_experiment.records import (
    RF_METRICS,
    RECORDS_CSV,
    _ensure_row,
    load_records,
    _save,
)

# ── config ─────────────────────────────────────────────────────────────────────

_RF_PARAM_GRID = {
    "n_estimators":      [200, 500],
    "max_features":      ["sqrt"],
    "max_depth":         [20],
    "min_samples_split": [2],
}

_NUC_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


# ── one-hot ────────────────────────────────────────────────────────────────────

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


# ── RF ──────────────────────────────────────────────────────────────────────────

def train_rf(X_train, y_train, n_classes):
    scoring = "roc_auc" if n_classes == 2 else "accuracy"
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    gs = GridSearchCV(
        RandomForestClassifier(n_jobs=4, random_state=42),
        _RF_PARAM_GRID, scoring=scoring, cv=cv, n_jobs=1, refit=True,
    )
    gs.fit(X_train, y_train)
    return gs


def eval_rf(model, X_test, y_test, n_classes):
    y_pred = model.predict(X_test)
    mcc  = matthews_corrcoef(y_test, y_pred)
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="macro")
    if n_classes == 2:
        auroc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        auroc = roc_auc_score(y_test, model.predict_proba(X_test),
                              multi_class="ovr", average="macro")
    return {"MCC": mcc, "AUROC": auroc, "F1": f1, "Accuracy": acc}


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
    parser.add_argument("--data-root", default="/data/genomic_bench/dna_foundation_benchmark/")
    parser.add_argument("--windows", nargs="+", type=int, default=[512, 1024, 2048],
                        metavar="W", help="Finestre centrali in bp (default: 512 1024 2048)")
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    all_datasets = discover_datasets(args.data_root)
    n_total = len(all_datasets) * len(args.windows)

    print(f"\nOne-hot RF Training")
    print(f"Dataset: {len(all_datasets)}")
    print(f"Finestre: {args.windows} bp")
    print(f"Risultati: {RECORDS_CSV}\n")

    records = load_records(RECORDS_CSV)
    pbar = tqdm(total=n_total, desc="Progresso", unit="task", ncols=70)

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
    print(f"\nCompletato! Risultati in {RECORDS_CSV}")


if __name__ == "__main__":
    main()
