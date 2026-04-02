"""
Script: genera feature k-mer e addestra RF per diversi k-values su tutti i dataset.

Mostra barra di progresso con dataset e k-values completati/mancanti.

Usage:
    python3 src/fm_experiment/train_kmer_rf.py --k-values 4 5 6 7
    python3 src/fm_experiment/train_kmer_rf.py --k-values 6 --n-workers 8
"""

import argparse
import sys
import os

import time
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.fm_experiment.kmer_features import extract_kmer_features, kmer_from_grids
from src.fm_experiment.records import load_records, has_kmer, write_kmer, RECORDS_CSV
from src.fm_experiment.efficiency import log_efficiency
from src.core.fcgr import batch_fcgr
from src.core.kmer import KmerEmbedder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, accuracy_score


_RF_PARAM_GRID = {
    "n_estimators":      [200, 500],
    "max_features":      ["sqrt"],
    "max_depth":         [20],
    "min_samples_split": [2],
}


def train_rf(X_train, y_train, n_classes):
    """Train RF con GridSearchCV."""
    scoring = "roc_auc" if n_classes == 2 else "accuracy"
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    gs = GridSearchCV(
        RandomForestClassifier(n_jobs=4, random_state=42),
        _RF_PARAM_GRID, scoring=scoring, cv=cv, n_jobs=1, refit=True,
    )
    gs.fit(X_train, y_train)
    return gs


def eval_rf(model, X_test, y_test, n_classes):
    """Valuta RF su test set."""
    y_pred = model.predict(X_test)
    mcc = matthews_corrcoef(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    if n_classes == 2:
        auroc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        auroc = roc_auc_score(y_test, model.predict_proba(X_test),
                              multi_class="ovr", average="macro")
    return {"MCC": mcc, "AUROC": auroc, "F1": f1, "Accuracy": acc}


def main():
    parser = argparse.ArgumentParser(
        description="Train RF su k-mer features per diversi k-values su tutti i dataset"
    )
    parser.add_argument("--data-root", default="/data/genomic_bench/dna_foundation_benchmark/")
    parser.add_argument("--k-values", nargs="+", type=int, required=True,
                       help="K-mer sizes to evaluate (e.g., 4 5 6)")
    parser.add_argument("--n-workers", type=int, default=8,
                       help="Workers per FCGR computation")
    args = parser.parse_args()

    # Carica lista dataset
    all_datasets = discover_datasets(args.data_root)
    n_datasets = len(all_datasets)
    n_k_values = len(args.k_values)
    n_total = n_datasets * n_k_values

    print(f"\n🧬 K-mer RF Training")
    print(f"📊 Dataset: {n_datasets}")
    print(f"📏 K-values: {args.k_values}")
    print(f"📁 Risultati: results/records.csv\n")

    records = load_records(RECORDS_CSV)
    completed = 0
    pbar = tqdm(total=n_total, desc="Progresso", unit="task", ncols=70)

    for ds in all_datasets:
        name = ds["name"]

        # Carica sequenze
        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )
        n_classes = len(set(train_labels))

        # Determina k mancanti
        pending_ks = [k for k in args.k_values if not has_kmer(name, k, records)]

        if not pending_ks:
            completed += len(args.k_values)
            pbar.update(len(args.k_values))
            pbar.set_description(f"⏭  {name}")
            continue

        # Calcola FCGR una sola volta alla risoluzione massima necessaria
        max_k = max(pending_ks)
        grid_size = max(128, 2 ** max_k)
        pbar.set_description(f"[{name}] FCGR grid={grid_size}")
        grids_train = batch_fcgr(train_seqs, grid_size=grid_size, n_workers=args.n_workers)
        grids_test = batch_fcgr(test_seqs, grid_size=grid_size, n_workers=args.n_workers)

        # Loop sui k-values
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
    print(f"\n✅ Completato! {n_total}/{n_total} task")


if __name__ == "__main__":
    main()
