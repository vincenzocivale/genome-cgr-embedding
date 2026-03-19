"""
Multiscale k-mer RF: combina vettori di frequenze k-mer a diverse scale.

Per ogni sequenza:
  1. Calcola FCGR una volta alla risoluzione massima
  2. Pool a k=4, k=5, k=6 → vettori di 256, 1024, 4096 features
  3. Normalizza L2 ciascuno
  4. Concatena → 5376 features totali
  5. Addestra RF con GridSearchCV

Usage:
    python3 src/fm_experiment/train_multiscale_kmer_rf.py --k-values 4 5 6
"""

import argparse
import sys
import os

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.fm_experiment.kmer_features import extract_multiscale_kmer_features
from src.fm_experiment.records import (
    load_records, has_multiscale, write_multiscale, RECORDS_CSV,
)
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
        description="Train RF su multiscale k-mer features per tutti i dataset"
    )
    parser.add_argument("--data-root", default="/data/genomic_bench/dna_foundation_benchmark/")
    parser.add_argument("--k-values", nargs="+", type=int, default=[4, 5, 6],
                       help="K-mer scales da combinare (default: 4 5 6)")
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    k_values = tuple(sorted(args.k_values))
    feat_dim = sum(4 ** k for k in k_values)

    all_datasets = discover_datasets(args.data_root)
    records = load_records(RECORDS_CSV)

    pending = [ds for ds in all_datasets if not has_multiscale(ds["name"], k_values, records)]

    print(f"\n🔬 Multiscale k-mer RF")
    print(f"   k-values : {list(k_values)}")
    print(f"   Features : {feat_dim} ({' + '.join(f'4^{k}={4**k}' for k in k_values)})")
    print(f"   Dataset  : {len(pending)} da calcolare, "
          f"{len(all_datasets) - len(pending)} già presenti")
    print(f"   Risultati: {RECORDS_CSV}\n")

    pbar = tqdm(pending, unit="ds", ncols=72)

    for ds in pbar:
        name = ds["name"]
        pbar.set_description(f"[{name}]")

        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )
        n_classes = len(set(train_labels))

        X_train = extract_multiscale_kmer_features(
            train_seqs, k_values=list(k_values), n_workers=args.n_workers
        )
        X_test = extract_multiscale_kmer_features(
            test_seqs, k_values=list(k_values), n_workers=args.n_workers
        )

        rf = train_rf(X_train, train_labels, n_classes)
        metrics = eval_rf(rf, X_test, test_labels, n_classes)
        write_multiscale(name, k_values, metrics)
        records = load_records(RECORDS_CSV)

        pbar.set_description(f"✓ {name} MCC={metrics['MCC']:.3f}")

    print(f"\n✅ Completato! Risultati in: {RECORDS_CSV}")


if __name__ == "__main__":
    main()
