"""
Train RF on wavelet / spatial pyramid CGR features.
"""

import argparse
import sys
import os
import time

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.fm_experiment.kmer_features import extract_wavelet_features
from src.fm_experiment.records import (
    load_records, has_wavelet, write_wavelet, RECORDS_CSV,
)
from src.fm_experiment.efficiency import log_efficiency
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
        description="Train RF su wavelet/spatial pyramid CGR per tutti i dataset"
    )
    parser.add_argument("--data-root", default="/data/genomic_bench/dna_foundation_benchmark/")
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Filter datasets (exact match)")
    args = parser.parse_args()

    all_datasets = discover_datasets(args.data_root)
    if args.datasets:
        all_datasets = [d for d in all_datasets if d["name"] in args.datasets]
    records = load_records(RECORDS_CSV)

    print("\n🌊 Wavelet/Spatial Pyramid RF")
    print(f"   Levels   : {args.levels}")
    print(f"   Grid     : {args.grid_size}")
    print(f"   Dataset  : {len(all_datasets)}")
    print(f"   Risultati: {RECORDS_CSV}\n")

    pbar = tqdm(all_datasets, unit="ds", ncols=72)

    for ds in pbar:
        name = ds["name"]
        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )
        n_classes = len(set(train_labels))

        for lvl in args.levels:
            if has_wavelet(name, lvl, records):
                continue

            t0 = time.perf_counter()
            X_train = extract_wavelet_features(
                train_seqs, levels=lvl, grid_size=args.grid_size, n_workers=args.n_workers
            )
            X_test = extract_wavelet_features(
                test_seqs, levels=lvl, grid_size=args.grid_size, n_workers=args.n_workers
            )
            t1 = time.perf_counter()

            log_efficiency(name, f"wavelet_l{lvl}", X_train.shape[1], t1 - t0)

            rf = train_rf(X_train, train_labels, n_classes)
            metrics = eval_rf(rf, X_test, test_labels, n_classes)
            write_wavelet(name, lvl, metrics)
            records = load_records(RECORDS_CSV)

        pbar.set_description(f"✓ {name}")

    print(f"\n✅ Completato! Risultati in: {RECORDS_CSV}")


if __name__ == "__main__":
    main()
