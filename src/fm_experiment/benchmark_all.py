"""
Benchmark script: misura le performance RF (k-mer e FM) su tutti i dataset.

Salva progressivamente i risultati in records.csv.
Salta i dataset già completati per quella configurazione (k, FM_model).

Usage:
    python3 src/fm_experiment/benchmark_all.py --k-values 6 --model InstaDeepAI/NTv3_650M_pre
    python3 src/fm_experiment/benchmark_all.py --k-values 4 6 8 --fm-batch-size 32
"""

import argparse
import sys
import os
from datetime import datetime

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.fm_experiment.fm_embedder import FMEmbedder
from src.fm_experiment.hyena_embedder import HyenaEmbedder
from src.fm_experiment.kmer_features import extract_kmer_features
from src.fm_experiment.records import (
    load_records, has_kmer, has_fm, has_ridge,
    write_kmer, write_fm, write_ridge,
    RECORDS_CSV,
)
from src.fm_experiment.ridge_mapping import fit_and_evaluate
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
        description="Benchmark RF su tutti i dataset — k-mer vs FM embeddings"
    )
    parser.add_argument("--data-root", default="/data/genomic_bench/dna_foundation_benchmark/")
    parser.add_argument("--k-values", nargs="+", type=int, default=[6])
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--fm-batch-size", type=int, default=32)
    parser.add_argument("--model", default="InstaDeepAI/NTv3_650M_pre",
                       help="FM model: NTv3_650M_pre, hyenadna-medium-160k, etc.")
    parser.add_argument("--datasets", nargs="*", default=None,
                       help="Filter datasets (exact match)")
    parser.add_argument("--fm-only", action="store_true",
                       help="Skip k-mer features and Ridge, run only FM embeddings")
    args = parser.parse_args()

    # Carica lista dataset
    all_datasets = discover_datasets(args.data_root)
    if args.datasets:
        all_datasets = [d for d in all_datasets if d["name"] in args.datasets]

    print(f"📊 Benchmark: {len(all_datasets)} dataset/i")
    print(f"   k-values: {args.k_values}")
    print(f"   FM model: {args.model}")
    print(f"   Results: results/records.csv\n")

    # Carica FM una sola volta (scegli embedder basato su model name)
    if "hyenadna" in args.model.lower():
        fm = HyenaEmbedder(model_name=args.model, cache_dir="cache/hyena_embeddings")
    else:
        fm = FMEmbedder(model_name=args.model, cache_dir="cache/fm_embeddings")

    records = load_records(os.path.join("results", "records.csv"))

    # Progress bar esterna per i dataset
    k_values = [] if args.fm_only else args.k_values
    total_work = len(all_datasets) * (1 + len(k_values))  # FM + k-mers per dataset
    pbar = tqdm(total=total_work, desc="Benchmark", unit="task", ncols=60)

    for ds in all_datasets:
        name = ds["name"]

        # Carica sequenze
        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )
        n_classes = len(set(train_labels))

        # FM embeddings (cached)
        Y_train = fm.embed_sequences(train_seqs, name, "train", args.fm_batch_size)
        Y_test = fm.embed_sequences(test_seqs, name, "test", args.fm_batch_size)

        # ── RF su FM (una sola volta per dataset) ──
        if not has_fm(name, args.model, records):
            rf_fm = train_rf(Y_train.astype(np.float32), train_labels, n_classes)
            res_fm = eval_rf(rf_fm, Y_test.astype(np.float32), test_labels, n_classes)
            write_fm(name, args.model, res_fm)
            records = load_records(os.path.join("results", "records.csv"))
        pbar.update(1)
        pbar.set_description(f"[{name}] FM")

        # ── Loop sui k-values ──
        for k in k_values:
            need_ridge = not has_ridge(name, k, args.model, records)
            need_kmer = not has_kmer(name, k, records)

            # Calcola feature k-mer una sola volta se servono (grid_size >= 2^k)
            if need_ridge or need_kmer:
                grid_size = max(128, 2 ** k)
                X_train = extract_kmer_features(train_seqs, k=k, grid_size=grid_size,
                                              n_workers=args.n_workers)
                X_test = extract_kmer_features(test_seqs, k=k, grid_size=grid_size,
                                             n_workers=args.n_workers)

            # Ridge
            if need_ridge:
                ridge_raw = fit_and_evaluate(
                    X_train, Y_train.astype(np.float32),
                    X_test, Y_test.astype(np.float32),
                )
                write_ridge(name, k, args.model,
                           {"R2": ridge_raw["r2_global"], "MSE": ridge_raw["mse_global"]})
                records = load_records(os.path.join("results", "records.csv"))

            # RF k-mer
            if need_kmer:
                rf_kmer = train_rf(X_train, train_labels, n_classes)
                res_kmer = eval_rf(rf_kmer, X_test, test_labels, n_classes)
                write_kmer(name, k, res_kmer)
                records = load_records(os.path.join("results", "records.csv"))

            pbar.update(1)
            pbar.set_description(f"[{name}] k={k}")

    pbar.close()

    print(f"\n✅ Benchmark completato!")
    print(f"   Risultati salvati in: results/records.csv")
    print(f"   Dataset: {len(records)}")
    print(f"   Colonne: {len(records.columns)}")


if __name__ == "__main__":
    main()
