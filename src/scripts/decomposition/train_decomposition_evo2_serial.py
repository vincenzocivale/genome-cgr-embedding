"""
Orthogonal decomposition for Evo2 with memory-safe serial processing.
Processes one dataset at a time with careful memory management.

Usage:
    conda run -n cgr_bench python3 src/scripts/decomposition/train_decomposition_evo2_serial.py \
        --model evo2_1b_base \
        --data-root /data/genomic_bench/dna_foundation_benchmark/ \
        --k 6 \
        --missing-only
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.features.kmer_features import extract_kmer_features

DATA_ROOT = "/data/genomic_bench/dna_foundation_benchmark/"
CACHE_DIR = "cache/fm_embeddings"
RESULTS_CSV = "results/decomposition/records_decomposition.csv"


def load_cached_embeddings_evo2(model_key, dataset_name, split='train'):
    """Load embeddings from cache, converting to float64."""
    safe_ds = dataset_name.replace("/", "__").replace("\\", "__")
    path = os.path.join(CACHE_DIR, model_key, safe_ds, f"{split}.npz")

    if not os.path.exists(path):
        return None

    data = np.load(path)
    embeddings = data['embeddings'].astype(np.float64)
    return embeddings


def load_data_from_csv(data_root, dataset_name, split='train'):
    """Load sequences and labels from CSV files."""
    parts = dataset_name.split('/')
    if len(parts) == 2:
        csv_path = os.path.join(data_root, parts[0], parts[1], f"{split}.csv")
    else:
        csv_path = os.path.join(data_root, dataset_name, f"{split}.csv")

    if not os.path.exists(csv_path):
        return None, None

    try:
        df = pd.read_csv(csv_path)
        seq_col = 'sequence' if 'sequence' in df.columns else 'Sequence'
        label_col = 'label' if 'label' in df.columns else 'Label'
        seqs = df[seq_col].values
        labels = df[label_col].values.astype(int)
        return seqs, labels
    except Exception as e:
        print(f"    Error loading CSV: {e}")
        return None, None


def _load(path=RESULTS_CSV):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["dataset"])


def _save(df, path=RESULTS_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _has(df, dataset, col):
    if df.empty or dataset not in df["dataset"].values:
        return False
    row = df[df["dataset"] == dataset].iloc[0]
    return col in df.columns and pd.notna(row.get(col, np.nan))


def _write(dataset, col, value, path=RESULTS_CSV):
    df = _load(path)
    if dataset not in df["dataset"].values:
        df = pd.concat([df, pd.DataFrame([{"dataset": dataset}])], ignore_index=True)
    idx = df[df["dataset"] == dataset].index[0]
    df.loc[idx, col] = value
    _save(df, path)


def process_dataset(dataset_name, model_tag, k, data_root, n_workers):
    """Process one dataset. Returns True if successful, False otherwise."""

    # Load data
    seqs_train, labels_train = load_data_from_csv(data_root, dataset_name, "train")
    seqs_test, labels_test = load_data_from_csv(data_root, dataset_name, "test")

    if seqs_train is None or seqs_test is None:
        return False

    # Load embeddings
    emb_train = load_cached_embeddings_evo2(model_tag, dataset_name, "train")
    emb_test = load_cached_embeddings_evo2(model_tag, dataset_name, "test")

    if emb_train is None or emb_test is None:
        return False

    try:
        # Extract k-mer features
        print(f"    Extracting k={k} features...", end=' ', flush=True)
        X_train_kmer = extract_kmer_features(seqs_train, k=k, n_workers=n_workers).astype(np.float64)
        X_test_kmer = extract_kmer_features(seqs_test, k=k, n_workers=n_workers).astype(np.float64)
        print(f"shape {X_test_kmer.shape}")
        gc.collect()

        # Ridge regression
        print(f"    Ridge regression...", end=' ', flush=True)
        ridge_model = Ridge(alpha=1.0, fit_intercept=True, solver='auto').fit(X_train_kmer, emb_train)
        del X_train_kmer
        gc.collect()

        y_pred = ridge_model.predict(X_test_kmer)
        ss_res_per_dim = np.sum((emb_test - y_pred) ** 2, axis=0)
        ss_tot_per_dim = np.sum((emb_test - np.mean(emb_test, axis=0)) ** 2, axis=0)
        ridge_r2 = 1.0 - (ss_res_per_dim / np.maximum(ss_tot_per_dim, 1e-12)).mean()
        print(f"R2={ridge_r2:.3f}")

        # Projections and residuals
        emb_proj = y_pred.astype(np.float64)
        emb_resid = (emb_test - emb_proj).astype(np.float64)
        del y_pred, emb_test
        gc.collect()

        # Train RF classifiers on projected features
        print(f"    RF on projected...", end=' ', flush=True)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        gs_proj = GridSearchCV(
            RandomForestClassifier(n_jobs=4, random_state=42),
            {'n_estimators': [200], 'max_depth': [15, 20], 'max_features': ['sqrt']},
            scoring='roc_auc' if len(np.unique(labels_test)) == 2 else 'f1_weighted',
            cv=cv, n_jobs=1, refit=True,
        )
        gs_proj.fit(emb_proj, labels_test)

        # Metrics for projected
        y_pred_proj = gs_proj.predict(emb_proj)
        y_proba_proj = gs_proj.predict_proba(emb_proj)[:, 1] if len(np.unique(labels_test)) == 2 else None
        mcc_proj = float(matthews_corrcoef(labels_test, y_pred_proj))
        auroc_proj = float(roc_auc_score(labels_test, y_proba_proj)) if y_proba_proj is not None else np.nan
        f1_proj = float(f1_score(labels_test, y_pred_proj, average='weighted'))
        print(f"MCC={mcc_proj:.3f}")

        del emb_proj
        gc.collect()

        # Train RF classifiers on residual features
        print(f"    RF on residual...", end=' ', flush=True)
        gs_resid = GridSearchCV(
            RandomForestClassifier(n_jobs=4, random_state=42),
            {'n_estimators': [200], 'max_depth': [15, 20], 'max_features': ['sqrt']},
            scoring='roc_auc' if len(np.unique(labels_test)) == 2 else 'f1_weighted',
            cv=cv, n_jobs=1, refit=True,
        )
        gs_resid.fit(emb_resid, labels_test)

        # Metrics for residual
        y_pred_resid = gs_resid.predict(emb_resid)
        y_proba_resid = gs_resid.predict_proba(emb_resid)[:, 1] if len(np.unique(labels_test)) == 2 else None
        mcc_resid = float(matthews_corrcoef(labels_test, y_pred_resid))
        auroc_resid = float(roc_auc_score(labels_test, y_proba_resid)) if y_proba_resid is not None else np.nan
        f1_resid = float(f1_score(labels_test, y_pred_resid, average='weighted'))
        print(f"MCC={mcc_resid:.3f}")

        del emb_resid, X_test_kmer
        gc.collect()

        # Save results
        ridge_col = f"ridge_k{k}_{model_tag}_R2"
        proj_col_mcc = f"proj_k{k}_{model_tag}_MCC"
        proj_col_auroc = f"proj_k{k}_{model_tag}_AUROC"
        proj_col_f1 = f"proj_k{k}_{model_tag}_F1"
        resid_col_mcc = f"resid_k{k}_{model_tag}_MCC"
        resid_col_auroc = f"resid_k{k}_{model_tag}_AUROC"
        resid_col_f1 = f"resid_k{k}_{model_tag}_F1"

        _write(dataset_name, ridge_col, ridge_r2)
        _write(dataset_name, proj_col_mcc, mcc_proj)
        _write(dataset_name, proj_col_auroc, auroc_proj)
        _write(dataset_name, proj_col_f1, f1_proj)
        _write(dataset_name, resid_col_mcc, mcc_resid)
        _write(dataset_name, resid_col_auroc, auroc_resid)
        _write(dataset_name, resid_col_f1, f1_resid)

        return True

    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Evo2 decomposition (serial, memory-safe)")
    parser.add_argument("--model", default="evo2_1b_base")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--missing-only", action='store_true')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Evo2 Decomposition (Serial, Memory-Safe)")
    print(f"{'='*70}")
    print(f"Model:        {args.model}")
    print(f"k:            {args.k}")
    print(f"Missing-only: {args.missing_only}")
    print(f"Results:      {RESULTS_CSV}\n")

    # Get dataset list
    cache_path = os.path.join(CACHE_DIR, args.model)
    if not os.path.exists(cache_path):
        print(f"Cache directory not found: {cache_path}")
        return

    datasets_cached = sorted([d for d in os.listdir(cache_path)
                             if os.path.isdir(os.path.join(cache_path, d))])
    print(f"Found {len(datasets_cached)} cached datasets\n")

    df_results = _load()

    # Filter to missing datasets
    if args.missing_only:
        target_col = f"ridge_k{args.k}_{args.model}_R2"
        datasets_to_process = [
            d for d in datasets_cached
            if not _has(df_results, d, target_col)
        ]
        print(f"Processing {len(datasets_to_process)} missing datasets\n")
    else:
        datasets_to_process = datasets_cached

    processed = 0
    errors = 0

    for dataset_name_safe in datasets_to_process:
        dataset_name = dataset_name_safe.replace("__", "/")
        print(f"  {dataset_name}")

        if process_dataset(dataset_name, args.model, args.k, args.data_root, args.n_workers):
            processed += 1
        else:
            errors += 1

    print(f"\n{'='*70}")
    print(f"Completed: {processed} | Errors: {errors}")
    print(f"Results in {RESULTS_CSV}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
