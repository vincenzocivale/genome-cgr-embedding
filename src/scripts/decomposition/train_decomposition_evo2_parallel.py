"""
Evo2 decomposition with parallel processing (4 workers) in reverse order.
Processes smaller datasets first to get quick wins.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from multiprocessing import Pool
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV

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


def process_dataset(args_tuple):
    """Process one dataset. Returns (dataset_name, success, metrics_dict)."""
    dataset_name, model_tag, k, data_root = args_tuple

    try:
        # Load data
        seqs_train, labels_train = load_data_from_csv(data_root, dataset_name, "train")
        seqs_test, labels_test = load_data_from_csv(data_root, dataset_name, "test")

        if seqs_train is None or seqs_test is None:
            return (dataset_name, False, {})

        # Load embeddings
        emb_train = load_cached_embeddings_evo2(model_tag, dataset_name, "train")
        emb_test = load_cached_embeddings_evo2(model_tag, dataset_name, "test")

        if emb_train is None or emb_test is None:
            return (dataset_name, False, {})

        # Extract k-mer features
        X_train_kmer = extract_kmer_features(seqs_train, k=k, n_workers=1).astype(np.float64)
        X_test_kmer = extract_kmer_features(seqs_test, k=k, n_workers=1).astype(np.float64)
        gc.collect()

        # Ridge regression on TRAINING data
        ridge_model = Ridge(alpha=1.0, fit_intercept=True, solver='auto').fit(X_train_kmer, emb_train)

        # Ridge R² evaluated on TEST data (multi-output: per-dimension R², then average)
        y_pred_test = ridge_model.predict(X_test_kmer)
        ss_res_per_dim = np.sum((emb_test - y_pred_test) ** 2, axis=0)
        ss_tot_per_dim = np.sum((emb_test - np.mean(emb_test, axis=0)) ** 2, axis=0)
        ridge_r2 = 1.0 - (ss_res_per_dim / np.maximum(ss_tot_per_dim, 1e-12)).mean()

        # Decompose TRAINING embeddings: proj and residual from training
        emb_proj_train = ridge_model.predict(X_train_kmer).astype(np.float64)
        emb_resid_train = (emb_train - emb_proj_train).astype(np.float64)

        # Also compute projections and residuals for TEST (for visualization)
        emb_proj_test = y_pred_test.astype(np.float64)
        emb_resid_test = (emb_test - emb_proj_test).astype(np.float64)

        del X_train_kmer, X_test_kmer, emb_train, emb_test, y_pred_test
        gc.collect()

        # Train RF classifiers on TRAINING decomposed embeddings — must match rf_pipeline.py exactly
        n_classes = len(np.unique(labels_test))
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        rf_param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['sqrt'],
            'max_depth': [20],
            'min_samples_split': [2],
        }
        scoring = 'roc_auc' if n_classes == 2 else 'accuracy'

        # Projected component: train on training, test on test
        gs_proj = GridSearchCV(
            RandomForestClassifier(n_jobs=4, random_state=42),
            rf_param_grid,
            scoring=scoring,
            cv=cv, n_jobs=1, refit=True,
        )
        gs_proj.fit(emb_proj_train, labels_train)
        y_pred_proj = gs_proj.predict(emb_proj_test)
        mcc_proj = float(matthews_corrcoef(labels_test, y_pred_proj))
        y_proba_proj = gs_proj.predict_proba(emb_proj_test)
        if n_classes == 2:
            auroc_proj = float(roc_auc_score(labels_test, y_proba_proj[:, 1]))
        else:
            auroc_proj = float(roc_auc_score(labels_test, y_proba_proj, multi_class='ovr', average='macro'))
        f1_proj = float(f1_score(labels_test, y_pred_proj, average='macro'))

        del emb_proj_train, emb_proj_test
        gc.collect()

        # Residual component: train on training, test on test
        gs_resid = GridSearchCV(
            RandomForestClassifier(n_jobs=4, random_state=42),
            rf_param_grid,
            scoring=scoring,
            cv=cv, n_jobs=1, refit=True,
        )
        gs_resid.fit(emb_resid_train, labels_train)
        y_pred_resid = gs_resid.predict(emb_resid_test)
        mcc_resid = float(matthews_corrcoef(labels_test, y_pred_resid))
        y_proba_resid = gs_resid.predict_proba(emb_resid_test)
        if n_classes == 2:
            auroc_resid = float(roc_auc_score(labels_test, y_proba_resid[:, 1]))
        else:
            auroc_resid = float(roc_auc_score(labels_test, y_proba_resid, multi_class='ovr', average='macro'))
        f1_resid = float(f1_score(labels_test, y_pred_resid, average='macro'))

        del emb_resid_train, emb_resid_test
        gc.collect()

        del emb_resid, X_test_kmer
        gc.collect()

        metrics = {
            f'ridge_k{k}_{model_tag}_R2': ridge_r2,
            f'proj_k{k}_{model_tag}_MCC': mcc_proj,
            f'proj_k{k}_{model_tag}_AUROC': auroc_proj,
            f'proj_k{k}_{model_tag}_F1': f1_proj,
            f'resid_k{k}_{model_tag}_MCC': mcc_resid,
            f'resid_k{k}_{model_tag}_AUROC': auroc_resid,
            f'resid_k{k}_{model_tag}_F1': f1_resid,
        }

        return (dataset_name, True, metrics)

    except Exception as e:
        print(f"ERROR {dataset_name}: {str(e)[:80]}", file=sys.stderr)
        return (dataset_name, False, {})


def main():
    parser = argparse.ArgumentParser(description="Evo2 decomposition (parallel, reverse order)")
    parser.add_argument("--model", default="evo2_1b_base")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--n-workers", type=int, default=4)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Evo2 Decomposition (Parallel, Reverse Order)")
    print(f"{'='*70}")
    print(f"Model:         {args.model}")
    print(f"k:             {args.k}")
    print(f"Parallel jobs: {args.n_workers}")
    print(f"Results:       {RESULTS_CSV}\n")

    # Get dataset list
    cache_path = os.path.join(CACHE_DIR, args.model)
    if not os.path.exists(cache_path):
        print(f"Cache directory not found: {cache_path}")
        return

    datasets_cached = sorted([d for d in os.listdir(cache_path)
                             if os.path.isdir(os.path.join(cache_path, d))])

    df_results = _load()

    # Filter to missing datasets
    target_col = f"ridge_k{args.k}_{args.model}_R2"
    datasets_missing = [
        d for d in datasets_cached
        if not _has(df_results, d, target_col)
    ]

    # **REVERSE ORDER** — small datasets first
    datasets_missing = list(reversed(datasets_missing))

    print(f"Processing {len(datasets_missing)} missing datasets (reverse order)\n")

    # Prepare args for parallel processing
    task_args = [
        (d.replace("__", "/"), args.model, args.k, args.data_root)
        for d in datasets_missing
    ]

    processed = 0
    errors = 0

    # Process in parallel
    with Pool(args.n_workers) as pool:
        for dataset_name, success, metrics in pool.imap_unordered(process_dataset, task_args):
            if success:
                # Write all metrics
                for col, val in metrics.items():
                    _write(dataset_name, col, val)
                processed += 1
                print(f"✓ {dataset_name}")
            else:
                errors += 1
                print(f"✗ {dataset_name}")

    print(f"\n{'='*70}")
    print(f"Completed: {processed} | Errors: {errors}")
    print(f"Results in {RESULTS_CSV}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
