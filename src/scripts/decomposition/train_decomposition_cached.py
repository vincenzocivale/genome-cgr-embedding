"""
Orthogonal decomposition (ridge regression) using CACHED FM embeddings.

Instead of loading the FM model and computing embeddings, this script
loads pre-computed embeddings from cache and performs ridge regression
to decompose: FM = ridge_proj(k-mer) + residual

Usage:
    conda run -n cgr_bench python3 src/scripts/decomposition/train_decomposition_cached.py \
        --model kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 \
        --data-root /data/genomic_bench/dna_foundation_benchmark/ \
        --k-values 6
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.features.kmer_features import extract_kmer_features

DATA_ROOT = "/data/genomic_bench/dna_foundation_benchmark/"
CACHE_DIR = "cache/fm_embeddings"
RESULTS_CSV = "results/decomposition/records_decomposition.csv"

# ── ridge decomposition ────────────────────────────────────────────────────

def train_ridge(X, y, alpha=1.0):
    """Fit ridge regression for projection."""
    return Ridge(alpha=alpha, fit_intercept=True, solver='auto').fit(X, y)


def eval_ridge_r2(model, X, y):
    """Compute R² of ridge projection."""
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return 0.0 if ss_res == 0 else -1.0
    return 1.0 - (ss_res / ss_tot)


def train_rf_classifier(X, y):
    """Train RF with GridSearchCV for classification."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        RandomForestClassifier(n_jobs=4, random_state=42),
        {'n_estimators': [200], 'max_depth': [15, 20], 'max_features': ['sqrt']},
        scoring='roc_auc' if len(np.unique(y)) == 2 else 'f1_weighted',
        cv=cv, n_jobs=1, refit=True,
    )
    gs.fit(X, y)
    return gs


def eval_rf_classifier(model, X, y):
    """Evaluate RF classifier: MCC, AUROC, F1."""
    from sklearn.metrics import matthews_corrcoef, f1_score
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if len(np.unique(y)) == 2 else None

    mcc = matthews_corrcoef(y, y_pred)
    auroc = roc_auc_score(y, y_proba) if y_proba is not None else np.nan
    f1 = f1_score(y, y_pred, average='weighted')

    return {'MCC': float(mcc), 'AUROC': float(auroc), 'F1': float(f1)}


# ── cache helpers ──────────────────────────────────────────────────────────

def load_cached_embeddings(model_key, dataset_name, split='train'):
    """Load embeddings from cache."""
    safe_model = model_key.replace("/", "__")
    safe_ds = dataset_name.replace("/", "__").replace("\\", "__")
    path = os.path.join(CACHE_DIR, safe_model, "pooling_mean", safe_ds, f"{split}.npz")

    if not os.path.exists(path):
        return None

    data = np.load(path)
    return data['embeddings']


# ── results helpers ────────────────────────────────────────────────────────

def _load(path=RESULTS_CSV):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["dataset"])


def _save(df, path=RESULTS_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _has(df, dataset, col_prefix):
    if df.empty or dataset not in df["dataset"].values:
        return False
    row = df[df["dataset"] == dataset].iloc[0]
    col = f"{col_prefix}_R2" if "ridge" in col_prefix else f"{col_prefix}_MCC"
    return col in df.columns and pd.notna(row.get(col, np.nan))


def _write(dataset, col_prefix, metrics, path=RESULTS_CSV):
    df = _load(path)
    if dataset not in df["dataset"].values:
        df = pd.concat([df, pd.DataFrame([{"dataset": dataset}])], ignore_index=True)
    idx = df[df["dataset"] == dataset].index[0]
    for m, v in metrics.items():
        df.loc[idx, f"{col_prefix}_{m}"] = v
    _save(df, path)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Decomposition using cached FM embeddings (ridge regression + RF evaluation)"
    )
    parser.add_argument("--model", default="kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--k-values", nargs="+", type=int, default=[6])
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    model_tag = args.model.split("/")[-1]

    print(f"\nDecomposition (Cached Embeddings)")
    print(f"Model  : {args.model}")
    print(f"Mapper : ridge (cached embeddings only)")
    print(f"k-values: {args.k_values}")
    print(f"Results: {RESULTS_CSV}\n")

    # Get dataset list
    datasets = sorted([d for d in os.listdir(args.data_root)
                      if os.path.isdir(os.path.join(args.data_root, d))])

    df_results = _load()
    processed = 0
    skipped = 0

    for dataset_name in tqdm(datasets, desc="Decomposition"):
        # Load sequences
        fasta_path = os.path.join(args.data_root, dataset_name, "sequences.fasta")
        if not os.path.exists(fasta_path):
            continue

        from pyfaidx import Fasta
        try:
            seqs_obj = Fasta(fasta_path)
            seqs = np.array([str(s[:]).upper() for s in seqs_obj])
        except Exception:
            skipped += 1
            continue

        # Load labels
        labels_path = os.path.join(args.data_root, dataset_name, "labels.txt")
        if not os.path.exists(labels_path):
            skipped += 1
            continue

        try:
            labels = np.loadtxt(labels_path, dtype=int)
        except Exception:
            skipped += 1
            continue

        # Load cached FM embeddings
        emb_train = load_cached_embeddings(args.model, dataset_name, "train")
        emb_test = load_cached_embeddings(args.model, dataset_name, "test")

        if emb_train is None or emb_test is None:
            skipped += 1
            continue

        # Split sequences into train/test (assuming first part is train)
        n_train = len(emb_train)
        seqs_train = seqs[:n_train]
        seqs_test = seqs[n_train:]
        labels_train = labels[:n_train]
        labels_test = labels[n_train:]

        # Process k-values
        for k in args.k_values:
            ridge_col = f"ridge_k{k}_{model_tag}"
            proj_col = f"proj_k{k}_{model_tag}"
            resid_col = f"resid_k{k}_{model_tag}"

            if _has(df_results, dataset_name, ridge_col):
                continue

            # Extract k-mer features
            X_train_kmer = extract_kmer_features(seqs_train, k=k, n_workers=args.n_workers)
            X_test_kmer = extract_kmer_features(seqs_test, k=k, n_workers=args.n_workers)

            # Ridge regression: FM = ridge_proj(k-mer) + residual
            ridge_model = train_ridge(X_train_kmer, emb_train)
            ridge_r2 = eval_ridge_r2(ridge_model, X_test_kmer, emb_test)

            # Compute projections and residuals
            emb_proj = ridge_model.predict(X_test_kmer)
            emb_resid = emb_test - emb_proj

            # Train RF on: full FM, projected (k-mer), residual
            rf_full = train_rf_classifier(emb_test.astype(np.float32), labels_test)
            rf_proj = train_rf_classifier(emb_proj.astype(np.float32), labels_test)
            rf_resid = train_rf_classifier(emb_resid.astype(np.float32), labels_test)

            # Evaluate
            metrics_ridge = {'R2': ridge_r2}
            metrics_proj = eval_rf_classifier(rf_proj, emb_proj.astype(np.float32), labels_test)
            metrics_resid = eval_rf_classifier(rf_resid, emb_resid.astype(np.float32), labels_test)

            # Save
            _write(dataset_name, ridge_col, metrics_ridge)
            _write(dataset_name, proj_col, metrics_proj)
            _write(dataset_name, resid_col, metrics_resid)

            processed += 1

    print(f"\nDone. Processed: {processed} | Skipped: {skipped}")
    print(f"Results in {RESULTS_CSV}")


if __name__ == "__main__":
    main()
