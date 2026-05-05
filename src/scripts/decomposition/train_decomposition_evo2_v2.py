"""
Orthogonal decomposition for Evo2 using cached embeddings + CSV data.

Loads:
- Cached Evo2 embeddings from: cache/fm_embeddings/evo2_1b_base/<dataset>/train.npz
- Sequences and labels from: /data/genomic_bench/dna_foundation_benchmark/<dataset>/train.csv

Usage:
    conda run -n cgr_bench python3 src/scripts/decomposition/train_decomposition_evo2_v2.py \
        --model evo2_1b_base \
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
from sklearn.model_selection import StratifiedKFold, GridSearchCV
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
    """Compute R² of ridge projection (multi-output: per-dim, then average)."""
    y_pred = model.predict(X)
    ss_res_per_dim = np.sum((y - y_pred) ** 2, axis=0)
    ss_tot_per_dim = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)
    if np.max(ss_tot_per_dim) == 0:
        return 0.0
    r2_per_dim = 1.0 - (ss_res_per_dim / np.maximum(ss_tot_per_dim, 1e-12))
    return r2_per_dim.mean()


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

def load_cached_embeddings_evo2(model_key, dataset_name, split='train'):
    """Load embeddings from flat Evo2 cache structure."""
    safe_ds = dataset_name.replace("/", "__").replace("\\", "__")
    path = os.path.join(CACHE_DIR, model_key, safe_ds, f"{split}.npz")

    if not os.path.exists(path):
        return None

    data = np.load(path)
    return data['embeddings']


# ── data loaders from CSV ──────────────────────────────────────────────────

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
        # Try both lowercase and capitalized column names
        seq_col = 'sequence' if 'sequence' in df.columns else 'Sequence'
        label_col = 'label' if 'label' in df.columns else 'Label'
        seqs = df[seq_col].values
        labels = df[label_col].values.astype(int)
        return seqs, labels
    except Exception:
        return None, None


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
        description="Decomposition for Evo2 using cached embeddings + CSV data"
    )
    parser.add_argument("--model", default="evo2_1b_base")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--k-values", nargs="+", type=int, default=[6])
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    model_tag = args.model

    print(f"\nDecomposition (Evo2 Cached Embeddings + CSV Data)")
    print(f"Model  : {args.model}")
    print(f"Mapper : ridge (cached embeddings only)")
    print(f"k-values: {args.k_values}")
    print(f"Results: {RESULTS_CSV}\n")

    # Get dataset list from cache
    cache_path = os.path.join(CACHE_DIR, model_tag)
    if not os.path.exists(cache_path):
        print(f"Cache directory not found: {cache_path}")
        return

    datasets_cached = sorted([d for d in os.listdir(cache_path)
                             if os.path.isdir(os.path.join(cache_path, d))])

    print(f"Found {len(datasets_cached)} cached datasets\n")

    df_results = _load()
    processed = 0

    for dataset_name_safe in tqdm(datasets_cached, desc="Decomposition"):
        # Convert safe name back to original format
        dataset_name = dataset_name_safe.replace("__", "/")

        # Load sequences and labels from CSV
        seqs_train, labels_train = load_data_from_csv(args.data_root, dataset_name, "train")
        seqs_test, labels_test = load_data_from_csv(args.data_root, dataset_name, "test")

        if seqs_train is None or seqs_test is None:
            continue

        # Load cached FM embeddings
        emb_train = load_cached_embeddings_evo2(model_tag, dataset_name, "train")
        emb_test = load_cached_embeddings_evo2(model_tag, dataset_name, "test")

        if emb_train is None or emb_test is None:
            continue

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

            # Ridge regression
            ridge_model = train_ridge(X_train_kmer, emb_train)
            ridge_r2 = eval_ridge_r2(ridge_model, X_test_kmer, emb_test)

            # Projections and residuals
            emb_proj = ridge_model.predict(X_test_kmer)
            emb_resid = emb_test - emb_proj

            # Train RF classifiers
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

    print(f"\nDone. Processed: {processed}")
    print(f"Results in {RESULTS_CSV}")


if __name__ == "__main__":
    main()
