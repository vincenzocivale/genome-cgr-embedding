"""
Orthogonal decomposition for Evo2 using cached embeddings + CSV data.
Final version with improved float handling and overflow prevention.

Usage:
    conda run -n cgr_bench python3 src/scripts/decomposition/train_decomposition_evo2_final.py \
        --model evo2_1b_base \
        --data-root /data/genomic_bench/dna_foundation_benchmark/ \
        --k-values 6 \
        --missing-only
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


def train_ridge(X, y, alpha=1.0):
    """Fit ridge regression for projection."""
    # Ensure X and y are float64 for numerical stability
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    return Ridge(alpha=alpha, fit_intercept=True, solver='auto').fit(X, y)


def eval_ridge_r2(model, X, y):
    """Compute R² of ridge projection."""
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    y_pred = model.predict(X)
    ss_res_per_dim = np.sum((y - y_pred) ** 2, axis=0)
    ss_tot_per_dim = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)
    if np.max(ss_tot_per_dim) == 0:
        return 0.0 if np.max(ss_res_per_dim) == 0 else -1.0
    r2_per_dim = 1.0 - (ss_res_per_dim / np.maximum(ss_tot_per_dim, 1e-12))
    return r2_per_dim.mean()


def train_rf_classifier(X, y):
    """Train RF with GridSearchCV for classification."""
    X = X.astype(np.float64)
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
    X = X.astype(np.float64)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if len(np.unique(y)) == 2 else None

    mcc = matthews_corrcoef(y, y_pred)
    auroc = roc_auc_score(y, y_proba) if y_proba is not None else np.nan
    f1 = f1_score(y, y_pred, average='weighted')

    return {'MCC': float(mcc), 'AUROC': float(auroc), 'F1': float(f1)}


def load_cached_embeddings_evo2(model_key, dataset_name, split='train'):
    """Load embeddings from flat Evo2 cache structure, converting to float64."""
    safe_ds = dataset_name.replace("/", "__").replace("\\", "__")
    path = os.path.join(CACHE_DIR, model_key, safe_ds, f"{split}.npz")

    if not os.path.exists(path):
        return None

    data = np.load(path)
    # Convert from float16 to float64 to avoid numerical issues
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
    except Exception:
        return None, None


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


def main():
    parser = argparse.ArgumentParser(
        description="Decomposition for Evo2 using cached embeddings + CSV data (final version)"
    )
    parser.add_argument("--model", default="evo2_1b_base")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--k-values", nargs="+", type=int, default=[6])
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--missing-only", action='store_true',
                        help="Only process datasets missing Evo2 results")
    args = parser.parse_args()

    model_tag = args.model

    print(f"\n{'='*60}")
    print(f"Decomposition (Evo2 Final)")
    print(f"{'='*60}")
    print(f"Model      : {args.model}")
    print(f"k-values   : {args.k_values}")
    print(f"Missing-only: {args.missing_only}")
    print(f"Results    : {RESULTS_CSV}\n")

    # Get dataset list from cache
    cache_path = os.path.join(CACHE_DIR, model_tag)
    if not os.path.exists(cache_path):
        print(f"Cache directory not found: {cache_path}")
        return

    datasets_cached = sorted([d for d in os.listdir(cache_path)
                             if os.path.isdir(os.path.join(cache_path, d))])

    print(f"Found {len(datasets_cached)} cached datasets\n")

    df_results = _load()

    # Filter to missing only if requested
    if args.missing_only:
        target_col = f"ridge_k6_{model_tag}_R2"
        datasets_to_process = [
            d for d in datasets_cached
            if d not in df_results["dataset"].values or
            pd.isna(df_results[df_results["dataset"] == d].get(target_col, np.nan).iloc[0] if len(df_results[df_results["dataset"] == d]) > 0 else np.nan)
        ]
        print(f"Processing {len(datasets_to_process)} missing datasets\n")
    else:
        datasets_to_process = datasets_cached

    processed = 0
    errors = 0

    for dataset_name_safe in tqdm(datasets_to_process, desc="Decomposition"):
        try:
            # Convert safe name back to original format
            dataset_name = dataset_name_safe.replace("__", "/")

            # Load sequences and labels from CSV
            seqs_train, labels_train = load_data_from_csv(args.data_root, dataset_name, "train")
            seqs_test, labels_test = load_data_from_csv(args.data_root, dataset_name, "test")

            if seqs_train is None or seqs_test is None:
                continue

            # Load cached FM embeddings (float64)
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

                # Extract k-mer features (ensure float64)
                X_train_kmer = extract_kmer_features(seqs_train, k=k, n_workers=args.n_workers).astype(np.float64)
                X_test_kmer = extract_kmer_features(seqs_test, k=k, n_workers=args.n_workers).astype(np.float64)

                # Ridge regression (on float64 data)
                ridge_model = train_ridge(X_train_kmer, emb_train)
                ridge_r2 = eval_ridge_r2(ridge_model, X_test_kmer, emb_test)

                # Projections and residuals (all float64)
                emb_proj = ridge_model.predict(X_test_kmer).astype(np.float64)
                emb_resid = (emb_test - emb_proj).astype(np.float64)

                # Train RF classifiers (with float64 inputs)
                rf_proj = train_rf_classifier(emb_proj, labels_test)
                rf_resid = train_rf_classifier(emb_resid, labels_test)

                # Evaluate
                metrics_ridge = {'R2': ridge_r2}
                metrics_proj = eval_rf_classifier(rf_proj, emb_proj, labels_test)
                metrics_resid = eval_rf_classifier(rf_resid, emb_resid, labels_test)

                # Save
                _write(dataset_name, ridge_col, metrics_ridge)
                _write(dataset_name, proj_col, metrics_proj)
                _write(dataset_name, resid_col, metrics_resid)

                processed += 1

        except Exception as e:
            print(f"  Error processing {dataset_name}: {e}")
            errors += 1
            continue

    print(f"\n{'='*60}")
    print(f"Processed: {processed} | Errors: {errors}")
    print(f"Results in {RESULTS_CSV}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
