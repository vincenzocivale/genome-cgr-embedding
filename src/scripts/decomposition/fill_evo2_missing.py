"""
Fill missing Evo2 decomposition results by reprocessing without skip logic.
"""
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
    return Ridge(alpha=alpha, fit_intercept=True, solver='auto').fit(X, y)

def eval_ridge_r2(model, X, y):
    y_pred = model.predict(X)
    ss_res_per_dim = np.sum((y - y_pred) ** 2, axis=0)
    ss_tot_per_dim = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)
    if np.max(ss_tot_per_dim) == 0:
        return 0.0 if np.max(ss_res_per_dim) == 0 else -1.0
    r2_per_dim = 1.0 - (ss_res_per_dim / np.maximum(ss_tot_per_dim, 1e-12))
    return r2_per_dim.mean()

def train_rf_classifier(X, y):
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
    from sklearn.metrics import matthews_corrcoef, f1_score
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if len(np.unique(y)) == 2 else None
    mcc = matthews_corrcoef(y, y_pred)
    auroc = roc_auc_score(y, y_proba) if y_proba is not None else np.nan
    f1 = f1_score(y, y_pred, average='weighted')
    return {'MCC': float(mcc), 'AUROC': float(auroc), 'F1': float(f1)}

def load_cached_embeddings_evo2(model_key, dataset_name, split='train'):
    safe_ds = dataset_name.replace("/", "__").replace("\\", "__")
    path = os.path.join(CACHE_DIR, model_key, safe_ds, f"{split}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return data['embeddings']

def load_data_from_csv(data_root, dataset_name, split='train'):
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

def _load(path=RESULTS_CSV):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["dataset"])

def _save(df, path=RESULTS_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def _write(dataset, col_prefix, metrics, path=RESULTS_CSV):
    df = _load(path)
    if dataset not in df["dataset"].values:
        df = pd.concat([df, pd.DataFrame([{"dataset": dataset}])], ignore_index=True)
    idx = df[df["dataset"] == dataset].index[0]
    for m, v in metrics.items():
        df.loc[idx, f"{col_prefix}_{m}"] = v
    _save(df, path)

def main():
    parser = argparse.ArgumentParser(description="Fill missing Evo2 decomposition")
    parser.add_argument("--model", default="evo2_1b_base")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--k-values", nargs="+", type=int, default=[6])
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    model_tag = args.model
    df_results = _load()

    # Find missing datasets
    missing = []
    for dataset_name in sorted([d for d in os.listdir(args.data_root)
                               if os.path.isdir(os.path.join(args.data_root, d))]):
        col = f"ridge_k{args.k_values[0]}_{model_tag}_R2"
        if col not in df_results.columns or pd.isna(df_results[df_results['dataset'] == dataset_name].get(col, np.nan).values[0] if len(df_results[df_results['dataset'] == dataset_name]) > 0 else np.nan):
            missing.append(dataset_name)

    print(f"Found {len(missing)} missing datasets for {model_tag}\n")

    processed = 0
    for dataset_name in tqdm(missing, desc="Filling missing"):
        seqs_train, labels_train = load_data_from_csv(args.data_root, dataset_name, "train")
        seqs_test, labels_test = load_data_from_csv(args.data_root, dataset_name, "test")

        if seqs_train is None or seqs_test is None:
            continue

        emb_train = load_cached_embeddings_evo2(model_tag, dataset_name, "train")
        emb_test = load_cached_embeddings_evo2(model_tag, dataset_name, "test")

        if emb_train is None or emb_test is None:
            continue

        for k in args.k_values:
            ridge_col = f"ridge_k{k}_{model_tag}"
            proj_col = f"proj_k{k}_{model_tag}"
            resid_col = f"resid_k{k}_{model_tag}"

            X_train_kmer = extract_kmer_features(seqs_train, k=k, n_workers=args.n_workers)
            X_test_kmer = extract_kmer_features(seqs_test, k=k, n_workers=args.n_workers)

            ridge_model = train_ridge(X_train_kmer, emb_train)
            ridge_r2 = eval_ridge_r2(ridge_model, X_test_kmer, emb_test)

            emb_proj = ridge_model.predict(X_test_kmer)
            emb_resid = emb_test - emb_proj

            rf_proj = train_rf_classifier(emb_proj.astype(np.float32), labels_test)
            rf_resid = train_rf_classifier(emb_resid.astype(np.float32), labels_test)

            metrics_ridge = {'R2': ridge_r2}
            metrics_proj = eval_rf_classifier(rf_proj, emb_proj.astype(np.float32), labels_test)
            metrics_resid = eval_rf_classifier(rf_resid, emb_resid.astype(np.float32), labels_test)

            _write(dataset_name, ridge_col, metrics_ridge)
            _write(dataset_name, proj_col, metrics_proj)
            _write(dataset_name, resid_col, metrics_resid)

            processed += 1

    print(f"\nDone. Processed: {processed}")
    print(f"Results in {RESULTS_CSV}")

if __name__ == "__main__":
    main()
