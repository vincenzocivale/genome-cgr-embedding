#!/usr/bin/env python3
"""Debug script to understand Evo2 Ridge R² anomaly."""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score

sys.path.insert(0, '/home/oem/genome-cgr-embedding')

from src.features.kmer_features import extract_kmer_features

DATA_ROOT = "/data/genomic_bench/dna_foundation_benchmark/"
CACHE_DIR = "cache/fm_embeddings"

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
        print(f"Error loading CSV: {e}")
        return None, None

# Test on a few datasets
cache_path = os.path.join(CACHE_DIR, "evo2_1b_base")
datasets = sorted([d for d in os.listdir(cache_path) if os.path.isdir(os.path.join(cache_path, d))])[:5]

output = []

for ds_safe in datasets:
    dataset_name = ds_safe.replace("__", "/")

    # Load data
    seqs_train, labels_train = load_data_from_csv(DATA_ROOT, dataset_name, "train")
    seqs_test, labels_test = load_data_from_csv(DATA_ROOT, dataset_name, "test")

    if seqs_train is None or seqs_test is None:
        output.append(f"{dataset_name}: FAILED to load data\n")
        continue

    # Load embeddings
    emb_train = load_cached_embeddings_evo2("evo2_1b_base", dataset_name, "train")
    emb_test = load_cached_embeddings_evo2("evo2_1b_base", dataset_name, "test")

    if emb_train is None or emb_test is None:
        output.append(f"{dataset_name}: FAILED to load embeddings\n")
        continue

    # Extract k-mer features
    X_train_kmer = extract_kmer_features(seqs_train, k=6, n_workers=1).astype(np.float64)
    X_test_kmer = extract_kmer_features(seqs_test, k=6, n_workers=1).astype(np.float64)

    output.append(f"\n{dataset_name}:")
    output.append(f"  Shapes: train={emb_train.shape}, test={emb_test.shape}, X_train={X_train_kmer.shape}, X_test={X_test_kmer.shape}")
    output.append(f"  Embedding ranges: train=[{emb_train.min():.4f}, {emb_train.max():.4f}], test=[{emb_test.min():.4f}, {emb_test.max():.4f}]")
    output.append(f"  Kmer ranges: train=[{X_train_kmer.min():.4f}, {X_train_kmer.max():.4f}], test=[{X_test_kmer.min():.4f}, {X_test_kmer.max():.4f}]")
    output.append(f"  Kmer sparsity: {(X_train_kmer == 0).mean()*100:.2f}% zeros in training")

    # Ridge regression with different alphas
    ridge_default = Ridge(alpha=1.0, fit_intercept=True).fit(X_train_kmer, emb_train)
    y_pred_test = ridge_default.predict(X_test_kmer)
    r2_test = r2_score(emb_test, y_pred_test, multioutput='uniform_average')

    # Try RidgeCV
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    ridgecv = RidgeCV(alphas=alphas, gcv_mode="auto").fit(X_train_kmer, emb_train)
    y_pred_ridgecv = ridgecv.predict(X_test_kmer)
    r2_ridgecv = r2_score(emb_test, y_pred_ridgecv, multioutput='uniform_average')

    output.append(f"  Ridge R² (alpha=1.0): {r2_test:.6f}")
    output.append(f"  RidgeCV R² (best alpha={ridgecv.alpha_:.4f}): {r2_ridgecv:.6f}")

    # Check per-dimension R² to see if any are NaN or suspicious
    per_dim_r2 = []
    for i in range(min(5, emb_test.shape[1])):
        r2_i = r2_score(emb_test[:, i], y_pred_test[:, i])
        per_dim_r2.append(r2_i)
    output.append(f"  Per-dim R² (first 5): {[f'{r:.4f}' for r in per_dim_r2]}")

# Write output
with open('/tmp/evo2_debug_output.txt', 'w') as f:
    f.write('\n'.join(output))

print("Debug output written to /tmp/evo2_debug_output.txt")
