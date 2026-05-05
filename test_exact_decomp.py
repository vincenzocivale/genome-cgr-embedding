#!/usr/bin/env python3
"""Test exact decomposition as done in parallel script."""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, '/home/oem/genome-cgr-embedding')
from src.features.kmer_features import extract_kmer_features

DATA_ROOT = "/data/genomic_bench/dna_foundation_benchmark/"
CACHE_DIR = "cache/fm_embeddings"

# Dataset
dataset_name = "EMP/Yeast_H3"
safe_ds = "EMP__Yeast_H3"

# Load data exactly as parallel script does
train_df = pd.read_csv(os.path.join(DATA_ROOT, "EMP", "Yeast_H3", "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_ROOT, "EMP", "Yeast_H3", "test.csv"))
seqs_train = train_df['sequence'].values
seqs_test = test_df['sequence'].values
labels_train = train_df['label'].values.astype(int)
labels_test = test_df['label'].values.astype(int)

# Load embeddings exactly as parallel script would
emb_train = np.load(os.path.join(CACHE_DIR, "evo2_1b_base", safe_ds, "train.npz"))['embeddings'].astype(np.float64)
emb_test = np.load(os.path.join(CACHE_DIR, "evo2_1b_base", safe_ds, "test.npz"))['embeddings'].astype(np.float64)

print(f"Embeddings loaded: train={emb_train.shape}, test={emb_test.shape}")

# Extract k-mer features EXACTLY as parallel script does (NO GRID_SIZE SPECIFIED)
# This will use the default grid_size=128
X_train_kmer = extract_kmer_features(seqs_train, k=6, n_workers=1).astype(np.float64)
X_test_kmer = extract_kmer_features(seqs_test, k=6, n_workers=1).astype(np.float64)

print(f"Kmer features: train={X_train_kmer.shape}, test={X_test_kmer.shape}")

# Ridge regression EXACTLY as parallel script does
ridge_model = Ridge(alpha=1.0, fit_intercept=True, solver='auto').fit(X_train_kmer, emb_train)

# Ridge R² evaluated on TEST data EXACTLY as parallel script does
y_pred_test = ridge_model.predict(X_test_kmer)
ss_res = np.sum((emb_test - y_pred_test) ** 2)
ss_tot = np.sum((emb_test - np.mean(emb_test)) ** 2)
ridge_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

print(f"\nRidge R² (exact as parallel script): {ridge_r2:.10f}")

# Also compute using sklearn's r2_score for comparison
ridge_r2_sklearn = r2_score(emb_test, y_pred_test, multioutput='uniform_average')
print(f"Ridge R² (sklearn multioutput): {ridge_r2_sklearn:.10f}")

# Per-dimension analysis
per_dim_r2 = []
for i in range(min(10, emb_test.shape[1])):
    r2_i = r2_score(emb_test[:, i], y_pred_test[:, i])
    per_dim_r2.append(r2_i)

print(f"Per-dim R² (first 10): {[f'{r:.4f}' for r in per_dim_r2]}")
print(f"  Mean: {np.mean(per_dim_r2):.6f}")
print(f"  Median: {np.median(per_dim_r2):.6f}")
print(f"  Min: {np.min(per_dim_r2):.6f}")
print(f"  Max: {np.max(per_dim_r2):.6f}")
print(f"  Num perfect (R²=1.0): {sum(1 for r in per_dim_r2 if r >= 0.99999)}")

# Check residuals
residuals = emb_test - y_pred_test
print(f"\nResiduals:")
print(f"  Mean: {residuals.mean():.6f}")
print(f"  Std: {residuals.std():.6f}")
print(f"  Min: {residuals.min():.6f}")
print(f"  Max: {residuals.max():.6f}")

# Check coef magnitude
print(f"\nRidge coefficients:")
print(f"  Shape: {ridge_model.coef_.shape}")
print(f"  Mean abs: {np.abs(ridge_model.coef_).mean():.6f}")
print(f"  Max abs: {np.abs(ridge_model.coef_).max():.6f}")
