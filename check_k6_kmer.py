#!/usr/bin/env python3
"""Check k=6 kmer extraction and Ridge R²."""

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

# Load embeddings
safe_ds = "EMP__Yeast_H3"
emb_train = np.load(os.path.join(CACHE_DIR, "evo2_1b_base", safe_ds, "train.npz"))['embeddings'].astype(np.float64)
emb_test = np.load(os.path.join(CACHE_DIR, "evo2_1b_base", safe_ds, "test.npz"))['embeddings'].astype(np.float64)

# Load sequences
train_df = pd.read_csv(os.path.join(DATA_ROOT, "EMP", "Yeast_H3", "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_ROOT, "EMP", "Yeast_H3", "test.csv"))
seqs_train = train_df['sequence'].values
seqs_test = test_df['sequence'].values

print(f"Loading sequences: {len(seqs_train)} train, {len(seqs_test)} test")

# Extract k=6 kmers
print("Extracting k=6 kmers...")
X_train = extract_kmer_features(seqs_train, k=6, n_workers=1).astype(np.float64)
X_test = extract_kmer_features(seqs_test, k=6, n_workers=1).astype(np.float64)

print(f"Kmer features (k=6): train={X_train.shape}, test={X_test.shape}")
print(f"  Train dtype: {X_train.dtype}")
print(f"  Train range: [{X_train.min()}, {X_train.max()}]")
print(f"  Train sparsity: {(X_train == 0).mean()*100:.2f}%")
print(f"  Train NaN count: {np.isnan(X_train).sum()}")
print(f"  Train inf count: {np.isinf(X_train).sum()}")

# Ridge regression
ridge = Ridge(alpha=1.0, fit_intercept=True).fit(X_train, emb_train)
y_pred = ridge.predict(X_test)
r2 = r2_score(emb_test, y_pred, multioutput='uniform_average')

print(f"\nRidge (alpha=1.0) R²: {r2:.10f}")

# Check per-dimension R²
per_dim_r2 = [r2_score(emb_test[:, i], y_pred[:, i]) for i in range(min(10, emb_test.shape[1]))]
print(f"Per-dim R² (first 10): {per_dim_r2}")
print(f"  Mean: {np.mean(per_dim_r2):.6f}")
print(f"  Min: {np.min(per_dim_r2):.6f}")
print(f"  Max: {np.max(per_dim_r2):.6f}")

# Check if X_train is full rank or nearly singular
rank = np.linalg.matrix_rank(X_train, tol=1e-10)
print(f"\nX_train rank: {rank}/{X_train.shape[1]}")

# Check covariate structure
print(f"\nCovariance structure:")
print(f"  X_train condition number: {np.linalg.cond(X_train @ X_train.T):.2e}")

# Try a smaller alpha
from sklearn.linear_model import RidgeCV
ridgecv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], gcv_mode='auto')
ridgecv.fit(X_train, emb_train)
y_pred_cv = ridgecv.predict(X_test)
r2_cv = r2_score(emb_test, y_pred_cv, multioutput='uniform_average')

print(f"\nRidgeCV (best alpha={ridgecv.alpha_:.4f}) R²: {r2_cv:.10f}")
