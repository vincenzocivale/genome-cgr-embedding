#!/usr/bin/env python3
"""Understand why parallel script gets high R² with wrong calculation."""

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

# Load exactly as before
dataset_name = "EMP/Yeast_H3"
safe_ds = "EMP__Yeast_H3"

train_df = pd.read_csv(os.path.join(DATA_ROOT, "EMP", "Yeast_H3", "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_ROOT, "EMP", "Yeast_H3", "test.csv"))
seqs_train = train_df['sequence'].values
seqs_test = test_df['sequence'].values

emb_train = np.load(os.path.join(CACHE_DIR, "evo2_1b_base", safe_ds, "train.npz"))['embeddings'].astype(np.float64)
emb_test = np.load(os.path.join(CACHE_DIR, "evo2_1b_base", safe_ds, "test.npz"))['embeddings'].astype(np.float64)

X_train_kmer = extract_kmer_features(seqs_train, k=6, n_workers=1).astype(np.float64)
X_test_kmer = extract_kmer_features(seqs_test, k=6, n_workers=1).astype(np.float64)

ridge_model = Ridge(alpha=1.0, fit_intercept=True, solver='auto').fit(X_train_kmer, emb_train)
y_pred_test = ridge_model.predict(X_test_kmer)

# Now let's understand the R² calculation
ss_res = np.sum((emb_test - y_pred_test) ** 2)
ss_tot = np.sum((emb_test - np.mean(emb_test)) ** 2)

print(f"Global R² calculation (WRONG):")
print(f"  ss_res = {ss_res:.2e}")
print(f"  ss_tot = {ss_tot:.2e}")
print(f"  R² = 1 - ss_res/ss_tot = {1.0 - (ss_res / ss_tot):.6f}")

# What about per-dimension calculation?
ss_res_per_dim = np.sum((emb_test - y_pred_test) ** 2, axis=0)
ss_tot_per_dim = np.sum((emb_test - np.mean(emb_test, axis=0)) ** 2, axis=0)
r2_per_dim = 1.0 - (ss_res_per_dim / np.maximum(ss_tot_per_dim, 1e-12))

print(f"\nPer-dimension R² (CORRECT):")
print(f"  ss_res per-dim: mean={ss_res_per_dim.mean():.2e}, sum={ss_res_per_dim.sum():.2e}")
print(f"  ss_tot per-dim: mean={ss_tot_per_dim.mean():.2e}, sum={ss_tot_per_dim.sum():.2e}")
print(f"  R² per-dim: mean={r2_per_dim.mean():.6f}, median={np.median(r2_per_dim):.6f}")

# Check why global gives high R² while per-dim gives low R²
print(f"\nWhy does global R² differ from per-dim average?")
print(f"  Global: {1.0 - (ss_res / ss_tot):.6f}")
print(f"  Per-dim mean: {r2_per_dim.mean():.6f}")
print(f"  sklearn: {r2_score(emb_test, y_pred_test, multioutput='uniform_average'):.6f}")

# The issue: global R² weights dimensions by their variance
print(f"\nDimension variances:")
print(f"  Min variance: {ss_tot_per_dim.min():.2e}")
print(f"  Max variance: {ss_tot_per_dim.max():.2e}")
print(f"  Mean variance: {ss_tot_per_dim.mean():.2e}")

# Compute global R² correctly: should be weighted by dimension variance
ss_tot_global_correct = ss_tot_per_dim.sum()
ss_res_global_correct = ss_res_per_dim.sum()
print(f"\nGlobal R² (summed per-dim, CORRECT for multioutput):")
print(f"  R² = 1 - {ss_res_global_correct:.2e} / {ss_tot_global_correct:.2e} = {1.0 - (ss_res_global_correct / ss_tot_global_correct):.6f}")

print(f"\n*** The bug: parallel script computes mean(emb_test) globally, not per-dimension! ***")
