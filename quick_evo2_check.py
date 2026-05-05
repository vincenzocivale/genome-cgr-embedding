#!/usr/bin/env python3
"""Quick check of Evo2 Ridge R² on one dataset."""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# Hard-coded test on EMP__Yeast_H3
DATA_ROOT = "/data/genomic_bench/dna_foundation_benchmark/"
CACHE_DIR = "cache/fm_embeddings"

# Load embeddings
safe_ds = "EMP__Yeast_H3"
emb_train = np.load(os.path.join(CACHE_DIR, "evo2_1b_base", safe_ds, "train.npz"))['embeddings'].astype(np.float64)
emb_test = np.load(os.path.join(CACHE_DIR, "evo2_1b_base", safe_ds, "test.npz"))['embeddings'].astype(np.float64)

print(f"Loaded embeddings: train={emb_train.shape}, test={emb_test.shape}")
print(f"  Train dtype: {emb_train.dtype}, min={emb_train.min():.4f}, max={emb_train.max():.4f}")
print(f"  Test dtype: {emb_test.dtype}, min={emb_test.min():.4f}, max={emb_test.max():.4f}")

# Create simple kmer features (k=2) for debugging
def simple_kmer(seqs, k=2):
    """Create simple k-mer feature matrix."""
    kmers = {}
    for seq in seqs:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            kmers[kmer] = kmers.get(kmer, 0) + 1

    kmer_list = sorted(kmers.keys())
    X = np.zeros((len(seqs), len(kmer_list)))
    for i, seq in enumerate(seqs):
        for j, kmer in enumerate(kmer_list):
            X[i, j] = seq.count(kmer)
    return X, kmer_list

# Load sequences
train_df = pd.read_csv(os.path.join(DATA_ROOT, "EMP", "Yeast_H3", "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_ROOT, "EMP", "Yeast_H3", "test.csv"))
seqs_train = train_df['sequence'].values
seqs_test = test_df['sequence'].values

X_train, kmer_list = simple_kmer(seqs_train, k=2)
X_test, _ = simple_kmer(seqs_test, k=2)

print(f"\nKmer features (k=2): train={X_train.shape}, test={X_test.shape}")
print(f"  Train sparsity: {(X_train == 0).mean()*100:.2f}%")

# Ridge regression
ridge = Ridge(alpha=1.0).fit(X_train, emb_train)
y_pred = ridge.predict(X_test)
r2 = r2_score(emb_test, y_pred, multioutput='uniform_average')

print(f"\nRidge R² (k=2, alpha=1.0): {r2:.6f}")

# Check prediction quality
print(f"Pred range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
print(f"Residual range: [{(emb_test - y_pred).min():.4f}, {(emb_test - y_pred).max():.4f}]")

# Per-dimension R²
per_dim_r2 = []
for i in range(min(5, emb_test.shape[1])):
    r2_i = r2_score(emb_test[:, i], y_pred[:, i])
    per_dim_r2.append(r2_i)

print(f"Per-dim R² (first 5): {per_dim_r2}")
