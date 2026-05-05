#!/usr/bin/env python3
"""Check the rank of Evo2 embeddings."""

import os
import numpy as np

# Load embeddings
safe_ds = "EMP__Yeast_H3"
emb_train = np.load(os.path.join("cache/fm_embeddings", "evo2_1b_base", safe_ds, "train.npz"))['embeddings'].astype(np.float64)

print(f"Embeddings shape: {emb_train.shape}")
print(f"Number of samples: {emb_train.shape[0]}")
print(f"Embedding dimension: {emb_train.shape[1]}")

# Compute rank using SVD
U, s, Vt = np.linalg.svd(emb_train, full_matrices=False)
rank = np.sum(s > 1e-10)
print(f"\nRank (SVD, tol=1e-10): {rank}")
print(f"Rank / min(N, D) = {rank} / {min(emb_train.shape[0], emb_train.shape[1])} = {rank / min(emb_train.shape[0], emb_train.shape[1]):.2%}")

# Check number of distinct rows
print(f"\nNumber of unique rows: {len(np.unique(emb_train, axis=0))}")
print(f"All unique? {len(np.unique(emb_train, axis=0)) == emb_train.shape[0]}")

# Check condition number
cond = np.linalg.cond(emb_train)
print(f"Condition number: {cond:.2e}")
