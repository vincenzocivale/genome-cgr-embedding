#!/usr/bin/env python3
"""Check k=6 kmer features dimensions and rank."""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/oem/genome-cgr-embedding')
from src.features.kmer_features import extract_kmer_features

DATA_ROOT = "/data/genomic_bench/dna_foundation_benchmark/"

# Load sequences
train_df = pd.read_csv(os.path.join(DATA_ROOT, "EMP", "Yeast_H3", "train.csv"))
seqs_train = train_df['sequence'].values

print(f"Loading {len(seqs_train)} sequences...")

# Extract k=6 kmers with grid_size (as used in the actual script)
grid_size = max(128, 2 ** 6)  # = 128
print(f"Using grid_size={grid_size}")

X_train = extract_kmer_features(seqs_train, k=6, grid_size=grid_size, n_workers=1)
print(f"X_train shape: {X_train.shape}")
print(f"X_train dtype: {X_train.dtype}")
print(f"X_train range: [{X_train.min()}, {X_train.max()}]")
print(f"X_train sparsity: {(X_train == 0).mean()*100:.2f}%")

# Check rank
rank = np.linalg.matrix_rank(X_train)
print(f"\nRank: {rank}")
print(f"Rank / min(N, D) = {rank} / {min(X_train.shape[0], X_train.shape[1])} = {rank / min(X_train.shape[0], X_train.shape[1]):.2%}")

# Check condition number
cond = np.linalg.cond(X_train @ X_train.T)
print(f"Condition number of X_train @ X_train.T: {cond:.2e}")

# Check unique rows
unique_rows = len(np.unique(X_train, axis=0))
print(f"Unique rows: {unique_rows} / {X_train.shape[0]}")
