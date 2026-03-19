"""
Batch k-mer feature extraction from DNA sequences via FCGR.
Thin wrapper around src/core/{fcgr, kmer}.
"""

import numpy as np
from src.core.fcgr import batch_fcgr
from src.core.kmer import KmerEmbedder


def extract_kmer_features(
    sequences: np.ndarray,
    k: int = 6,
    grid_size: int = 128,
    n_workers: int = 1,
) -> np.ndarray:
    """
    Extract k-mer frequency vectors from DNA sequences.

    Returns (N, 4^k) float32 array of normalised k-mer frequencies.
    """
    grids = batch_fcgr(sequences, grid_size=grid_size, n_workers=n_workers)
    embedder = KmerEmbedder(k_size=k, normalize=True)
    features = np.array([embedder.compute_embedding(g) for g in grids])
    return features
