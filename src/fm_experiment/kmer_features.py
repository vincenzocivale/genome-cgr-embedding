"""
Batch k-mer feature extraction from DNA sequences via FCGR.
Thin wrapper around src/core/{fcgr, kmer}.
"""

import numpy as np
from sklearn.preprocessing import normalize
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


def extract_multiscale_kmer_features(
    sequences: np.ndarray,
    k_values: list[int] = (4, 5, 6),
    grid_size: int = 128,
    n_workers: int = 1,
) -> np.ndarray:
    """
    Multiscale k-mer features: FCGR calcolata una volta, pooling a diverse scale,
    normalizzazione L2 per scala, concatenazione.

    Returns (N, sum(4^k for k in k_values)) float32 array.
    """
    # Calcola FCGR una sola volta alla risoluzione necessaria
    max_k = max(k_values)
    needed_grid = max(grid_size, 2 ** max_k)
    grids = batch_fcgr(sequences, grid_size=needed_grid, n_workers=n_workers)

    # Estrai e normalizza L2 per ogni scala
    parts = []
    for k in k_values:
        embedder = KmerEmbedder(k_size=k, normalize=True)
        feats = np.array([embedder.compute_embedding(g) for g in grids])
        # L2 normalize: ogni campione ha norma 1
        feats = normalize(feats, norm="l2", axis=1)
        parts.append(feats)

    return np.concatenate(parts, axis=1).astype(np.float32)
