"""
Batch k-mer feature extraction from DNA sequences via FCGR.
Thin wrapper around src/core/{fcgr, kmer}.
"""

import numpy as np
from src.core.fcgr import batch_fcgr
from src.core.quadtree import quadtree_features


def _pool_kmer_from_grids(grids: np.ndarray, k: int) -> np.ndarray:
    """
    Vectorized pooling from FCGR grids to k-mer frequencies.
    grids: (N, H, W) float32
    returns: (N, 4^k) float32 (not normalized)
    """
    if grids.ndim != 3:
        raise ValueError("grids must be a 3D array (N, H, W)")
    n, h, w = grids.shape
    if h != w:
        raise ValueError("FCGR grid must be square")
    target = 2 ** k
    if h < target:
        raise ValueError(f"Grid size {h} too small for k={k}")
    if h == target:
        counts = grids.reshape(n, -1)
    else:
        block = h // target
        if h % target != 0:
            raise ValueError(f"Grid size {h} not divisible by target {target}")
        reshaped = grids.reshape(n, target, block, target, block)
        counts = reshaped.sum(axis=(2, 4)).reshape(n, -1)
    return counts.astype(np.float32)


def _normalize_l1(counts: np.ndarray) -> np.ndarray:
    total = counts.sum(axis=1, keepdims=True)
    total = np.where(total == 0, 1.0, total)
    return (counts / total).astype(np.float32)


def _normalize_l2(counts: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(counts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (counts / norms).astype(np.float32)


def kmer_from_grids(
    grids: list[np.ndarray],
    k: int,
    normalize: str = "l1",
) -> np.ndarray:
    """
    Vectorized k-mer extraction from a list of FCGR grids.
    normalize: "l1", "l2", or "none"
    """
    grid_arr = np.stack(grids, axis=0)
    counts = _pool_kmer_from_grids(grid_arr, k)
    if normalize == "l1":
        return _normalize_l1(counts)
    if normalize == "l2":
        return _normalize_l2(counts)
    return counts.astype(np.float32)


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
    return kmer_from_grids(grids, k, normalize="l1")


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

    # Extract and L2-normalise each scale
    parts = []
    for k in k_values:
        feats = kmer_from_grids(grids, k, normalize="l1")
        feats = _normalize_l2(feats)
        parts.append(feats)

    return np.concatenate(parts, axis=1).astype(np.float32)


def extract_weighted_multiscale_kmer_features(
    sequences: np.ndarray,
    k_values: list[int] = (4, 5, 6),
    weights: list[float] | None = None,
    grid_size: int = 128,
    n_workers: int = 1,
) -> np.ndarray:
    """
    Weighted multiscale k-mer features: per-scale L2 normalization,
    optional scalar weights, then concatenation.

    Returns (N, sum(4^k for k in k_values)) float32 array.
    """
    if weights is None:
        weights = [1.0 for _ in k_values]
    if len(weights) != len(k_values):
        raise ValueError("weights must have same length as k_values")

    max_k = max(k_values)
    needed_grid = max(grid_size, 2 ** max_k)
    grids = batch_fcgr(sequences, grid_size=needed_grid, n_workers=n_workers)

    parts = []
    for k, w in zip(k_values, weights):
        feats = kmer_from_grids(grids, k, normalize="l1")
        feats = _normalize_l2(feats)
        feats = feats * float(w)
        parts.append(feats)

    return np.concatenate(parts, axis=1).astype(np.float32)


def extract_quadtree_features(
    sequences: np.ndarray,
    max_depth: int = 7,
    p_threshold: float = 0.05,
    min_count: int = 8,
    grid_size: int | None = None,
    n_workers: int = 1,
) -> np.ndarray:
    """
    Adaptive QuadTree features with fixed-length Z-order layout.

    Returns (N, 4^max_depth) float32 array.
    """
    if grid_size is None:
        grid_size = 2 ** max_depth
    grids = batch_fcgr(sequences, grid_size=grid_size, n_workers=n_workers)
    feats = [
        quadtree_features(g, max_depth=max_depth,
                          p_threshold=p_threshold, min_count=min_count)
        for g in grids
    ]
    return np.array(feats, dtype=np.float32)


def _fcgr_index_to_kmer(idx: int, k: int) -> str:
    """Convert flattened FCGR index to k-mer string."""
    grid_size = 2 ** k
    i = idx // grid_size
    j = idx % grid_size
    inv_map = {(0, 0): 'A', (1, 0): 'C', (1, 1): 'G', (0, 1): 'T'}
    kmer = []
    for t in range(k):
        vx = (i >> t) & 1
        vy = (j >> t) & 1
        kmer.append(inv_map[(vx, vy)])
    return ''.join(kmer)


def _kmer_to_fcgr_index(kmer: str) -> int:
    """Convert k-mer string to flattened FCGR index."""
    k = len(kmer)
    mapping = {'A': (0, 0), 'C': (1, 0), 'G': (1, 1), 'T': (0, 1)}
    grid_size = 2 ** k
    i, j = 0, 0
    for t, b in enumerate(kmer):
        vx, vy = mapping[b]
        i |= (vx << t)
        j |= (vy << t)
    return i * grid_size + j


def _reverse_complement(seq: str) -> str:
    """Compute reverse complement of a DNA sequence."""
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(comp[b] for b in reversed(seq))


_canonical_cache: dict = {}


def _build_canonical_map(k: int) -> tuple:
    """
    Build canonical k-mer map (reverse-complement aware).

    Returns:
        canonical_map: (4^k,) int array mapping each kmer index to a canonical group index
        n_canonical:   number of canonical groups (k=4->136, k=5->512, k=6->2080)
    """
    if k in _canonical_cache:
        return _canonical_cache[k]

    n_kmers = 4 ** k
    canonical_idx = np.empty(n_kmers, dtype=np.int32)
    for idx in range(n_kmers):
        kmer = _fcgr_index_to_kmer(idx, k)
        rc_idx = _kmer_to_fcgr_index(_reverse_complement(kmer))
        canonical_idx[idx] = min(idx, rc_idx)

    unique_canonical = sorted(set(canonical_idx.tolist()))
    canon_to_group = {c: g for g, c in enumerate(unique_canonical)}
    n_canonical = len(unique_canonical)
    canonical_map = np.array(
        [canon_to_group[canonical_idx[i]] for i in range(n_kmers)], dtype=np.int32
    )
    _canonical_cache[k] = (canonical_map, n_canonical)
    return canonical_map, n_canonical


def canonicalize_kmer_features(kmer_freqs: np.ndarray, k: int) -> np.ndarray:
    """
    Convert standard k-mer frequency vectors to canonical (RC-aware) features.

    Input:  (N, 4^k)       L1-normalised k-mer frequencies (from kmer_from_grids)
    Output: (N, n_canonical) L1-normalised canonical k-mer frequencies

    n_canonical: k=4 -> 136, k=5 -> 512, k=6 -> 2080
    """
    n_kmers = 4 ** k
    assert kmer_freqs.shape[1] == n_kmers, f"Expected {n_kmers} features, got {kmer_freqs.shape[1]}"

    canonical_map, n_canonical = _build_canonical_map(k)

    # Aggregation matrix (n_kmers, n_canonical): one-hot assignment
    agg = np.zeros((n_kmers, n_canonical), dtype=np.float32)
    agg[np.arange(n_kmers), canonical_map] = 1.0

    result = kmer_freqs.astype(np.float32) @ agg  # (N, n_canonical)
    return _normalize_l1(result)


def extract_wavelet_features(
    sequences: np.ndarray,
    levels: int = 2,
    grid_size: int = 128,
    n_workers: int = 1,
) -> np.ndarray:
    """
    Wavelet/Spatial Pyramid features using simple 2D Haar-like pooling.

    For each level, compute the approximation (block sums) at that scale
    and concatenate all levels (level=0 is full resolution).
    """
    if levels < 1:
        raise ValueError("levels must be >= 1")

    grids = batch_fcgr(sequences, grid_size=grid_size, n_workers=n_workers)
    features = []
    for g in grids:
        parts = []
        current = g
        # Level 1: full resolution
        parts.append(current.flatten())
        for _ in range(1, levels):
            h, w = current.shape
            if h % 2 != 0 or w % 2 != 0:
                # Pad to even if needed
                pad_h = 0 if h % 2 == 0 else 1
                pad_w = 0 if w % 2 == 0 else 1
                current = np.pad(current, ((0, pad_h), (0, pad_w)), mode="constant")
                h, w = current.shape
            current = current.reshape(h // 2, 2, w // 2, 2).sum(axis=(1, 3))
            parts.append(current.flatten())
        feat = np.concatenate(parts, axis=0).astype(np.float32)
        features.append(feat)
    return np.array(features, dtype=np.float32)
