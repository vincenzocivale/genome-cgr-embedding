"""
FCGR (Frequency Chaos Game Representation) computation.

Core primitives shared by all feature extraction methods.
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial

MAPPING = {'A': (0, 0), 'C': (1, 0), 'G': (1, 1), 'T': (0, 1)}


def compute_fcgr(seq: str, grid_size: int = 128) -> np.ndarray:
    """
    Compute the FCGR matrix (raw counts) for a DNA sequence.

    Args:
        seq: DNA sequence (A, C, G, T characters).
        grid_size: Grid side length. Must be a power of 2.

    Returns:
        (grid_size, grid_size) matrix of raw counts.
    """
    mat = np.zeros((grid_size, grid_size), dtype=np.float32)
    x, y = 0.5, 0.5

    for b in seq.upper():
        if b not in MAPPING:
            continue
        vx, vy = MAPPING[b]
        x = (x + vx) / 2
        y = (y + vy) / 2
        i = min(int(x * grid_size), grid_size - 1)
        j = min(int(y * grid_size), grid_size - 1)
        mat[i, j] += 1

    return mat


def compute_cgr_coords(seq: str) -> np.ndarray:
    """
    Compute the stream of CGR 2D coordinates for a DNA sequence.

    Returns an (N, 2) array where N is the number of valid bases.
    Coordinates are in [0, 1]^2.
    Used by codebook-based methods (KMeans, GMM) that operate
    directly on the point cloud without materialising the grid.
    """
    coords = []
    x, y = 0.5, 0.5

    for b in seq.upper():
        if b not in MAPPING:
            continue
        vx, vy = MAPPING[b]
        x = (x + vx) / 2
        y = (y + vy) / 2
        coords.append((x, y))

    if not coords:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array(coords, dtype=np.float32)


def _fcgr_worker(seq: str, grid_size: int) -> np.ndarray:
    return compute_fcgr(seq, grid_size)


def batch_fcgr(
    sequences,
    grid_size: int = 128,
    n_workers: int = 1,
    desc: str = "FCGR",
) -> list:
    """
    Compute FCGR matrices for a list of sequences, optionally in parallel.

    Returns a list of (grid_size, grid_size) arrays.
    """
    from tqdm import tqdm

    func = partial(_fcgr_worker, grid_size=grid_size)

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            grids = list(tqdm(
                ex.map(func, sequences, chunksize=128),
                total=len(sequences),
                desc=desc,
            ))
    else:
        grids = [func(s) for s in tqdm(sequences, desc=desc)]

    return grids
