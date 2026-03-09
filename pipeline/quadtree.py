import numpy as np
from tqdm import tqdm

from .feature_extraction import compute_fcgr


def _quadtree_recursive(grid, row, col, size, depth, max_depth, threshold, stats):
    """
    Suddivisione ricorsiva QuadTree adattiva.

    Se la densità della cella corrente supera la soglia E non siamo
    alla profondità massima, suddividi in 4 sotto-celle.
    Altrimenti registra questa cella come foglia al livello corrente.

    Args:
        grid: Matrice FCGR (conteggi)
        row, col: Angolo top-left della cella corrente
        size: Dimensione lato della cella corrente
        depth: Profondità corrente (0 = radice)
        max_depth: Profondità massima consentita
        threshold: Soglia di densità per suddividere
        stats: Lista di dict, uno per livello di profondità
    """
    region = grid[row:row + size, col:col + size]
    density = region.sum()

    if depth >= max_depth or density <= threshold or size <= 1:
        # Foglia: registra statistiche al livello corrente
        stats[depth]["n_leaves"] += 1
        stats[depth]["total_density"] += density
        stats[depth]["max_density"] = max(stats[depth]["max_density"], density)
        stats[depth]["densities"].append(density)
        return

    # Suddividi in 4 quadranti
    half = size // 2
    _quadtree_recursive(grid, row, col, half, depth + 1, max_depth, threshold, stats)
    _quadtree_recursive(grid, row, col + half, half, depth + 1, max_depth, threshold, stats)
    _quadtree_recursive(grid, row + half, col, half, depth + 1, max_depth, threshold, stats)
    _quadtree_recursive(grid, row + half, col + half, half, depth + 1, max_depth, threshold, stats)


def quadtree_features(grid: np.ndarray, max_depth: int = 7, threshold: float = 0.0) -> np.ndarray:
    """
    Estrae features da una matrice FCGR usando QuadTree adattivo.

    Per ogni livello di profondità (0..max_depth) calcola:
    - n_leaves: numero di foglie a quel livello
    - total_density: somma delle densità delle foglie
    - mean_density: densità media delle foglie
    - max_density: densità massima tra le foglie
    - std_density: deviazione standard delle densità

    Args:
        grid: Matrice FCGR (NxN)
        max_depth: Profondità massima del QuadTree (griglia deve essere >= 2^max_depth)
        threshold: Densità minima per suddividere una cella

    Returns:
        Vettore 1D di dimensione (max_depth + 1) * 5
    """
    n_levels = max_depth + 1
    stats = [
        {"n_leaves": 0, "total_density": 0.0, "max_density": 0.0, "densities": []}
        for _ in range(n_levels)
    ]

    size = grid.shape[0]
    _quadtree_recursive(grid, 0, 0, size, 0, max_depth, threshold, stats)

    # Normalizza densità rispetto al totale
    total = grid.sum()
    if total == 0:
        total = 1.0

    feature_vec = []
    for level in range(n_levels):
        s = stats[level]
        n = s["n_leaves"]
        if n > 0:
            densities = np.array(s["densities"])
            feature_vec.extend([
                n,
                s["total_density"] / total,
                (s["total_density"] / n) / total,
                s["max_density"] / total,
                densities.std() / total if n > 1 else 0.0,
            ])
        else:
            feature_vec.extend([0, 0.0, 0.0, 0.0, 0.0])

    return np.array(feature_vec, dtype=np.float64)


def extract_quadtree_features(
    sequences,
    grid_size: int = 128,
    max_depth: int = 7,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Estrae features QuadTree adattive per un set di sequenze.

    Args:
        sequences: Lista di sequenze DNA
        grid_size: Dimensione griglia FCGR
        max_depth: Profondità massima QuadTree
        threshold: Soglia densità per suddivisione

    Returns:
        Matrice (n_sequences, (max_depth+1)*5)
    """
    features = []
    for seq in tqdm(sequences, desc=f"QuadTree (depth={max_depth})"):
        grid = compute_fcgr(seq, grid_size)
        vec = quadtree_features(grid, max_depth=max_depth, threshold=threshold)
        features.append(vec)

    return np.array(features)
