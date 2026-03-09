"""
QuadTree adattivo per partizionamento della matrice FCGR.

Criterio di suddivisione: test chi-quadrato tra i 4 quadranti figli.
Verifica se la distribuzione dei conteggi devia significativamente
dall'uniforme, tenendo conto della varianza naturale dovuta al
basso numero di conteggi (robusto su griglie sparse).

Il parametro `split_threshold` è il p-value sotto il quale la cella
viene suddivisa: con p-value basso la distribuzione è significativamente
non-uniforme → c'è struttura → vale la pena suddividere.

Default: split_threshold=0.05 (95% di confidenza).

Il vettore di feature ha dimensione fissa = 4^max_depth. Le foglie
vengono mappate in Z-order (Morton order). Una foglia a profondità
< max_depth propaga la propria densità equamente ai discendenti terminali.

Con split_threshold=0.0 → non suddivide mai.
Con split_threshold=1.0 → suddivide sempre → k-mer a risoluzione fissa.
"""

import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from scipy.stats import chisquare

from .feature_extraction import compute_fcgr


def _should_split(grid, row, col, size, split_threshold, min_count):
    """
    Test chi-quadrato: i 4 quadranti figli deviano dall'uniforme?

    Returns:
        True se la distribuzione è significativamente non-uniforme
        (p-value < split_threshold) e c'è abbastanza segnale.
    """
    half = size // 2
    observed = np.array([
        grid[row:row + half, col:col + half].sum(),
        grid[row:row + half, col + half:col + size].sum(),
        grid[row + half:row + size, col:col + half].sum(),
        grid[row + half:row + size, col + half:col + size].sum(),
    ])
    total = observed.sum()

    # Troppo pochi conteggi → il test non è affidabile → non suddividere
    if total < min_count:
        return False

    _, p_value = chisquare(observed)
    return p_value < split_threshold


def _qt_recursive(
    grid, row, col, size, depth, max_depth, split_threshold, min_count,
    out, offset, stride,
):
    """
    Partizionamento ricorsivo guidato dal chi-quadrato.

    Suddivide se:
      1. Non siamo a max_depth
      2. I 4 figli hanno distribuzione significativamente non-uniforme
    """
    density = grid[row:row + size, col:col + size].sum()

    # Foglia: propaga densità equamente a tutti gli slot discendenti
    if depth >= max_depth or size <= 1:
        out[offset:offset + stride] = density / stride
        return

    if not _should_split(grid, row, col, size, split_threshold, min_count):
        out[offset:offset + stride] = density / stride
        return

    # Suddividi in Z-order
    half = size // 2
    cs = stride // 4
    _qt_recursive(grid, row,        col,        half, depth + 1, max_depth, split_threshold, min_count, out, offset,          cs)
    _qt_recursive(grid, row,        col + half, half, depth + 1, max_depth, split_threshold, min_count, out, offset + cs,     cs)
    _qt_recursive(grid, row + half, col,        half, depth + 1, max_depth, split_threshold, min_count, out, offset + 2 * cs, cs)
    _qt_recursive(grid, row + half, col + half, half, depth + 1, max_depth, split_threshold, min_count, out, offset + 3 * cs, cs)


def adaptive_fcgr_features(
    grid: np.ndarray,
    max_depth: int = 6,
    split_threshold: float = 0.05,
    min_count: int = 8,
    normalize: bool = True,
) -> np.ndarray:
    """
    Estrae feature adattive da una matrice FCGR tramite QuadTree.

    Il QuadTree suddivide una cella solo se la distribuzione dei conteggi
    tra i 4 quadranti figli è significativamente non-uniforme (chi-quadrato).

    Args:
        grid:             Matrice FCGR (NxN, conteggi grezzi).
        max_depth:        Profondità massima. Output dim = 4^max_depth.
        split_threshold:  P-value del chi-quadrato (default: 0.05).
                          Suddivide se p < soglia. 0.0 = non suddivide mai,
                          1.0 = suddivide sempre (= k-mer fisso).
        min_count:        Conteggi minimi per eseguire il test (default: 8).
                          Sotto questa soglia il test non è affidabile.
        normalize:        Se True, normalizza il vettore a somma 1.

    Returns:
        Vettore 1D di dimensione 4^max_depth.
    """
    out_dim = 4 ** max_depth
    out = np.zeros(out_dim, dtype=np.float64)

    _qt_recursive(
        grid, 0, 0, grid.shape[0], 0, max_depth,
        split_threshold, min_count, out, 0, out_dim,
    )

    if normalize:
        total = out.sum()
        if total > 0:
            out /= total

    return out


def _adaptive_single(
    seq: str,
    grid_size: int,
    max_depth: int,
    split_threshold: float,
    min_count: int,
    normalize: bool,
) -> np.ndarray:
    """Worker per parallelizzazione."""
    grid = compute_fcgr(seq, grid_size)
    return adaptive_fcgr_features(grid, max_depth, split_threshold, min_count, normalize)


def extract_adaptive_features(
    sequences,
    grid_size: int = 128,
    max_depth: int = 6,
    split_threshold: float = 0.05,
    min_count: int = 8,
    normalize: bool = True,
    n_workers: int = 1,
) -> np.ndarray:
    """
    Estrae feature QuadTree adattive per un set di sequenze.

    Args:
        sequences:        Lista/array di sequenze DNA.
        grid_size:        Dimensione griglia FCGR (default 128). Deve essere >= 2^max_depth.
        max_depth:        Profondità massima QuadTree (default 6 → 4096 feature).
        split_threshold:  P-value chi-quadrato per suddivisione (default 0.05).
        min_count:        Conteggi minimi per suddivisione (default 8).
        normalize:        Se True, normalizza ogni vettore a somma 1.
        n_workers:        Worker paralleli (1 = sequenziale).

    Returns:
        Matrice (n_sequences, 4^max_depth).
    """
    if grid_size < 2 ** max_depth:
        raise ValueError(
            f"grid_size ({grid_size}) deve essere >= 2^max_depth ({2**max_depth})"
        )

    desc = f"Adaptive FCGR (depth={max_depth}, p<{split_threshold})"

    if n_workers > 1:
        func = partial(
            _adaptive_single,
            grid_size=grid_size,
            max_depth=max_depth,
            split_threshold=split_threshold,
            min_count=min_count,
            normalize=normalize,
        )
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(tqdm(
                executor.map(func, sequences, chunksize=64),
                total=len(sequences),
                desc=desc,
            ))
        return np.array(results)

    features = []
    for seq in tqdm(sequences, desc=desc):
        grid = compute_fcgr(seq, grid_size)
        vec = adaptive_fcgr_features(grid, max_depth, split_threshold, min_count, normalize)
        features.append(vec)

    return np.array(features)
