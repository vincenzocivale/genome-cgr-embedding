"""
Rappresentazione combinata: k-mer uniforme + QuadTree adattivo.

Dalla stessa matrice FCGR estrae e concatena:
- k-mer uniforme   : frequenze a risoluzione fissa k        (4^k feature)
- QuadTree adattivo: frequenze a risoluzione variabile       (4^qt_depth feature)
                     guidato dal chi-quadrato sui quadranti figli

Il vettore risultante è [kmer_k | adaptive_qt].
"""

import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from core.kmer import KmerEmbedder
from pipeline.feature_extraction import compute_fcgr
from pipeline.quadtree import adaptive_fcgr_features


def _combined_single(
    seq: str,
    k: int,
    grid_size: int,
    qt_max_depth: int,
    qt_split_threshold: float,
) -> np.ndarray:
    """Vettore combinato [kmer | adaptive_qt] per una singola sequenza."""
    grid = compute_fcgr(seq, grid_size)
    kmer_vec = KmerEmbedder(k_size=k, normalize=True).compute_embedding(grid)
    qt_vec = adaptive_fcgr_features(
        grid, max_depth=qt_max_depth, split_threshold=qt_split_threshold,
    )
    return np.concatenate([kmer_vec, qt_vec])


def extract_combined_features(
    sequences,
    k: int,
    grid_size: int = 128,
    qt_max_depth: int = 4,
    qt_split_threshold: float = 0.05,
    n_workers: int = 1,
) -> np.ndarray:
    """
    Estrae il vettore combinato [k-mer | QuadTree adattivo].

    Args:
        sequences:            Lista/array di sequenze DNA.
        k:                    Lunghezza k-mero (output = 4^k).
        grid_size:            Dimensione griglia FCGR (default 128).
        qt_max_depth:         Profondità massima QuadTree (default 4 → 256 feature).
        qt_split_threshold:   P-value chi-quadrato per suddivisione (default 0.05).
        n_workers:            Worker paralleli (1 = sequenziale).

    Returns:
        Matrice (n_sequences, 4^k + 4^qt_max_depth).
    """
    if grid_size < 2 ** k:
        raise ValueError(
            f"grid_size ({grid_size}) deve essere >= 2^k ({2**k}) per k={k}"
        )
    if grid_size < 2 ** qt_max_depth:
        raise ValueError(
            f"grid_size ({grid_size}) deve essere >= 2^qt_max_depth ({2**qt_max_depth})"
        )

    func = partial(
        _combined_single,
        k=k,
        grid_size=grid_size,
        qt_max_depth=qt_max_depth,
        qt_split_threshold=qt_split_threshold,
    )

    desc = f"FCGR k={k} + Adaptive d={qt_max_depth}"

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(tqdm(
                executor.map(func, sequences, chunksize=64),
                total=len(sequences),
                desc=desc,
            ))
        return np.array(results)

    features = []
    for seq in tqdm(sequences, desc=desc):
        features.append(func(seq))

    return np.array(features)
