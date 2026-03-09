import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from core.kmer import KmerEmbedder


MAPPING = {'A': (0, 0), 'C': (1, 0), 'G': (1, 1), 'T': (0, 1)}


def compute_fcgr(seq: str, grid_size: int = 128) -> np.ndarray:
    """
    Calcola la matrice FCGR (conteggi grezzi) per una sequenza DNA.

    Args:
        seq: Sequenza DNA (caratteri A, C, G, T)
        grid_size: Dimensione della griglia (lato). Deve essere potenza di 2.

    Returns:
        Matrice (grid_size x grid_size) con i conteggi grezzi.
    """
    mat = np.zeros((grid_size, grid_size), dtype=np.float64)
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


def _process_single(seq: str, k: int, grid_size: int) -> np.ndarray:
    """Calcola FCGR e estrae k-mer features per una singola sequenza."""
    embedder = KmerEmbedder(k_size=k, normalize=True)
    grid = compute_fcgr(seq, grid_size)
    return embedder.compute_embedding(grid)


def extract_kmer_features(
    sequences,
    k: int,
    grid_size: int = 128,
    n_workers: int = 1,
) -> np.ndarray:
    """
    Estrae vettori di frequenze k-mer dalle sequenze via FCGR.

    Args:
        sequences: Lista/array di sequenze DNA
        k: Lunghezza del k-mero. Output dim = 4^k.
        grid_size: Dimensione griglia FCGR. Deve essere >= 2^k.
        n_workers: Numero di processi paralleli (1 = sequenziale).

    Returns:
        Matrice (n_sequences, 4^k) di frequenze k-mer normalizzate.
    """
    if grid_size < 2 ** k:
        raise ValueError(
            f"grid_size ({grid_size}) deve essere >= 2^k ({2**k}) per k={k}"
        )

    if n_workers > 1:
        func = partial(_process_single, k=k, grid_size=grid_size)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(tqdm(
                executor.map(func, sequences, chunksize=64),
                total=len(sequences),
                desc=f"FCGR k={k}",
            ))
        return np.array(results)

    embedder = KmerEmbedder(k_size=k, normalize=True)
    features = []
    for seq in tqdm(sequences, desc=f"FCGR k={k}"):
        grid = compute_fcgr(seq, grid_size)
        vec = embedder.compute_embedding(grid)
        features.append(vec)

    return np.array(features)
