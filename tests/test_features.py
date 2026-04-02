import numpy as np

from src.fm_experiment.kmer_features import (
    extract_weighted_multiscale_kmer_features,
    extract_wavelet_features,
)


def test_weighted_multiscale_dim():
    seqs = np.array(["ACGTACGT", "TGCATGCA"])
    feats = extract_weighted_multiscale_kmer_features(
        seqs, k_values=[2, 3], weights=[1.0, 2.0], grid_size=8, n_workers=1
    )
    assert feats.shape == (2, 4 ** 2 + 4 ** 3)


def test_wavelet_dim():
    seqs = np.array(["ACGTACGT"])
    feats = extract_wavelet_features(seqs, levels=3, grid_size=8, n_workers=1)
    # level dims: 8x8 + 4x4 + 2x2
    assert feats.shape == (1, 64 + 16 + 4)
