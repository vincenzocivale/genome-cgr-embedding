"""
K-mer embedder: extracts k-mer frequency vectors from FCGR grids.
"""

import numpy as np


class KmerEmbedder:
    """
    Extracts a k-mer frequency vector from a CGR grid.

    A CGR grid of size (2^k x 2^k) encodes k-mer frequencies spatially.
    For grids larger than 2^k, sum-pooling downsamples to the target resolution.
    """

    def __init__(self, k_size: int = 3, normalize: bool = True):
        self.k = k_size
        self.normalize = normalize
        self.target_dim = 2 ** k_size
        self.feature_dim = 4 ** k_size

    def compute_embedding(self, cgr_grid: np.ndarray) -> np.ndarray:
        H, W = cgr_grid.shape

        if H < self.target_dim or W < self.target_dim:
            raise ValueError(
                f"CGR grid ({H}x{W}) too small for k={self.k} "
                f"(need at least {self.target_dim}x{self.target_dim})"
            )

        if H == self.target_dim and W == self.target_dim:
            counts = cgr_grid.flatten()
        else:
            block_h = H // self.target_dim
            block_w = W // self.target_dim
            reshaped = cgr_grid.reshape(
                self.target_dim, block_h, self.target_dim, block_w
            )
            counts = reshaped.sum(axis=(1, 3)).flatten()

        if self.normalize:
            total = counts.sum()
            if total > 0:
                return (counts / total).astype(np.float32)
            return counts.astype(np.float32)

        return counts.astype(np.float32)
