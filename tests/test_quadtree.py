import numpy as np

from src.core.quadtree import quadtree_features


def test_quadtree_dim_and_sum():
    grid = np.ones((8, 8), dtype=np.float32)
    feat = quadtree_features(grid, max_depth=3, p_threshold=0.05, min_count=0)
    assert feat.shape == (4 ** 3,)
    assert np.isclose(feat.sum(), 1.0)


def test_quadtree_zero_grid():
    grid = np.zeros((8, 8), dtype=np.float32)
    feat = quadtree_features(grid, max_depth=3, p_threshold=0.05, min_count=0)
    assert np.allclose(feat, 0.0)
