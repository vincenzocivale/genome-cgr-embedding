"""
Adaptive QuadTree features on FCGR grids.

Implements chi-square splitting with fixed-length Z-order (Morton) output.
"""

from __future__ import annotations

import math
import numpy as np

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False


def _chi2_pvalue_df3(chi2: float) -> float:
    """
    Survival function (p-value) for chi-square with df=3.

    CDF for df=3:
      F(x) = erf(sqrt(x/2)) - sqrt(2/pi) * sqrt(x) * exp(-x/2)
    p-value = 1 - F(x)
    """
    if chi2 <= 0:
        return 1.0
    t = math.sqrt(chi2 / 2.0)
    cdf = math.erf(t) - math.sqrt(2.0 / math.pi) * math.sqrt(chi2) * math.exp(-chi2 / 2.0)
    p = 1.0 - cdf
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return p


def _morton_prefix(row: int, col: int, depth: int) -> int:
    """Interleave row/col bits to a Morton (Z-order) prefix."""
    idx = 0
    for i in range(depth - 1, -1, -1):
        r = (row >> i) & 1
        c = (col >> i) & 1
        idx = (idx << 2) | (r << 1) | c
    return idx


def _integral_image(grid: np.ndarray) -> np.ndarray:
    return grid.cumsum(axis=0).cumsum(axis=1)


def _sum_region(integ: np.ndarray, r0: int, c0: int, size: int) -> float:
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    total = float(integ[r1, c1])
    if r0 > 0:
        total -= float(integ[r0 - 1, c1])
    if c0 > 0:
        total -= float(integ[r1, c0 - 1])
    if r0 > 0 and c0 > 0:
        total += float(integ[r0 - 1, c0 - 1])
    return total


def quadtree_features(
    grid: np.ndarray,
    max_depth: int = 7,
    p_threshold: float = 0.05,
    min_count: int = 8,
    use_numba: bool | None = None,
) -> np.ndarray:
    """
    Compute adaptive QuadTree features with fixed-length Z-order layout.

    Args:
        grid: FCGR grid (2^max_depth x 2^max_depth preferred).
        max_depth: maximum quadtree depth.
        p_threshold: split if p-value < p_threshold.
        min_count: minimum counts to allow split.

    Returns:
        (4^max_depth,) float32 vector.
    """
    grid = np.asarray(grid, dtype=np.float32)
    target = 2 ** max_depth

    if grid.shape[0] != grid.shape[1]:
        raise ValueError("FCGR grid must be square.")

    if grid.shape[0] != target:
        # Sum-pool to target resolution if grid is larger and divisible.
        if grid.shape[0] < target or grid.shape[0] % target != 0:
            raise ValueError(
                f"Grid size {grid.shape[0]} not compatible with max_depth={max_depth}."
            )
        factor = grid.shape[0] // target
        grid = grid.reshape(target, factor, target, factor).sum(axis=(1, 3))

    total = float(grid.sum())
    out_dim = 4 ** max_depth
    out = np.zeros(out_dim, dtype=np.float32)
    if total <= 0:
        return out

    integ = _integral_image(grid)

    if use_numba is None:
        use_numba = _NUMBA_AVAILABLE

    if use_numba and _NUMBA_AVAILABLE:
        return _quadtree_features_numba(
            integ, total, max_depth, p_threshold, min_count
        )

    def assign(row: int, col: int, depth: int, count: float) -> None:
        scale = 4 ** (max_depth - depth)
        val = (count / total) / scale
        start = _morton_prefix(row, col, depth) * scale
        out[start:start + scale] = val

    def recurse(r0: int, c0: int, size: int, depth: int, row: int, col: int) -> None:
        count = _sum_region(integ, r0, c0, size)
        if depth >= max_depth or count < min_count:
            assign(row, col, depth, count)
            return

        if count <= 0:
            assign(row, col, depth, 0.0)
            return

        half = size // 2
        q00 = _sum_region(integ, r0, c0, half)
        q01 = _sum_region(integ, r0, c0 + half, half)
        q10 = _sum_region(integ, r0 + half, c0, half)
        q11 = _sum_region(integ, r0 + half, c0 + half, half)

        expected = count / 4.0
        if expected <= 0:
            assign(row, col, depth, count)
            return

        chi2 = ((q00 - expected) ** 2 + (q01 - expected) ** 2 +
                (q10 - expected) ** 2 + (q11 - expected) ** 2) / expected
        p_val = _chi2_pvalue_df3(chi2)

        if p_val >= p_threshold:
            assign(row, col, depth, count)
            return

        recurse(r0, c0, half, depth + 1, row * 2, col * 2)
        recurse(r0, c0 + half, half, depth + 1, row * 2, col * 2 + 1)
        recurse(r0 + half, c0, half, depth + 1, row * 2 + 1, col * 2)
        recurse(r0 + half, c0 + half, half, depth + 1, row * 2 + 1, col * 2 + 1)

    recurse(0, 0, target, 0, 0, 0)
    return out


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _morton_prefix_nb(row: int, col: int, depth: int) -> int:
        idx = 0
        for i in range(depth - 1, -1, -1):
            r = (row >> i) & 1
            c = (col >> i) & 1
            idx = (idx << 2) | (r << 1) | c
        return idx


    @njit(cache=True)
    def _sum_region_nb(integ: np.ndarray, r0: int, c0: int, size: int) -> float:
        r1 = r0 + size - 1
        c1 = c0 + size - 1
        total = float(integ[r1, c1])
        if r0 > 0:
            total -= float(integ[r0 - 1, c1])
        if c0 > 0:
            total -= float(integ[r1, c0 - 1])
        if r0 > 0 and c0 > 0:
            total += float(integ[r0 - 1, c0 - 1])
        return total


    @njit(cache=True)
    def _chi2_pvalue_df3_nb(chi2: float) -> float:
        if chi2 <= 0:
            return 1.0
        t = math.sqrt(chi2 / 2.0)
        cdf = math.erf(t) - math.sqrt(2.0 / math.pi) * math.sqrt(chi2) * math.exp(-chi2 / 2.0)
        p = 1.0 - cdf
        if p < 0.0:
            return 0.0
        if p > 1.0:
            return 1.0
        return p


    @njit(cache=True)
    def _quadtree_features_numba(
        integ: np.ndarray,
        total: float,
        max_depth: int,
        p_threshold: float,
        min_count: int,
    ) -> np.ndarray:
        out_dim = 4 ** max_depth
        out = np.zeros(out_dim, dtype=np.float32)
        if total <= 0:
            return out

        max_nodes = (4 ** (max_depth + 1) - 1) // 3
        r0_stack = np.empty(max_nodes, dtype=np.int64)
        c0_stack = np.empty(max_nodes, dtype=np.int64)
        size_stack = np.empty(max_nodes, dtype=np.int64)
        depth_stack = np.empty(max_nodes, dtype=np.int64)
        row_stack = np.empty(max_nodes, dtype=np.int64)
        col_stack = np.empty(max_nodes, dtype=np.int64)

        sp = 0
        r0_stack[sp] = 0
        c0_stack[sp] = 0
        size_stack[sp] = integ.shape[0]
        depth_stack[sp] = 0
        row_stack[sp] = 0
        col_stack[sp] = 0
        sp += 1

        while sp > 0:
            sp -= 1
            r0 = r0_stack[sp]
            c0 = c0_stack[sp]
            size = size_stack[sp]
            depth = depth_stack[sp]
            row = row_stack[sp]
            col = col_stack[sp]

            count = _sum_region_nb(integ, r0, c0, size)
            if depth >= max_depth or count < min_count or count <= 0.0:
                scale = 4 ** (max_depth - depth)
                val = (count / total) / scale if total > 0 else 0.0
                start = _morton_prefix_nb(row, col, depth) * scale
                for i in range(start, start + scale):
                    out[i] = val
                continue

            half = size // 2
            q00 = _sum_region_nb(integ, r0, c0, half)
            q01 = _sum_region_nb(integ, r0, c0 + half, half)
            q10 = _sum_region_nb(integ, r0 + half, c0, half)
            q11 = _sum_region_nb(integ, r0 + half, c0 + half, half)

            expected = count / 4.0
            if expected <= 0:
                scale = 4 ** (max_depth - depth)
                val = (count / total) / scale if total > 0 else 0.0
                start = _morton_prefix_nb(row, col, depth) * scale
                for i in range(start, start + scale):
                    out[i] = val
                continue

            chi2 = ((q00 - expected) ** 2 + (q01 - expected) ** 2 +
                    (q10 - expected) ** 2 + (q11 - expected) ** 2) / expected
            p_val = _chi2_pvalue_df3_nb(chi2)

            if p_val >= p_threshold:
                scale = 4 ** (max_depth - depth)
                val = (count / total) / scale if total > 0 else 0.0
                start = _morton_prefix_nb(row, col, depth) * scale
                for i in range(start, start + scale):
                    out[i] = val
                continue

            # Push children (reverse order not important)
            r0_stack[sp] = r0
            c0_stack[sp] = c0
            size_stack[sp] = half
            depth_stack[sp] = depth + 1
            row_stack[sp] = row * 2
            col_stack[sp] = col * 2
            sp += 1

            r0_stack[sp] = r0
            c0_stack[sp] = c0 + half
            size_stack[sp] = half
            depth_stack[sp] = depth + 1
            row_stack[sp] = row * 2
            col_stack[sp] = col * 2 + 1
            sp += 1

            r0_stack[sp] = r0 + half
            c0_stack[sp] = c0
            size_stack[sp] = half
            depth_stack[sp] = depth + 1
            row_stack[sp] = row * 2 + 1
            col_stack[sp] = col * 2
            sp += 1

            r0_stack[sp] = r0 + half
            c0_stack[sp] = c0 + half
            size_stack[sp] = half
            depth_stack[sp] = depth + 1
            row_stack[sp] = row * 2 + 1
            col_stack[sp] = col * 2 + 1
            sp += 1

        return out
