"""
Correctness tests for pseudo-Laplacian via accumulate_pseudo_laplacians.

The Python reference is the same formula used by the Rust implementation:
    L = zeros(n, n)
    for members in nodes:
        for i in members:
            for j in members:
                L[i, j] += 1 if i == j else -1
"""

import numpy as np

from pulsar._pulsar import BallMapper, accumulate_pseudo_laplacians
from tests.conftest import pseudo_laplacian_py


def test_single_ballmap_matches_reference():
    """Single BallMapper accumulated matches Python reference."""
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((20, 2)).astype(np.float64)
    n = len(pts)

    bm = BallMapper(eps=1.0)
    bm.fit(pts)

    rust_L = np.array(accumulate_pseudo_laplacians([bm], n))
    py_L = pseudo_laplacian_py(bm.nodes, n)

    np.testing.assert_array_equal(rust_L, py_L)


def test_multiple_ballmaps_matches_reference():
    """Multiple BallMappers accumulated matches summed Python reference."""
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((30, 3)).astype(np.float64)
    n = len(pts)

    ball_maps = []
    for eps in [0.5, 1.0, 1.5, 2.0]:
        bm = BallMapper(eps=eps)
        bm.fit(pts)
        ball_maps.append(bm)

    rust_L = np.array(accumulate_pseudo_laplacians(ball_maps, n))

    py_L = np.zeros((n, n), dtype=np.int64)
    for bm in ball_maps:
        py_L += pseudo_laplacian_py(bm.nodes, n)

    np.testing.assert_array_equal(rust_L, py_L)


def test_symmetry():
    """Accumulated pseudo-Laplacian is symmetric."""
    rng = np.random.default_rng(123)
    pts = rng.standard_normal((25, 2)).astype(np.float64)
    n = len(pts)

    ball_maps = []
    for eps in [0.8, 1.2]:
        bm = BallMapper(eps=eps)
        bm.fit(pts)
        ball_maps.append(bm)

    L = np.array(accumulate_pseudo_laplacians(ball_maps, n))
    np.testing.assert_array_equal(L, L.T)


def test_diagonal_is_membership_count():
    """Diagonal equals total membership count across all ball maps."""
    rng = np.random.default_rng(7)
    pts = rng.standard_normal((15, 2)).astype(np.float64)
    n = len(pts)

    ball_maps = []
    for eps in [0.5, 1.0]:
        bm = BallMapper(eps=eps)
        bm.fit(pts)
        ball_maps.append(bm)

    L = np.array(accumulate_pseudo_laplacians(ball_maps, n))

    # Count memberships manually
    for i in range(n):
        expected_count = sum(1 for bm in ball_maps for node in bm.nodes if i in node)
        assert L[i, i] == expected_count


def test_empty_ballmaps_zero_matrix():
    """Empty ball map list produces zero matrix."""
    n = 10
    L = np.array(accumulate_pseudo_laplacians([], n))
    np.testing.assert_array_equal(L, np.zeros((n, n), dtype=np.int64))
