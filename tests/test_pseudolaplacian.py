"""Tests for pseudo-Laplacian computation via accumulate_pseudo_laplacians."""
import numpy as np
import pytest

from pulsar._pulsar import BallMapper, accumulate_pseudo_laplacians
from tests.conftest import pseudo_laplacian_py


def test_diagonal_equals_membership_count():
    n = 5
    nodes = [[0, 1, 2], [0, 3, 4]]
    L = pseudo_laplacian_py(nodes, n)

    assert L[0, 0] == 2  # Point 0 in 2 balls
    assert L[1, 1] == 1
    assert L[2, 2] == 1
    assert L[3, 3] == 1
    assert L[4, 4] == 1


def test_offdiag_negative_shared_memberships():
    n = 5
    nodes = [[0, 1, 2], [0, 3, 4]]
    L = pseudo_laplacian_py(nodes, n)

    assert L[1, 2] == -1
    assert L[2, 1] == -1
    assert L[0, 1] == -1


def test_no_cross_ball_negative():
    n = 4
    nodes = [[0, 1], [2, 3]]
    L = pseudo_laplacian_py(nodes, n)
    assert L[0, 2] == 0
    assert L[1, 3] == 0


def test_symmetry():
    n = 6
    nodes = [[0, 1, 2], [1, 2, 3], [3, 4, 5]]
    L = pseudo_laplacian_py(nodes, n)
    np.testing.assert_array_equal(L, L.T)


def test_accumulate_matches_python():
    """Test that accumulate_pseudo_laplacians matches Python reference."""
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((50, 3)).astype(np.float64)
    n = len(pts)

    # Build multiple ball mappers
    ball_maps = []
    for eps in [0.5, 1.0, 1.5]:
        bm = BallMapper(eps=eps)
        bm.fit(pts)
        ball_maps.append(bm)

    # Rust accumulation
    rust_L = np.array(accumulate_pseudo_laplacians(ball_maps, n))

    # Python reference
    py_L = np.zeros((n, n), dtype=np.int64)
    for bm in ball_maps:
        py_L += pseudo_laplacian_py(bm.nodes, n)

    np.testing.assert_array_equal(rust_L, py_L)


def test_empty_nodes_zero_matrix():
    n = 4
    L = pseudo_laplacian_py([], n)
    np.testing.assert_array_equal(L, np.zeros((n, n), dtype=np.int64))
