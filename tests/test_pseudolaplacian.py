"""Tests for pseudo-Laplacian computation via accumulate_pseudo_laplacians."""

import numpy as np

from pulsar._pulsar import (
    BallMapper,
    accumulate_pseudo_laplacians,
    accumulate_pseudo_laplacians_sparse,
)
from tests.conftest import pseudo_laplacian_py


def _spl_to_dense(spl, n):
    """Reconstruct the dense pseudo-Laplacian from a SparsePseudoLaplacian."""
    L = np.zeros((n, n), dtype=np.int64)
    diag = np.asarray(spl.diag)
    for i in range(n):
        L[i, i] = diag[i]
    for i, j, count in spl.offdiag:
        L[i, j] -= count
        L[j, i] -= count
    return L


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


def test_sparse_accumulate_matches_dense():
    """accumulate_pseudo_laplacians_sparse must reconstruct the exact dense matrix."""
    rng = np.random.default_rng(7)
    pts = rng.standard_normal((80, 4)).astype(np.float64)
    n = len(pts)

    ball_maps = []
    for eps in [0.4, 0.7, 1.1, 1.6]:
        bm = BallMapper(eps=eps)
        bm.fit(pts)
        ball_maps.append(bm)

    dense_L = np.array(accumulate_pseudo_laplacians(ball_maps, n))
    spl = accumulate_pseudo_laplacians_sparse(ball_maps, n)

    np.testing.assert_array_equal(_spl_to_dense(spl, n), dense_L)
    # Off-diagonal entries are the upper triangle only, sorted, deduped.
    offdiag = spl.offdiag
    assert all(i < j for i, j, _ in offdiag)
    assert offdiag == sorted(offdiag, key=lambda e: (e[0], e[1]))


def test_sparse_merge_in_place_matches_concatenated_accumulate():
    """merge_in_place over batches equals accumulating all ball maps at once."""
    rng = np.random.default_rng(11)
    pts = rng.standard_normal((60, 3)).astype(np.float64)
    n = len(pts)

    def bms(epsilons):
        out = []
        for eps in epsilons:
            bm = BallMapper(eps=eps)
            bm.fit(pts)
            out.append(bm)
        return out

    batch_a = bms([0.5, 0.9])
    batch_b = bms([1.3])

    full = accumulate_pseudo_laplacians_sparse(batch_a + batch_b, n)

    acc = accumulate_pseudo_laplacians_sparse(batch_a, n)
    acc.merge_in_place(accumulate_pseudo_laplacians_sparse(batch_b, n))

    np.testing.assert_array_equal(_spl_to_dense(acc, n), _spl_to_dense(full, n))
