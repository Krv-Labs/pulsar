"""Tests for CosmicGraph construction."""
import numpy as np
import pytest

from pulsar._pulsar import CosmicGraph, BallMapper, accumulate_pseudo_laplacians
from tests.conftest import pseudo_laplacian_py


def make_simple_laplacian():
    """Build a small pseudo-Laplacian from known nodes."""
    nodes = [[0, 1, 2], [1, 2, 3], [3, 4]]
    n = 5
    return pseudo_laplacian_py(nodes, n)


def test_output_shapes():
    L = make_simple_laplacian()
    cg = CosmicGraph.from_pseudo_laplacian(L, threshold=0.0)
    n = L.shape[0]
    assert np.array(cg.weighted_adj).shape == (n, n)
    assert np.array(cg.adj).shape == (n, n)
    assert cg.n == n


def test_zero_diagonal():
    L = make_simple_laplacian()
    cg = CosmicGraph.from_pseudo_laplacian(L, threshold=0.0)
    adj = np.array(cg.adj)
    np.testing.assert_array_equal(np.diag(adj), 0)


def test_zero_diagonal_weighted():
    L = make_simple_laplacian()
    cg = CosmicGraph.from_pseudo_laplacian(L, threshold=0.0)
    wadj = np.array(cg.weighted_adj)
    np.testing.assert_array_equal(np.diag(wadj), 0.0)


def test_weights_in_range():
    L = make_simple_laplacian()
    cg = CosmicGraph.from_pseudo_laplacian(L, threshold=0.0)
    wadj = np.array(cg.weighted_adj)
    assert np.all(wadj >= 0.0)
    assert np.all(wadj <= 1.0 + 1e-10)


def test_threshold_zero_gives_binary_adj():
    L = make_simple_laplacian()
    cg = CosmicGraph.from_pseudo_laplacian(L, threshold=0.0)
    adj = np.array(cg.adj)
    wadj = np.array(cg.weighted_adj)
    np.testing.assert_array_equal((wadj > 0).astype(np.uint8), adj)


def test_high_threshold_empty_adj():
    L = make_simple_laplacian()
    cg = CosmicGraph.from_pseudo_laplacian(L, threshold=2.0)
    adj = np.array(cg.adj)
    np.testing.assert_array_equal(adj, 0)


def test_symmetry():
    L = make_simple_laplacian()
    cg = CosmicGraph.from_pseudo_laplacian(L, threshold=0.0)
    wadj = np.array(cg.weighted_adj)
    np.testing.assert_allclose(wadj, wadj.T, atol=1e-12)


def test_with_accumulated_laplacian():
    """Test CosmicGraph with Rust-accumulated Laplacian."""
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((30, 2)).astype(np.float64)
    n = len(pts)

    ball_maps = []
    for eps in [0.5, 1.0, 1.5]:
        bm = BallMapper(eps=eps)
        bm.fit(pts)
        ball_maps.append(bm)

    galactic_L = np.array(accumulate_pseudo_laplacians(ball_maps, n))
    cg = CosmicGraph.from_pseudo_laplacian(galactic_L, threshold=0.1)

    wadj = np.array(cg.weighted_adj)
    assert wadj.shape == (n, n)
    assert np.all(wadj >= 0)
    assert np.all(wadj <= 1.0 + 1e-10)
