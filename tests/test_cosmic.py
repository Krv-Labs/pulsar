"""Tests for CosmicGraph construction."""

import numpy as np

from pulsar._pulsar import CosmicGraph, BallMapper, accumulate_pseudo_laplacians
from pulsar.analysis import cosmic_to_networkx
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


def test_weighted_edges_for_dense_graph():
    L = make_simple_laplacian()
    cg = CosmicGraph.from_pseudo_laplacian(L, threshold=0.0)
    edges = cg.weighted_edges()
    wadj = np.array(cg.weighted_adj)

    assert len(edges) == cg.n_edges
    for i, j, w in edges:
        assert i < j
        assert w == wadj[i, j]
        assert w > 0.0


def test_spectral_sparsify_deterministic_and_symmetric():
    nodes = [list(range(18))]
    L = pseudo_laplacian_py(nodes, 18)
    cg = CosmicGraph.from_pseudo_laplacian(L, threshold=0.0)

    sp1 = cg.spectral_sparsify(1.5, seed=123, sketch_dim=8, sample_count=40)
    sp2 = cg.spectral_sparsify(1.5, seed=123, sketch_dim=8, sample_count=40)

    assert sp1.n == cg.n
    assert sp1.n_edges <= cg.n_edges
    assert sp1.weighted_edges() == sp2.weighted_edges()
    wadj = np.array(sp1.weighted_adj)
    np.testing.assert_allclose(wadj, wadj.T, atol=1e-12)
    np.testing.assert_array_equal(np.diag(wadj), 0.0)


def test_spectral_sparsify_quadratic_forms_relaxed():
    nodes = [list(range(14))]
    L = pseudo_laplacian_py(nodes, 14)
    cg = CosmicGraph.from_pseudo_laplacian(L, threshold=0.0)
    sp = cg.spectral_sparsify(1.2, seed=5, sketch_dim=10, sample_count=80)

    w_full = np.array(cg.weighted_adj)
    w_sparse = np.array(sp.weighted_adj)
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.standard_normal(14)
        q_full = sum(
            w_full[i, j] * (x[i] - x[j]) ** 2
            for i in range(14)
            for j in range(i + 1, 14)
        )
        q_sparse = sum(
            w_sparse[i, j] * (x[i] - x[j]) ** 2
            for i in range(14)
            for j in range(i + 1, 14)
        )
        assert q_sparse > 0
        assert 0.15 * q_full <= q_sparse <= 3.5 * q_full


def test_spectral_sparsify_disconnected_graph_has_no_cross_edges():
    L = pseudo_laplacian_py([list(range(5)), list(range(5, 10))], 10)
    cg = CosmicGraph.from_pseudo_laplacian(L, threshold=0.0)
    sp = cg.spectral_sparsify(1.0, seed=9, sketch_dim=6, sample_count=30)

    for i, j, _ in sp.weighted_edges():
        assert (i < 5 and j < 5) or (i >= 5 and j >= 5)


def test_cosmic_to_networkx_uses_weighted_edges():
    L = make_simple_laplacian()
    cg = CosmicGraph.from_pseudo_laplacian(L, threshold=0.0)
    graph = cosmic_to_networkx(cg)

    assert graph.number_of_edges() == cg.n_edges
    for i, j, w in cg.weighted_edges():
        assert graph[i][j]["weight"] == w
