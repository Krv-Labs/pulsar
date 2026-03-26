import numpy as np
import pytest
import sys
import os

from pulsar._pulsar import CosmicGraph, pseudo_laplacian


def make_simple_laplacian():
    """Build a small pseudo-Laplacian from known nodes."""
    nodes = [[0, 1, 2], [1, 2, 3], [3, 4]]
    n = 5
    L = np.array(pseudo_laplacian(nodes, n))
    return L


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
    # Every position with weight > 0 should have adj = 1
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


def test_matches_thema_reference():
    """Compare against Thema's normalize_cosmicGraph."""
    thema_path = os.path.join(os.path.dirname(__file__), "..", "Thema")
    if not os.path.isdir(thema_path):
        pytest.skip("Thema submodule not available")

    sys.path.insert(0, thema_path)
    try:
        from thema.multiverse.universe.utils.starHelpers import normalize_cosmicGraph
    except ImportError:
        pytest.skip("Cannot import Thema starHelpers")

    L = make_simple_laplacian()
    threshold = 0.0

    cg = CosmicGraph.from_pseudo_laplacian(L, threshold=threshold)
    pulsar_wadj = np.array(cg.weighted_adj)

    _, thema_wadj, _ = normalize_cosmicGraph(L, threshold=threshold)

    np.testing.assert_allclose(pulsar_wadj, thema_wadj, atol=1e-12)
