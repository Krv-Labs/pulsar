"""
Correctness tests for cosmic.rs — CosmicGraph.

The Python reference implements the weight formula from Thema's
normalize_cosmicGraph:

    denom = L[i,i] + L[j,j] + L[i,j]
    W[i,j] = -L[i,j] / denom    if denom > 0
           = 0                   otherwise

    adj[i,j] = 1 if W[i,j] > threshold else 0
    adj[i,i] = 0  (diagonal is always zero)

Because this is pure arithmetic on integer inputs, the floating-point results
are deterministic and Rust should match Python to machine precision (atol=1e-12).
"""
import numpy as np
import pytest

from pulsar._pulsar import CosmicGraph, pseudo_laplacian


# ---------------------------------------------------------------------------
# Python reference implementation
# ---------------------------------------------------------------------------

def py_cosmic(L, threshold):
    """Cosmic Graph weight formula — direct translation of the Rust loop.

    Parameters
    ----------
    L : np.ndarray[int64, 2D], shape (n, n)  — pseudo-Laplacian
    threshold : float  — minimum weight for a binary edge

    Returns
    -------
    W   : np.ndarray[float64, 2D] — weighted adjacency matrix
    adj : np.ndarray[uint8, 2D]  — binary adjacency matrix
    """
    n = L.shape[0]
    W = np.zeros((n, n), dtype=np.float64)
    adj = np.zeros((n, n), dtype=np.uint8)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            denom = int(L[i, i]) + int(L[j, j]) + int(L[i, j])
            if denom > 0:
                W[i, j] = -int(L[i, j]) / denom
            if W[i, j] > threshold:
                adj[i, j] = 1

    return W, adj


def make_laplacian():
    """Build a small pseudo-Laplacian from known nodes."""
    nodes = [[0, 1, 2], [1, 2, 3], [3, 4]]
    n = 5
    return np.array(pseudo_laplacian(nodes, n))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_weighted_adj_matches_reference():
    """Rust weighted adjacency must match the Python reference formula."""
    L = make_laplacian()
    threshold = 0.0
    expected_W, _ = py_cosmic(L, threshold)

    cg = CosmicGraph.from_pseudo_laplacian(L, threshold)
    actual_W = np.array(cg.weighted_adj)

    np.testing.assert_allclose(actual_W, expected_W, atol=1e-12,
                               err_msg="weighted_adj: Rust differs from Python reference")


def test_adj_matches_reference():
    """Binary adjacency must match the Python reference threshold rule."""
    L = make_laplacian()
    threshold = 0.0
    _, expected_adj = py_cosmic(L, threshold)

    cg = CosmicGraph.from_pseudo_laplacian(L, threshold)
    actual_adj = np.array(cg.adj)

    np.testing.assert_array_equal(actual_adj, expected_adj,
                                  err_msg="adj: Rust differs from Python reference")


def test_threshold_filtering_matches_reference():
    """Higher threshold → fewer edges; must match reference for several thresholds."""
    L = make_laplacian()
    for threshold in [0.0, 0.2, 0.5, 0.8, 1.0]:
        expected_W, expected_adj = py_cosmic(L, threshold)

        cg = CosmicGraph.from_pseudo_laplacian(L, threshold)
        actual_W = np.array(cg.weighted_adj)
        actual_adj = np.array(cg.adj)

        np.testing.assert_allclose(actual_W, expected_W, atol=1e-12,
                                   err_msg=f"weighted_adj mismatch at threshold={threshold}")
        np.testing.assert_array_equal(actual_adj, expected_adj,
                                      err_msg=f"adj mismatch at threshold={threshold}")


def test_diagonal_is_zero():
    """Diagonal must always be zero (no self-loops)."""
    L = make_laplacian()
    cg = CosmicGraph.from_pseudo_laplacian(L, threshold=0.0)
    W = np.array(cg.weighted_adj)
    adj = np.array(cg.adj)

    np.testing.assert_array_equal(np.diag(W), 0.0)
    np.testing.assert_array_equal(np.diag(adj), 0)


def test_weights_in_unit_interval():
    """All weights must be in [0, 1]."""
    L = make_laplacian()
    cg = CosmicGraph.from_pseudo_laplacian(L, threshold=0.0)
    W = np.array(cg.weighted_adj)

    assert np.all(W >= 0.0), f"Negative weight found: {W.min()}"
    assert np.all(W <= 1.0 + 1e-12), f"Weight > 1 found: {W.max()}"


def test_symmetry():
    """Weighted adjacency must be symmetric: W[i,j] == W[j,i]."""
    L = make_laplacian()
    cg = CosmicGraph.from_pseudo_laplacian(L, threshold=0.0)
    W = np.array(cg.weighted_adj)
    np.testing.assert_allclose(W, W.T, atol=1e-12,
                               err_msg="weighted_adj must be symmetric")


def test_zero_laplacian_gives_zero_adj():
    """A zero Laplacian (no balls, no connectivity) produces a zero adjacency."""
    n = 5
    L = np.zeros((n, n), dtype=np.int64)
    cg = CosmicGraph.from_pseudo_laplacian(L, threshold=0.0)
    W = np.array(cg.weighted_adj)
    adj = np.array(cg.adj)

    np.testing.assert_array_equal(W, 0.0)
    np.testing.assert_array_equal(adj, 0)


def test_matches_reference_on_accumulated_laplacian():
    """End-to-end: accumulate pseudo-Laplacians from multiple BallMappers,
    build CosmicGraph, and check against the Python reference."""
    from pulsar._pulsar import BallMapper

    rng = np.random.default_rng(0)
    pts = rng.standard_normal((30, 2)).astype(np.float64)
    n = len(pts)

    # Build 3 BallMappers and accumulate
    galactic_L = np.zeros((n, n), dtype=np.int64)
    for eps in [0.5, 1.0, 1.5]:
        bm = BallMapper(eps=eps)
        bm.fit(pts)
        galactic_L += np.array(pseudo_laplacian(bm.nodes, n))

    threshold = 0.1
    expected_W, expected_adj = py_cosmic(galactic_L, threshold)

    cg = CosmicGraph.from_pseudo_laplacian(galactic_L, threshold)
    np.testing.assert_allclose(np.array(cg.weighted_adj), expected_W, atol=1e-12)
    np.testing.assert_array_equal(np.array(cg.adj), expected_adj)
