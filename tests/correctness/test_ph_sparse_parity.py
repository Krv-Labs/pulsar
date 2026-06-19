"""Parity tests for the sparse threshold-selection path (ph.rs).

``find_stable_thresholds_sparse(n, edges)`` must produce results identical to the
dense ``find_stable_thresholds(weighted_adj)`` on the same graph. This is the most
important guarantee in the sparse refactor: it gates the pipeline's auto-threshold.
"""

import numpy as np

from pulsar._pulsar import (
    BallMapper,
    CosmicGraph,
    accumulate_pseudo_laplacians,
    find_stable_thresholds,
    find_stable_thresholds_sparse,
)


def _dense_to_edges(W):
    """Upper-triangle (i<j) positive-weight edges from a dense adjacency."""
    n = W.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            w = W[i, j]
            if w > 0.0:
                edges.append((i, j, float(w)))
    return edges


def _assert_results_equal(dense, sparse):
    assert dense.optimal_threshold == sparse.optimal_threshold
    np.testing.assert_array_equal(
        np.asarray(dense.thresholds), np.asarray(sparse.thresholds)
    )
    np.testing.assert_array_equal(
        np.asarray(dense.component_counts), np.asarray(sparse.component_counts)
    )


def test_sparse_threshold_matches_dense_on_random_graph():
    rng = np.random.default_rng(5)
    n = 60
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.15:
                w = float(rng.random())
                W[i, j] = W[j, i] = w

    dense = find_stable_thresholds(W)
    sparse = find_stable_thresholds_sparse(n, _dense_to_edges(W))
    _assert_results_equal(dense, sparse)


def test_sparse_threshold_matches_dense_end_to_end():
    """From a real ball-mapper cosmic graph, the normalized edge list and the dense
    normalized adjacency must yield the identical stability result."""
    rng = np.random.default_rng(9)
    pts = rng.standard_normal((90, 3)).astype(np.float64)
    n = len(pts)
    ball_maps = []
    for eps in [0.6, 1.0, 1.5]:
        bm = BallMapper(eps=eps)
        bm.fit(pts)
        ball_maps.append(bm)

    L = np.array(accumulate_pseudo_laplacians(ball_maps, n))
    cg = CosmicGraph.from_pseudo_laplacian(L, 0.0)
    W = np.array(cg.weighted_adj)
    scale = max(1.0, float(W.max()))
    W_norm = W / scale

    dense = find_stable_thresholds(W_norm)
    norm_edges = [(i, j, w / scale) for i, j, w in cg.weighted_edges()]
    sparse = find_stable_thresholds_sparse(n, norm_edges)
    _assert_results_equal(dense, sparse)


def test_sparse_threshold_degenerate_cases():
    # No edges -> all singletons, optimal 0.5 (mirrors dense path).
    res = find_stable_thresholds_sparse(4, [])
    assert res.optimal_threshold == 0.5
    # n == 1
    res1 = find_stable_thresholds_sparse(1, [])
    assert res1.optimal_threshold == 0.5
    # n == 0
    res0 = find_stable_thresholds_sparse(0, [])
    assert res0.optimal_threshold == 0.5


def test_sparse_threshold_rejects_out_of_range_edge():
    import pytest

    with pytest.raises(Exception):
        find_stable_thresholds_sparse(3, [(0, 5, 0.4)])
