"""
Tests for pulsar.hooks module.

Tests the cosmic_to_networkx function to ensure correctness of the
vectorized weight assignment (np.where) vs the old loop approach.
"""

import numpy as np
import pytest

from pulsar.hooks import cosmic_to_networkx
from pulsar._pulsar import CosmicGraph


def make_simple_laplacian(n: int = 10, density: float = 0.3, seed: int = 42) -> np.ndarray:
    """Create a simple symmetric Laplacian-like matrix for testing (int64 format)."""
    rng = np.random.default_rng(seed)
    # Build a sparse symmetric matrix
    L = rng.random((n, n)) < density
    L = L | L.T  # Make symmetric
    L = L.astype(np.int64)
    # Add self-loops (diagonal) to make it look like a pseudo-Laplacian
    for i in range(n):
        L[i, i] = np.sum(np.abs(L[i, :]))  # Self-loop = sum of off-diagonal magnitudes
    # Return as C-contiguous int64 (CosmicGraph.from_pseudo_laplacian requires int64)
    return np.ascontiguousarray(L, dtype=np.int64)


def test_weight_parity():
    """Assert every edge in the result has correct weight from weighted_adj."""
    L = make_simple_laplacian(n=15, density=0.25, seed=42)
    cg = CosmicGraph.from_pseudo_laplacian(L, threshold=0.0)
    g = cosmic_to_networkx(cg)

    wadj = np.array(cg.weighted_adj, dtype=np.float64)
    for u, v in g.edges():
        # Check that weight matches the weighted adjacency
        assert "weight" in g[u][v], f"Edge ({u}, {v}) missing weight attribute"
        expected_weight = wadj[u, v]
        actual_weight = g[u][v]["weight"]
        assert np.isclose(actual_weight, expected_weight, rtol=1e-6), (
            f"Weight mismatch for edge ({u}, {v}): "
            f"expected {expected_weight}, got {actual_weight}"
        )


def test_threshold_mask_respected():
    """Assert g.number_of_edges() matches nonzero count of binary adj (÷2 for undirected)."""
    L = make_simple_laplacian(n=20, density=0.3, seed=123)
    # Use a non-zero threshold to create a sparse binary adjacency
    cg = CosmicGraph.from_pseudo_laplacian(L, threshold=0.5)
    g = cosmic_to_networkx(cg)

    adj = np.array(cg.adj, dtype=bool)
    expected_edge_count = np.sum(adj) // 2  # Undirected: count upper triangle only
    actual_edge_count = g.number_of_edges()

    assert actual_edge_count == expected_edge_count, (
        f"Edge count mismatch: expected {expected_edge_count}, got {actual_edge_count}. "
        f"This suggests the threshold mask is not being respected."
    )


def test_no_zero_weight_edges():
    """Assert all edges have positive weight (no zero-weight or missing weights)."""
    L = make_simple_laplacian(n=12, density=0.35, seed=99)
    cg = CosmicGraph.from_pseudo_laplacian(L, threshold=0.0)
    g = cosmic_to_networkx(cg)

    for u, v, data in g.edges(data=True):
        weight = data.get("weight", None)
        assert weight is not None, f"Edge ({u}, {v}) missing weight attribute"
        assert weight > 0, f"Edge ({u}, {v}) has zero or negative weight: {weight}"


def test_high_threshold_produces_sparse_graph():
    """Test that high threshold produces fewer edges (extreme case of mask respect)."""
    L = make_simple_laplacian(n=15, density=0.4, seed=42)

    # Two graphs: one with low threshold (many edges), one with high threshold (few edges)
    cg_low = CosmicGraph.from_pseudo_laplacian(L, threshold=0.1)
    g_low = cosmic_to_networkx(cg_low)

    cg_high = CosmicGraph.from_pseudo_laplacian(L, threshold=0.8)
    g_high = cosmic_to_networkx(cg_high)

    # High threshold should have fewer or equal edges
    assert g_high.number_of_edges() <= g_low.number_of_edges(), (
        f"High threshold produced more edges ({g_high.number_of_edges()}) "
        f"than low threshold ({g_low.number_of_edges()})"
    )


def test_symmetry_preserved():
    """Assert the resulting graph is symmetric (undirected)."""
    L = make_simple_laplacian(n=10, density=0.3, seed=55)
    cg = CosmicGraph.from_pseudo_laplacian(L, threshold=0.2)
    g = cosmic_to_networkx(cg)

    # For an undirected graph, if (u, v) is an edge, (v, u) is the same edge
    # Check by verifying adjacency is symmetric
    for u, v in g.edges():
        assert g.has_edge(v, u), f"Graph not symmetric: edge ({u}, {v}) exists but ({v}, {u}) does not"
        # Weights should also match (for symmetric matrix)
        w_uv = g[u][v].get("weight", None)
        w_vu = g[v][u].get("weight", None)
        assert w_uv == w_vu, (
            f"Weight asymmetry for edge ({u}, {v}): "
            f"({u}, {v}) has {w_uv}, ({v}, {u}) has {w_vu}"
        )


def test_all_zero_threshold_includes_all_nonzero_weights():
    """At threshold=0.0, the graph should include all nonzero weighted entries."""
    L = make_simple_laplacian(n=8, density=0.25, seed=42)
    cg = CosmicGraph.from_pseudo_laplacian(L, threshold=0.0)
    g = cosmic_to_networkx(cg)

    wadj = np.array(cg.weighted_adj, dtype=np.float64)
    # At threshold 0.0, adj should be (wadj > 0)
    expected_edges = np.sum(np.triu(wadj > 0, k=1))
    actual_edges = g.number_of_edges()

    assert actual_edges == expected_edges, (
        f"At threshold=0.0, edge count should be {expected_edges}, got {actual_edges}"
    )
