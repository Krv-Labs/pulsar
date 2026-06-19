"""Spectral clustering robustness on disconnected affinity graphs.

Verifies that ``_cluster_spectral`` no longer hard-fails when the affinity
graph has natural outliers/singletons: it clusters the giant connected
component and isolates the residual instead of raising.
"""

import numpy as np
import pytest

from pulsar.mcp.interpreter import SpectralClusterCutError, _cluster_spectral


def _two_cluster_block(n_per: int = 8, intra: float = 0.9) -> np.ndarray:
    """Build a giant component: two tight blocks with a weak inter-block bridge."""
    n = 2 * n_per
    adj = np.zeros((n, n), dtype=float)
    # Dense intra-block affinity.
    for block_start in (0, n_per):
        for i in range(block_start, block_start + n_per):
            for j in range(block_start, block_start + n_per):
                if i != j:
                    adj[i, j] = intra
    # A single weak bridge so the two blocks form one connected component.
    bridge = 0.05
    adj[n_per - 1, n_per] = bridge
    adj[n_per, n_per - 1] = bridge
    return adj


def _embed_with_singletons(giant: np.ndarray, n_singletons: int) -> np.ndarray:
    """Append disconnected singleton nodes (no edges) to a giant component."""
    g = giant.shape[0]
    n = g + n_singletons
    adj = np.zeros((n, n), dtype=float)
    adj[:g, :g] = giant
    return adj


def test_spectral_disconnected_labels_all_nodes_without_raising():
    """Two-cluster giant + singletons: every node gets a label, no raise."""
    giant = _two_cluster_block(n_per=8)
    n_singletons = 2
    adj = _embed_with_singletons(giant, n_singletons)
    n = adj.shape[0]

    result = _cluster_spectral(adj, n, max_k=6)

    # Every node receives a label.
    assert len(result.labels) == n
    assert not result.labels.isna().any()
    assert result.method_used == "spectral"

    # The giant component (first 16 nodes) is split into >= 2 spectral clusters.
    giant_labels = result.labels.to_numpy()[: giant.shape[0]]
    assert len(np.unique(giant_labels)) >= 2

    # The two singletons land in their own clusters, distinct from each other
    # and from every giant-component label.
    singleton_labels = result.labels.to_numpy()[giant.shape[0] :]
    assert len(np.unique(singleton_labels)) == n_singletons
    assert set(singleton_labels).isdisjoint(set(giant_labels))

    # Residual is surfaced in failure_reason rather than blocking the call.
    assert result.failure_reason is not None
    assert "residual" in result.failure_reason.lower()


def test_spectral_giant_component_split_is_sensible():
    """Spectral cut on the giant component recovers the two planted blocks."""
    giant = _two_cluster_block(n_per=8)
    adj = _embed_with_singletons(giant, 1)
    result = _cluster_spectral(adj, adj.shape[0], max_k=6)

    labels = result.labels.to_numpy()
    block_a = labels[:8]
    block_b = labels[8:16]
    # Each planted block should be internally homogeneous (single label) and
    # the two blocks should differ from each other.
    assert len(np.unique(block_a)) == 1
    assert len(np.unique(block_b)) == 1
    assert block_a[0] != block_b[0]


def test_spectral_connected_graph_unchanged_behavior():
    """A fully connected affinity graph takes the original single-pass path."""
    giant = _two_cluster_block(n_per=8)  # connected (single component)
    n = giant.shape[0]
    result = _cluster_spectral(giant, n, max_k=6)

    assert len(result.labels) == n
    assert result.method_used == "spectral"
    # Connected path leaves failure_reason unset (no residual to report).
    assert result.failure_reason is None
    assert result.n_clusters >= 2


def test_spectral_all_singletons_finds_no_cut():
    """A graph with no giant structure cannot produce a stable cut."""
    adj = np.zeros((5, 5), dtype=float)  # 5 isolated nodes
    with pytest.raises(SpectralClusterCutError, match="No stable spectral cut") as exc:
        _cluster_spectral(adj, 5, max_k=4)

    diagnostics = exc.value.diagnostics
    assert diagnostics["affinity_component_count"] == 5
    assert diagnostics["giant_component_size"] == 1
    assert diagnostics["residual_node_count"] == 4
    assert diagnostics["k_min"] == 2
    assert diagnostics["max_k"] == 4
    assert "candidate_scores" in diagnostics
