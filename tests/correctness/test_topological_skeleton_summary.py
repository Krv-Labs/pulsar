"""
Correctness tests for the precomputed topological graph summary and high-SNR skeleton.
"""

import networkx as nx
import numpy as np
import pytest

from pulsar.mcp.diagnostics import _build_graph_summary, _skeleton_graph_payload


class MockModel:
    """Mock model with a cosmic_graph and resolved_construction_threshold."""

    def __init__(self, graph: nx.Graph, threshold: float = 0.5):
        self.cosmic_graph = graph
        self.resolved_construction_threshold = threshold
        # Build dummy weighted adjacency matrix
        n = graph.number_of_nodes()
        self.weighted_adjacency = np.zeros((n, n), dtype=float)
        for u, v, data in graph.edges(data=True):
            w = data.get("weight", 1.0)
            self.weighted_adjacency[int(u), int(v)] = w
            self.weighted_adjacency[int(v), int(u)] = w


@pytest.fixture
def bridge_and_clique_graph() -> nx.Graph:
    """
    Builds a single connected component with two dense regions (cliques)
    connected via a bridge structure:
    Clique 1: [0, 1, 2, 3] (fully connected)
    Bridge links: 3-4 and 4-5
    Clique 2: [5, 6, 7, 8] (fully connected)

    In this graph:
    - Nodes 3, 4, 5 are articulation points (bridges).
    - Nodes 3 and 5 are also major hubs.
    """
    g = nx.Graph()
    # Add Clique 1
    clique1 = [0, 1, 2, 3]
    for i in range(len(clique1)):
        for j in range(i + 1, len(clique1)):
            g.add_edge(clique1[i], clique1[j], weight=1.0)

    # Add Clique 2
    clique2 = [5, 6, 7, 8]
    for i in range(len(clique2)):
        for j in range(i + 1, len(clique2)):
            g.add_edge(clique2[i], clique2[j], weight=1.0)

    # Add Bridge path
    g.add_edge(3, 4, weight=0.5)
    g.add_edge(4, 5, weight=0.5)

    return g


def test_topological_summary_precomputation(bridge_and_clique_graph):
    """
    Assert that the precomputed topological summary accurately detects
    articulation points, key hubs, and degree distributions.
    """
    model = MockModel(bridge_and_clique_graph)
    summary = _build_graph_summary(model)

    assert "topological_summary" in summary
    topo = summary["topological_summary"]

    # 1. Articulation points detection (3, 4, and 5)
    assert topo["bridges"] == [3, 4, 5]

    # 2. Overall hubs identification (sorted by weighted degree)
    # Node 3 and Node 5 have highest degree in their cliques + connection to the bridge
    overall_hubs = topo["overall_hubs"]
    hub_nodes = [h["node"] for h in overall_hubs]
    assert 3 in hub_nodes
    assert 5 in hub_nodes

    # 3. Degree distribution
    degree_dist = topo["degree_distribution"]
    assert degree_dist["mean"] == pytest.approx(28 / 9)  # sum of degrees (28) / 9 nodes
    assert degree_dist["max"] == 4  # Node 3 and Node 5 have degree 4
    assert len(degree_dist["sparkline"]) > 0


def test_skeleton_graph_payload_detail_levels(bridge_and_clique_graph):
    """
    Assert that _skeleton_graph_payload correctly gates fields based on 'detail'
    and supports both raw arrays and precomputed topological summaries.
    """
    model = MockModel(bridge_and_clique_graph)
    summary = _build_graph_summary(model)

    # A. detail="summary" (no nodes, no edges, no summary payload)
    payload_sum = _skeleton_graph_payload(
        summary, detail="summary", max_edges=10, max_nodes=10
    )
    assert "nodes" not in payload_sum
    assert "edges" not in payload_sum
    assert "topological_summary" not in payload_sum

    # B. detail="nodes" (returns high-SNR precomputed topological summary, but no raw nodes array)
    payload_nodes = _skeleton_graph_payload(
        summary, detail="nodes", max_edges=10, max_nodes=10
    )
    assert "nodes" not in payload_nodes
    assert "edges" not in payload_nodes
    assert "topological_summary" in payload_nodes
    assert payload_nodes["topological_summary"]["bridges"] == [3, 4, 5]

    # C. detail="full_nodes" (returns raw flat node list)
    payload_fn = _skeleton_graph_payload(
        summary, detail="full_nodes", max_edges=10, max_nodes=10
    )
    assert "nodes" in payload_fn
    assert len(payload_fn["nodes"]) == 9
    assert "edges" not in payload_fn

    # D. detail="edges" (returns raw flat edge list)
    payload_ed = _skeleton_graph_payload(
        summary, detail="edges", max_edges=10, max_nodes=10
    )
    assert "nodes" not in payload_ed
    assert "edges" in payload_ed

    # E. detail="full" (returns both raw nodes and edges for backward compatibility)
    payload_full = _skeleton_graph_payload(
        summary, detail="full", max_edges=50, max_nodes=50
    )
    assert "nodes" in payload_full
    assert "edges" in payload_full


def test_topological_directed_graph_guard():
    """
    Assert that directed graphs are gracefully handled and don't raise articulation points exceptions.
    """
    dg = nx.DiGraph()
    dg.add_edges_from([(0, 1), (1, 2), (2, 3)])
    model = MockModel(dg)

    summary = _build_graph_summary(model)
    topo = summary["topological_summary"]
    # Should be empty since DiGraph is directed
    assert topo["bridges"] == []


def test_topological_scale_guards():
    """
    Assert that graphs exceeding scale limits skip articulation point searches.
    """
    # 1. Node limit guard: 1001 nodes
    large_nodes_g = nx.Graph()
    large_nodes_g.add_nodes_from(range(1001))
    # Add a couple of edges so it doesn't fail basic checks
    large_nodes_g.add_edge(0, 1)
    large_nodes_g.add_edge(1, 2)
    model1 = MockModel(large_nodes_g)

    summary1 = _build_graph_summary(model1)
    assert summary1["topological_summary"]["bridges"] == []

    # 2. Edge limit guard: 25001 edges
    large_edges_g = nx.Graph()
    # Build a bipartite graph or complete-ish graph that stays under 1000 nodes but has >25000 edges
    # A graph with 250 nodes has 250 * 249 / 2 = 31125 edges if fully connected.
    for i in range(250):
        for j in range(i + 1, 250):
            large_edges_g.add_edge(i, j)

    model2 = MockModel(large_edges_g)
    summary2 = _build_graph_summary(model2)
    assert summary2["topological_summary"]["bridges"] == []
