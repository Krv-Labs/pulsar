"""Connectivity-preserving backbone pipeline for the cosmic_graph viz payload.

Replaces the naive "top-N edges by weight" cap with a backbone + strength-budget
selection computed over the FULL weighted-adjacency edge set. The rendered graph has
far fewer edges but the IDENTICAL connected-component structure as the fitted
cosmic_graph (the faithfulness invariant — the backbone never merges or splits a
component).

Pipeline (build-spec viz revamp; contract = isomorph packages/contracts CosmicGraph*):
  S1. Authoritative components over the full thresholded edge set
      (``weighted_adjacency[i,j] > resolved_construction_threshold`` — STRICT ``>``,
      matching ``src/cosmic.rs`` so the component count AGREES with diagnose_model).
      Ranked by size desc → ``component`` ids (0 = giant). Isolated nodes are their
      own singleton components.
  S2. Backbone = maximum spanning forest (one max spanning tree per component).
      Forest invariant: ``len(backbone) == n_nodes - n_components``.
  S3. Context budget = per node, its top-k strongest NON-backbone incident edges,
      capped at a global budget ``B = min(totalEdges, EDGE_BUDGET)``. LOCAL (per-node)
      selection, not a global threshold; undirected dedup.

Output keys are camelCase and flow VERBATIM to the client.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

# --------------------------------------------------------------------------- #
# Tuning constants
# --------------------------------------------------------------------------- #
# Per-node context fan-out: each node contributes up to this many of its strongest
# non-backbone incident edges. k=4 keeps local neighbourhoods legible (the backbone
# already guarantees the component is connected) without re-densifying the graph.
CONTEXT_K = 4
# Total edge budget for context edges + backbone. ~6*n keeps the rendered graph
# sparse and force-layout-friendly (avg degree ~12) while leaving room above the
# n-1-ish backbone for meaningful local structure. Tunable single source of truth.
EDGE_BUDGET_PER_NODE = 6


def _edge_budget(n_nodes: int) -> int:
    return EDGE_BUDGET_PER_NODE * max(n_nodes, 1)


# --------------------------------------------------------------------------- #
# S1 — authoritative components over the FULL thresholded edge set
# --------------------------------------------------------------------------- #
def _thresholded_edges(W: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """Upper-triangle edges with ``W[i,j] > threshold`` (STRICT, matches src/cosmic.rs:69).

    Returns ``[(u, v, w), ...]`` with ``u < v`` (undirected, deduped).
    """
    n = W.shape[0]
    if n == 0:
        return []
    iu, iv = np.triu_indices(n, k=1)
    w = W[iu, iv]
    mask = w > threshold
    us = iu[mask]
    vs = iv[mask]
    ws = w[mask]
    return [(int(u), int(v), float(wt)) for u, v, wt in zip(us, vs, ws)]


def _components_by_size(
    n_nodes: int, edges: list[tuple[int, int, float]]
) -> tuple[dict[int, int], list[dict]]:
    """Connected components over ALL nodes, ranked by size desc (0 = giant).

    Returns ``(component_of_node, components_meta)`` where ``components_meta`` is the
    contract ``components[]`` list (``{id, size, isSingleton}``) sorted by size desc.
    Isolated nodes are their own singleton components.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    # Rank by size desc; deterministic tiebreak on the component's min node id so a
    # given fitted graph always yields the same component ids.
    comps = sorted(
        (sorted(int(x) for x in c) for c in nx.connected_components(G)),
        key=lambda c: (-len(c), c[0]),
    )
    component_of_node: dict[int, int] = {}
    components_meta: list[dict] = []
    for cid, members in enumerate(comps):
        size = len(members)
        for node in members:
            component_of_node[node] = cid
        components_meta.append(
            {"id": cid, "size": size, "isSingleton": bool(size == 1)}
        )
    return component_of_node, components_meta


# --------------------------------------------------------------------------- #
# S2 — backbone (maximum spanning forest)
# --------------------------------------------------------------------------- #
def _backbone_edges(edges: list[tuple[int, int, float]]) -> set[tuple[int, int]]:
    """Maximum spanning forest over the thresholded graph.

    One maximum spanning tree per connected component (singletons contribute none),
    so the union is a spanning forest with ``n_nodes - n_components`` edges and no
    edge ever joins two different components. Returns a set of ``(u, v)`` keys
    (``u < v``).
    """
    if not edges:
        return set()
    G = nx.Graph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    forest = nx.maximum_spanning_tree(G, weight="weight")
    return {(min(int(u), int(v)), max(int(u), int(v))) for u, v in forest.edges()}


# --------------------------------------------------------------------------- #
# S3 — context budget (per-node top-k strongest non-backbone incident edges)
# --------------------------------------------------------------------------- #
def _context_edges(
    edges: list[tuple[int, int, float]],
    backbone: set[tuple[int, int]],
    *,
    n_nodes: int,
    k: int,
    budget: int,
) -> set[tuple[int, int]]:
    """Top-k strongest non-backbone incident edges per node, capped at ``budget`` total.

    LOCAL selection: each node independently nominates its k strongest non-backbone
    incident edges. Nominated edges are then merged (undirected dedup) and the
    strongest are kept until the global budget (which already accounts for the
    backbone) is exhausted.
    """
    remaining = max(budget - len(backbone), 0)
    if remaining <= 0 or k <= 0:
        return set()

    # Group non-backbone edges by incident node, strongest first.
    by_node: dict[int, list[tuple[float, int, int]]] = {i: [] for i in range(n_nodes)}
    for u, v, w in edges:
        key = (u, v) if u < v else (v, u)
        if key in backbone:
            continue
        by_node[u].append((w, u, v))
        by_node[v].append((w, u, v))

    nominated: set[tuple[int, int]] = set()
    nominated_list: list[tuple[float, int, int]] = []
    for node in range(n_nodes):
        incident = by_node[node]
        if not incident:
            continue
        incident.sort(key=lambda e: (-e[0], e[1], e[2]))
        for w, u, v in incident[:k]:
            key = (min(u, v), max(u, v))
            if key not in nominated:
                nominated.add(key)
                nominated_list.append((w, key[0], key[1]))

    if len(nominated_list) <= remaining:
        return {(u, v) for _, u, v in nominated_list}

    # Over budget: keep globally strongest nominees (deterministic tiebreak).
    nominated_list.sort(key=lambda e: (-e[0], e[1], e[2]))
    return {(u, v) for _, u, v in nominated_list[:remaining]}


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_cosmic_graph_payload(view, labels, *, provenance=None) -> dict:
    """Build the backbone cosmic_graph viz payload from ``view.weighted_adjacency``.

    Args:
        view: ``ArtifactView`` (or fitted ThemaRS) exposing ``weighted_adjacency``,
            ``resolved_construction_threshold``, and ``n``.
        labels: per-node interpretation cluster labels (or ``None``).
        provenance: ClusterProvenance object (snake_case) attached VERBATIM.
    """
    W = np.asarray(view.weighted_adjacency, dtype=np.float64)
    n_nodes = int(view.n)
    threshold = float(view.resolved_construction_threshold)
    lab = [int(x) for x in labels] if labels is not None else []

    # S1 — full thresholded edge set + authoritative components.
    all_edges = _thresholded_edges(W, threshold)
    total_edges = len(all_edges)
    component_of_node, components_meta = _components_by_size(n_nodes, all_edges)
    n_components = len(components_meta)

    # S2 — backbone (maximum spanning forest).
    backbone = _backbone_edges(all_edges)

    # S3 — context budget.
    budget = min(total_edges, _edge_budget(n_nodes))
    context = _context_edges(
        all_edges, backbone, n_nodes=n_nodes, k=CONTEXT_K, budget=budget
    )

    # Weight lookup for emitted edges (undirected key).
    weight_of = {(u, v): w for u, v, w in all_edges}

    out_edges: list[dict] = []
    degree = np.zeros(n_nodes, dtype=np.int64)
    for (u, v) in backbone:
        w = weight_of.get((u, v), weight_of.get((v, u), 1.0))
        out_edges.append({"u": u, "v": v, "w": float(w), "role": "backbone"})
        degree[u] += 1
        degree[v] += 1
    for (u, v) in context:
        w = weight_of.get((u, v), weight_of.get((v, u), 1.0))
        out_edges.append({"u": u, "v": v, "w": float(w), "role": "context"})
        degree[u] += 1
        degree[v] += 1

    nodes = [
        {
            "id": i,
            "component": int(component_of_node.get(i, 0)),
            "cluster": (lab[i] if i < len(lab) else -1),
            "size": int(1 + degree[i]),
        }
        for i in range(n_nodes)
    ]

    rendered_edges = len(out_edges)
    pruned_edges = total_edges - rendered_edges

    viz = {
        "kind": "cosmic_graph",
        "nodes": nodes,
        "edges": out_edges,
        "components": components_meta,
        "renderedEdges": rendered_edges,
        "totalEdges": total_edges,
        "prunedEdges": pruned_edges,
        "backboneEdges": len(backbone),
        # Back-compat camelCase flag (the OLD code wrongly emitted snake_case, which
        # never reached the client). True iff anything was pruned.
        "edgesTruncated": pruned_edges > 0,
    }
    if provenance is not None:
        viz["provenance"] = provenance
    return viz
