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

# Fixed seed + iteration count for the per-component force layout. Determinism is a
# hard contract (same fitted view → identical coords), so NOTHING here may be random.
LAYOUT_SEED = 42
LAYOUT_ITERATIONS = 60
# Packing geometry. Components are placed on a deterministic Archimedean spiral with
# bounding circles that NEVER overlap; the giant (largest) sits at the origin.
_PACK_PADDING = 0.25  # extra gap between adjacent bounding circles (in radius units)
_PACK_SPIRAL_B = 1.0  # spiral tightness; larger = looser turns
# Singleton periphery band: tiled on concentric rings OUTSIDE every packed component.
_SINGLETON_BAND_GAP = 1.5  # gap (radius units) between the packed region and ring 1
_SINGLETON_RING_STEP = 0.6  # radial spacing between successive singleton rings


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
# Layout — server-precomputed island layout (D3-disjoint analog, deterministic)
# --------------------------------------------------------------------------- #
def _component_force_layout(
    members: list[int], member_edges: list[tuple[int, int]]
) -> dict[int, tuple[float, float]]:
    """Seeded force layout of ONE multi-node component, centred on its centroid.

    Uses ``nx.spring_layout`` with a FIXED ``seed`` and ``iterations`` so the same
    component (same node set + edge set) always yields identical coordinates.
    """
    g = nx.Graph()
    g.add_nodes_from(members)
    g.add_edges_from(member_edges)
    # spring_layout is deterministic given a fixed seed AND a deterministic initial
    # node ordering — networkx seeds its RNG from ``seed`` and lays out in graph
    # insertion order, which we control via the sorted ``members`` list above.
    pos = nx.spring_layout(g, seed=LAYOUT_SEED, iterations=LAYOUT_ITERATIONS)
    coords = {int(node): (float(xy[0]), float(xy[1])) for node, xy in pos.items()}
    # Centre on the centroid so the bounding radius below is measured about (0,0).
    cx = sum(x for x, _ in coords.values()) / len(coords)
    cy = sum(y for _, y in coords.values()) / len(coords)
    # spring_layout normalizes EVERY component to ~unit radius, which crushes a large component into a
    # dense "blob" while a tiny one stays airy. Scale each component's extent by √size so node DENSITY
    # is ~constant across islands (radius ∝ √size ⇒ area ∝ size) — readable, breathing components.
    # Packing measures the scaled radius, so islands stay non-overlapping regardless of this factor.
    spread = float(len(coords)) ** 0.5
    return {
        node: ((x - cx) * spread, (y - cy) * spread) for node, (x, y) in coords.items()
    }


def _bounding_radius(local: dict[int, tuple[float, float]]) -> float:
    """Radius of the smallest origin-centred circle enclosing ``local`` coords.

    ``local`` is already centroid-centred. For a degenerate (single point at origin
    after centring) component we fall back to a small positive radius proportional to
    √size so packing still leaves room.
    """
    if not local:
        return 0.0
    r = max((x * x + y * y) ** 0.5 for x, y in local.values())
    # Floor proportional to √size: a tight/degenerate component still claims space.
    return max(r, 0.5 * (len(local) ** 0.5))


def _spiral_pack_centers(radii: list[float]) -> list[tuple[float, float]]:
    """Place components (sorted by size DESC → ``radii`` index 0 = giant) so their
    bounding circles never overlap. Giant at the origin; the rest on a deterministic
    Archimedean spiral, each pushed outward until it clears ALL already-placed circles.

    Returns one ``(cx, cy)`` per input radius, in the same order. Deterministic: no
    randomness, fixed spiral parameters.
    """
    centers: list[tuple[float, float]] = []
    if not radii:
        return centers
    centers.append((0.0, 0.0))  # giant at origin
    placed: list[tuple[float, float, float]] = [(0.0, 0.0, radii[0])]

    # Archimedean spiral r(θ) = b·θ. We advance θ in small steps and, for each new
    # component, find the first angle whose radial distance also clears every placed
    # circle (so the spiral only ever expands outward, never collides).
    theta = 0.0
    d_theta = 0.35
    for i in range(1, len(radii)):
        ri = radii[i]
        while True:
            theta += d_theta
            rad = _PACK_SPIRAL_B * theta
            cx = rad * float(np.cos(theta))
            cy = rad * float(np.sin(theta))
            ok = True
            for px, py, pr in placed:
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if dist < (pr + ri + _PACK_PADDING):
                    ok = False
                    break
            if ok:
                centers.append((cx, cy))
                placed.append((cx, cy, ri))
                break
    return centers


def _outer_packed_radius(
    centers: list[tuple[float, float]], radii: list[float]
) -> float:
    """Farthest extent (center distance + bounding radius) of any packed component."""
    if not centers:
        return 0.0
    return max(((cx * cx + cy * cy) ** 0.5) + r for (cx, cy), r in zip(centers, radii))


def compute_island_layout(
    nodes: list[dict],
    edges: list[dict],
    components: list[dict],
) -> dict[int, tuple[float, float]]:
    """Deterministic island layout: per-component force layout + non-overlapping
    spiral packing of components + a tidy singleton periphery band.

    Args:
        nodes: emitted node dicts (``id``, ``component``, ...).
        edges: emitted edge dicts (``u``, ``v``, ``role``, ...) — both backbone and
            context edges feed the per-component force sim.
        components: ``components_meta`` (``id``, ``size``, ``isSingleton``), size desc.

    Returns ``{node_id: (x, y)}`` normalized into a stable unit-ish coordinate space
    centred on the giant. No unseeded randomness anywhere → same input, same output.
    """
    comp_of = {int(nd["id"]): int(nd["component"]) for nd in nodes}
    members_by_comp: dict[int, list[int]] = {}
    for nd in nodes:
        members_by_comp.setdefault(int(nd["component"]), []).append(int(nd["id"]))
    for cid in members_by_comp:
        members_by_comp[cid].sort()  # deterministic node ordering for the sim

    edges_by_comp: dict[int, list[tuple[int, int]]] = {}
    for e in edges:
        u, v = int(e["u"]), int(e["v"])
        cu = comp_of.get(u)
        if cu is not None and cu == comp_of.get(v):
            edges_by_comp.setdefault(cu, []).append((u, v))

    # Partition multi-node components (force-sim + pack) from singletons (periphery).
    multi = [c for c in components if int(c["size"]) > 1]
    singleton_comp_ids = [int(c["id"]) for c in components if int(c["size"]) == 1]
    # Already size-desc in components_meta; sort defensively (size desc, id asc).
    multi.sort(key=lambda c: (-int(c["size"]), int(c["id"])))

    local_layouts: dict[int, dict[int, tuple[float, float]]] = {}
    radii: list[float] = []
    for c in multi:
        cid = int(c["id"])
        local = _component_force_layout(
            members_by_comp.get(cid, []), edges_by_comp.get(cid, [])
        )
        local_layouts[cid] = local
        radii.append(_bounding_radius(local))

    centers = _spiral_pack_centers(radii)

    positions: dict[int, tuple[float, float]] = {}
    for c, (cx, cy) in zip(multi, centers):
        cid = int(c["id"])
        for node, (x, y) in local_layouts[cid].items():
            positions[node] = (cx + x, cy + y)

    # Singletons: tidy concentric periphery rings OUTSIDE every packed component.
    if singleton_comp_ids:
        # Singletons map 1:1 to their single member node (sorted for determinism).
        singleton_nodes = sorted(
            members_by_comp.get(cid, [cid])[0] for cid in singleton_comp_ids
        )
        outer = _outer_packed_radius(centers, radii)
        base = outer + _SINGLETON_BAND_GAP
        idx = 0
        ring = 0
        while idx < len(singleton_nodes):
            ring_r = base + ring * _SINGLETON_RING_STEP
            # Circumference grows with radius → more slots per outer ring.
            capacity = max(1, int(round(2.0 * np.pi * ring_r)))
            count = min(capacity, len(singleton_nodes) - idx)
            for j in range(count):
                ang = 2.0 * np.pi * (j / count)
                node = singleton_nodes[idx + j]
                positions[node] = (
                    ring_r * float(np.cos(ang)),
                    ring_r * float(np.sin(ang)),
                )
            idx += count
            ring += 1

    # Any node not placed (defensive — should not happen) → origin.
    for nd in nodes:
        positions.setdefault(int(nd["id"]), (0.0, 0.0))

    # Normalize to a stable unit-ish space: scale by max |coord| so coords land in
    # roughly [-1, 1]. Centred on the giant (origin) already; the client re-normalizes
    # to the viewport. Deterministic — pure function of the (deterministic) positions.
    extent = max((abs(x) for x, _ in positions.values()), default=0.0)
    extent = max(extent, max((abs(y) for _, y in positions.values()), default=0.0))
    if extent > 0:
        positions = {nid: (x / extent, y / extent) for nid, (x, y) in positions.items()}
    return positions


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
    for u, v in backbone:
        w = weight_of.get((u, v), weight_of.get((v, u), 1.0))
        out_edges.append({"u": u, "v": v, "w": float(w), "role": "backbone"})
        degree[u] += 1
        degree[v] += 1
    for u, v in context:
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

    # Server-precomputed deterministic island layout → x,y on every node.
    layout = compute_island_layout(nodes, out_edges, components_meta)
    for nd in nodes:
        x, y = layout[nd["id"]]
        nd["x"] = float(x)
        nd["y"] = float(y)

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
        "layoutSeed": LAYOUT_SEED,
        # Back-compat camelCase flag (the OLD code wrongly emitted snake_case, which
        # never reached the client). True iff anything was pruned.
        "edgesTruncated": pruned_edges > 0,
    }
    if provenance is not None:
        viz["provenance"] = provenance
    return viz
