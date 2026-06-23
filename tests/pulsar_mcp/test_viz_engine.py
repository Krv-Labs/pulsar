"""Viz-engine tests — server island layout + real manifold3d DR.

Covers:
  - cosmic_graph island layout: deterministic coords, non-overlapping component
    bounding circles, singletons tiled in the periphery, layoutSeed emitted.
  - manifold3d: PCA when >3 dims (deterministic, 3-D points) else raw passthrough.

Run: `uv run pytest tests/pulsar_mcp/test_viz_engine.py -q`
"""

from __future__ import annotations

import numpy as np

from pulsar.mcp.viz_graph import (
    LAYOUT_SEED,
    _bounding_radius,
    build_cosmic_graph_payload,
    compute_island_layout,
)


# --------------------------------------------------------------------------- #
# Fixture views
# --------------------------------------------------------------------------- #
def _block_adjacency(block_sizes: list[int], intra: float = 0.9) -> np.ndarray:
    """Block-diagonal weighted adjacency → each block is a connected component."""
    n = sum(block_sizes)
    W = np.zeros((n, n), dtype=np.float64)
    off = 0
    for s in block_sizes:
        for i in range(off, off + s):
            for j in range(i + 1, off + s):
                W[i, j] = W[j, i] = intra
        off += s
    return W


class _View:
    def __init__(self, W: np.ndarray, threshold: float = 0.5):
        self.weighted_adjacency = W
        self.resolved_construction_threshold = threshold
        self.n = W.shape[0]


# --------------------------------------------------------------------------- #
# Task A — island layout
# --------------------------------------------------------------------------- #
def _make_payload():
    # Three multi-node components (sizes 6, 4, 3) + 5 singletons.
    W = _block_adjacency([6, 4, 3])
    # pad with 5 isolated singleton nodes
    n_multi = W.shape[0]
    n = n_multi + 5
    big = np.zeros((n, n), dtype=np.float64)
    big[:n_multi, :n_multi] = W
    return build_cosmic_graph_payload(_View(big), labels=None)


def test_layout_seed_and_coords_emitted():
    viz = _make_payload()
    assert viz["layoutSeed"] == LAYOUT_SEED
    for nd in viz["nodes"]:
        assert "x" in nd and "y" in nd
        assert isinstance(nd["x"], float) and isinstance(nd["y"], float)
        # Normalized unit-ish space.
        assert -1.0001 <= nd["x"] <= 1.0001
        assert -1.0001 <= nd["y"] <= 1.0001


def test_layout_deterministic_across_two_calls():
    a = _make_payload()
    b = _make_payload()
    coords_a = {nd["id"]: (nd["x"], nd["y"]) for nd in a["nodes"]}
    coords_b = {nd["id"]: (nd["x"], nd["y"]) for nd in b["nodes"]}
    assert coords_a == coords_b
    # And the low-level layout function is itself deterministic.
    l1 = compute_island_layout(a["nodes"], a["edges"], a["components"])
    l2 = compute_island_layout(a["nodes"], a["edges"], a["components"])
    assert l1 == l2


def test_component_islands_do_not_overlap():
    viz = _make_payload()
    coords = {nd["id"]: np.array([nd["x"], nd["y"]]) for nd in viz["nodes"]}
    comp_of = {nd["id"]: nd["component"] for nd in viz["nodes"]}

    # Bounding circle (centroid + radius) per MULTI-node component.
    multi_ids = [c["id"] for c in viz["components"] if c["size"] > 1]
    circles = {}
    for cid in multi_ids:
        pts = np.array([coords[nid] for nid, c in comp_of.items() if c == cid])
        center = pts.mean(axis=0)
        radius = float(np.max(np.linalg.norm(pts - center, axis=1)))
        circles[cid] = (center, radius)

    eps = 1e-6
    ids = list(circles)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            (c1, r1), (c2, r2) = circles[ids[i]], circles[ids[j]]
            dist = float(np.linalg.norm(c1 - c2))
            # Disjoint: center distance >= sum of radii (minus tiny epsilon).
            assert dist >= (r1 + r2) - eps, (
                f"components {ids[i]} and {ids[j]} overlap: "
                f"dist={dist}, r1+r2={r1 + r2}"
            )


def test_singletons_in_periphery():
    viz = _make_payload()
    coords = {nd["id"]: np.array([nd["x"], nd["y"]]) for nd in viz["nodes"]}
    comp_of = {nd["id"]: nd["component"] for nd in viz["nodes"]}
    singleton_comp_ids = {c["id"] for c in viz["components"] if c["size"] == 1}
    multi_comp_ids = {c["id"] for c in viz["components"] if c["size"] > 1}

    singleton_nodes = [n for n, c in comp_of.items() if c in singleton_comp_ids]
    multi_nodes = [n for n, c in comp_of.items() if c in multi_comp_ids]
    assert singleton_nodes and multi_nodes

    # Outer extent of the packed (multi) region, measured from the origin (giant centre).
    packed_extent = max(float(np.linalg.norm(coords[n])) for n in multi_nodes)
    # Every singleton sits beyond the packed region (a tidy periphery band).
    for n in singleton_nodes:
        assert float(np.linalg.norm(coords[n])) > packed_extent, (
            f"singleton {n} not in periphery"
        )


def test_giant_central():
    viz = _make_payload()
    coords = {nd["id"]: np.array([nd["x"], nd["y"]]) for nd in viz["nodes"]}
    comp_of = {nd["id"]: nd["component"] for nd in viz["nodes"]}
    # Component 0 is the giant; its centroid should be at/near the origin.
    giant_pts = np.array([coords[n] for n, c in comp_of.items() if c == 0])
    centroid = giant_pts.mean(axis=0)
    assert float(np.linalg.norm(centroid)) < 0.2


def test_bounding_radius_floor_for_degenerate_component():
    # All-at-origin component → radius falls back to √size floor (> 0).
    local = {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0), 3: (0.0, 0.0)}
    assert _bounding_radius(local) >= 0.5 * (4**0.5)


# --------------------------------------------------------------------------- #
# Task B — manifold3d DR
# --------------------------------------------------------------------------- #
def test_sklearn_available():
    from sklearn.decomposition import PCA  # noqa: F401


class _EmbView:
    def __init__(self, emb: np.ndarray):
        self._embeddings = [emb]
        self.n = emb.shape[0]


def test_manifold3d_pca_when_high_dim():
    from pulsar.mcp.tools.curated import _viz_manifold3d

    rng = np.random.default_rng(7)
    emb = rng.standard_normal((30, 8))  # 8-D → must PCA to 3-D
    viz = _viz_manifold3d(_EmbView(emb), labels=None)
    assert viz["method"] == "pca"
    pts = np.array(viz["points"])
    assert pts.shape == (30, 3)
    assert len(viz["ids"]) == 30
    assert len(viz["cluster"]) == 30


def test_manifold3d_pca_deterministic():
    from pulsar.mcp.tools.curated import _viz_manifold3d

    rng = np.random.default_rng(11)
    emb = rng.standard_normal((25, 6))
    a = _viz_manifold3d(_EmbView(emb.copy()), labels=None)
    b = _viz_manifold3d(_EmbView(emb.copy()), labels=None)
    assert a["method"] == b["method"] == "pca"
    assert np.allclose(np.array(a["points"]), np.array(b["points"]))


def test_manifold3d_raw_when_low_dim():
    from pulsar.mcp.tools.curated import _viz_manifold3d

    emb = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]], dtype=np.float32)  # 2-D
    viz = _viz_manifold3d(_EmbView(emb), labels=None)
    assert viz["method"] == "raw"
    pts = np.array(viz["points"])
    assert pts.shape == (3, 3)  # zero-padded to 3 cols
    assert np.allclose(pts[:, 2], 0.0)


def test_manifold3d_raw_when_exactly_3_dim():
    from pulsar.mcp.tools.curated import _viz_manifold3d

    emb = np.arange(15, dtype=float).reshape(5, 3)
    viz = _viz_manifold3d(_EmbView(emb), labels=None)
    assert viz["method"] == "raw"
    assert np.allclose(np.array(viz["points"]), emb)
