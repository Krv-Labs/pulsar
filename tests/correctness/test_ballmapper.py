"""
Correctness tests for ballmapper.rs — BallMapper and ball_mapper_grid.

The Python reference implements the same three-step algorithm as the Rust code:
  1. Greedy centre selection: a point becomes a centre if no existing centre
     is within distance eps of it.
  2. Membership: every point within distance eps of a centre belongs to that ball.
  3. Edges: two balls are connected if they share at least one point.

Because the algorithm is fully deterministic (no RNG), the Rust and Python
outputs must agree exactly: same node sets, same edges.
"""

import numpy as np

from pulsar._pulsar import BallMapper, ball_mapper_grid


# ---------------------------------------------------------------------------
# Python reference implementation
# ---------------------------------------------------------------------------


def py_ball_mapper(pts, eps):
    """Ball Mapper: greedy cover → membership → edges.

    Parameters
    ----------
    pts : np.ndarray, shape (n_points, n_dims)
    eps : float — ball radius

    Returns
    -------
    nodes : list[list[int]]   — nodes[k] = list of point indices in ball k
    edges : list[tuple[int,int]] — (a, b) pairs with a < b, sharing ≥1 point
    """
    n = len(pts)
    eps_sq = eps * eps

    # Step 1: greedy centre selection
    centers = []
    for i in range(n):
        if all(np.sum((pts[i] - pts[c]) ** 2) > eps_sq for c in centers):
            centers.append(i)

    # Step 2: membership (all points within eps of each centre)
    nodes = []
    for c in centers:
        members = [i for i in range(n) if np.sum((pts[i] - pts[c]) ** 2) <= eps_sq]
        nodes.append(members)

    # Step 3: edges (pairs of balls sharing at least one point)
    edges = []
    for a in range(len(nodes)):
        set_a = set(nodes[a])
        for b in range(a + 1, len(nodes)):
            if any(x in set_a for x in nodes[b]):
                edges.append((a, b))

    return nodes, edges


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_pts(n=30, d=2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float64)


def test_nodes_match_reference():
    """Every ball must contain exactly the same point indices as the Python reference."""
    pts = _make_pts()
    eps = 0.8
    expected_nodes, _ = py_ball_mapper(pts, eps)

    bm = BallMapper(eps=eps)
    bm.fit(pts)
    actual_nodes = bm.nodes

    assert len(actual_nodes) == len(expected_nodes), (
        f"Number of balls: Rust={len(actual_nodes)}, Python={len(expected_nodes)}"
    )
    for k, (rust_members, py_members) in enumerate(zip(actual_nodes, expected_nodes)):
        assert sorted(rust_members) == sorted(py_members), (
            f"Ball {k}: Rust members {sorted(rust_members)} != "
            f"Python members {sorted(py_members)}"
        )


def test_edges_match_reference():
    """Edge sets must be identical (order-independent comparison)."""
    pts = _make_pts()
    eps = 0.8
    _, expected_edges = py_ball_mapper(pts, eps)

    bm = BallMapper(eps=eps)
    bm.fit(pts)

    assert set(bm.edges) == set(expected_edges), (
        f"Edges differ:\n  Rust: {sorted(bm.edges)}\n  Python: {sorted(expected_edges)}"
    )


def test_nodes_and_edges_match_multiple_eps():
    """Run the comparison for several epsilon values to catch edge cases."""
    pts = _make_pts(n=40, seed=7)
    for eps in [0.3, 0.7, 1.5]:
        expected_nodes, expected_edges = py_ball_mapper(pts, eps)

        bm = BallMapper(eps=eps)
        bm.fit(pts)

        assert len(bm.nodes) == len(expected_nodes), f"eps={eps}: node count mismatch"
        for k, (rust_m, py_m) in enumerate(zip(bm.nodes, expected_nodes)):
            assert sorted(rust_m) == sorted(py_m), (
                f"eps={eps}, ball {k}: member mismatch"
            )
        assert set(bm.edges) == set(expected_edges), f"eps={eps}: edge mismatch"


def test_single_ball_large_eps():
    """With eps much larger than the data spread, all points end up in one ball."""
    pts = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]], dtype=np.float64)
    expected_nodes, _ = py_ball_mapper(pts, eps=10.0)

    bm = BallMapper(eps=10.0)
    bm.fit(pts)

    assert len(bm.nodes) == 1
    assert sorted(bm.nodes[0]) == [0, 1, 2]


def test_isolated_balls_no_edges():
    """Well-separated clusters should produce no edges (no shared members)."""
    pts = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]], dtype=np.float64)
    expected_nodes, expected_edges = py_ball_mapper(pts, eps=0.5)

    bm = BallMapper(eps=0.5)
    bm.fit(pts)

    assert len(bm.nodes) == len(expected_nodes)
    assert len(bm.edges) == 0
    assert len(expected_edges) == 0


def test_ball_mapper_grid_matches_individual_fits():
    """ball_mapper_grid should produce the same nodes/edges as fitting individually."""
    embs = [_make_pts(seed=i) for i in range(3)]
    epsilons = [0.5, 1.0]

    grid_results = ball_mapper_grid(embs, epsilons)

    idx = 0
    for pts in embs:
        for eps in epsilons:
            bm_single = BallMapper(eps=eps)
            bm_single.fit(pts)

            bm_grid = grid_results[idx]

            assert len(bm_grid.nodes) == len(bm_single.nodes), (
                f"Grid vs single: node count mismatch at index {idx}"
            )
            for k, (gn, sn) in enumerate(zip(bm_grid.nodes, bm_single.nodes)):
                assert sorted(gn) == sorted(sn), (
                    f"Grid vs single: ball {k} members differ at index {idx}"
                )
            assert set(bm_grid.edges) == set(bm_single.edges), (
                f"Grid vs single: edges differ at index {idx}"
            )
            idx += 1
