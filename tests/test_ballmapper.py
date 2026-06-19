import numpy as np
from pulsar._pulsar import BallMapper, ball_mapper_grid


def make_grid_points(n=20, d=2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float64)


def test_fit_returns_nodes():
    pts = make_grid_points()
    bm = BallMapper(eps=1.0)
    bm.fit(pts)
    assert bm.n_nodes() > 0
    assert len(bm.nodes) == bm.n_nodes()


def test_large_eps_covers_all_points():
    pts = make_grid_points()
    bm = BallMapper(eps=100.0)
    bm.fit(pts)
    covered = {pt for members in bm.nodes for pt in members}
    assert covered == set(range(len(pts)))


def test_small_eps_many_balls():
    pts = make_grid_points()
    bm_large = BallMapper(eps=10.0)
    bm_small = BallMapper(eps=0.01)
    bm_large.fit(pts)
    bm_small.fit(pts)
    assert bm_small.n_nodes() >= bm_large.n_nodes()


def test_membership_within_eps():
    pts = make_grid_points()
    eps = 0.8
    bm = BallMapper(eps=eps)
    bm.fit(pts)
    for ball_id, members in enumerate(bm.nodes):
        # Find the center for this ball: the center is the first point that
        # created this ball. We verify all members are within eps of each other's
        # center by checking against the first member (the center's point index
        # isn't directly exposed, but all members must be within eps of center).
        # We use the fact that the center itself is always a member.
        pts_arr = pts[list(members)]
        # centroid as proxy check: max pairwise distance should be <= 2*eps
        for i in range(len(pts_arr)):
            for j in range(len(pts_arr)):
                dist = np.linalg.norm(pts_arr[i] - pts_arr[j])
                assert dist <= 2 * eps + 1e-9, (
                    f"Ball {ball_id}: points {i},{j} too far apart: {dist:.4f} > 2*{eps}"
                )


def test_edges_share_members():
    pts = make_grid_points()
    bm = BallMapper(eps=1.0)
    bm.fit(pts)
    nodes = bm.nodes
    for a, b in bm.edges:
        shared = set(nodes[a]) & set(nodes[b])
        assert len(shared) > 0, f"Edge ({a},{b}) has no shared members"


def test_no_duplicate_edges():
    pts = make_grid_points()
    bm = BallMapper(eps=1.0)
    bm.fit(pts)
    edge_set = set(bm.edges)
    assert len(edge_set) == len(bm.edges), "Duplicate edges found"


def test_ball_mapper_grid_output_count():
    embs = [make_grid_points(d=2, seed=i) for i in range(3)]
    epsilons = [0.5, 1.0]
    results = ball_mapper_grid(embs, epsilons)
    # 3 embeddings × 2 epsilons = 6 results
    assert len(results) == 6


def test_ball_mapper_grid_eps_preserved():
    embs = [make_grid_points()]
    epsilons = [0.3, 0.7, 1.2]
    results = ball_mapper_grid(embs, epsilons)
    returned_eps = sorted(bm.eps for bm in results)
    assert np.allclose(returned_eps, sorted(epsilons))


def test_single_point_cloud():
    pts = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0]])
    bm = BallMapper(eps=0.5)
    bm.fit(pts)
    # Points 0 and 1 are within 0.5, point 2 is far away
    assert bm.n_nodes() == 2


def test_kd_tree_path_4d_membership_and_edges():
    pts = make_grid_points(n=40, d=4, seed=11)
    bm = BallMapper(eps=1.2)
    bm.fit(pts)
    for members in bm.nodes:
        assert members == sorted(members)
        pts_arr = pts[members]
        for i in range(len(pts_arr)):
            for j in range(len(pts_arr)):
                assert np.linalg.norm(pts_arr[i] - pts_arr[j]) <= 2.4 + 1e-9
    for a, b in bm.edges:
        assert set(bm.nodes[a]) & set(bm.nodes[b])


def test_linear_fallback_17d_remains_correct():
    pts = make_grid_points(n=35, d=17, seed=12)
    bm = BallMapper(eps=4.0)
    bm.fit(pts)
    covered = {pt for members in bm.nodes for pt in members}
    assert covered == set(range(len(pts)))
    for members in bm.nodes:
        pts_arr = pts[members]
        for i in range(len(pts_arr)):
            for j in range(len(pts_arr)):
                assert np.linalg.norm(pts_arr[i] - pts_arr[j]) <= 8.0 + 1e-9


def test_kd_path_matches_linear_path_including_boundary():
    # The KD-tree path (d<=16) and the linear-scan fallback (d>=17) must agree
    # exactly. Zero-padding extra columns preserves Euclidean distances, so the
    # only difference between the two fits is the code path. Point idx 1 sits at
    # distance exactly eps from the first center, pinning the inclusive (`<=`)
    # boundary: it must land in the center's ball under both implementations.
    eps = 1.0
    pts_low = np.array(
        [
            [0.0, 0.0],  # first greedy center
            [1.0, 0.0],  # distance == eps -> must be included
            [0.5, 0.0],  # inside
            [5.0, 5.0],  # far -> its own center
            [5.3, 5.0],  # inside the far center's ball
        ],
        dtype=np.float64,
    )
    pts_high = np.hstack([pts_low, np.zeros((pts_low.shape[0], 15))])
    assert pts_high.shape[1] == 17  # forces the linear fallback path

    bm_kd = BallMapper(eps=eps)
    bm_kd.fit(pts_low)
    bm_linear = BallMapper(eps=eps)
    bm_linear.fit(pts_high)

    assert bm_kd.nodes == bm_linear.nodes
    assert bm_kd.edges == bm_linear.edges
    assert 1 in bm_kd.nodes[0]  # boundary point included by the KD path


def test_ball_mapper_grid_matches_individual_fits():
    embs = [make_grid_points(n=25, d=3, seed=20), make_grid_points(n=25, d=4, seed=21)]
    epsilons = [0.7, 1.1]
    grid = ball_mapper_grid(embs, epsilons)

    expected = []
    for emb in embs:
        for eps in epsilons:
            bm = BallMapper(eps=eps)
            bm.fit(emb)
            expected.append((eps, bm.nodes, bm.edges))

    observed = [(bm.eps, bm.nodes, bm.edges) for bm in grid]
    assert observed == expected
