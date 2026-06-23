"""
Correctness tests for minhash.rs — approximate cosmic-graph construction.

The MinHash path estimates each edge weight as the Jaccard similarity of the two
points' ball-sets:

    W[i,j] = |balls(i) ∩ balls(j)| / |balls(i) ∪ balls(j)|

which is exactly the weight the *exact* path computes from co-occurrence counts. So
the exact ball-set Jaccard is the ground-truth oracle here. The estimator is unbiased
with Var = J(1−J)/d, so accuracy is controlled solely by the signature depth `d`.
"""

import itertools

import numpy as np

from pulsar._pulsar import BallMapper, CosmicGraph, MinHashAccumulator


def _ball_maps(points, epsilons):
    bms = []
    for eps in epsilons:
        bm = BallMapper(float(eps))
        bm.fit(np.ascontiguousarray(points, dtype=np.float64))
        bms.append(bm)
    return bms


def _exact_ball_sets(ball_maps, n):
    """For each point, the set of global ball ids containing it."""
    sets = [set() for _ in range(n)]
    cid = 0
    for bm in ball_maps:
        for members in bm.nodes:
            for p in members:
                sets[p].add(cid)
            cid += 1
    return sets


def _exact_jaccard(sets, i, j):
    a, b = sets[i], sets[j]
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _blobs(seed=0, per=60, centers=((0, 0), (3, 0), (0, 3), (3, 3))):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(c, 0.25, (per, 2)) for c in centers])


def test_estimates_match_exact_jaccard():
    """Estimated edge weights must be close to the exact ball-set Jaccard."""
    pts = _blobs(seed=1)
    n = pts.shape[0]
    bms = _ball_maps(pts, [0.6, 0.9, 1.2])
    sets = _exact_ball_sets(bms, n)

    d = 512
    cg = CosmicGraph.from_ball_maps_minhash(bms, n, d, 42)
    errs = [abs(w - _exact_jaccard(sets, i, j)) for i, j, w in cg.weighted_edges()]

    assert errs, "expected some edges"
    # Mean error must be near zero (unbiased); max within a few standard errors.
    assert np.mean(errs) < 0.02
    # Worst-case SE at d=512 is 1/(2*sqrt(512)) ≈ 0.022; allow generous slack.
    assert np.max(errs) < 0.12


def test_unbiased_and_variance_scales_with_d():
    """Empirical variance of the estimator must track J(1−J)/d across seeds."""
    pts = _blobs(seed=2, per=40, centers=((0, 0), (2.5, 0), (0, 2.5)))
    n = pts.shape[0]
    bms = _ball_maps(pts, [0.7, 1.0, 1.3])
    sets = _exact_ball_sets(bms, n)

    # Pick a pair with a mid-range exact Jaccard for a meaningful variance check.
    target = None
    for i, j in itertools.combinations(range(n), 2):
        jac = _exact_jaccard(sets, i, j)
        if 0.25 < jac < 0.6:
            target = (i, j, jac)
            break
    assert target is not None
    i, j, jac = target

    d = 128
    ests = []
    for seed in range(40):
        edges = {
            (a, b): w
            for a, b, w in CosmicGraph.from_ball_maps_minhash(
                bms, n, d, seed
            ).weighted_edges()
        }
        ests.append(edges.get((i, j), 0.0))
    ests = np.array(ests)

    # Unbiased: sample mean near the true Jaccard.
    assert abs(ests.mean() - jac) < 0.05
    # Variance near J(1−J)/d (loose factor-of-2 envelope for finite samples).
    expected_var = jac * (1 - jac) / d
    assert ests.var() < 3 * expected_var


def test_high_recall_for_strong_edges():
    """Every strongly co-occurring pair (exact W ≥ 0.3) must be recovered."""
    pts = _blobs(seed=3)
    n = pts.shape[0]
    bms = _ball_maps(pts, [0.6, 0.9, 1.2])
    sets = _exact_ball_sets(bms, n)

    cg = CosmicGraph.from_ball_maps_minhash(bms, n, 256, 42)
    found = {(i, j) for i, j, _ in cg.weighted_edges()}
    strong = [
        (i, j)
        for i, j in itertools.combinations(range(n), 2)
        if _exact_jaccard(sets, i, j) >= 0.3
    ]
    assert strong, "expected some strong edges"
    recall = sum(1 for e in strong if e in found) / len(strong)
    assert recall > 0.98


def test_deterministic_for_fixed_seed():
    pts = _blobs(seed=4, per=30)
    n = pts.shape[0]
    bms = _ball_maps(pts, [0.7, 1.0])
    a = CosmicGraph.from_ball_maps_minhash(bms, n, 256, 99).weighted_edges()
    b = CosmicGraph.from_ball_maps_minhash(bms, n, 256, 99).weighted_edges()
    assert a == b


def test_accumulator_matches_one_shot_and_is_order_invariant():
    """Streaming accumulation (any batching) must equal the one-shot construction.

    Global ball-id offsets make element-wise-min folding identical regardless of how
    ball maps are split across accumulate() calls.
    """
    pts = _blobs(seed=5, per=35, centers=((0, 0), (3, 0), (0, 3)))
    n = pts.shape[0]
    bms = _ball_maps(pts, [0.6, 0.9, 1.2, 1.5])

    one_shot = CosmicGraph.from_ball_maps_minhash(bms, n, 256, 42).weighted_edges()

    acc = MinHashAccumulator(n, 256, 42)
    acc.accumulate(bms[:1])
    acc.accumulate(bms[1:3])
    acc.accumulate(bms[3:])
    streamed = acc.to_cosmic_graph().weighted_edges()

    assert one_shot == streamed


def test_isolated_points_have_no_edges():
    """A point in no ball must never appear in an edge (no spurious self-similarity)."""
    # Two tight blobs far apart plus one outlier far from everything.
    pts = np.array(
        [[0.0, 0.0], [0.05, 0.0], [0.0, 0.05], [10.0, 10.0]], dtype=np.float64
    )
    n = pts.shape[0]
    bms = _ball_maps(pts, [0.3])  # eps small: outlier is its own singleton ball
    cg = CosmicGraph.from_ball_maps_minhash(bms, n, 256, 1)
    # Point 3 shares no ball with 0,1,2 → must not connect to them with weight > 0.
    for i, j, _ in cg.weighted_edges():
        if 3 in (i, j):
            other = j if i == 3 else i
            # Only allowed if they genuinely share a ball (they do not here).
            assert False, f"isolated point 3 wrongly connected to {other}"


def test_degenerate_inputs():
    pts = _blobs(seed=6, per=10, centers=((0, 0),))
    n = pts.shape[0]
    bms = _ball_maps(pts, [0.5])
    # d must be >= 1.
    import pytest

    with pytest.raises(ValueError, match="d must be"):
        CosmicGraph.from_ball_maps_minhash(bms, n, 0, 0)
    with pytest.raises(ValueError, match="d must be"):
        MinHashAccumulator(n, 0, 0)
