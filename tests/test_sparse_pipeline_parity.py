"""End-to-end fidelity: the MinHash-constructed pipeline must approximate the exact
dense cosmic graph.

The pipeline now builds the cosmic graph by MinHash/LSH (an approximate, randomized
construction) rather than exact co-occurrence counting. Edge weights are unbiased
Jaccard estimates with Var = J(1−J)/d, so we assert *approximate* parity against an
independently computed exact dense reference: every strong reference edge is recovered
(high recall) and recovered weights are close to the exact Jaccard. Exact bit-for-bit
parity is intentionally no longer expected.
"""

import numpy as np
import pandas as pd

from pulsar import ThemaRS, load_config
from pulsar._pulsar import CosmicGraph, accumulate_pseudo_laplacians

# Reference edges at or above this exact Jaccard must be recovered by the MinHash path.
_STRONG = 0.3
_MIN_RECALL = 0.95
_MAX_MEAN_WEIGHT_ERR = 0.05


def _config():
    return load_config(
        {
            "preprocessing": {},
            "sweep": {
                "projection": {
                    "method": "jl",
                    "dimensions": {"values": [2, 3]},
                    "seed": {"values": [42]},
                },
                "ball_mapper": {"epsilon": {"values": [0.8, 1.2]}},
            },
            "cosmic_graph": {"construction_threshold": "auto"},
        }
    )


def _frame():
    rng = np.random.default_rng(17)
    return pd.DataFrame(rng.standard_normal((150, 4)), columns=list("abcd"))


def _dense_reference_jaccard(model) -> np.ndarray:
    """Exact ball-set Jaccard matrix from the fitted ball maps (the oracle)."""
    n = model.cosmic_rust.n
    dense_L = np.array(accumulate_pseudo_laplacians(model.ball_maps, n))
    return np.array(CosmicGraph.from_pseudo_laplacian(dense_L, 0.0).weighted_adj)


def _assert_approximate_parity(model) -> None:
    W = _dense_reference_jaccard(model)
    # MinHash weighted graph before thresholding/sparsification.
    est = {(i, j): w for i, j, w in model.dense_cosmic_rust.weighted_edges()}

    n = W.shape[0]
    strong = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if W[i, j] >= _STRONG
    ]
    assert strong, "expected some strong reference edges"

    recall = sum(1 for e in strong if e in est) / len(strong)
    assert recall >= _MIN_RECALL, f"recall {recall:.3f} < {_MIN_RECALL}"

    errs = [abs(est[(i, j)] - W[i, j]) for (i, j) in strong if (i, j) in est]
    assert np.mean(errs) < _MAX_MEAN_WEIGHT_ERR, f"mean weight error {np.mean(errs):.4f}"


def test_pipeline_minhash_approximates_dense_reference():
    model = ThemaRS(_config()).fit(data=_frame())
    _assert_approximate_parity(model)
    # Resolved threshold must be a valid fraction in [0, 1].
    assert 0.0 <= model.resolved_construction_threshold <= 1.0


def test_fit_batched_matches_all_at_once():
    data = _frame()
    batched = ThemaRS(_config()).fit(data=data)
    all_at_once = ThemaRS(_config()).fit(data=data, ballmap_batch_size=None)

    assert batched.resolved_construction_threshold == (
        all_at_once.resolved_construction_threshold
    )
    np.testing.assert_allclose(
        batched.weighted_adjacency, all_at_once.weighted_adjacency, atol=1e-12
    )
    assert set(batched.cosmic_graph.edges()) == set(all_at_once.cosmic_graph.edges())


def test_weighted_adjacency_is_lazy():
    model = ThemaRS(_config()).fit(data=_frame())
    # Not materialized right after fit.
    assert model._weighted_adjacency is None
    # First property access materializes and caches it.
    _ = model.weighted_adjacency
    assert model._weighted_adjacency is not None


def test_fit_multi_minhash_approximates_dense_reference():
    cfg = _config()
    ds = [_frame(), _frame() + 0.5]
    model = ThemaRS(cfg).fit_multi(ds, store_ball_maps=True)
    _assert_approximate_parity(model)
    assert 0.0 <= model.resolved_construction_threshold <= 1.0
