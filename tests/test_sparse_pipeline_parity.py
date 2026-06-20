"""End-to-end parity: the sparse-backbone pipeline must reproduce the dense path.

The pipeline now keeps the cosmic graph sparse end-to-end (no n×n on the hot path)
and materializes ``weighted_adjacency`` lazily. These tests assert the observable
results — resolved threshold, exposed graph edges, and the dense adjacency view —
are identical to an independently computed dense reference.
"""

import networkx as nx
import numpy as np
import pandas as pd

from pulsar import ThemaRS, load_config
from pulsar._pulsar import (
    CosmicGraph,
    accumulate_pseudo_laplacians,
    find_stable_thresholds,
)
from pulsar.analysis import cosmic_to_networkx


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


def test_pipeline_sparse_matches_dense_reference():
    model = ThemaRS(_config()).fit(data=_frame())

    # Independent dense reference from the same fitted ball maps.
    n = model.cosmic_rust.n
    dense_L = np.array(accumulate_pseudo_laplacians(model.ball_maps, n))
    cg_dense = CosmicGraph.from_pseudo_laplacian(dense_L, 0.0)
    W = np.array(cg_dense.weighted_adj)
    scale = max(1.0, float(W.max()))
    ref = find_stable_thresholds(W / scale)

    # Resolved construction threshold parity.
    assert model.resolved_construction_threshold == ref.optimal_threshold

    # Dense weighted-adjacency view parity (lazily materialized inside the property).
    np.testing.assert_allclose(model.weighted_adjacency, W / scale, atol=1e-12)

    # Exposed networkx graph parity.
    ref_graph = cosmic_to_networkx(
        cg_dense, threshold=ref.optimal_threshold, scale=scale
    )
    assert model.cosmic_graph.number_of_edges() == ref_graph.number_of_edges()
    assert set(model.cosmic_graph.edges()) == set(ref_graph.edges())
    # Edge weights match too.
    ref_w = nx.to_numpy_array(ref_graph)
    got_w = nx.to_numpy_array(model.cosmic_graph)
    np.testing.assert_allclose(got_w, ref_w, atol=1e-12)


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


def test_fit_multi_sparse_matches_dense_reference():
    cfg = _config()
    ds = [_frame(), _frame() + 0.5]
    model = ThemaRS(cfg).fit_multi(ds, store_ball_maps=True)

    n = model.cosmic_rust.n
    dense_L = np.array(accumulate_pseudo_laplacians(model.ball_maps, n))
    cg_dense = CosmicGraph.from_pseudo_laplacian(dense_L, 0.0)
    W = np.array(cg_dense.weighted_adj)
    scale = max(1.0, float(W.max()))
    ref = find_stable_thresholds(W / scale)

    assert model.resolved_construction_threshold == ref.optimal_threshold
    np.testing.assert_allclose(model.weighted_adjacency, W / scale, atol=1e-12)
