"""Threshold-scale normalization in ThemaRS.

Spectral sparsification can reweight cosmic-graph edges above 1.0. The pipeline
rescales by ``max(1, max_weight)`` so the exposed weights and the resolved
construction threshold stay on a single [0, 1] scale — which is also the domain
``find_stable_thresholds`` quantizes over. These tests pin that contract and
confirm the un-sparsified path is left untouched.
"""

import numpy as np
import pandas as pd

from pulsar.config import load_config
from pulsar.pipeline import ThemaRS


def _clustered_frame(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pts = np.vstack(
        [
            rng.normal(c, 0.3, (40, 4))
            for c in ([0, 0, 0, 0], [8, 8, 8, 8], [16, 0, 16, 0])
        ]
    )
    return pd.DataFrame(pts, columns=list("abcd"))


def _config(sparsify: bool = True) -> dict:
    return {
        "preprocessing": {},
        "sweep": {
            "projection": {
                "method": "jl",
                "dimensions": {"values": [2, 3, 4]},
                "seed": {"values": [42, 7, 13]},
            },
            "ball_mapper": {"epsilon": {"range": {"min": 0.5, "max": 2.5, "steps": 8}}},
        },
        "cosmic_graph": {"construction_threshold": "auto", "sparsify": sparsify},
    }


def test_sparsified_weights_above_one_are_normalized():
    model = ThemaRS(load_config(_config(sparsify=True))).fit(data=_clustered_frame())

    # Precondition: sparsification actually pushed raw weights past 1.0, so this
    # exercises the normalization path rather than a no-op.
    raw_max = float(np.array(model.cosmic_rust.weighted_adj).max())
    assert raw_max > 1.0
    assert model._weight_scale == raw_max

    # Exposed weights and the resolved threshold live on the same [0, 1] scale.
    assert model.weighted_adjacency.max() <= 1.0 + 1e-9
    assert 0.0 <= model.resolved_construction_threshold <= 1.0
    edge_weights = [w for _, _, w in model.weighted_edges()]
    assert edge_weights and max(edge_weights) <= 1.0 + 1e-9
    assert all(
        d["weight"] <= 1.0 + 1e-9 for _, _, d in model.cosmic_graph.edges(data=True)
    )


def test_unsparsified_path_weights_unchanged():
    model = ThemaRS(load_config(_config(sparsify=False))).fit(data=_clustered_frame())

    # Dense co-membership weights are already in [0, 1], so scale is a no-op and
    # the exposed adjacency matches the raw Rust weights byte-for-byte.
    assert model._weight_scale == 1.0
    np.testing.assert_array_equal(
        model.weighted_adjacency, np.array(model.cosmic_rust.weighted_adj)
    )
