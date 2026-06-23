"""Ground-truth validation for threshold stability on known-structure data.

Regression guard for the failure mode that motivated the trivial-plateau guard
(`src/ph.rs`), the [0,1] weight normalization, and the lower default
`sparsify_epsilon`: on sparsified graphs the auto construction threshold could
land in the all-singleton tail, and the exposed graph could fragment into
near-all-singletons (e.g. 151 components / 93 singletons on 344 rows).

These tests assert the *qualitative* guarantees that were violated — a
nontrivial, low-singleton auto-threshold graph — rather than an exact k, which
legitimately depends on data geometry (softly-overlapping groups merge).
"""

import networkx as nx
import numpy as np
import pandas as pd

from pulsar.config import load_config
from pulsar.pipeline import ThemaRS


def _structured_frame(seed: int = 0) -> pd.DataFrame:
    """One well-separated group plus a softly-overlapping pair (320 rows)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        np.vstack(
            [
                rng.normal([0, 0, 0, 0], 0.6, (120, 4)),
                rng.normal([2.2, 2.2, 0, 0], 0.6, (110, 4)),  # overlaps group 1
                rng.normal([8, 8, 8, 8], 0.6, (90, 4)),  # well separated
            ]
        ),
        columns=list("abcd"),
    )


def _fit(*, eps: float, sparsify: bool, seed: int = 1) -> ThemaRS:
    cfg = load_config(
        {
            "preprocessing": {},
            "sweep": {
                "projection": {
                    "method": "jl",
                    "dimensions": {"values": [3, 4]},
                    "seed": {"values": [42, 7]},
                },
                "ball_mapper": {
                    "epsilon": {"range": {"min": 0.5, "max": 2.5, "steps": 8}}
                },
            },
            "cosmic_graph": {
                "construction_threshold": "auto",
                "sparsify": sparsify,
                "sparsify_epsilon": eps,
                "sparsify_seed": seed,
            },
        }
    )
    return ThemaRS(cfg).fit(data=_structured_frame())


def _graph_health(model: ThemaRS) -> tuple[int, int, float]:
    """(n_components, n_nodes, singleton_fraction) of the exposed cosmic graph."""
    g = model.cosmic_graph
    n = g.number_of_nodes()
    singletons = sum(1 for _, deg in g.degree() if deg == 0)
    return nx.number_connected_components(g), n, singletons / n


def test_sparsified_auto_threshold_is_nontrivial_and_low_singleton():
    """The guard prevents the all-singleton-tail auto-threshold; the exposed
    graph is nontrivial with a low singleton residual, not a fragmentation cliff."""
    ncomp, n, singleton_frac = _graph_health(_fit(eps=0.3, sparsify=True))
    assert 1 < ncomp < n  # neither one blob nor all singletons
    assert singleton_frac < 0.05
    assert ncomp <= 10  # coarse structure recovered, not 100+ noise components


def test_dense_path_low_singleton():
    """The un-sparsified path is equally well-behaved."""
    ncomp, n, singleton_frac = _graph_health(_fit(eps=1.0, sparsify=False))
    assert 1 < ncomp < n
    assert singleton_frac < 0.05


def test_no_fragmentation_explosion_on_sparsified_path():
    """When spectral sparsification is enabled, the auto-threshold never explodes
    into a singleton-dominated graph (the original bug), and it is seed-stable
    across the random sparsification seed."""
    for seed in (1, 2, 3):
        ncomp, n, singleton_frac = _graph_health(
            _fit(eps=1.0, sparsify=True, seed=seed)
        )
        assert singleton_frac < 0.05, f"seed={seed}: {singleton_frac:.2%}"
        assert ncomp < n
