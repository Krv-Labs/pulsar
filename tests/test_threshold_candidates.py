"""Scale-adaptive ranking of stable-plateau threshold candidates.

These tests exercise the public helpers in ``pulsar.mcp.thresholds`` directly,
using minimal fake plateau objects that mirror the ``PyPlateau`` fields produced
by ``src/ph.rs`` (``start_threshold``, ``end_threshold``, ``component_count``,
``length``, ``midpoint``).

The regression they guard: after spectral sparsification the meaningful plateau is
compressed into a narrow threshold window, so an *absolute*-width persistence scale
under-ranks the true (clean) plateau and lets a wide, singleton-heavy cliff slice
win. With relative-to-active-range persistence, the clean plateau must out-rank the
fragmented one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from pulsar.mcp.thresholds import (
    _active_threshold_range,
    _mass_shape_metrics,
    _rank_plateau_candidate,
    agent_threshold_options,
    component_mass_profile,
    plateau_threshold_candidates,
)


@dataclass(frozen=True)
class FakePlateau:
    """Stand-in for ``ph.rs`` ``PyPlateau`` (constant-count threshold interval).

    Plateaus run high -> low threshold, so ``length = start - end`` is positive,
    matching ``PyPlateau::length``.
    """

    start_threshold: float
    end_threshold: float
    component_count: int

    @property
    def length(self) -> float:
        return self.start_threshold - self.end_threshold

    @property
    def midpoint(self) -> float:
        return (self.start_threshold + self.end_threshold) / 2.0


def _build_two_cluster_adjacency(n_per_cluster: int, n_singletons: int) -> np.ndarray:
    """A weighted graph: two equal dense clusters joined weakly, plus singletons.

    Edge-weight tiers (each tier "breaks" once the threshold exceeds its weight):
    - Within-cluster edges strong (~0.9): survive high thresholds, keeping each
      cluster intact across the whole nontrivial range.
    - Inter-cluster bridge weak (~0.2): breaks above 0.2, so above that the graph
      splits into exactly two nontrivial components.
    - Singleton attachment edges moderate (~0.5): break above 0.5. So:
        * threshold in (0.2, 0.5): two clean clusters, singletons ABSORBED -> the
          clean nontrivial plateau.
        * threshold in (0.5, 0.9): singletons shed AND clusters still split -> a
          wide singleton-heavy fragmented tail.
    """
    n_core = 2 * n_per_cluster
    n = n_core + n_singletons
    adj = np.zeros((n, n), dtype=np.float64)

    # Two dense intra-cluster blocks.
    for offset in (0, n_per_cluster):
        for i in range(offset, offset + n_per_cluster):
            for j in range(i + 1, offset + n_per_cluster):
                adj[i, j] = adj[j, i] = 0.9

    # Weak bridge between the two clusters (breaks above 0.2).
    adj[0, n_per_cluster] = adj[n_per_cluster, 0] = 0.2

    # Singletons attach to cluster 0 via moderate edges (break above 0.5).
    for s in range(n_core, n):
        adj[0, s] = adj[s, 0] = 0.5

    return adj


def test_active_threshold_range_excludes_trivial_cliffs():
    # n_max == 4. Counts of 1 (connected) and 4 (fully fragmented) are trivial.
    thresholds = [0.9, 0.7, 0.5, 0.3, 0.1]
    component_counts = [4, 2, 2, 1, 1]
    # Nontrivial counts (2) occur at thresholds 0.7 and 0.5 -> span 0.2.
    assert _active_threshold_range(thresholds, component_counts) == pytest.approx(0.2)

    # No nontrivial window -> 0.0.
    assert _active_threshold_range([0.9, 0.1], [3, 3]) == 0.0
    assert _active_threshold_range([], []) == 0.0


# Shared "compressed active range" scenario used by the ranking tests.
#
# This models the post-sparsification regime the fix targets: the meaningful
# (nontrivial) threshold window is squeezed into a NARROW band. The clean plateau
# is narrow in absolute width but spans almost the whole active window; the
# fragmented competitor is much WIDER in absolute width (so an absolute-scale
# persistence over-rewards it) yet carries singleton dust.
_N_PER_CLUSTER = 150
_N_SINGLETONS = 20  # modest dust, so the fragmented slice stays a real competitor

# Clean two-cluster plateau: narrow band (~0.02 wide) around midpoint 0.35, where
# singletons are absorbed -> 2 balanced components, zero singletons.
_CLEAN = FakePlateau(start_threshold=0.36, end_threshold=0.34, component_count=2)

# Fragmented plateau: a WIDE high-threshold band (~0.15 wide) around midpoint 0.70,
# where singleton edges have broken. Wider in absolute terms but singleton-tainted.
_FRAGMENTED = FakePlateau(
    start_threshold=0.775,
    end_threshold=0.625,
    component_count=2 + _N_SINGLETONS,
)

# Component curve consistent with the two plateaus. The fragmented count
# (2 + n_singletons) is the maximum, so it is the trivial fragmentation cliff and
# is excluded from the active range. The nontrivial (count == 2) band is confined
# to the narrow clean plateau (0.34-0.36), giving a compressed active range
# (~0.02) -- the regime where absolute-width persistence misfires.
_THRESHOLDS = [0.775, 0.700, 0.625, 0.500, 0.360, 0.350, 0.340, 0.300, 0.100]
_COMPONENT_COUNTS = [
    2 + _N_SINGLETONS,  # 0.775  fragmented cliff (== n_max, trivial)
    2 + _N_SINGLETONS,  # 0.700  fragmented cliff
    2 + _N_SINGLETONS,  # 0.625  fragmented cliff
    2 + _N_SINGLETONS,  # 0.500  singletons still shed
    2,  # 0.360  clean plateau (nontrivial)
    2,  # 0.350  clean plateau (nontrivial)
    2,  # 0.340  clean plateau (nontrivial)
    1,  # 0.300  fully connected (trivial)
    1,  # 0.100  fully connected (trivial)
]


def test_clean_narrow_plateau_outranks_wide_singleton_tail():
    """The clean two-cluster plateau must rank above the wider fragmented tail."""
    adj = _build_two_cluster_adjacency(_N_PER_CLUSTER, _N_SINGLETONS)
    assert _FRAGMENTED.length > _CLEAN.length  # the tail really is absolutely wider

    active_range = _active_threshold_range(_THRESHOLDS, _COMPONENT_COUNTS)
    assert 0.0 < active_range < 0.05  # compressed window: narrower than legacy 0.05

    candidates = plateau_threshold_candidates(
        adj,
        [_FRAGMENTED, _CLEAN],
        policy="balanced",
        max_candidates=5,
        active_range=active_range,
    )

    ranks = {round(c["threshold"], 4): c["rank_within_policy"] for c in candidates}
    clean_rank = ranks[round(_CLEAN.midpoint, 4)]
    frag_rank = ranks[round(_FRAGMENTED.midpoint, 4)]

    assert clean_rank > frag_rank, (
        f"clean plateau ({clean_rank}) should outrank fragmented "
        f"tail ({frag_rank}); candidates={candidates}"
    )
    assert candidates[0]["threshold"] == _CLEAN.midpoint


def test_relative_scaling_flips_outcome_versus_absolute():
    """The relative scale is *load-bearing*: the absolute scale picks the wrong one.

    Under the legacy absolute 0.05 scale the narrow clean plateau (width ~0.02)
    gets low persistence while the wide fragmented plateau (width ~0.15) saturates,
    so the fragmented slice wins despite its singleton dust. Under the new
    relative scale the clean plateau spans most of the compressed active window and
    correctly wins. This test asserts the outcome actually flips -- guarding the
    fix itself, not just the singleton penalty.
    """
    adj = _build_two_cluster_adjacency(_N_PER_CLUSTER, _N_SINGLETONS)
    clean_profile = component_mass_profile(adj, _CLEAN.midpoint, top_k=5)
    clean_metrics = _mass_shape_metrics(clean_profile)
    frag_profile = component_mass_profile(adj, _FRAGMENTED.midpoint, top_k=5)
    frag_metrics = _mass_shape_metrics(frag_profile)

    active_range = _active_threshold_range(_THRESHOLDS, _COMPONENT_COUNTS)

    def rank(plateau: FakePlateau, profile, metrics, *, active: float | None):
        return _rank_plateau_candidate(
            plateau_width=plateau.length,
            profile=profile,
            metrics=metrics,
            policy="balanced",
            active_range=active,
        )

    # active_range=None forces the legacy absolute 0.05 fallback.
    abs_clean = rank(_CLEAN, clean_profile, clean_metrics, active=None)
    abs_frag = rank(_FRAGMENTED, frag_profile, frag_metrics, active=None)
    rel_clean = rank(_CLEAN, clean_profile, clean_metrics, active=active_range)
    rel_frag = rank(_FRAGMENTED, frag_profile, frag_metrics, active=active_range)

    assert abs_frag > abs_clean, (
        "precondition: under the absolute scale the fragmented tail wins "
        f"(clean={abs_clean}, frag={abs_frag})"
    )
    assert rel_clean > rel_frag, (
        "relative scale should flip the result to the clean plateau "
        f"(clean={rel_clean}, frag={rel_frag})"
    )


def test_agent_threshold_options_threads_active_range():
    """End-to-end: agent_threshold_options recommends the clean plateau."""
    adj = _build_two_cluster_adjacency(_N_PER_CLUSTER, _N_SINGLETONS)

    options = agent_threshold_options(
        adj,
        [_FRAGMENTED, _CLEAN],
        _THRESHOLDS,
        _COMPONENT_COUNTS,
        policy="balanced",
    )

    stable = options["stable_plateau_candidates"]
    assert stable, "expected at least one stable plateau candidate"
    # The auto/top stable candidate should be the clean plateau, not the tail.
    assert stable[0]["threshold"] == _CLEAN.midpoint
