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
    first_report_ready_candidate,
    plateau_threshold_candidates,
    prepare_threshold_graph_from_edges,
    significant_component_sizes,
    useful_component_size_floor,
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


def test_balanced_options_exclude_near_singleton_frontier_lenses():
    n = 100
    edges = _ring_edges(0, 49, 0.4)
    edges += _ring_edges(49, 98, 0.4)
    edges.append((98, 99, 0.98))
    graph = prepare_threshold_graph_from_edges(n, edges)

    singleton_frontier = FakePlateau(
        start_threshold=1.0,
        end_threshold=0.95,
        component_count=99,
    )
    clean = FakePlateau(
        start_threshold=0.45,
        end_threshold=0.25,
        component_count=3,
    )
    thresholds = [1.0, 0.98, 0.95, 0.45, 0.35, 0.25, 0.0]
    component_counts = [100, 99, 99, 3, 3, 3, 1]

    balanced = agent_threshold_options(
        graph,
        [singleton_frontier, clean],
        thresholds,
        component_counts,
        policy="balanced",
    )
    assert all(
        float(candidate["component_mass_profile"]["singleton_fraction"]) < 0.95
        for candidate in balanced["candidates"]
        if "component_mass_profile" in candidate
    )
    assert all(
        candidate["threshold"] != singleton_frontier.midpoint
        for candidate in balanced["candidates"]
    )

    assert any(
        candidate["threshold"] == singleton_frontier.midpoint
        for candidate in balanced["stable_plateau_candidates"]
    )


# --- significant_component_sizes: relative, gap-based significance --------------
#
# Replaces the old absolute size-floor (max(3, min(sqrt(n), 0.5%*n))) as the test
# for "is this component a real mode?". Significance is now relative to the size
# *distribution* at the slice: a component counts when a clear multiplicative gap
# separates the head modes from the dust tail. The motivating failure: a real
# ~101-node minority cohort in a large (100k) graph fell below the 316-node floor
# and was silently discarded. These cases pin the contract and its two knobs
# (DOMINANCE=0.5, CLIFF_RATIO=3.0).


def test_significance_rescues_minority_mode_under_giant():
    # The headline regression: giant (83%) + a clearly gap-separated 101-node
    # cohort (101 -> 8 is a >10x cliff) + dust. The old n-floor (316 at n=100k)
    # dropped the 101; gap-based significance keeps it.
    sizes = [83000, 101, 8, 8, 5, 4]
    significant, _ = significant_component_sizes(sizes)
    assert significant == [83000, 101]


def test_significance_drops_pure_dust_under_giant():
    # Same giant, but the tail is a smooth low ramp with no cliff: all dust.
    sizes = [83000, 8, 8, 5, 4]
    significant, _ = significant_component_sizes(sizes)
    assert significant == [83000]


def test_significance_treats_smooth_tail_as_dust():
    # No internal cliff among the non-giant components -> no secondary mode,
    # even though each tail component is well above any tiny absolute floor.
    sizes = [83000, 50, 48, 45, 43, 40]
    significant, _ = significant_component_sizes(sizes)
    assert significant == [83000]


def test_significance_keeps_coequal_clusters():
    # No dominant blob (largest fraction < 0.5): genuine balanced multi-cluster
    # structure -> every cluster is a real mode.
    sizes = [200, 200, 200]
    significant, _ = significant_component_sizes(sizes)
    assert significant == [200, 200, 200]


def test_significance_cuts_dust_in_coequal_regime():
    # Co-equal head with a dust tail: the clean 200 -> 4 cliff separates modes
    # from dust even when there is no single dominant component.
    sizes = [200, 200, 200, 4, 4, 4]
    significant, _ = significant_component_sizes(sizes)
    assert significant == [200, 200, 200]


def test_significance_keeps_multiple_minority_modes():
    sizes = [83000, 500, 480, 7, 6]
    significant, _ = significant_component_sizes(sizes)
    assert significant == [83000, 500, 480]


def test_significance_single_giant_is_one_component():
    significant, _ = significant_component_sizes([14010])
    assert significant == [14010]


def test_significance_noise_gate_drops_specks_and_singletons():
    # Tiny specks (size < 3) never count, regardless of how many there are.
    sizes = [500, 2, 1, 1, 1]
    significant, _ = significant_component_sizes(sizes)
    assert significant == [500]


def test_significance_empty_input():
    significant, reason = significant_component_sizes([])
    assert significant == []
    assert isinstance(reason, str)


# --- end-to-end: minority mode at scale on the real PH + candidate pipeline ----


def _ring_edges(start, stop, weight, degree=4):
    """Sparse high-weight ring+skip connectivity over nodes [start, stop)."""
    edges, size = [], stop - start
    for off in range(start, stop):
        for d in range(1, degree + 1):
            edges.append((off, start + ((off - start + d) % size), weight))
    return edges


def test_minority_mode_surfaces_at_scale_on_real_pipeline():
    """A 101-node minority cohort, gap-separated from dust, must surface as a clean
    component lens in a ~25k-node graph -- a scale where the OLD absolute floor
    (max(3, min(sqrt(n), 0.005*n))) exceeds 101 and would have discarded it.

    Drives the real find_stable_thresholds_sparse + agent_threshold_options path
    (not synthetic metrics), so it guards the whole handoff, not just the helper.
    """
    from pulsar._pulsar import find_stable_thresholds_sparse

    n_total, n_minority = 25000, 101
    n_giant = int(n_total * 0.78)  # dominant but < 95%, so a mode can lift it to a lens

    edges = _ring_edges(0, n_giant, 0.9)
    edges += _ring_edges(n_giant, n_giant + n_minority, 0.9)  # intact minority
    edges.append((0, n_giant, 0.15))  # weak giant<->minority bridge
    nxt = n_giant + n_minority
    while nxt + 1 < n_total:  # weak dust pairs fill the remainder
        edges += [(0, nxt, 0.1), (nxt, nxt + 1, 0.9)]
        nxt += 2

    # Regression boundary: the legacy floor would have dropped the 101-node mode.
    assert useful_component_size_floor(nxt) > n_minority

    tg = prepare_threshold_graph_from_edges(nxt, edges)
    stability = find_stable_thresholds_sparse(nxt, edges)
    thresholds = [float(t) for t in stability.thresholds]
    component_counts = [int(c) for c in stability.component_counts]

    options = agent_threshold_options(
        tg,
        stability.top_k_plateaus(20),
        thresholds,
        component_counts,
        policy="balanced",
    )
    clean = first_report_ready_candidate(options["candidates"])
    assert clean is not None, "expected a report_ready/balanced component lens"

    profile = component_mass_profile(tg, clean["threshold"], top_k=8)
    significant, _ = significant_component_sizes(profile["top_component_sizes"])
    assert any(abs(size - n_minority) <= 1 for size in significant), (
        f"the {n_minority}-node minority must be a significant component; "
        f"got {significant}"
    )
