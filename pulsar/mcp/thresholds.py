"""Shared threshold mass-dynamics helpers for MCP diagnostics and reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

THRESHOLD_CANDIDATE_POLICIES = {
    "balanced",
    "report_ready",
    "detail_seeking",
    "outlier_mining",
}
_COHORT_READY_TIERS = frozenset({"report_ready", "balanced"})


def first_report_ready_candidate(
    candidates: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Return the first candidate suitable for component-based cohort reading."""
    for candidate in candidates or []:
        if candidate.get("interpretability_tier") in _COHORT_READY_TIERS:
            return candidate
    return None


@dataclass(frozen=True)
class _PreparedThresholdGraph:
    """One-time sparse representation for repeated threshold component queries."""

    dense_adj: np.ndarray
    weighted_csr: csr_matrix
    n_nodes: int
    nnz: int
    density: float

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_nodes, self.n_nodes)


ThresholdGraph = np.ndarray | _PreparedThresholdGraph


def prepare_threshold_graph(
    adj: np.ndarray | _PreparedThresholdGraph,
) -> _PreparedThresholdGraph:
    """Prepare a weighted graph once for repeated threshold scans."""
    if isinstance(adj, _PreparedThresholdGraph):
        return adj
    dense_adj = np.asarray(adj)
    n_nodes = int(dense_adj.shape[0])
    weighted_csr = csr_matrix(dense_adj)
    possible_edges = max(n_nodes * n_nodes, 1)
    return _PreparedThresholdGraph(
        dense_adj=dense_adj,
        weighted_csr=weighted_csr,
        n_nodes=n_nodes,
        nnz=int(weighted_csr.nnz),
        density=round(float(weighted_csr.nnz) / possible_edges, 6),
    )


def prepare_threshold_graph_from_edges(
    n_nodes: int,
    edges: list[tuple[int, int, float]],
) -> _PreparedThresholdGraph:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i, j, weight in edges:
        rows.extend([int(i), int(j)])
        cols.extend([int(j), int(i)])
        data.extend([float(weight), float(weight)])
    weighted_csr = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    possible_edges = max(n_nodes * n_nodes, 1)
    return _PreparedThresholdGraph(
        dense_adj=np.empty((0, 0), dtype=float),
        weighted_csr=weighted_csr,
        n_nodes=n_nodes,
        nnz=int(weighted_csr.nnz),
        density=round(float(weighted_csr.nnz) / possible_edges, 6),
    )


def _dense_component_state_at_threshold(
    adj: np.ndarray,
    threshold: float,
) -> tuple[list[int], np.ndarray]:
    n_nodes = int(adj.shape[0])
    labels = np.full(n_nodes, -1, dtype=np.int32)
    sizes: list[int] = []

    for start in range(n_nodes):
        if labels[start] != -1:
            continue
        label = len(sizes)
        labels[start] = label
        stack = [start]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in np.flatnonzero(adj[node] > threshold):
                if labels[int(neighbor)] == -1:
                    labels[int(neighbor)] = label
                    stack.append(int(neighbor))
        sizes.append(size)

    return sorted(sizes, reverse=True), labels


def component_state_at_threshold(
    adj: ThresholdGraph,
    threshold: float,
) -> tuple[list[int], np.ndarray]:
    if isinstance(adj, _PreparedThresholdGraph):
        if threshold < 0.0:
            return _dense_component_state_at_threshold(adj.dense_adj, threshold)
        graph = adj.weighted_csr > threshold
        _n_components, labels = connected_components(
            graph,
            directed=False,
            return_labels=True,
        )
        labels = np.asarray(labels, dtype=np.int32)
        sizes = np.bincount(labels, minlength=adj.n_nodes).astype(int).tolist()
        sizes = [size for size in sizes if size > 0]
        return sorted(sizes, reverse=True), labels

    return _dense_component_state_at_threshold(np.asarray(adj), threshold)


def _threshold_graph_shape(adj: ThresholdGraph) -> tuple[int, int]:
    return adj.shape


def component_mass_profile(
    adj: ThresholdGraph,
    threshold: float,
    *,
    top_k: int = 8,
    small_component_max_fraction: float = 0.01,
) -> dict[str, Any]:
    sizes, _ = component_state_at_threshold(adj, threshold)
    n_nodes = int(_threshold_graph_shape(adj)[0])
    top_sizes = sizes[:top_k]
    singleton_count = sum(1 for size in sizes if size == 1)
    small_size_limit = max(1, int(n_nodes * small_component_max_fraction))
    small_sizes = [size for size in sizes if size <= small_size_limit]

    return {
        "n_nodes": n_nodes,
        "component_count": len(sizes),
        "top_component_sizes": top_sizes,
        "top_component_sizes_omitted": max(len(sizes) - len(top_sizes), 0),
        "largest_component_fraction": round(
            (top_sizes[0] / n_nodes) if n_nodes and top_sizes else 0.0,
            4,
        ),
        "singleton_count": singleton_count,
        "singleton_fraction": round(singleton_count / n_nodes if n_nodes else 0.0, 4),
        "small_component_count": len(small_sizes),
        "small_component_mass_fraction": round(
            sum(small_sizes) / n_nodes if n_nodes else 0.0,
            4,
        ),
    }


def mass_profile_hint(profile: dict[str, Any]) -> str:
    largest_fraction = float(profile["largest_component_fraction"])
    singleton_fraction = float(profile["singleton_fraction"])
    small_mass_fraction = float(profile["small_component_mass_fraction"])
    component_count = int(profile["component_count"])

    if component_count <= 1:
        return "mostly connected graph"
    if largest_fraction >= 0.95:
        return "mostly connected graph with a small tail"
    if singleton_fraction >= 0.2:
        return "singleton-heavy fragmentation"
    if largest_fraction >= 0.8 and small_mass_fraction >= 0.05:
        return "giant component with small-tail fragmentation"
    if largest_fraction <= 0.8 and singleton_fraction <= 0.05:
        return "stable nontrivial multi-component structure"
    return "mixed component-size structure"


def useful_component_size_floor(n_nodes: int) -> int:
    if n_nodes <= 0:
        return 1
    return max(3, min(round(np.sqrt(n_nodes)), round(0.005 * n_nodes)))


def _mass_shape_metrics(profile: dict[str, Any]) -> dict[str, float | int]:
    n_nodes = int(profile["n_nodes"])
    top_sizes = [int(size) for size in profile["top_component_sizes"]]
    floor = useful_component_size_floor(n_nodes)
    nontrivial_sizes = [size for size in top_sizes if size >= floor]
    nontrivial_mass = sum(nontrivial_sizes)
    multi_component_mass = sum(nontrivial_sizes[1:]) if len(nontrivial_sizes) > 1 else 0

    if top_sizes:
        probs = np.asarray(top_sizes, dtype=float) / max(sum(top_sizes), 1)
        entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        normalized_entropy = (
            entropy / np.log(len(top_sizes)) if len(top_sizes) > 1 else 0.0
        )
    else:
        normalized_entropy = 0.0

    if len(nontrivial_sizes) >= 2:
        a, b = nontrivial_sizes[:2]
        balance = 1.0 - abs(a - b) / max(a + b, 1)
    else:
        balance = 0.0

    return {
        "useful_component_size_floor": floor,
        "nontrivial_component_count": len(nontrivial_sizes),
        "nontrivial_mass_fraction": round(
            nontrivial_mass / n_nodes if n_nodes else 0.0, 4
        ),
        "multi_component_coverage": round(
            multi_component_mass / n_nodes if n_nodes else 0.0, 4
        ),
        "component_size_entropy": round(normalized_entropy, 4),
        "top_two_balance": round(balance, 4),
    }


def threshold_morphology_profile(
    adj: ThresholdGraph,
    threshold: float,
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """Compact mass-distribution row for one threshold cut.

    This is intentionally descriptive, not prescriptive: downstream tools and
    agents decide what to do with the shape in context.
    """
    profile = component_mass_profile(adj, threshold, top_k=top_k)
    metrics = _mass_shape_metrics(profile)
    top_sizes = [int(size) for size in profile["top_component_sizes"]]
    giant_size = top_sizes[0] if top_sizes else 0
    second_size = top_sizes[1] if len(top_sizes) > 1 else 0
    second_largest_ratio = second_size / giant_size if giant_size else 0.0
    return {
        "threshold": float(threshold),
        "component_count": int(profile["component_count"]),
        "top_component_sizes": top_sizes,
        "top_component_sizes_omitted": profile["top_component_sizes_omitted"],
        "giant_fraction": profile["largest_component_fraction"],
        "second_largest_ratio": round(second_largest_ratio, 4),
        "singleton_count": int(profile["singleton_count"]),
        "singleton_fraction": profile["singleton_fraction"],
        "small_component_mass_fraction": profile["small_component_mass_fraction"],
        "nontrivial_component_count": int(metrics["nontrivial_component_count"]),
        "nontrivial_mass_fraction": metrics["nontrivial_mass_fraction"],
        "multi_component_coverage": metrics["multi_component_coverage"],
        "component_size_entropy": metrics["component_size_entropy"],
        "interpretation_hint": mass_profile_hint(profile),
    }


def _plateau_tier(profile: dict[str, Any], metrics: dict[str, float | int]) -> str:
    largest = float(profile["largest_component_fraction"])
    singletons = float(profile["singleton_fraction"])
    nontrivial_count = int(metrics["nontrivial_component_count"])
    nontrivial_mass = float(metrics["nontrivial_mass_fraction"])

    if singletons >= 0.35:
        return "outlier_frontier"
    if largest >= 0.95:
        return "giant_component_with_dust"
    if nontrivial_count >= 2 and singletons <= 0.05 and nontrivial_mass >= 0.8:
        return "report_ready"
    if nontrivial_count >= 2 and singletons <= 0.2 and nontrivial_mass >= 0.55:
        return "balanced"
    if nontrivial_count >= 2:
        return "exploratory"
    return "weak_candidate"


# Fraction of the active (nontrivial) threshold range a plateau must span to be
# considered fully "persistent". A plateau covering >= this fraction of the active
# range saturates persistence at 1.0. Chosen at 0.25 so that roughly a quarter of
# the meaningful threshold window is "very stable" -- this keeps persistence
# scale-relative instead of tied to an absolute width, so spectral sparsification
# (which compresses the meaningful window) no longer under-ranks the true plateau.
_PERSISTENCE_ACTIVE_RANGE_FRACTION = 0.25


def _active_threshold_range(
    thresholds: list[float],
    component_counts: list[int],
) -> float:
    """Span of thresholds over which the component count is nontrivial.

    "Nontrivial" mirrors ``ph.rs`` optimal-threshold selection: a count strictly
    between 1 (fully connected) and ``n`` (fully fragmented). The active range is
    the width of the threshold window where ``1 < component_count < n``; the
    cliff regions where everything is connected or everything is a singleton are
    excluded. Returns 0.0 if no such window exists.
    """
    if not thresholds or not component_counts:
        return 0.0
    n_max = max(int(c) for c in component_counts)
    nontrivial = [
        float(t)
        for t, c in zip(thresholds, component_counts, strict=False)
        if 1 < int(c) < n_max
    ]
    if not nontrivial:
        return 0.0
    return abs(max(nontrivial) - min(nontrivial))


def _rank_plateau_candidate(
    *,
    plateau_width: float,
    profile: dict[str, Any],
    metrics: dict[str, float | int],
    policy: str,
    active_range: float | None = None,
) -> float:
    # Persistence is the plateau width as a fraction of the active threshold range
    # (the window where structure is nontrivial), not a fixed absolute width. This
    # keeps ranking scale-adaptive: after sparsification the active window narrows,
    # so a clean-but-narrow plateau still scores as persistent. Falls back to the
    # legacy absolute scale (0.05) only when the active range is unavailable/zero.
    eps = 1e-9
    if active_range and active_range > eps:
        denom = max(float(active_range) * _PERSISTENCE_ACTIVE_RANGE_FRACTION, eps)
    else:
        denom = 0.05
    persistence = min(max(float(plateau_width) / denom, 0.0), 1.0)
    balance = float(metrics["top_two_balance"])
    coverage = float(metrics["nontrivial_mass_fraction"])
    entropy = float(metrics["component_size_entropy"])
    singletons = float(profile["singleton_fraction"])
    small_mass = float(profile["small_component_mass_fraction"])

    if policy == "report_ready":
        penalty = 0.55 * singletons + 0.25 * small_mass
        score = 0.35 * persistence + 0.25 * balance + 0.3 * coverage + 0.1 * entropy
    elif policy == "detail_seeking":
        penalty = 0.2 * singletons + 0.1 * small_mass
        score = 0.25 * persistence + 0.3 * balance + 0.25 * coverage + 0.2 * entropy
    elif policy == "outlier_mining":
        penalty = 0.05 * small_mass
        score = (
            0.15 * persistence
            + 0.1 * balance
            + 0.2 * coverage
            + 0.35 * entropy
            + 0.2 * singletons
        )
    else:
        penalty = 0.35 * singletons + 0.15 * small_mass
        score = 0.3 * persistence + 0.25 * balance + 0.3 * coverage + 0.15 * entropy

    return round(max(score - penalty, 0.0), 4)


def _candidate_use_guidance(
    tier: str, singleton_fraction: float
) -> tuple[list[str], list[str], str]:
    if tier == "report_ready":
        return (
            ["report_ready", "balanced"],
            ["outlier_mining"],
            "Candidate has broad nontrivial coverage with low singleton residual.",
        )
    if tier == "balanced":
        return (
            ["balanced", "detail_seeking"],
            [],
            "Candidate exposes nontrivial components with manageable residual mass.",
        )
    if tier == "exploratory":
        return (
            ["detail_seeking"],
            ["report_ready"],
            "Candidate may expose detailed archetypes, but needs caution before final naming.",
        )
    if tier == "outlier_frontier" or singleton_fraction >= 0.35:
        return (
            ["outlier_mining"],
            ["report_ready"],
            "Candidate is singleton-rich; use for anomaly or frontier discovery, not final cohort reporting.",
        )
    return (
        ["detail_seeking"],
        ["report_ready"],
        "Candidate has weak cohort structure; treat as a diagnostic lens only.",
    )


def _active_range_from_plateaus(plateaus: list[Any], n_nodes: int) -> float:
    """Approximate the active (nontrivial) threshold range from plateaus alone.

    Used when explicit ``thresholds``/``component_counts`` are not threaded in.
    Each plateau covers ``[end_threshold, start_threshold]`` with a constant
    ``component_count``; we take the union span of plateaus whose count is
    nontrivial (``1 < count < n``). This is an approximation of the per-threshold
    computation in ``_active_threshold_range`` -- the fuller approach is to pass
    the raw ``thresholds``/``component_counts`` (see ``agent_threshold_options``),
    which avoids relying on plateau boundaries lining up with the true window.
    """
    bounds: list[float] = []
    for plateau in plateaus:
        count = int(plateau.component_count)
        if 1 < count < max(n_nodes, 2):
            bounds.append(float(plateau.start_threshold))
            bounds.append(float(plateau.end_threshold))
    if not bounds:
        return 0.0
    return abs(max(bounds) - min(bounds))


def plateau_threshold_candidates(
    adj: ThresholdGraph,
    plateaus: list[Any],
    *,
    policy: str = "balanced",
    max_candidates: int = 3,
    active_range: float | None = None,
) -> list[dict[str, Any]]:
    if policy not in THRESHOLD_CANDIDATE_POLICIES:
        raise ValueError(f"unknown threshold candidate policy: {policy}")

    n_nodes = int(_threshold_graph_shape(adj)[0])
    if active_range is None:
        active_range = _active_range_from_plateaus(plateaus, n_nodes)

    candidates = []
    for plateau in plateaus:
        midpoint = float(plateau.midpoint)
        width = float(plateau.length)
        profile = component_mass_profile(adj, midpoint, top_k=5)
        metrics = _mass_shape_metrics(profile)
        tier = _plateau_tier(profile, metrics)
        rank = _rank_plateau_candidate(
            plateau_width=width,
            profile=profile,
            metrics=metrics,
            policy=policy,
            active_range=active_range,
        )
        best_for, avoid_for, why = _candidate_use_guidance(
            tier,
            float(profile["singleton_fraction"]),
        )
        candidates.append(
            {
                "candidate_kind": "stable_plateau",
                "threshold": midpoint,
                "plateau_start": float(plateau.start_threshold),
                "plateau_end": float(plateau.end_threshold),
                "persistence_width": round(width, 4),
                "component_count": int(plateau.component_count),
                "interpretability_tier": tier,
                "rank_within_policy": rank,
                "component_mass_profile": profile,
                "mass_shape": metrics,
                "best_for": best_for,
                "avoid_for": avoid_for,
                "why": why,
            }
        )

    candidates.sort(key=lambda row: float(row["rank_within_policy"]), reverse=True)
    return candidates[:max_candidates]


def _transition_candidate_tier(
    profile: dict[str, Any], metrics: dict[str, float | int]
) -> str:
    singletons = float(profile["singleton_fraction"])
    multi_coverage = float(metrics["multi_component_coverage"])
    balance = float(metrics["top_two_balance"])
    nontrivial_count = int(metrics["nontrivial_component_count"])

    if singletons >= 0.35:
        return "outlier_frontier"
    if nontrivial_count >= 2 and multi_coverage >= 0.2 and balance >= 0.35:
        return "exploratory"
    if nontrivial_count >= 2 and multi_coverage >= 0.05:
        return "detail_lens"
    return "weak_candidate"


def _rank_transition_candidate(
    *,
    profile: dict[str, Any],
    metrics: dict[str, float | int],
    event: dict[str, Any],
    policy: str,
) -> float:
    multi_coverage = float(metrics["multi_component_coverage"])
    balance = float(metrics["top_two_balance"])
    entropy = float(metrics["component_size_entropy"])
    singletons = float(profile["singleton_fraction"])
    absorbed = float(event.get("absorbed_mass_fraction", 0.0))

    if policy == "outlier_mining":
        score = (
            0.3 * entropy + 0.25 * absorbed + 0.25 * multi_coverage + 0.2 * singletons
        )
        penalty = 0.05 * float(profile["small_component_mass_fraction"])
    elif policy == "detail_seeking":
        score = 0.3 * multi_coverage + 0.25 * balance + 0.25 * absorbed + 0.2 * entropy
        penalty = 0.12 * singletons
    else:
        score = (
            0.35 * multi_coverage + 0.25 * balance + 0.25 * absorbed + 0.15 * entropy
        )
        penalty = 0.35 * singletons

    return round(max(score - penalty, 0.0), 4)


def transition_adjacent_candidates(
    adj: ThresholdGraph,
    thresholds: list[float],
    component_counts: list[int],
    *,
    policy: str = "balanced",
    max_candidates: int = 3,
) -> list[dict[str, Any]]:
    if policy not in THRESHOLD_CANDIDATE_POLICIES:
        raise ValueError(f"unknown threshold candidate policy: {policy}")

    events = structural_breakpoints(
        adj,
        thresholds,
        component_counts,
        max_breakpoints=max(max_candidates * 3, 6),
        max_candidates=32,
    )
    candidates: list[dict[str, Any]] = []
    for event in events:
        if (
            event["event"] == "small_component_absorption"
            and policy != "outlier_mining"
        ):
            continue
        threshold = float(event["threshold"])
        profile = component_mass_profile(adj, threshold, top_k=5)
        metrics = _mass_shape_metrics(profile)
        tier = _transition_candidate_tier(profile, metrics)
        rank = _rank_transition_candidate(
            profile=profile,
            metrics=metrics,
            event=event,
            policy=policy,
        )
        best_for, avoid_for, why = _candidate_use_guidance(
            tier,
            float(profile["singleton_fraction"]),
        )
        if policy == "detail_seeking" and tier in {"detail_lens", "exploratory"}:
            best_for = sorted(set(best_for + ["detail_seeking"]))
        if policy == "outlier_mining":
            best_for = sorted(set(best_for + ["outlier_mining"]))
        candidates.append(
            {
                "candidate_kind": "transition_adjacent",
                "threshold": threshold,
                "side": "stricter_before_merge",
                "event": event["event"],
                "component_count_before": event["component_count_before"],
                "component_count_after": event["component_count_after"],
                "top_component_sizes": profile["top_component_sizes"],
                "singleton_fraction": profile["singleton_fraction"],
                "absorbed_mass_fraction": event["absorbed_mass_fraction"],
                "resulting_component_fraction": event["resulting_component_fraction"],
                "interpretability_tier": tier,
                "sort_key": rank,
                "mass_shape": metrics,
                "best_for": best_for,
                "avoid_for": avoid_for,
                "why": why,
            }
        )

    candidates.sort(key=lambda row: float(row["sort_key"]), reverse=True)
    return candidates[:max_candidates]


def _policy_caps(policy: str, max_candidates: int) -> tuple[int, int]:
    if policy == "report_ready":
        return min(max_candidates, 3), min(max_candidates, 1)
    if policy == "detail_seeking":
        return 1, max_candidates
    if policy == "outlier_mining":
        return 1, max_candidates
    return min(max_candidates, 2), min(max_candidates, 4)


def _deduplicate_candidates(
    candidates: list[dict[str, Any]],
    n_nodes: int,
) -> list[dict[str, Any]]:
    """Filter out candidates that are topologically or threshold-wise redundant."""
    if not candidates:
        return []

    def _get_largest_frac(c: dict[str, Any]) -> float:
        if "component_mass_profile" in c:
            return float(c["component_mass_profile"]["largest_component_fraction"])
        if "top_component_sizes" in c and c["top_component_sizes"]:
            return float(c["top_component_sizes"][0]) / max(n_nodes, 1)
        return 0.0

    def _get_component_count(c: dict[str, Any]) -> int:
        if "component_count" in c:
            return int(c["component_count"])
        if "component_count_after" in c:
            return int(c["component_count_after"])
        return 1

    unique_list: list[dict[str, Any]] = []
    for cand in candidates:
        t_cand = float(cand["threshold"])
        f_cand = _get_largest_frac(cand)
        cc_cand = _get_component_count(cand)
        tier_cand = cand.get("interpretability_tier", "")

        is_redundant = False
        for existing in unique_list:
            t_exist = float(existing["threshold"])
            f_exist = _get_largest_frac(existing)
            cc_exist = _get_component_count(existing)
            tier_exist = existing.get("interpretability_tier", "")

            # Rule 1: Very close thresholds (practically the same threshold)
            if abs(t_cand - t_exist) < 0.03:
                is_redundant = True
                break

            # Rule 2: Both are giant-dominated (largest component has >= 90% of nodes).
            # No need to show multiple giant-dominated candidates as separate options.
            if f_cand >= 0.90 and f_exist >= 0.90:
                is_redundant = True
                break

            # Rule 3: Very similar topology and tier
            if tier_cand == tier_exist and abs(f_cand - f_exist) < 0.05:
                # If component count is very similar
                if (
                    abs(cc_cand - cc_exist) <= 2
                    or abs(cc_cand - cc_exist) / max(cc_cand, cc_exist, 1) < 0.15
                ):
                    is_redundant = True
                    break

        if not is_redundant:
            unique_list.append(cand)

    return unique_list


def agent_threshold_options(
    adj: ThresholdGraph,
    plateaus: list[Any],
    thresholds: list[float],
    component_counts: list[int],
    *,
    policy: str = "balanced",
    max_candidates: int = 5,
) -> dict[str, Any]:
    n_nodes = int(_threshold_graph_shape(adj)[0])
    stable_cap, transition_cap = _policy_caps(policy, max_candidates)
    # Exact active (nontrivial) range from the raw component curve; threaded into
    # plateau ranking so persistence is relative to the meaningful window.
    active_range = _active_threshold_range(thresholds, component_counts)
    stable_candidates = plateau_threshold_candidates(
        adj,
        plateaus,
        policy=policy,
        max_candidates=stable_cap,
        active_range=active_range,
    )
    transition_candidates = transition_adjacent_candidates(
        adj,
        thresholds,
        component_counts,
        policy=policy,
        max_candidates=transition_cap,
    )

    # Pre-deduplicate individual lists for cleaner sub-components
    stable_candidates = _deduplicate_candidates(stable_candidates, n_nodes)
    transition_candidates = _deduplicate_candidates(transition_candidates, n_nodes)

    if policy == "detail_seeking":
        candidates = transition_candidates + [
            row
            for row in stable_candidates
            if row["interpretability_tier"]
            not in {"giant_component_with_dust", "weak_candidate"}
        ]
        strategy = "transition_adjacent"
    elif policy == "outlier_mining":
        candidates = transition_candidates + stable_candidates[:1]
        strategy = "transition_adjacent"
    elif policy == "report_ready":
        candidates = [
            row
            for row in stable_candidates
            if row["interpretability_tier"] in {"report_ready", "balanced"}
        ]
        strategy = "stable_plateau"
    else:
        useful_stable = [
            row
            for row in stable_candidates
            if row["interpretability_tier"]
            not in {"giant_component_with_dust", "weak_candidate"}
        ]
        candidates = useful_stable or transition_candidates[:2] or stable_candidates[:1]
        strategy = "stable_plateau" if useful_stable else "transition_adjacent"

    # Post-deduplicate merged recommendations list to ensure distinct options are presented
    candidates = _deduplicate_candidates(candidates, n_nodes)
    candidates = candidates[:max_candidates]
    auto = stable_candidates[0] if stable_candidates else None
    if auto is None:
        readout = "No stable threshold candidates were available."
        action = "Run a sweep before interpreting threshold options."
    elif auto["interpretability_tier"] == "giant_component_with_dust":
        largest_pct = (
            float(auto["component_mass_profile"]["largest_component_fraction"]) * 100
        )
        readout = (
            f"Auto/stable threshold is giant-dominated: {largest_pct:.2f}% of rows "
            "sit in the largest component."
        )
        action = "Use alternatives as research lenses; do not name final cohorts from a dusty giant component alone."
    elif auto["interpretability_tier"] in {"report_ready", "balanced"}:
        readout = (
            "Stable threshold candidates include nontrivial multi-component structure."
        )
        action = "Use the report-ready or balanced candidate for cohort naming, then validate features."
    else:
        readout = (
            "Stable threshold candidates are exploratory rather than report-ready."
        )
        action = (
            "Use candidates for detailed or outlier analysis, not final cohort naming."
        )

    return {
        "policy": policy,
        "selection_strategy": strategy,
        "auto_readout": readout,
        "recommended_action": action,
        "stable_plateau_candidates": stable_candidates,
        "transition_adjacent_candidates": transition_candidates,
        "stable_plateau_candidates_omitted": max(
            len(plateaus) - len(stable_candidates),
            0,
        ),
        "transition_adjacent_candidates_omitted": 0,
        "candidates": candidates,
    }


def structural_breakpoints(
    adj: ThresholdGraph,
    thresholds: list[float],
    component_counts: list[int],
    *,
    max_breakpoints: int = 5,
    max_candidates: int = 16,
) -> list[dict[str, Any]]:
    n_nodes = int(_threshold_graph_shape(adj)[0])
    if n_nodes == 0:
        return []

    states: dict[float, tuple[list[int], np.ndarray]] = {}

    def _state(threshold: float) -> tuple[list[int], np.ndarray]:
        if threshold not in states:
            states[threshold] = component_state_at_threshold(adj, threshold)
        return states[threshold]

    transitions = [
        {
            "index": i,
            "count_delta": abs(int(component_counts[i - 1]) - int(component_counts[i])),
            "threshold_gap": abs(float(thresholds[i - 1]) - float(thresholds[i])),
        }
        for i in range(1, len(thresholds))
        if int(component_counts[i - 1]) != int(component_counts[i])
    ]
    if len(transitions) > max_candidates:
        by_count = sorted(
            transitions,
            key=lambda row: (row["count_delta"], row["threshold_gap"]),
            reverse=True,
        )
        by_gap = sorted(
            transitions,
            key=lambda row: (row["threshold_gap"], row["count_delta"]),
            reverse=True,
        )
        selected_indices = {row["index"] for row in by_count[: max_candidates // 2]} | {
            row["index"] for row in by_gap[: max_candidates // 2]
        }
    else:
        selected_indices = {row["index"] for row in transitions}

    candidates: list[dict[str, Any]] = []
    for i in sorted(selected_indices):
        before_threshold = float(thresholds[i - 1])
        after_threshold = float(thresholds[i])
        before_count = int(component_counts[i - 1])
        after_count = int(component_counts[i])

        before_sizes, before_labels = _state(before_threshold)
        after_sizes, after_labels = _state(after_threshold)
        before_size_by_label = np.bincount(before_labels)

        transition_mass = 0
        absorbed_mass = 0
        merged_prior_sizes: list[int] = []
        for after_label in np.unique(after_labels):
            members = after_labels == after_label
            prior_labels = np.unique(before_labels[members])
            if len(prior_labels) <= 1:
                continue
            prior_sizes = sorted(
                (int(before_size_by_label[int(label)]) for label in prior_labels),
                reverse=True,
            )
            mass = int(sum(prior_sizes))
            smaller_mass = int(sum(prior_sizes[1:]))
            if smaller_mass > absorbed_mass:
                transition_mass = mass
                absorbed_mass = smaller_mass
                merged_prior_sizes = prior_sizes

        if absorbed_mass == 0:
            continue

        absorbed_fraction = absorbed_mass / n_nodes
        resulting_fraction = transition_mass / n_nodes
        second_prior_fraction = (
            merged_prior_sizes[1] / n_nodes if len(merged_prior_sizes) > 1 else 0.0
        )
        if absorbed_fraction >= 0.05 and second_prior_fraction >= 0.05:
            event = "large_component_transition"
            hint = (
                "Major structural transition; lowering the threshold merges large "
                "components, while raising it splits them."
            )
        elif after_count < before_count and second_prior_fraction < 0.01:
            event = "small_component_absorption"
            hint = "Mostly small components or singletons joining larger structure."
        else:
            event = "mixed_component_transition"
            hint = "Component count changes with moderate mass redistribution."

        candidates.append(
            {
                "threshold": after_threshold,
                "event": event,
                "component_count_before": before_count,
                "component_count_after": after_count,
                "before_top_component_sizes": before_sizes[:5],
                "after_top_component_sizes": after_sizes[:5],
                "absorbed_mass_fraction": round(absorbed_fraction, 4),
                "resulting_component_fraction": round(resulting_fraction, 4),
                "affected_mass_fraction": round(absorbed_fraction, 4),
                "interpretation_hint": hint,
            }
        )

    priority = {
        "large_component_transition": 0,
        "mixed_component_transition": 1,
        "small_component_absorption": 2,
    }
    candidates.sort(
        key=lambda row: (
            priority.get(str(row["event"]), 9),
            -float(row["absorbed_mass_fraction"]),
        )
    )
    non_absorption = [
        row for row in candidates if row["event"] != "small_component_absorption"
    ]
    if non_absorption:
        return non_absorption[:max_breakpoints]
    return candidates[:max_breakpoints]
