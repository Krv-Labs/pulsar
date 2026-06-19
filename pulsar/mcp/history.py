"""
Session sweep history synthesis for Pulsar MCP.

Pure rule-based pattern detection over recorded SweepRecords. Returns
observations + rationale only — no recommended-config field, by design, to
avoid fabricating numbers and to keep agent decisions explicit.
"""

from __future__ import annotations

import math
from typing import Any

import yaml


_HAIRBALL_DENSITY = 0.8
_SPARSE_DENSITY = 0.05
_FRAGMENTED_SINGLETON_FRACTION = 0.25
_DOMINANT_GIANT = 0.95
_ICE_CHIP_NONTRIVIAL_DELTA = 0.02
_MEANINGFUL_NONTRIVIAL_DELTA = 0.03
_SINGLETON_SPIKE_DELTA = 0.05


def _extract_pca_dims(config_yaml: str) -> list[int]:
    try:
        cfg = yaml.safe_load(config_yaml) or {}
    except yaml.YAMLError:
        return []
    pca = cfg.get("sweep", {}).get("pca", {}).get("dimensions", {})
    values = pca.get("values", [])
    if isinstance(values, list):
        return [int(v) for v in values if isinstance(v, (int, float))]
    return []


def _extract_construction_threshold(config_yaml: str) -> str | float | None:
    try:
        cfg = yaml.safe_load(config_yaml) or {}
    except yaml.YAMLError:
        return None
    return cfg.get("cosmic_graph", {}).get("construction_threshold")


def _health_label(metrics: dict[str, Any]) -> str:
    density = float(metrics.get("density", 0.0))
    singleton_fraction = float(metrics.get("singleton_fraction", 0.0))
    giant_fraction = float(metrics.get("giant_fraction", 0.0))
    n_edges = int(metrics.get("n_edges", 0))
    if n_edges == 0:
        return "empty"
    if density > _HAIRBALL_DENSITY:
        return "hairball"
    if singleton_fraction > _FRAGMENTED_SINGLETON_FRACTION:
        return "fragmented"
    if density < _SPARSE_DENSITY:
        return "sparse"
    if giant_fraction > _DOMINANT_GIANT:
        return "giant_dominant"
    return "connected"


def _component_profile(metrics: dict[str, Any]) -> dict[str, Any]:
    sizes = [int(size) for size in metrics.get("component_sizes", []) if int(size) > 0]
    n_total = int(metrics.get("n_nodes", 0) or sum(sizes))
    if n_total <= 0 or not sizes:
        return {
            "n_nodes": n_total,
            "component_count": int(metrics.get("component_count", 0) or 0),
            "giant_fraction": float(metrics.get("giant_fraction", 0.0)),
            "non_giant_mass": 0.0,
            "nontrivial_component_count": 0,
            "nontrivial_component_mass": 0.0,
            "largest_non_giant_component_pct": 0.0,
            "singleton_fraction": float(metrics.get("singleton_fraction", 0.0)),
            "small_component_mass": 0.0,
        }

    sizes = sorted(sizes, reverse=True)
    giant_size = sizes[0]
    tail = sizes[1:]
    nontrivial_min = max(10, math.ceil(n_total * 0.01))
    nontrivial = [size for size in tail if size >= nontrivial_min]
    small = [size for size in tail if size < nontrivial_min]

    return {
        "n_nodes": n_total,
        "component_count": int(
            metrics.get("component_count", len(sizes)) or len(sizes)
        ),
        "giant_fraction": float(giant_size / n_total),
        "non_giant_mass": float((n_total - giant_size) / n_total),
        "nontrivial_component_count": len(nontrivial),
        "nontrivial_component_mass": float(sum(nontrivial) / n_total),
        "largest_non_giant_component_pct": float(max(tail, default=0) / n_total),
        "singleton_fraction": float(metrics.get("singleton_fraction", 0.0)),
        "small_component_mass": float(sum(small) / n_total),
    }


def _fragmentation_trend(history: list[Any]) -> dict[str, Any]:
    if len(history) < 2:
        return {
            "status": "insufficient_history",
            "next_decision": "run_one_more_sweep",
            "agent_action": "Run at least two sweeps before judging fragmentation trend.",
        }

    previous = _component_profile(getattr(history[-2], "metrics", {}) or {})
    current = _component_profile(getattr(history[-1], "metrics", {}) or {})
    n_total = int(current.get("n_nodes", 0) or previous.get("n_nodes", 0))
    significant_pct = max(25, math.ceil(max(n_total, 1) * 0.03)) / max(n_total, 1)

    nontrivial_delta = (
        current["nontrivial_component_mass"] - previous["nontrivial_component_mass"]
    )
    singleton_delta = current["singleton_fraction"] - previous["singleton_fraction"]
    largest_non_giant = current["largest_non_giant_component_pct"]
    component_count_delta = current["component_count"] - previous["component_count"]

    if singleton_delta > _SINGLETON_SPIKE_DELTA:
        status = "over_fragmentation"
        next_decision = "back_off_threshold"
        action = (
            "Back off the latest refinement: singleton mass increased faster than "
            "coherent non-giant structure."
        )
    elif (
        largest_non_giant >= significant_pct
        or nontrivial_delta >= _MEANINGFUL_NONTRIVIAL_DELTA
    ):
        status = "meaningful_resolution"
        next_decision = "inspect_features"
        action = (
            "Inspect the emerging non-giant component(s) with generate_cluster_dossier "
            "or get_cluster_profile before further tuning."
        )
    elif component_count_delta > 0 and nontrivial_delta < _ICE_CHIP_NONTRIVIAL_DELTA:
        status = "ice_chipping"
        next_decision = "run_one_more_sweep_or_back_off"
        action = (
            "Do not treat higher component count as progress. The refinement mostly "
            "chipped off tiny components; shift the grid only if a non-trivial "
            "component starts gaining mass."
        )
    else:
        status = "stable"
        next_decision = "compare_feature_evidence"
        action = (
            "No strong fragmentation trend detected. Compare feature evidence before "
            "making another refinement."
        )

    return {
        "status": status,
        "previous": previous,
        "current": current,
        "nontrivial_component_mass_delta": round(nontrivial_delta, 4),
        "singleton_fraction_delta": round(singleton_delta, 4),
        "largest_non_giant_component_pct": round(largest_non_giant, 4),
        "component_count_delta": component_count_delta,
        "significant_component_threshold_pct": round(significant_pct, 4),
        "next_decision": next_decision,
        "agent_action": action,
    }


def summarize_history(history: list[Any]) -> dict[str, Any]:
    """Synthesize observations from a session's SweepRecord list.

    ``history`` items are expected to expose ``config_yaml`` and ``metrics``
    attributes (the SweepRecord dataclass shape). Empty history is handled
    gracefully.
    """
    if not history:
        return {
            "n_runs": 0,
            "observations": ["No sweeps recorded in this session."],
            "rationale": "",
            "fragmentation_trend": {
                "status": "insufficient_history",
                "next_decision": "run_one_more_sweep",
                "agent_action": "Run at least two sweeps before judging fragmentation trend.",
            },
        }

    n = len(history)
    health_counts: dict[str, int] = {}
    pca_dim_health: dict[int, list[str]] = {}
    thresholds_with_singletons: list[float | str] = []

    for record in history:
        metrics = getattr(record, "metrics", {}) or {}
        config_yaml = getattr(record, "config_yaml", "") or ""
        health = _health_label(metrics)
        health_counts[health] = health_counts.get(health, 0) + 1

        for dim in _extract_pca_dims(config_yaml):
            pca_dim_health.setdefault(dim, []).append(health)

        if (
            float(metrics.get("singleton_fraction", 0.0))
            > _FRAGMENTED_SINGLETON_FRACTION
        ):
            threshold = _extract_construction_threshold(config_yaml)
            if threshold is not None:
                thresholds_with_singletons.append(threshold)

    observations: list[str] = []

    hairball_dims = sorted(
        dim
        for dim, healths in pca_dim_health.items()
        if sum(1 for h in healths if h == "hairball") >= max(1, len(healths) // 2)
    )
    if hairball_dims:
        observations.append(
            f"Projection dimensions {hairball_dims} appeared in hairball regimes in "
            f"the majority of runs containing them."
        )

    fragmented_dims = sorted(
        dim
        for dim, healths in pca_dim_health.items()
        if sum(1 for h in healths if h in {"fragmented", "empty"})
        >= max(1, len(healths) // 2)
    )
    if fragmented_dims:
        observations.append(
            f"Projection dimensions {fragmented_dims} appeared in fragmented or empty "
            f"graphs in the majority of runs containing them."
        )

    if thresholds_with_singletons:
        observations.append(
            f"Construction thresholds {sorted(set(map(str, thresholds_with_singletons)))} "
            f"coincided with elevated singleton fractions."
        )

    health_breakdown = ", ".join(
        f"{label}×{count}" for label, count in sorted(health_counts.items())
    )
    observations.append(f"Graph health across {n} runs: {health_breakdown}.")
    # Health labels here intentionally mirror server._graph_health_summary so
    # observations agree with the per-run health field agents already saw.

    fragmentation_trend = _fragmentation_trend(history)
    observations.append(f"Fragmentation trend: {fragmentation_trend['status']}.")

    rationale_parts: list[str] = []
    if hairball_dims:
        rationale_parts.append(
            "Low or extreme projection dimensions correlate with over-connected hairballs."
        )
    if fragmented_dims:
        rationale_parts.append(
            "Some projection dimensions consistently shatter the graph; consider shifting away from them."
        )
    if thresholds_with_singletons:
        rationale_parts.append(
            "High construction thresholds collapsed the graph into singletons."
        )

    return {
        "n_runs": n,
        "observations": observations,
        "rationale": " ".join(rationale_parts),
        "fragmentation_trend": fragmentation_trend,
    }
