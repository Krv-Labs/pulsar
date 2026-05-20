"""
Session sweep history synthesis for Pulsar MCP.

Pure rule-based pattern detection over recorded SweepRecords. Returns
observations + rationale only — no recommended-config field, by design, to
avoid fabricating numbers and to keep agent decisions explicit.
"""

from __future__ import annotations

from typing import Any

import yaml


_HAIRBALL_DENSITY = 0.8
_SPARSE_DENSITY = 0.05
_FRAGMENTED_SINGLETON_FRACTION = 0.25
_DOMINANT_GIANT = 0.95


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

        if float(metrics.get("singleton_fraction", 0.0)) > _FRAGMENTED_SINGLETON_FRACTION:
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
            f"PCA dimensions {hairball_dims} produced hairball regimes in "
            f"the majority of runs they appeared in."
        )

    fragmented_dims = sorted(
        dim
        for dim, healths in pca_dim_health.items()
        if sum(1 for h in healths if h in {"fragmented", "empty"})
        >= max(1, len(healths) // 2)
    )
    if fragmented_dims:
        observations.append(
            f"PCA dimensions {fragmented_dims} produced fragmented or empty "
            f"graphs in the majority of runs they appeared in."
        )

    if thresholds_with_singletons:
        observations.append(
            f"Construction thresholds {sorted(set(map(str, thresholds_with_singletons)))} "
            f"coincided with >50% singletons."
        )

    health_breakdown = ", ".join(
        f"{label}×{count}" for label, count in sorted(health_counts.items())
    )
    observations.append(f"Graph health across {n} runs: {health_breakdown}.")
    # Health labels here intentionally mirror server._graph_health_summary so
    # observations agree with the per-run health field agents already saw.

    rationale_parts: list[str] = []
    if hairball_dims:
        rationale_parts.append(
            "Low or extreme PCA dimensions correlate with over-connected hairballs."
        )
    if fragmented_dims:
        rationale_parts.append(
            "Some PCA dimensions consistently shatter the graph; consider shifting away from them."
        )
    if thresholds_with_singletons:
        rationale_parts.append(
            "High construction thresholds collapsed the graph into singletons."
        )

    return {
        "n_runs": n,
        "observations": observations,
        "rationale": " ".join(rationale_parts),
    }
