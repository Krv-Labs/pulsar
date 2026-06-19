from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from typing import Any, Literal

import numpy as np
from fastmcp import Context
from fastmcp.exceptions import ToolError

from pulsar.config import THRESHOLD_RANGE_MESSAGE
from pulsar.mcp.diagnostics import (
    _finalization_gate,
    _skeleton_graph_payload,
    diagnose_model,
)
from pulsar.mcp.errors import mcp_error, unknown_handle_error
from pulsar.mcp.payloads import singleton_count_at_threshold
from pulsar.mcp.registry import registry
from pulsar.mcp.session import _get_session
from pulsar.mcp.thresholds import (
    THRESHOLD_CANDIDATE_POLICIES,
    agent_threshold_options,
    component_mass_profile,
    mass_profile_hint,
    prepare_threshold_graph,
    structural_breakpoints,
)

logger = logging.getLogger(__name__)


def _validate_unit_threshold(value: float, *, name: str) -> float:
    threshold = float(value)
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ToolError(f"{name}: {THRESHOLD_RANGE_MESSAGE}")
    return threshold


async def diagnose_cosmic_graph(ctx: Context) -> str:
    """GraphMetrics for the fitted cosmic graph (density, components, etc).
    Interpret given dataset N."""
    session = _get_session(ctx)

    if session.model is None:
        raise ToolError("No model found. Run run_topological_sweep() first.")

    try:
        result = diagnose_model(session.model)
        payload = dataclasses.asdict(result)
        # B3: these metrics describe the persisted graph @ construction_threshold,
        # NOT any interpretation_edge_weight_threshold slice used for clustering.
        payload["graph_surface"] = (
            "persisted cosmic graph @ resolved_construction_threshold "
            f"({result.resolved_construction_threshold:.4f})"
        )
        if session.active_config_yaml:
            gate = _finalization_gate(
                payload,
                sweep_count=len(session.sweep_history),
                config_yaml=session.active_config_yaml,
            )
            payload["finalization_gate"] = gate
            if gate["status"] == "blocked":
                payload["advisories"].append(
                    {
                        "code": gate["code"],
                        "severity": "warning",
                        "message": gate["message"],
                        "agent_action": gate["suggested_refinement"]["strategy"],
                    }
                )
        return json.dumps(payload, indent=2)
    except Exception as e:
        logger.error(f"Error diagnosing graph: {e}")
        return mcp_error("diagnose_cosmic_graph", str(e))


def _sparse_threshold_curve(
    thresholds: list[float],
    component_counts: list[int],
    singleton_counts: list[int],
    *,
    max_points: int = 25,
) -> list[dict[str, Any]]:
    n_points = len(thresholds)
    if n_points == 0:
        return []

    if n_points <= max_points:
        indices = range(n_points)
    else:
        indices = sorted({int(i) for i in np.linspace(0, n_points - 1, num=max_points)})

    return [
        {
            "threshold": thresholds[i],
            "component_count": component_counts[i],
            "singleton_count": singleton_counts[i],
        }
        for i in indices
    ]


def _threshold_agent_readout(
    selected_profile: dict[str, Any] | None,
    breakpoints: list[dict[str, Any]],
) -> str:
    if not selected_profile:
        return "No threshold mass profile was available."

    largest_pct = float(selected_profile["largest_component_fraction"]) * 100
    small_pct = float(selected_profile["small_component_mass_fraction"]) * 100
    large_breakpoints = [
        row for row in breakpoints if row["event"] == "large_component_transition"
    ]

    if largest_pct >= 95:
        if large_breakpoints:
            return (
                f"Auto threshold is stable but giant-component dominated "
                f"({largest_pct:.2f}% of rows in the largest component); inspect "
                "large structural breakpoints before naming cohorts."
            )
        return (
            f"Auto threshold is stable but produces one giant component containing "
            f"{largest_pct:.2f}% of rows; only {small_pct:.2f}% of rows sit in "
            "small components."
        )
    if large_breakpoints:
        return (
            "Auto threshold exposes nontrivial component structure and large "
            "threshold transitions are available for parameter discussion."
        )
    return "Auto threshold exposes nontrivial component structure without large split markers."


async def get_threshold_stability_curve(
    detail: Literal["summary", "full"] = "summary",
    threshold_candidate_policy: Literal[
        "balanced", "report_ready", "detail_seeking", "outlier_mining"
    ] = "balanced",
    ctx: Context = None,
) -> str:
    """H0 persistent-homology stability of components vs edge-weight threshold.
    Use to reason about alternative thresholds after auto-clustering.
    `detail='full'` returns raw arrays.
    """
    if threshold_candidate_policy not in THRESHOLD_CANDIDATE_POLICIES:
        return mcp_error(
            "get_threshold_stability_curve",
            "threshold_candidate_policy must be one of "
            f"{sorted(THRESHOLD_CANDIDATE_POLICIES)}, got "
            f"'{threshold_candidate_policy}'",
        )

    session = _get_session(ctx)

    if session.model is None:
        raise ToolError("No model found. Run run_topological_sweep() first.")

    try:
        from pulsar._pulsar import find_stable_thresholds

        adj = session.model.weighted_adjacency
        threshold_graph = prepare_threshold_graph(adj)
        stability = await asyncio.to_thread(find_stable_thresholds, adj)

        thresholds = [float(t) for t in stability.thresholds]
        resolved_construction_threshold = _validate_unit_threshold(
            session.model.resolved_construction_threshold,
            name="resolved_construction_threshold",
        )
        optimal_threshold = float(stability.optimal_threshold)
        matches_current = bool(
            np.isclose(resolved_construction_threshold, optimal_threshold)
        )

        def _singleton_counts() -> list[int]:
            # adj is symmetric; a singleton at threshold t has no row-wise
            # neighbor above t. Vectorized: (n_thresholds, n) → bool.
            row_max = adj.max(axis=1)
            ts = np.asarray(thresholds, dtype=adj.dtype)
            # singleton iff row_max <= t for that threshold
            mask = row_max[None, :] <= ts[:, None]
            return [int(c) for c in mask.sum(axis=1)]

        singleton_counts = await asyncio.to_thread(_singleton_counts)

        plateau_limit = 10 if detail == "full" else 4
        plateaus = []
        for p in stability.top_k_plateaus(plateau_limit):
            singleton_count = singleton_count_at_threshold(adj, float(p.midpoint))
            mass_profile = component_mass_profile(threshold_graph, float(p.midpoint))
            plateaus.append(
                {
                    "start": float(p.start_threshold),
                    "end": float(p.end_threshold),
                    "component_count": int(p.component_count),
                    "length": float(p.length),
                    "midpoint": float(p.midpoint),
                    "singleton_count": singleton_count,
                    "singleton_fraction": round(
                        singleton_count / max(int(adj.shape[0]), 1),
                        4,
                    ),
                    "component_mass_profile": mass_profile,
                    "interpretation_hint": mass_profile_hint(mass_profile),
                }
            )

        component_counts = [int(c) for c in stability.component_counts]
        curve_sample = _sparse_threshold_curve(
            thresholds,
            component_counts,
            singleton_counts,
            max_points=25 if detail == "full" else 15,
        )
        breakpoints = structural_breakpoints(
            threshold_graph,
            thresholds,
            component_counts,
        )
        threshold_options = agent_threshold_options(
            threshold_graph,
            stability.top_k_plateaus(20 if detail == "full" else 10),
            thresholds,
            component_counts,
            policy=threshold_candidate_policy,
            max_candidates=12 if detail == "full" else 7,
        )
        selected_profile = (
            plateaus[0].get("component_mass_profile") if plateaus else None
        )
        if detail == "summary":
            for p in plateaus:
                p.pop("component_mass_profile", None)
            for list_key in [
                "stable_plateau_candidates",
                "transition_adjacent_candidates",
                "candidates",
            ]:
                if list_key in threshold_options:
                    for cand in threshold_options[list_key]:
                        cand.pop("component_mass_profile", None)
                        cand.pop("mass_shape", None)

        payload: dict[str, Any] = {
            "status": "ok",
            "detail": detail,
            "resolved_construction_threshold": resolved_construction_threshold,
            "optimal_threshold": optimal_threshold,
            "matches_current_threshold": matches_current,
            "threshold_guidance": (
                "Current graph construction matches the stability optimum."
                if matches_current
                else "Stability optimum differs from the current constructed graph; "
                "treat it as a rerun candidate, not a change to the fitted graph."
            ),
            "agent_readout": _threshold_agent_readout(selected_profile, breakpoints),
            "agent_threshold_options": threshold_options,
            "plateaus": plateaus,
            "structural_breakpoints": breakpoints,
            "structural_breakpoints_guidance": (
                "Breakpoints are capped structural transition candidates ranked "
                "first by large-component transitions, then by the smaller mass "
                "that actually joins or splits."
            ),
            "curve_point_count": len(thresholds),
            "curve_sample": curve_sample,
            "curve_sample_omitted": max(len(thresholds) - len(curve_sample), 0),
            "curve_sample_guidance": (
                "curve_sample is an evenly spaced sketch of the full threshold "
                "curve. Use detail='full' only when exact per-threshold arrays "
                "are needed."
            ),
        }
        if detail == "full":
            payload["thresholds"] = thresholds
            payload["component_counts"] = component_counts
            payload["singleton_counts"] = singleton_counts
        return json.dumps(payload, indent=2)
    except Exception as e:
        logger.error(f"Error computing stability curve: {e}")
        return mcp_error("get_threshold_stability_curve", str(e))


async def get_topological_skeleton(
    run_id: str = "",
    detail: Literal["summary", "nodes", "edges", "full", "full_nodes"] = "summary",
    max_edges: int = 100,
    max_nodes: int = 100,
    ctx: Context = None,
) -> str:
    """Structured graph connectivity for latest run or explicit `run_id`.

    Detail modes are summary-first: `nodes` returns a compact topological summary,
    `edges` returns capped raw edges, `full_nodes` returns capped raw nodes, and
    `full` returns capped raw nodes, capped raw edges, and config YAML.
    """
    try:
        if max_edges < 1:
            raise ToolError(f"max_edges must be >= 1, got '{max_edges}'")
        if max_nodes < 1:
            raise ToolError(f"max_nodes must be >= 1, got '{max_nodes}'")

        session = _get_session(ctx)
        target_run_id = run_id or session.latest_run_id
        if not target_run_id:
            raise ToolError("No run available. Run run_topological_sweep() first.")
        record = registry.get_run(target_run_id)
        if record is None:
            return unknown_handle_error(
                "get_topological_skeleton", "run_id", target_run_id
            )
        payload = {
            "run_id": record.run_id,
            "dataset_id": record.dataset_id,
            "config_yaml_omitted": detail != "full",
            "resolved_construction_threshold": record.resolved_construction_threshold,
            "graph_surface": (
                "persisted cosmic graph @ resolved_construction_threshold "
                f"({record.resolved_construction_threshold}); not an "
                "interpretation_edge_weight_threshold slice"
            ),
            "graph": _skeleton_graph_payload(
                record.graph_summary,
                detail=detail,
                max_edges=max_edges,
                max_nodes=max_nodes,
            ),
            "recommended_next_tools": [
                "diagnose_cosmic_graph",
                "get_threshold_stability_curve",
                "compare_sweeps",
            ],
        }
        if detail == "full":
            payload["config_yaml"] = record.config_yaml
        return json.dumps(payload, indent=2)
    except Exception as e:
        return mcp_error("get_topological_skeleton", str(e))
