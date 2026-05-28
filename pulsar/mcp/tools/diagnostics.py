from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from typing import Any

import numpy as np
from fastmcp import Context
from fastmcp.exceptions import ToolError

from pulsar.config import THRESHOLD_RANGE_MESSAGE
from pulsar.mcp.diagnostics import (
    _finalization_gate,
    _skeleton_graph_payload,
    _threshold_stability_summary,
    diagnose_model,
)
from pulsar.mcp.errors import mcp_error, unknown_handle_error
from pulsar.mcp.payloads import singleton_count_at_threshold
from pulsar.mcp.registry import registry
from pulsar.mcp.session import _get_session

logger = logging.getLogger(__name__)


def _validate_unit_threshold(value: float, *, name: str) -> float:
    threshold = float(value)
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ToolError(f"{name}: {THRESHOLD_RANGE_MESSAGE}")
    return threshold


async def diagnose_cosmic_graph(ctx: Context) -> str:
    """
    Diagnose the fitted cosmic graph quality by returning pure GraphMetrics.
    Interpret these metrics (e.g. density, component distribution) given N.
    """
    session = _get_session(ctx)

    if session.model is None:
        raise ToolError("No model found. Run run_topological_sweep() first.")

    try:
        result = diagnose_model(session.model)
        payload = dataclasses.asdict(result)
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


async def get_threshold_stability_curve(
    detail: str = "summary",
    ctx: Context = None,
) -> str:
    """
    Return component-count-vs-edge-weight-threshold stability.

    Uses H0 persistent homology on the cosmic graph's weighted adjacency
    to show how many connected components exist at each edge weight threshold.
    Use this to reason about alternative clustering thresholds after the
    initial auto-clustering.

    Returns:
        Summary JSON with top plateaus, sparse curve sample, and selected
        threshold by default. Pass detail="full" for raw arrays.
    """
    if detail not in {"summary", "full"}:
        return mcp_error(
            "get_threshold_stability_curve",
            f"detail must be 'summary' or 'full', got '{detail}'",
        )

    session = _get_session(ctx)

    if session.model is None:
        raise ToolError("No model found. Run run_topological_sweep() first.")

    try:
        from pulsar._pulsar import find_stable_thresholds

        adj = session.model.weighted_adjacency
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

        plateaus = []
        for p in stability.top_k_plateaus(10):
            singleton_count = singleton_count_at_threshold(adj, float(p.midpoint))
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
                }
            )

        component_counts = [int(c) for c in stability.component_counts]
        curve_sample = _sparse_threshold_curve(
            thresholds,
            component_counts,
            singleton_counts,
        )
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
            "plateaus": plateaus,
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
    detail: str = "summary",
    max_edges: int = 100,
    max_nodes: int = 100,
    ctx: Context = None,
) -> str:
    """
    Return structured graph connectivity for the latest run or an explicit run_id.

    Defaults to summary-only output for chat-first agent loops. Use
    detail="edges", "nodes", or "full" with explicit caps for late-stage graph
    inspection.
    """
    try:
        if detail not in {"summary", "nodes", "edges", "full"}:
            raise ToolError(
                f"detail must be 'summary', 'nodes', 'edges', or 'full', got '{detail}'"
            )
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
