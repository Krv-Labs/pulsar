from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import math
import time
from typing import Any, Literal
import uuid

import numpy as np
from fastmcp import Context
from fastmcp.exceptions import ToolError

from pulsar.analysis import cosmic_to_networkx
from pulsar.config import THRESHOLD_RANGE_MESSAGE
from pulsar.mcp.diagnostics import (
    _build_graph_summary_from_graph,
    _finalization_gate,
    _graph_metrics_from_graph,
    _skeleton_graph_payload,
    diagnose_model,
)
from pulsar.mcp.errors import mcp_error, unknown_handle_error
from pulsar.mcp.payloads import size_summary
from pulsar.mcp.registry import registry
from pulsar.mcp.session import GraphArtifact, _get_session
from pulsar.mcp.thresholds import (
    THRESHOLD_CANDIDATE_POLICIES,
    agent_threshold_options,
    component_mass_profile,
    mass_profile_hint,
    prepare_threshold_graph_from_edges,
    structural_breakpoints,
    threshold_morphology_profile,
)

logger = logging.getLogger(__name__)


def _validate_unit_threshold(value: float, *, name: str) -> float:
    threshold = float(value)
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ToolError(f"{name}: {THRESHOLD_RANGE_MESSAGE}")
    return threshold


_DENSE_NODE_CAP = 2500
_DEFAULT_MEMORY_BUDGET_MB = 512.0


def _artifact_staleness(
    artifact: GraphArtifact,
    latest_run_id: str | None,
) -> dict[str, Any]:
    stale = bool(latest_run_id and artifact.run_id != latest_run_id)
    return {
        "stale": stale,
        "artifact_run_id": artifact.run_id,
        "latest_run_id": latest_run_id,
        "warning": (
            "Artifact was built from a superseded run; inspect for audit only or rebuild "
            "from the latest run."
            if stale
            else None
        ),
    }


# Rough throughput for the spectral-sparsifier inner loop (one preconditioned-CG
# solve per JL sketch row over the edge set). Calibrated so n=800 lands at ~2s
# (epsilon=1.0) and ~20s (epsilon=0.3), matching the PR benchmark order of
# magnitude. It is an order-of-magnitude projection — enough to tell a 2s build
# from a 30s one — not a precise timing.
_SPARSIFY_OPS_PER_S = 4.5e6


def _estimate_sparsify_runtime_s(sketch_dim: int, edge_count: int) -> float:
    return round((sketch_dim * max(edge_count, 1)) / _SPARSIFY_OPS_PER_S, 2)


def _dense_cost(n_nodes: int) -> dict[str, Any]:
    peak_mem_mb = (n_nodes * n_nodes * 8) / (1024 * 1024)
    # Dense affinity is a materialization, not the iterative sparsifier loop;
    # runtime is memory-bound and dominated by the O(n^2) allocation.
    return {
        "runtime_s": round((n_nodes * n_nodes) / 2.5e8, 2),
        "runtime_basis": "rough projection; dominated by O(n^2) dense materialization",
        "peak_mem_mb": round(peak_mem_mb, 2),
        "edge_count": n_nodes * (n_nodes - 1) // 2,
    }


def _artifact_estimate(n_nodes: int, edge_count: int, epsilon: float) -> dict[str, Any]:
    eps = max(float(epsilon), 1e-9)
    sketch_dim = max(
        1, int(math.ceil(24.0 * max(math.log(max(n_nodes, 1)), 1.0) / (eps * eps)))
    )
    sample_count = max(
        1,
        int(
            math.ceil(9.0 * n_nodes * max(math.log(max(n_nodes, 1)), 1.0) / (eps * eps))
        ),
    )
    return {
        "runtime_s": _estimate_sparsify_runtime_s(sketch_dim, edge_count),
        "runtime_basis": (
            "rough projection; scales ~sketch_dim x edges (so ~1/epsilon^2). "
            f"Estimated at epsilon={eps:g}."
        ),
        "peak_mem_mb": round(
            ((edge_count * 24) + (n_nodes * 8 * 3)) / (1024 * 1024), 2
        ),
        "edge_count": min(edge_count, sample_count),
        "sketch_dim": sketch_dim,
        "sample_count": sample_count,
    }


def _advisory_codes(payload: dict[str, Any]) -> set[str]:
    return {str(row.get("code")) for row in payload.get("advisories", [])}


def _build_recommendation(
    payload: dict[str, Any],
    *,
    task: str,
    constraints: dict[str, Any] | None,
) -> dict[str, Any]:
    constraints = constraints or {}
    n_nodes = int(payload.get("n_nodes", 0) or 0)
    n_edges = int(payload.get("n_edges", 0) or 0)
    codes = _advisory_codes(payload)
    max_memory_mb = float(constraints.get("max_memory_mb", _DEFAULT_MEMORY_BUDGET_MB))
    allow_dense = bool(constraints.get("allow_dense", True))
    allow_artifacts = bool(constraints.get("allow_artifacts", True))
    excluded: list[dict[str, Any]] = []

    if {"HAIRBALL_DENSITY", "DOMINANT_COMPONENT"} & codes:
        excluded.append(
            {
                "surface": "spectral_sparsifier",
                "reason": (
                    "Spectral sparsification preserves Laplacian quadratic forms, "
                    "cuts, and effective-resistance structure; it will not separate "
                    "a dominant component or de-hairball H0 topology."
                ),
            }
        )
        return {
            "recommended_surface": "re_grid",
            "why": (
                "The constructed graph is over-connected. Use a stricter threshold "
                "slice for interpretation or rerun a more discriminating projection/"
                "epsilon grid; do not use spectral sparsification as the remedy."
            ),
            "estimated_cost": {
                "runtime_s": None,
                "peak_mem_mb": None,
                "edge_count": n_edges,
            },
            "next_tool_call": {
                "tool": "get_threshold_stability_curve",
                "args": {"source": "persisted"},
            },
            "excluded": excluded,
            "supported_future_surfaces": [
                {
                    "kind": "structural_backbone",
                    "status": "planned, not callable",
                    "use_when": "de-hairball while preserving visible 1-skeleton structure",
                }
            ],
        }

    if {"EMPTY_GRAPH", "HIGH_SINGLETONS"} & codes:
        return {
            "recommended_surface": "re_grid",
            "why": (
                "The constructed graph is fragmented. Lower the construction "
                "threshold or broaden neighborhoods before interpreting clusters."
            ),
            "estimated_cost": {
                "runtime_s": None,
                "peak_mem_mb": None,
                "edge_count": n_edges,
            },
            "next_tool_call": {
                "tool": "get_threshold_stability_curve",
                "args": {"source": "persisted"},
            },
            "excluded": excluded,
            "supported_future_surfaces": [],
        }

    if task == "dense_affinity_required":
        dense_cost = _dense_cost(n_nodes)
        if (
            allow_dense
            and n_nodes <= _DENSE_NODE_CAP
            and dense_cost["peak_mem_mb"] <= max_memory_mb
        ):
            return {
                "recommended_surface": "dense_full_affinity",
                "why": "The task explicitly requires exact dense affinity and fits the configured memory gate.",
                "estimated_cost": dense_cost,
                "next_tool_call": {
                    "tool": "generate_cluster_dossier",
                    "args": {"method": "spectral"},
                },
                "excluded": excluded,
                "supported_future_surfaces": [],
            }
        excluded.append(
            {
                "surface": "dense_full_affinity",
                "reason": "Dense n x n affinity exceeds node or memory gate.",
            }
        )

    if task == "spectral_analysis" and allow_artifacts:
        target = max(n_nodes * max(math.log(max(n_nodes, 2)), 1.0), 1.0)
        if n_edges > 4 * target:
            return {
                "recommended_surface": "build_artifact",
                "why": "Repeated spectral analysis on a graph with edges far above n log n can amortize a spectral sparsifier.",
                "estimated_cost": _artifact_estimate(n_nodes, n_edges, 1.0),
                "next_tool_call": {
                    "tool": "create_graph_artifact",
                    "args": {"kind": "spectral_sparsifier", "estimate_only": True},
                },
                "excluded": excluded,
                "supported_future_surfaces": [],
            }

    if task in {"clustering", "reporting", "anomaly_mining"}:
        return {
            "recommended_surface": "threshold_slice",
            "why": "Use the retained sparse weighted graph at an interpretation threshold for this analysis task.",
            "estimated_cost": {
                "runtime_s": None,
                "peak_mem_mb": None,
                "edge_count": n_edges,
            },
            "next_tool_call": {
                "tool": "generate_cluster_dossier",
                "args": {"detail": "summary"},
            },
            "excluded": excluded,
            "supported_future_surfaces": [],
        }

    return {
        "recommended_surface": "exact_sparse",
        "why": "The constructed sparse cosmic graph is the right default surface for structural inspection.",
        "estimated_cost": {
            "runtime_s": None,
            "peak_mem_mb": None,
            "edge_count": n_edges,
        },
        "next_tool_call": {
            "tool": "get_topological_skeleton",
            "args": {"surface": "constructed"},
        },
        "excluded": excluded,
        "supported_future_surfaces": [],
    }


async def diagnose_cosmic_graph(
    surface: Literal["constructed", "artifact"] = "constructed",
    artifact_id: str = "",
    detail: Literal["summary", "full"] = "summary",
    task: Literal[
        "structure",
        "clustering",
        "spectral_analysis",
        "dense_affinity_required",
        "reporting",
        "anomaly_mining",
    ]
    | None = None,
    constraints: dict[str, Any] | None = None,
    ctx: Context = None,
) -> str:
    """Graph health/gating for the fitted cosmic graph: density, components,
    advisories, finalization gate, and optional task-specific surface advice."""
    session = _get_session(ctx)
    if detail not in {"summary", "full"}:
        return mcp_error(
            "diagnose_cosmic_graph",
            "detail must be 'summary' or 'full'.",
        )

    if surface == "constructed" and session.model is None:
        raise ToolError("No model found. Run run_topological_sweep() first.")

    try:
        staleness = None
        if surface == "artifact":
            if not artifact_id:
                raise ToolError("artifact_id is required when surface='artifact'.")
            artifact = session.graph_artifacts.get(artifact_id)
            if artifact is None:
                return unknown_handle_error(
                    "diagnose_cosmic_graph", "artifact_id", artifact_id
                )
            payload = dict(artifact.metrics)
            staleness = _artifact_staleness(artifact, session.latest_run_id)
            payload["artifact_id"] = artifact.artifact_id
            payload["artifact_kind"] = artifact.kind
            payload["artifact_staleness"] = staleness
            payload["graph_surface"] = f"graph artifact {artifact_id} ({artifact.kind})"
        else:
            result = diagnose_model(session.model)
            payload = dataclasses.asdict(result)
            # B3: these metrics describe the persisted graph @ construction_threshold,
            # NOT any interpretation_edge_weight_threshold slice used for clustering.
            payload["graph_surface"] = (
                "persisted cosmic graph @ resolved_construction_threshold "
                f"({result.resolved_construction_threshold:.4f})"
            )

        if detail == "summary":
            component_sizes = list(payload.get("component_sizes", []) or [])
            payload["component_sizes_summary"] = size_summary(component_sizes)
            payload.pop("component_sizes", None)
        payload["detail"] = detail

        if surface == "constructed" and session.active_config_yaml:
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
                        "diagnostic_interpretation": gate["suggested_refinement"][
                            "strategy"
                        ],
                    }
                )
        if task is not None:
            payload["recommendation"] = _build_recommendation(
                payload,
                task=task,
                constraints=constraints,
            )
        return json.dumps(payload, indent=2)
    except Exception as e:
        logger.error(f"Error diagnosing graph: {e}")
        return mcp_error("diagnose_cosmic_graph", str(e))


def _singleton_counts_from_edges(
    n_nodes: int,
    edges: list[tuple[int, int, float]],
    thresholds: list[float],
) -> list[int]:
    row_max = [0.0] * n_nodes
    for i, j, weight in edges:
        row_max[int(i)] = max(row_max[int(i)], float(weight))
        row_max[int(j)] = max(row_max[int(j)], float(weight))
    return [
        sum(1 for value in row_max if value <= threshold) for threshold in thresholds
    ]


def _summary_structural_breakpoints(
    breakpoints: list[dict[str, Any]],
    *,
    max_breakpoints: int = 3,
) -> list[dict[str, Any]]:
    """Summary mode shows only non-dust component transitions."""
    return [
        row for row in breakpoints if row.get("event") != "small_component_absorption"
    ][:max_breakpoints]


def _threshold_curve_payload(
    model,
    *,
    detail: Literal["summary", "full"],
    threshold_candidate_policy: str,
) -> dict[str, Any]:
    from pulsar._pulsar import find_stable_thresholds_sparse

    # --- Shared core (cheap; computed for every tier) ---
    n_nodes = int(model.cosmic_rust.n)
    edges = model.weighted_edges(threshold=0.0)
    threshold_graph = prepare_threshold_graph_from_edges(n_nodes, edges)
    stability = find_stable_thresholds_sparse(n_nodes, edges)

    thresholds = [float(t) for t in stability.thresholds]
    component_counts = [int(c) for c in stability.component_counts]
    resolved_construction_threshold = _validate_unit_threshold(
        model.resolved_construction_threshold,
        name="resolved_construction_threshold",
    )
    optimal_threshold = float(stability.optimal_threshold)
    matches_current = bool(
        np.isclose(resolved_construction_threshold, optimal_threshold)
    )
    breakpoints = structural_breakpoints(threshold_graph, thresholds, component_counts)
    current_threshold_morphology = threshold_morphology_profile(
        threshold_graph,
        resolved_construction_threshold,
        top_k=10,
    )
    h0_longest_plateau_morphology = threshold_morphology_profile(
        threshold_graph,
        optimal_threshold,
        top_k=10,
    )

    def _profile_rows(max_points: int | None) -> list[dict[str, Any]]:
        if not thresholds:
            return []
        if max_points is None or len(thresholds) <= max_points:
            indices = range(len(thresholds))
        else:
            indices = sorted(
                {int(i) for i in np.linspace(0, len(thresholds) - 1, num=max_points)}
            )
        return [
            threshold_morphology_profile(threshold_graph, thresholds[i], top_k=10)
            for i in indices
        ]

    def _candidate_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for candidate in candidates:
            threshold = candidate.get("threshold", candidate.get("midpoint"))
            if threshold is None:
                continue
            row = _compact_threshold_candidate(candidate)
            row["morphology"] = threshold_morphology_profile(
                threshold_graph,
                float(threshold),
                top_k=10,
            )
            rows.append(row)
        return rows

    def _plateau_row(plateau, *, with_profile: bool) -> dict[str, Any]:
        midpoint = float(plateau.midpoint)
        singleton_count = _singleton_counts_from_edges(n_nodes, edges, [midpoint])[0]
        row = {
            "start": float(plateau.start_threshold),
            "end": float(plateau.end_threshold),
            "component_count": int(plateau.component_count),
            "length": float(plateau.length),
            "midpoint": midpoint,
            "singleton_count": singleton_count,
            "singleton_fraction": round(singleton_count / max(n_nodes, 1), 4),
        }
        if with_profile:
            mass_profile = component_mass_profile(threshold_graph, midpoint)
            row["component_mass_profile"] = mass_profile
            row["interpretation_hint"] = mass_profile_hint(mass_profile)
        return row

    if detail == "summary":
        # Decision packet: bounded morphology rows and top-N lenses only. Raw
        # arrays, curve_sample, and full candidate families stay in full mode.
        top_plateaus = list(stability.top_k_plateaus(4))
        selected_profile = (
            component_mass_profile(threshold_graph, float(top_plateaus[0].midpoint))
            if top_plateaus
            else None
        )
        threshold_options = agent_threshold_options(
            threshold_graph,
            stability.top_k_plateaus(10),
            thresholds,
            component_counts,
            policy=threshold_candidate_policy,
            max_candidates=7,
        )
        all_candidates = threshold_options.get("candidates", [])
        candidates = _candidate_rows(all_candidates[:3])
        threshold_profiles = _profile_rows(10)
        summary_breakpoints = _summary_structural_breakpoints(breakpoints)
        return {
            "status": "ok",
            "detail": detail,
            "source": "live",
            "resolved_construction_threshold": resolved_construction_threshold,
            "h0_longest_plateau_threshold": optimal_threshold,
            "matches_current_threshold": matches_current,
            "current_threshold_morphology": current_threshold_morphology,
            "h0_longest_plateau_morphology": h0_longest_plateau_morphology,
            "h0_plateau_readout": _threshold_agent_readout(
                selected_profile, breakpoints
            ),
            "threshold_candidate_policy": threshold_candidate_policy,
            "selection_strategy": threshold_options.get("selection_strategy"),
            "threshold_candidates": candidates,
            "threshold_candidates_omitted": max(
                len(all_candidates) - len(candidates), 0
            ),
            "threshold_profiles": threshold_profiles,
            "threshold_profiles_omitted": max(
                len(thresholds) - len(threshold_profiles), 0
            ),
            "structural_breakpoints": summary_breakpoints,
            "structural_breakpoints_omitted": max(
                len(breakpoints) - len(summary_breakpoints), 0
            ),
            "structural_breakpoints_filter": (
                "summary hides small_component_absorption dust events; use "
                "detail='full' for all breakpoints"
            ),
            "plateaus_returned": min(len(top_plateaus), 3),
            "plateaus_omitted": max(len(top_plateaus) - 3, 0),
            "curve_point_count": len(thresholds),
            "full_detail_available": "get_threshold_stability_curve(detail='full')",
        }

    # --- Full tier: lossless arrays, curve sample, full plateaus/candidates ---
    singleton_counts = _singleton_counts_from_edges(n_nodes, edges, thresholds)
    threshold_profiles = _profile_rows(None)
    plateaus = [
        _plateau_row(p, with_profile=True) for p in stability.top_k_plateaus(10)
    ]
    selected_profile = plateaus[0].get("component_mass_profile") if plateaus else None
    curve_sample = _sparse_threshold_curve(
        thresholds, component_counts, singleton_counts, max_points=25
    )
    threshold_options = agent_threshold_options(
        threshold_graph,
        stability.top_k_plateaus(20),
        thresholds,
        component_counts,
        policy=threshold_candidate_policy,
        max_candidates=12,
    )
    all_candidates = threshold_options.get("candidates", [])
    enriched_candidates = _candidate_rows(all_candidates)
    # threshold_candidates is the canonical, morphology-enriched candidate list.
    # Drop the duplicate raw `candidates` from agent_threshold_options so the same
    # lenses are not shipped twice; selection_strategy and the other candidate
    # families are retained for audit.
    audit_threshold_options = {
        key: value for key, value in threshold_options.items() if key != "candidates"
    }
    return {
        "status": "ok",
        "detail": detail,
        "source": "live",
        "resolved_construction_threshold": resolved_construction_threshold,
        "h0_longest_plateau_threshold": optimal_threshold,
        "matches_current_threshold": matches_current,
        "current_threshold_morphology": current_threshold_morphology,
        "h0_longest_plateau_morphology": h0_longest_plateau_morphology,
        "threshold_guidance": (
            "Current graph construction matches the H0 longest plateau."
            if matches_current
            else "H0 longest plateau differs from the current constructed graph; "
            "treat it as a rerun candidate, not a change to the fitted graph."
        ),
        "h0_plateau_readout": _threshold_agent_readout(selected_profile, breakpoints),
        "agent_threshold_options": audit_threshold_options,
        "threshold_candidates": enriched_candidates,
        "plateaus": plateaus,
        "structural_breakpoints": breakpoints,
        "structural_breakpoints_guidance": (
            "Breakpoints are capped structural transition candidates ranked "
            "first by large-component transitions, then by the smaller mass "
            "that actually joins or splits."
        ),
        "curve_point_count": len(thresholds),
        "threshold_profiles": threshold_profiles,
        "threshold_profiles_omitted": 0,
        "curve_sample": curve_sample,
        "curve_sample_omitted": max(len(thresholds) - len(curve_sample), 0),
        "curve_sample_guidance": (
            "curve_sample is an evenly spaced sketch of the full threshold "
            "curve. Use detail='full' only when exact per-threshold arrays "
            "are needed."
        ),
        "thresholds": thresholds,
        "component_counts": component_counts,
        "singleton_counts": singleton_counts,
    }


def _threshold_curve_to_markdown(payload: dict[str, Any]) -> str:
    payload = _normalize_threshold_curve_payload(payload)
    h0_threshold = payload.get(
        "h0_longest_plateau_threshold",
        payload.get("optimal_threshold"),
    )
    lines = [
        "# Threshold Stability",
        "",
        f"- Source: {payload.get('source')}",
        f"- Detail: {payload.get('detail')}",
        "- Construction threshold: "
        f"{_format_threshold_value(payload.get('resolved_construction_threshold'))}",
        f"- H0 longest plateau: {_format_threshold_value(h0_threshold)}",
        f"- Matches current threshold: {payload.get('matches_current_threshold')}",
        f"- Curve points: {payload.get('curve_point_count', 0)}",
    ]
    if payload.get("run_id"):
        lines.insert(2, f"- Run ID: `{payload['run_id']}`")
    current_morphology = payload.get("current_threshold_morphology")
    if current_morphology:
        lines.append(
            "- Current construction morphology: "
            + _format_morphology_summary(current_morphology)
        )
    h0_morphology = payload.get("h0_longest_plateau_morphology")
    if h0_morphology:
        lines.append(
            "- H0 plateau morphology: " + _format_morphology_summary(h0_morphology)
        )
    readout = payload.get("h0_plateau_readout", payload.get("agent_readout"))
    if readout:
        lines.append(f"- H0 plateau readout: {readout}")

    profiles = payload.get("threshold_profiles") or []
    lines.extend(
        [
            "",
            "## Threshold Morphology Sample",
            "",
            "| Threshold | Components | Top sizes | Giant % | SLR | Singleton % |",
            "|---:|---:|---|---:|---:|---:|",
        ]
    )
    if profiles:
        for row in profiles:
            lines.append(_format_morphology_table_row(row))
    else:
        lines.append("| - | - | - | - | - | - |")
    omitted = int(payload.get("threshold_profiles_omitted", 0) or 0)
    if omitted:
        lines.append("")
        lines.append(f"- Threshold rows omitted: {omitted}")
    lines.append(
        "- Sample rows are evenly spaced across the curve; candidate thresholds "
        "may not appear in this sample."
    )

    candidates = payload.get("threshold_candidates") or []
    if candidates:
        lines.extend(
            [
                "",
                "## Candidate Lenses",
                "",
                "| Threshold | Kind | Tier | Components | Top sizes | Giant % | SLR | Singleton % | Why |",
                "|---:|---|---|---:|---|---:|---:|---:|---|",
            ]
        )
        for row in candidates:
            threshold = row.get("threshold", row.get("midpoint"))
            morphology = row.get("morphology") or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        _format_threshold_value(threshold),
                        _table_cell(row.get("candidate_kind", "candidate")),
                        _table_cell(row.get("interpretability_tier", "unknown")),
                        str(morphology.get("component_count", "n/a")),
                        _format_top_sizes(morphology.get("top_component_sizes", [])),
                        _format_pct(morphology.get("giant_fraction")),
                        _format_ratio(morphology.get("second_largest_ratio")),
                        _format_pct(morphology.get("singleton_fraction")),
                        _table_cell(row.get("why", "")),
                    ]
                )
                + " |"
            )
        if not any(row.get("morphology") for row in candidates):
            lines.append("")
            lines.append(
                "- Candidate morphology is unavailable in this stored summary; "
                "rerun with `source='live'` or run a new sweep to persist enriched rows."
            )

    breakpoints = payload.get("structural_breakpoints") or []
    if breakpoints:
        lines.extend(["", "## Structural Breakpoints", ""])
        for row in breakpoints:
            lines.append(
                f"- `{_format_threshold_value(row.get('threshold'))}`: "
                f"{row.get('event')} "
                f"({row.get('component_count_before')} -> {row.get('component_count_after')} components); "
                f"{row.get('interpretation_hint', '')}"
            )
    elif int(payload.get("structural_breakpoints_omitted", 0) or 0):
        lines.extend(
            [
                "",
                "## Structural Breakpoints",
                "",
                "- No non-dust structural breakpoints returned in summary; "
                "dust absorption events are available with `detail='full'`.",
            ]
        )

    lines.extend(
        [
            "",
            "## Compatible Next Tools",
            "",
            *_threshold_next_tool_lines(payload),
        ]
    )
    if payload.get("full_detail_available"):
        lines.append(
            f"- `{payload['full_detail_available']}` for raw arrays and full rows."
        )
    return "\n".join(lines).strip()


def _format_threshold_value(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _format_morphology_summary(row: dict[str, Any]) -> str:
    return (
        f"components={row.get('component_count', 'n/a')}, "
        f"top_sizes={_format_top_sizes(row.get('top_component_sizes', []))}, "
        f"giant={_format_pct(row.get('giant_fraction'))}, "
        f"SLR={_format_ratio(row.get('second_largest_ratio'))}, "
        f"singletons={_format_pct(row.get('singleton_fraction'))}"
    )


def _format_morphology_table_row(row: dict[str, Any]) -> str:
    return (
        "| "
        + " | ".join(
            [
                _format_threshold_value(row.get("threshold")),
                str(row.get("component_count", "")),
                _format_top_sizes(row.get("top_component_sizes", [])),
                _format_pct(row.get("giant_fraction")),
                _format_ratio(row.get("second_largest_ratio")),
                _format_pct(row.get("singleton_fraction")),
            ]
        )
        + " |"
    )


def _format_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def _format_ratio(value: Any) -> str:
    """Render a unit ratio (e.g. SLR = 2nd/largest) as a ratio, not a percent."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _table_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_top_sizes(values: Any) -> str:
    if not values:
        return "-"
    return ", ".join(str(int(v)) for v in list(values)[:10])


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


def _threshold_next_tool_lines(payload: dict[str, Any]) -> list[str]:
    current = payload.get("current_threshold_morphology") or {}
    giant = float(current.get("giant_fraction", 0.0) or 0.0)
    slr = float(current.get("second_largest_ratio", 0.0) or 0.0)
    singletons = float(current.get("singleton_fraction", 0.0) or 0.0)

    if giant >= 0.85 and slr < 0.05:
        return [
            '- `generate_cluster_dossier(method="spectral")` for latent structure inside the dominant component.',
            '- `generate_cluster_dossier(method="components")` only when the tail/outlier components are the object of study.',
            "- `refine_config` if this giant-plus-tail topology is not expected for the dataset.",
        ]
    if singletons >= 0.25:
        return [
            "- `refine_config` first if the current construction threshold is too fragmented for the analysis goal.",
            '- `generate_cluster_dossier(method="components")` when singleton/tail components are the object of study.',
            '- `generate_cluster_dossier(method="spectral")` only after confirming the full weighted graph is connected enough for the question.',
        ]
    return [
        '- `generate_cluster_dossier(method="components")` when hard disconnected components are the object of study.',
        '- `generate_cluster_dossier(method="spectral")` when the question is latent structure inside a dominant component.',
        "- `refine_config` when construction-time topology appears over- or under-connected.",
    ]


def _threshold_agent_readout(
    selected_profile: dict[str, Any] | None,
    breakpoints: list[dict[str, Any]],
) -> str:
    if not selected_profile:
        return "No threshold mass profile was available."

    largest_pct = float(selected_profile["largest_component_fraction"]) * 100
    small_pct = float(selected_profile["small_component_mass_fraction"]) * 100
    singleton_pct = float(selected_profile["singleton_fraction"]) * 100
    large_breakpoints = [
        row for row in breakpoints if row["event"] == "large_component_transition"
    ]

    if singleton_pct >= 75:
        return (
            "H0 longest plateau is singleton-dominated "
            f"({singleton_pct:.2f}% singletons); treat it as an outlier/frontier "
            "lens, not a construction or cohort threshold."
        )
    if singleton_pct >= 35:
        return (
            "H0 longest plateau is singleton-rich "
            f"({singleton_pct:.2f}% singletons); use only as an exploratory "
            "frontier lens unless the task is anomaly mining."
        )
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


def _compact_threshold_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "candidate_kind",
        "threshold",
        "midpoint",
        "start",
        "end",
        "component_count",
        "component_count_before",
        "component_count_after",
        "top_component_sizes",
        "singleton_fraction",
        "interpretability_tier",
        "best_for",
        "avoid_for",
        "why",
        "event",
        "side",
    ]
    return {key: candidate[key] for key in keep if key in candidate}


def _strip_state_label(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out.pop("state_label", None)
    return out


def _normalize_threshold_curve_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize older persisted summaries to the current public contract."""
    out = dict(payload)
    if "h0_longest_plateau_threshold" not in out and "optimal_threshold" in out:
        out["h0_longest_plateau_threshold"] = out.pop("optimal_threshold")
    if "h0_plateau_readout" not in out and "agent_readout" in out:
        out["h0_plateau_readout"] = out.pop("agent_readout")

    for key in (
        "threshold_profiles",
        "current_threshold_morphology",
        "h0_longest_plateau_morphology",
    ):
        value = out.get(key)
        if isinstance(value, list):
            out[key] = [_strip_state_label(row) for row in value]
        elif isinstance(value, dict):
            out[key] = _strip_state_label(value)

    candidates = out.get("threshold_candidates")
    if isinstance(candidates, list):
        normalized_candidates = []
        for candidate in candidates:
            row = dict(candidate)
            morphology = row.get("morphology")
            if isinstance(morphology, dict):
                row["morphology"] = _strip_state_label(morphology)
            normalized_candidates.append(row)
        out["threshold_candidates"] = normalized_candidates

    return out


async def get_threshold_stability_curve(
    detail: Literal["summary", "full"] = "summary",
    run_id: str = "",
    source: Literal["persisted", "live"] = "persisted",
    threshold_candidate_policy: Literal[
        "balanced", "report_ready", "detail_seeking", "outlier_mining"
    ] = "balanced",
    response_format: Literal["markdown", "json"] = "markdown",
    ctx: Context = None,
) -> str:
    """Threshold morphology/lenses across the H0 component stability curve.
    Summary mode returns bounded component-mass rows; `detail='full'` returns
    raw arrays.
    """
    if threshold_candidate_policy not in THRESHOLD_CANDIDATE_POLICIES:
        return mcp_error(
            "get_threshold_stability_curve",
            "threshold_candidate_policy must be one of "
            f"{sorted(THRESHOLD_CANDIDATE_POLICIES)}, got "
            f"'{threshold_candidate_policy}'",
        )
    if response_format not in {"markdown", "json"}:
        return mcp_error(
            "get_threshold_stability_curve",
            "response_format must be 'markdown' or 'json'.",
        )

    try:
        session = _get_session(ctx)
        target_run_id = run_id or session.latest_run_id
        if source == "persisted" and detail == "summary":
            if target_run_id:
                record = registry.get_run(target_run_id)
                if record is None:
                    return unknown_handle_error(
                        "get_threshold_stability_curve", "run_id", target_run_id
                    )
                if record.threshold_stability_summary is not None:
                    payload = _normalize_threshold_curve_payload(
                        dict(record.threshold_stability_summary)
                    )
                    payload["source"] = "persisted"
                    payload["run_id"] = target_run_id
                    if response_format == "markdown":
                        return _threshold_curve_to_markdown(payload)
                    return json.dumps(payload, indent=2)

        if session.model is None:
            raise ToolError("No model found. Run run_topological_sweep() first.")
        if target_run_id and target_run_id != session.latest_run_id:
            return mcp_error(
                "get_threshold_stability_curve",
                "Live threshold-curve recomputation is only available for the current run.",
                error_code="RUN_NOT_LIVE",
                details={
                    "requested_run_id": target_run_id,
                    "latest_run_id": session.latest_run_id,
                    "available_source": (
                        "persisted"
                        if source == "persisted" and detail == "summary"
                        else None
                    ),
                },
            )
        payload = await asyncio.to_thread(
            _threshold_curve_payload,
            session.model,
            detail=detail,
            threshold_candidate_policy=threshold_candidate_policy,
        )
        payload = _normalize_threshold_curve_payload(payload)
        if run_id:
            payload["run_id"] = run_id
        if response_format == "markdown":
            return _threshold_curve_to_markdown(payload)
        return json.dumps(payload, indent=2)
    except Exception as e:
        logger.error(f"Error computing stability curve: {e}")
        return mcp_error("get_threshold_stability_curve", str(e))


async def create_graph_artifact(
    kind: Literal["spectral_sparsifier"] = "spectral_sparsifier",
    run_id: str = "",
    epsilon: float = 1.0,
    seed: int = 42,
    sketch_dim: int | None = None,
    sample_count: int | None = None,
    pcg_tol: float = 1e-6,
    max_iter: int = 1000,
    estimate_only: bool = True,
    ctx: Context = None,
) -> str:
    """Estimate or build a read-only graph artifact for the live fitted run.

    Expensive by design: estimate_only defaults true, so agents must explicitly
    opt in before paying for spectral sparsification.
    """
    session = _get_session(ctx)
    if session.model is None:
        raise ToolError("No model found. Run run_topological_sweep() first.")
    target_run_id = run_id or session.latest_run_id
    if not target_run_id:
        raise ToolError("No run available. Run run_topological_sweep() first.")
    if target_run_id != session.latest_run_id:
        return mcp_error(
            "create_graph_artifact",
            "Artifacts can only be built from the current live run.",
            error_code="RUN_NOT_LIVE",
            agent_action="Rerun or select the latest run before building a graph artifact.",
            details={
                "requested_run_id": target_run_id,
                "latest_run_id": session.latest_run_id,
            },
        )
    if kind != "spectral_sparsifier":
        return mcp_error(
            "create_graph_artifact",
            f"Unsupported graph artifact kind: {kind!r}",
            error_code="GRAPH_ARTIFACT_KIND_UNSUPPORTED",
            agent_action="Use kind='spectral_sparsifier'. structural_backbone is planned but not callable.",
        )
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        return mcp_error(
            "create_graph_artifact", "epsilon must be finite and positive."
        )

    n_nodes = int(session.model.cosmic_rust.n)
    edge_count = int(session.model.cosmic_rust.n_edges)
    estimate = _artifact_estimate(n_nodes, edge_count, epsilon)
    params = {
        "epsilon": epsilon,
        "seed": seed,
        "sketch_dim": sketch_dim,
        "sample_count": sample_count,
        "pcg_tol": pcg_tol,
        "max_iter": max_iter,
    }
    if estimate_only:
        return json.dumps(
            {
                "status": "ok",
                "estimate_only": True,
                "kind": kind,
                "run_id": target_run_id,
                "input": {"n_nodes": n_nodes, "edge_count": edge_count},
                "estimated_cost": estimate,
                "params": params,
                "build_call": {
                    "tool": "create_graph_artifact",
                    "args": {
                        "kind": kind,
                        "run_id": target_run_id,
                        "epsilon": epsilon,
                        "seed": seed,
                        "estimate_only": False,
                    },
                },
            },
            indent=2,
        )

    started = time.perf_counter()
    sparse = await asyncio.to_thread(
        session.model.spectral_sparsify,
        epsilon,
        seed=seed,
        sketch_dim=sketch_dim,
        sample_count=sample_count,
        pcg_tol=pcg_tol,
        max_iter=max_iter,
        update=False,
    )
    elapsed = time.perf_counter() - started
    raw_edges = sparse.weighted_edges()
    scale = max(1.0, max((float(w) for _, _, w in raw_edges), default=0.0))
    graph = cosmic_to_networkx(sparse, threshold=0.0, scale=scale)
    normalized_edges = [(int(i), int(j), float(w) / scale) for i, j, w in raw_edges]
    metrics = dataclasses.asdict(
        _graph_metrics_from_graph(
            graph,
            resolved_construction_threshold=0.0,
            n_ball_maps=0,
            full_weight_edges=normalized_edges,
        )
    )
    artifact_id = f"graph_{uuid.uuid4().hex[:12]}"
    graph_summary = _build_graph_summary_from_graph(
        graph,
        resolved_construction_threshold=0.0,
    )
    artifact = GraphArtifact(
        artifact_id=artifact_id,
        run_id=target_run_id,
        kind=kind,
        created_at=time.time(),
        params=params,
        graph_summary=graph_summary,
        metrics=metrics,
    )
    session.graph_artifacts[artifact_id] = artifact
    return json.dumps(
        {
            "status": "ok",
            "estimate_only": False,
            "artifact_id": artifact_id,
            "kind": kind,
            "run_id": target_run_id,
            "runtime_s": round(elapsed, 3),
            "input_edge_count": edge_count,
            "artifact_edge_count": graph.number_of_edges(),
            "params": params,
            "staleness": _artifact_staleness(artifact, session.latest_run_id),
        },
        indent=2,
    )


async def get_topological_skeleton(
    run_id: str = "",
    surface: Literal["constructed", "artifact"] = "constructed",
    artifact_id: str = "",
    detail: Literal["summary", "nodes", "edges", "full", "full_nodes"] = "summary",
    max_edges: int = 100,
    max_nodes: int = 100,
    ctx: Context = None,
) -> str:
    """Graph connectivity structure for latest run or explicit `run_id`.

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
        if surface == "artifact":
            if not artifact_id:
                raise ToolError("artifact_id is required when surface='artifact'.")
            artifact = session.graph_artifacts.get(artifact_id)
            if artifact is None:
                return unknown_handle_error(
                    "get_topological_skeleton", "artifact_id", artifact_id
                )
            payload = {
                "run_id": artifact.run_id,
                "artifact_id": artifact.artifact_id,
                "artifact_kind": artifact.kind,
                "config_yaml_omitted": True,
                "graph_surface": f"graph artifact {artifact_id} ({artifact.kind})",
                "artifact_staleness": _artifact_staleness(
                    artifact,
                    session.latest_run_id,
                ),
                "graph": _skeleton_graph_payload(
                    artifact.graph_summary,
                    detail=detail,
                    max_edges=max_edges,
                    max_nodes=max_nodes,
                ),
                "available_followup_tools": [
                    "diagnose_cosmic_graph",
                    "compare_sweeps",
                ],
            }
            return json.dumps(payload, indent=2)

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
            "available_followup_tools": [
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
