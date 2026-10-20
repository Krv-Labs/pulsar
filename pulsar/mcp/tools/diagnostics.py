from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import time
from typing import Annotated, Any, Literal
import uuid

import numpy as np
from fastmcp import Context
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from pydantic import Field

from pulsar.analysis import cosmic_to_networkx
from pulsar.config import THRESHOLD_RANGE_MESSAGE
from pulsar.mcp.diagnostics import (
    _build_graph_summary_from_graph,
    _graph_metrics_from_graph,
    _skeleton_graph_payload,
    diagnose_model,
)
from pulsar.mcp.errors import mcp_error, unknown_handle_error
from pulsar.mcp.registry import registry
from pulsar.mcp.results import mcp_result
from pulsar.mcp.session import (
    GraphArtifact,
    ResolvedRun,
    SessionConflictError,
    _get_session,
    is_latest,
    resolve_run,
    run_session_worker,
)
from pulsar.mcp.thresholds import (
    THRESHOLD_CANDIDATE_POLICIES,
    agent_threshold_options,
    component_mass_profile,
    first_report_ready_candidate,
    mass_profile_hint,
    prepare_threshold_graph_from_edges,
    significant_component_sizes,
    structural_breakpoints,
    threshold_morphology_profile,
    useful_component_size_floor,
)
from pulsar.mcp.toolmeta import (
    ANY,
    BOOL,
    INT,
    NUM,
    STR,
    arr,
    enum,
    nullable,
    obj,
    tool_meta,
)

logger = logging.getLogger(__name__)

OPEN_OBJ = obj({}, additional=True)


def model_not_loaded_error(tool: str) -> ToolResult:
    return mcp_error(
        tool,
        "No fitted model is loaded in this session.",
        error_code="MODEL_NOT_LOADED",
        agent_action=(
            "Run run_topological_sweep first, or pass run_id of a cached run."
        ),
    )


def resolve_run_or_error(
    session: Any, tool: str, run_id: str
) -> tuple[ResolvedRun | None, ToolResult | None]:
    """Resolve ``run_id`` (empty = live) to a run, or return the error envelope."""
    try:
        return resolve_run(session, run_id), None
    except LookupError:
        if run_id:
            return None, unknown_handle_error(tool, "run_id", run_id)
        return None, model_not_loaded_error(tool)


def stale_write_error(tool: str, noun: str) -> ToolResult:
    return mcp_error(
        tool,
        f"Refusing to publish a {noun} for a run that is not the session's latest.",
        error_code="SESSION_REVISION_CONFLICT",
        agent_action=(
            "Omit run_id or re-run the sweep; dossiers and artifacts can only be "
            "published for the session's latest run."
        ),
    )


def _validate_unit_threshold(value: float, *, name: str) -> float:
    threshold = float(value)
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ToolError(f"{name}: {THRESHOLD_RANGE_MESSAGE}")
    return threshold


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


def _component_morphology(metrics: dict[str, Any], *, detail: str) -> dict[str, Any]:
    sizes = sorted(
        (int(size) for size in metrics.get("component_sizes", [])), reverse=True
    )
    n_nodes = int(metrics.get("n_nodes", 0) or 0)
    floor = useful_component_size_floor(n_nodes)  # descriptive context only
    # Single source of truth for "real mode vs. dust": relative, gap-based
    # significance (not the absolute n-scaled floor). significant is the size-
    # ranked prefix, so the remainder is the dust tail.
    nontrivial, _ = significant_component_sizes(sizes)
    tail = sizes[len(nontrivial) :]
    largest = sizes[0] if sizes else 0
    second = sizes[1] if len(sizes) > 1 else 0
    probs = np.asarray(sizes, dtype=float) / max(sum(sizes), 1)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum()) if len(probs) else 0.0
    normalized_entropy = entropy / np.log(len(sizes)) if len(sizes) > 1 else 0.0
    top_two_balance = (
        1.0 - abs(largest - second) / max(largest + second, 1) if second else 0.0
    )
    out = {
        "component_count": int(metrics.get("component_count", 0) or 0),
        "top_component_sizes": sizes[:10],
        "top_component_sizes_omitted": max(len(sizes) - 10, 0),
        "largest_component_fraction": float(metrics.get("giant_fraction", 0.0) or 0.0),
        "second_largest_ratio": round(second / largest if largest else 0.0, 4),
        "singleton_count": int(metrics.get("singleton_count", 0) or 0),
        "singleton_fraction": float(metrics.get("singleton_fraction", 0.0) or 0.0),
        "nontrivial_component_floor": floor,
        "nontrivial_component_count": len(nontrivial),
        "nontrivial_mass_fraction": round(sum(nontrivial) / max(n_nodes, 1), 4),
        "multi_component_coverage": round(sum(nontrivial[1:]) / max(n_nodes, 1), 4),
        "tail_component_count": len(tail),
        "tail_mass_fraction": round(sum(tail) / max(n_nodes, 1), 4),
        "component_size_entropy": round(normalized_entropy, 4),
        "top_two_balance": round(top_two_balance, 4),
    }
    if detail == "full":
        out["component_sizes"] = sizes
    return out


def _observed_patterns(
    metrics: dict[str, Any], morphology: dict[str, Any]
) -> list[str]:
    patterns: list[str] = []
    if int(metrics.get("n_edges", 0) or 0) == 0:
        patterns.append("empty_graph")
    if morphology["nontrivial_component_count"] >= 2:
        patterns.append("multiple_nontrivial_components")
    if morphology["largest_component_fraction"] >= 0.95:
        patterns.append("dominant_component")
    else:
        patterns.append("no_dominant_component")
    if morphology["singleton_fraction"] >= 0.25:
        patterns.append("singleton_residual_present")
    else:
        patterns.append("low_singleton_residual")
    if morphology["top_two_balance"] >= 0.7:
        patterns.append("high_top_component_balance")
    if float(metrics.get("density", 0.0) or 0.0) > 0.8:
        patterns.append("dense_connectivity")
    grid_status = str(metrics.get("grid_adequacy_status", "unknown"))
    if grid_status == "sample_count_ok":
        patterns.append("adequate_grid_sample")
    elif grid_status in {"under_sampled", "thin_grid"}:
        patterns.append(grid_status)
    return patterns


def _risk_factors(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    n_edges = int(metrics.get("n_edges", 0) or 0)
    density = float(metrics.get("density", 0.0) or 0.0)
    singleton_fraction = float(metrics.get("singleton_fraction", 0.0) or 0.0)
    giant_fraction = float(metrics.get("giant_fraction", 0.0) or 0.0)
    grid_status = str(metrics.get("grid_adequacy_status", "unknown"))
    if n_edges == 0:
        risks.append({"code": "empty_graph", "severity": "error"})
    if density > 0.8 and n_edges > 0:
        risks.append({"code": "dense_hairball", "severity": "warning"})
    if singleton_fraction > 0.8:
        risks.append({"code": "singleton_heavy", "severity": "warning"})
    if giant_fraction > 0.95:
        risks.append({"code": "dominant_component", "severity": "warning"})
    if grid_status in {"under_sampled", "thin_grid"}:
        risks.append({"code": grid_status, "severity": "warning"})
    return risks


def _diagnosis_payload(
    metrics: dict[str, Any],
    *,
    detail: str,
    source: str,
    run_id: str | None,
    graph_surface: dict[str, Any],
    artifact_staleness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    morphology = _component_morphology(metrics, detail=detail)
    payload = {
        "status": "ok",
        "detail": detail,
        "source": source,
        "run_id": run_id,
        "graph_surface": graph_surface,
        "scale": {
            "n_nodes": int(metrics.get("n_nodes", 0) or 0),
            "n_edges": int(metrics.get("n_edges", 0) or 0),
            "density": float(metrics.get("density", 0.0) or 0.0),
            "avg_degree": float(metrics.get("avg_degree", 0.0) or 0.0),
            "nonzero_fraction_full_weighted": float(
                metrics.get("nonzero_fraction", 0.0) or 0.0
            ),
        },
        "component_morphology": morphology,
        "weight_distribution": {
            "p25": float(metrics.get("weight_p25", 0.0) or 0.0),
            "p50": float(metrics.get("weight_p50", 0.0) or 0.0),
            "p95": float(metrics.get("weight_p95", 0.0) or 0.0),
            "sparkline": metrics.get("weight_distribution_sparkline", ""),
            "basis": "upper-triangle node-pair distribution; zeros include absent edges",
        },
        "sweep_support": {
            "n_ball_maps": int(metrics.get("n_ball_maps", 0) or 0),
            "grid_adequacy_status": metrics.get("grid_adequacy_status", "unknown"),
            "grid_adequacy_note": metrics.get("grid_adequacy_note", ""),
        },
        "observed_patterns": _observed_patterns(metrics, morphology),
        "risk_factors": _risk_factors(metrics),
    }
    # Realized accuracy of the MinHash-constructed weights at the depth used. Present
    # only on the live constructed graph built via MinHash; describes the construction
    # surface (distinct from the interpretation threshold above).
    minhash_profile = metrics.get("minhash_profile")
    if minhash_profile:
        payload["minhash_profile"] = minhash_profile
    if artifact_staleness is not None:
        payload["artifact_staleness"] = artifact_staleness
    return payload


def _diagnosis_to_markdown(payload: dict[str, Any]) -> str:
    surface = payload["graph_surface"]
    scale = payload["scale"]
    morph = payload["component_morphology"]
    weights = payload["weight_distribution"]
    support = payload["sweep_support"]
    risk_codes = [row["code"] for row in payload["risk_factors"]]
    lines = [
        "# Cosmic Graph Diagnosis",
        "",
        f"- Source: {payload['source']}",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Surface: {surface['kind']} @ construction threshold "
        f"{surface['construction_threshold']:.4f}",
        f"- Nodes / edges: {scale['n_nodes']} / {scale['n_edges']}",
        f"- Density: {scale['density']:.4f}",
        f"- Components: {morph['component_count']}",
        "- Top component sizes: "
        + ", ".join(str(size) for size in morph["top_component_sizes"]),
        f"- Largest component: {morph['largest_component_fraction']:.1%}",
        f"- Singleton fraction: {morph['singleton_fraction']:.1%}",
        f"- Nontrivial components: {morph['nontrivial_component_count']} "
        f"(floor={morph['nontrivial_component_floor']})",
        f"- Grid support: {support['grid_adequacy_status']} "
        f"({support['n_ball_maps']} ball maps)",
        f"- Weight p25/p50/p95: {weights['p25']:.4f} / "
        f"{weights['p50']:.4f} / {weights['p95']:.4f}",
        "- Observed patterns: " + ", ".join(payload["observed_patterns"]),
        "- Risk factors: " + (", ".join(risk_codes) if risk_codes else "none"),
    ]
    mh = payload.get("minhash_profile")
    if mh:
        lines.append(
            f"- MinHash construction (d={mh['d']}): 95% CI "
            f"±{mh['ci95_half_width_worst']}, signature {mh['signature_memory_human']}"
        )
    return "\n".join(lines)


_DIAGNOSE_SCHEMA = obj(
    {
        "status": enum(["ok"]),
        "detail": enum(["summary", "full"]),
        "source": enum(["live", "artifact"]),
        "run_id": nullable(STR),
        "graph_surface": obj(
            {
                "kind": STR,
                "artifact_id": STR,
                "construction_threshold": NUM,
                "threshold_role": STR,
                "preserves_current_fit": BOOL,
            },
            ["kind", "construction_threshold", "threshold_role"],
            additional=True,
        ),
        "scale": obj(
            {
                "n_nodes": INT,
                "n_edges": INT,
                "density": NUM,
                "avg_degree": NUM,
                "nonzero_fraction_full_weighted": NUM,
            },
            ["n_nodes", "n_edges", "density", "avg_degree"],
            additional=True,
        ),
        "component_morphology": obj(
            {
                "component_count": INT,
                "top_component_sizes": arr(INT),
                "top_component_sizes_omitted": INT,
                "largest_component_fraction": NUM,
                "second_largest_ratio": NUM,
                "singleton_count": INT,
                "singleton_fraction": NUM,
                "nontrivial_component_floor": NUM,
                "nontrivial_component_count": INT,
                "nontrivial_mass_fraction": NUM,
                "multi_component_coverage": NUM,
                "tail_component_count": INT,
                "tail_mass_fraction": NUM,
                "component_size_entropy": NUM,
                "top_two_balance": NUM,
                "component_sizes": arr(INT),
            },
            ["component_count", "top_component_sizes", "largest_component_fraction"],
            additional=True,
        ),
        "weight_distribution": obj(
            {"p25": NUM, "p50": NUM, "p95": NUM, "sparkline": STR, "basis": STR},
            ["p25", "p50", "p95"],
            additional=True,
        ),
        "sweep_support": obj(
            {
                "n_ball_maps": INT,
                "grid_adequacy_status": STR,
                "grid_adequacy_note": STR,
            },
            ["n_ball_maps", "grid_adequacy_status"],
            additional=True,
        ),
        "observed_patterns": arr(STR),
        "risk_factors": arr(
            obj({"code": STR, "severity": STR}, ["code"], additional=True)
        ),
        "minhash_profile": OPEN_OBJ,
        "artifact_staleness": obj(
            {
                "stale": BOOL,
                "artifact_run_id": nullable(STR),
                "latest_run_id": nullable(STR),
                "warning": nullable(STR),
            },
            ["stale"],
            additional=True,
        ),
    },
    [
        "status",
        "detail",
        "source",
        "graph_surface",
        "scale",
        "component_morphology",
        "weight_distribution",
        "sweep_support",
        "observed_patterns",
        "risk_factors",
    ],
)


@tool_meta(
    read_only=True,
    idempotent=True,
    output_schema=_DIAGNOSE_SCHEMA,
    title="Diagnose Cosmic Graph",
)
async def diagnose_cosmic_graph(
    surface: Annotated[
        Literal["constructed", "artifact"],
        Field(
            description=(
                "'constructed' (default): the fitted cosmic graph at its construction threshold. 'artifact': a create_graph_artifact graph; artifact_id is then required."
            )
        ),
    ] = "constructed",
    artifact_id: Annotated[
        str,
        Field(
            description=(
                "Artifact handle from create_graph_artifact (e.g. 'graph_ab12cd34ef56'). "
                "Only used, and then required, when surface='artifact'."
            )
        ),
    ] = "",
    detail: Annotated[
        Literal["summary", "full"],
        Field(
            description=(
                "'summary' (default) returns the top 10 component sizes. 'full' "
                "additionally lists every component size (O(components) entries)."
            )
        ),
    ] = "summary",
    response_format: Annotated[
        Literal["json", "markdown"],
        Field(
            description=(
                "Text view: 'json' (default) or 'markdown'. structured_content is identical."
            )
        ),
    ] = "json",
    run_id: Annotated[
        str,
        Field(
            description=(
                "Run to read, from run_topological_sweep. Omit for the latest run. "
                "Ignored when surface='artifact' (artifacts carry their own run_id)."
            )
        ),
    ] = "",
    ctx: Context = None,
) -> ToolResult:
    """Graph-state observables for a fitted cosmic graph or graph artifact.

    Returns scale, component morphology, weight distribution, sweep support,
    observed patterns and risk factors (plus MinHash weight accuracy when used).
    Measurement only, no recommended action. Use after run_topological_sweep;
    use get_threshold_stability_curve to compare construction thresholds.
    """
    tool = "diagnose_cosmic_graph"
    session = _get_session(ctx)
    if detail not in {"summary", "full"}:
        return mcp_error(
            tool,
            "detail must be 'summary' or 'full'.",
            error_code="INVALID_ARGUMENT",
            agent_action="Pass detail='summary' or detail='full'.",
        )
    if response_format not in {"json", "markdown"}:
        return mcp_error(
            tool,
            "response_format must be 'json' or 'markdown'.",
            error_code="INVALID_ARGUMENT",
            agent_action="Pass response_format='json' or 'markdown'.",
        )

    try:
        if surface == "artifact":
            if not artifact_id:
                return mcp_error(
                    tool,
                    "artifact_id is required when surface='artifact'.",
                    error_code="ARTIFACT_ID_REQUIRED",
                    agent_action=(
                        "Pass the artifact_id returned by create_graph_artifact, "
                        "or use surface='constructed'."
                    ),
                )
            artifact = session.graph_artifacts.get(artifact_id)
            if artifact is None:
                return unknown_handle_error(tool, "artifact_id", artifact_id)
            staleness = _artifact_staleness(artifact, session.latest_run_id)
            payload = _diagnosis_payload(
                dict(artifact.metrics),
                detail=detail,
                source="artifact",
                run_id=artifact.run_id,
                graph_surface={
                    "kind": artifact.kind,
                    "artifact_id": artifact.artifact_id,
                    "construction_threshold": float(
                        artifact.metrics.get("resolved_construction_threshold", 0.0)
                    ),
                    "threshold_role": "artifact_construction",
                    "preserves_current_fit": False,
                },
                artifact_staleness=staleness,
            )
        else:
            resolved, err = resolve_run_or_error(session, tool, run_id)
            if err is not None:
                return err
            result = await asyncio.to_thread(diagnose_model, resolved.model)
            metrics = dataclasses.asdict(result)
            payload = _diagnosis_payload(
                metrics,
                detail=detail,
                source="live",
                run_id=resolved.run_id or None,
                graph_surface={
                    "kind": "constructed_cosmic_graph",
                    "construction_threshold": float(
                        result.resolved_construction_threshold
                    ),
                    "threshold_role": "construction",
                    "preserves_current_fit": True,
                },
            )
        if response_format == "markdown":
            return mcp_result(payload, text=_diagnosis_to_markdown(payload))
        return mcp_result(payload)
    except Exception as e:
        logger.error(f"Error diagnosing graph: {e}")
        return mcp_error(
            tool,
            str(e),
            error_code="DIAGNOSE_FAILED",
            agent_action="Check get_runtime_context for the session state, then retry.",
        )


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
        top_k = 3 if max_points is None else 10
        rows = []
        for i in indices:
            row = threshold_morphology_profile(
                threshold_graph, thresholds[i], top_k=top_k
            )
            if max_points is None:
                row = {
                    "threshold": round(float(row["threshold"]), 6),
                    "component_count": row["component_count"],
                    "top_component_sizes": row["top_component_sizes"],
                    "giant_fraction": row["giant_fraction"],
                    "second_largest_ratio": row["second_largest_ratio"],
                    "singleton_fraction": row["singleton_fraction"],
                }
            rows.append(row)
        return rows

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
            "unthresholded_component_count": (
                int(component_counts[-1]) if component_counts else None
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
    candidate = first_report_ready_candidate(payload.get("threshold_candidates"))
    if candidate is not None:
        # Stable-plateau candidates store the (midpoint) cut under "threshold";
        # only transition-adjacent candidates use "midpoint". Prefer "threshold"
        # to match the rest of this module's key ordering.
        threshold = candidate.get("threshold", candidate.get("midpoint"))
        tier = candidate.get("interpretability_tier", "candidate")
        return [
            '- `generate_cluster_dossier(method="components", '
            f"interpretation_edge_weight_threshold={_format_threshold_value(threshold)})` "
            f"for natural H0 components at the {tier} threshold lens.",
            '- `generate_cluster_dossier(method="spectral")` only if the question is latent structure inside a still-continuous dominant component.',
            "- `get_feature_signal` / `compare_clusters` to validate that the component slice is domain-relevant.",
        ]

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


_MORPH = OPEN_OBJ
_CURVE_SCHEMA = obj(
    {
        "status": enum(["ok"]),
        "detail": enum(["summary", "full"]),
        "source": enum(["live", "persisted"]),
        "run_id": nullable(STR),
        "resolved_construction_threshold": nullable(NUM),
        "h0_longest_plateau_threshold": nullable(NUM),
        "matches_current_threshold": BOOL,
        "current_threshold_morphology": _MORPH,
        "h0_longest_plateau_morphology": _MORPH,
        "h0_plateau_readout": STR,
        "threshold_guidance": STR,
        "threshold_candidate_policy": STR,
        "selection_strategy": ANY,
        "threshold_candidates": arr(OPEN_OBJ),
        "threshold_candidates_omitted": INT,
        "threshold_profiles": arr(OPEN_OBJ),
        "threshold_profiles_omitted": INT,
        "unthresholded_component_count": nullable(INT),
        "structural_breakpoints": arr(OPEN_OBJ),
        "structural_breakpoints_omitted": INT,
        "structural_breakpoints_filter": STR,
        "structural_breakpoints_guidance": STR,
        "plateaus_returned": INT,
        "plateaus_omitted": INT,
        "curve_point_count": INT,
        "full_detail_available": STR,
        "agent_threshold_options": OPEN_OBJ,
        "plateaus": arr(OPEN_OBJ),
        "curve_sample": arr(OPEN_OBJ),
        "curve_sample_omitted": INT,
        "curve_sample_guidance": STR,
        "thresholds": arr(NUM),
        "component_counts": arr(INT),
        "singleton_counts": arr(INT),
    },
    ["status", "source"],
    additional=True,  # persisted summaries from older runs may carry extra keys
)


@tool_meta(
    read_only=True,
    idempotent=True,
    output_schema=_CURVE_SCHEMA,
    title="Threshold Stability Curve",
)
async def get_threshold_stability_curve(
    detail: Annotated[
        Literal["summary", "full"],
        Field(
            description=(
                "'summary' (default): bounded decision packet. 'full': adds plateaus, a curve sample and raw O(thresholds) arrays; recomputes live."
            )
        ),
    ] = "summary",
    run_id: Annotated[
        str,
        Field(
            description=(
                "Run to read, from run_topological_sweep. Omit for the latest run. Live recomputation (source='live' or detail='full') needs the latest run, else RUN_NOT_LIVE."
            )
        ),
    ] = "",
    source: Annotated[
        Literal["persisted", "live"],
        Field(
            description=(
                "'persisted' (default): summary stored at sweep time (cheap, works for old runs). 'live': recompute from the in-memory model."
            )
        ),
    ] = "persisted",
    threshold_candidate_policy: Annotated[
        Literal["balanced", "report_ready", "detail_seeking", "outlier_mining"],
        Field(
            description=(
                "Lens for ranking candidates (not a score): 'balanced' (default), 'report_ready' (few large components), 'detail_seeking' (finer splits), 'outlier_mining' (singleton tails)."
            )
        ),
    ] = "balanced",
    response_format: Annotated[
        Literal["markdown", "json"],
        Field(
            description=(
                "Text view: 'markdown' (default) or 'json'. structured_content is identical."
            )
        ),
    ] = "markdown",
    ctx: Context = None,
) -> ToolResult:
    """Threshold morphology and candidate lenses along the H0 component-stability curve.

    Compares the fitted construction threshold with H0 plateaus and structural
    breakpoints. Use to judge or choose a construction threshold before refine_config;
    use diagnose_cosmic_graph for the graph already built.
    """
    tool = "get_threshold_stability_curve"
    if threshold_candidate_policy not in THRESHOLD_CANDIDATE_POLICIES:
        return mcp_error(
            tool,
            "threshold_candidate_policy must be one of "
            f"{sorted(THRESHOLD_CANDIDATE_POLICIES)}, got "
            f"'{threshold_candidate_policy}'",
            error_code="INVALID_ARGUMENT",
            agent_action="Pass one of the listed threshold_candidate_policy values.",
        )
    if response_format not in {"markdown", "json"}:
        return mcp_error(
            tool,
            "response_format must be 'markdown' or 'json'.",
            error_code="INVALID_ARGUMENT",
            agent_action="Pass response_format='markdown' or 'json'.",
        )

    def _render(payload: dict[str, Any]) -> ToolResult:
        if response_format == "markdown":
            return mcp_result(payload, text=_threshold_curve_to_markdown(payload))
        return mcp_result(payload)

    try:
        session = _get_session(ctx)
        target_run_id = run_id or session.latest_run_id
        if source == "persisted" and detail == "summary":
            if target_run_id:
                record = registry.get_run(target_run_id)
                if record is None:
                    return unknown_handle_error(tool, "run_id", target_run_id)
                if record.threshold_stability_summary is not None:
                    payload = _normalize_threshold_curve_payload(
                        dict(record.threshold_stability_summary)
                    )
                    payload["source"] = "persisted"
                    payload["run_id"] = target_run_id
                    return _render(payload)

        resolved, err = resolve_run_or_error(session, tool, run_id)
        if err is not None:
            return err
        if run_id and not is_latest(session, resolved):
            return mcp_error(
                tool,
                "Live threshold-curve recomputation is only available for the current run.",
                error_code="RUN_NOT_LIVE",
                agent_action=(
                    "Use source='persisted' with detail='summary' for this run, or "
                    "omit run_id to recompute for the latest run."
                ),
                details={
                    "requested_run_id": run_id,
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
            resolved.model,
            detail=detail,
            threshold_candidate_policy=threshold_candidate_policy,
        )
        payload = _normalize_threshold_curve_payload(payload)
        payload["run_id"] = resolved.run_id or None
        return _render(payload)
    except Exception as e:
        logger.error(f"Error computing stability curve: {e}")
        return mcp_error(
            tool,
            str(e),
            error_code="THRESHOLD_CURVE_FAILED",
            agent_action="Check get_runtime_context for the session state, then retry.",
        )


def _build_sparsifier_artifact_worker(
    model: Any,
    target_run_id: str,
    kind: str,
    epsilon: float,
    seed: int,
    sketch_dim: int | None,
    sample_count: int | None,
    pcg_tol: float,
    max_iter: int,
    params: dict[str, Any],
) -> tuple[GraphArtifact, float, int]:
    started = time.perf_counter()
    sparse = model.spectral_sparsify(
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
    # Pure compute: the caller publishes the artifact on the event loop.
    return artifact, elapsed, graph.number_of_edges()


_ARTIFACT_PARAMS = obj(
    {
        "epsilon": NUM,
        "seed": INT,
        "sketch_dim": nullable(INT),
        "sample_count": nullable(INT),
        "pcg_tol": NUM,
        "max_iter": INT,
    },
    additional=True,
)
_ARTIFACT_SCHEMA = obj(
    {
        "status": enum(["ok"]),
        "estimate_only": BOOL,
        "kind": STR,
        "run_id": STR,
        "input": obj({"n_nodes": INT, "edge_count": INT}, additional=True),
        "estimated_cost": obj(
            {
                "runtime_s": NUM,
                "runtime_basis": STR,
                "peak_mem_mb": NUM,
                "edge_count": INT,
                "sketch_dim": INT,
                "sample_count": INT,
            },
            additional=True,
        ),
        "params": _ARTIFACT_PARAMS,
        "build_call": obj({"tool": STR, "args": OPEN_OBJ}, additional=True),
        "artifact_id": STR,
        "runtime_s": NUM,
        "input_edge_count": INT,
        "artifact_edge_count": INT,
        "staleness": obj(
            {
                "stale": BOOL,
                "artifact_run_id": nullable(STR),
                "latest_run_id": nullable(STR),
                "warning": nullable(STR),
            },
            additional=True,
        ),
    },
    ["status", "estimate_only", "kind", "run_id", "params"],
)


@tool_meta(
    read_only=False,
    idempotent=False,
    output_schema=_ARTIFACT_SCHEMA,
    title="Graph Artifact",
)
async def create_graph_artifact(
    kind: Annotated[
        Literal["spectral_sparsifier"],
        Field(
            description=(
                "Artifact type. 'spectral_sparsifier' builds an edge-sparsified "
                "approximation of the weighted graph (the only supported kind)."
            )
        ),
    ] = "spectral_sparsifier",
    run_id: Annotated[
        str,
        Field(
            description=(
                "Run to use, from run_topological_sweep. Omit for the latest run. Any cached run works with estimate_only=True (a pure read); building publishes into the session, so it must be the latest run (else SESSION_REVISION_CONFLICT)."
            )
        ),
    ] = "",
    epsilon: Annotated[
        float,
        Field(
            description=(
                "Sparsifier accuracy in (0, inf). Smaller is more faithful and "
                "costs roughly 1/epsilon^2 more time and edges. Default 1.0."
            )
        ),
    ] = 1.0,
    seed: Annotated[
        int, Field(description="Random seed for the sparsifier sketch. Default 42.")
    ] = 42,
    sketch_dim: Annotated[
        int | None,
        Field(
            description=(
                "Johnson-Lindenstrauss sketch rows. Omit to derive from epsilon; "
                "set a small value only for quick tests."
            )
        ),
    ] = None,
    sample_count: Annotated[
        int | None,
        Field(
            description=(
                "Number of edge samples drawn. Omit to derive from epsilon and node count."
            )
        ),
    ] = None,
    pcg_tol: Annotated[
        float,
        Field(description="Convergence tolerance of the preconditioned CG solves."),
    ] = 1e-6,
    max_iter: Annotated[
        int, Field(description="Maximum CG iterations per solve.")
    ] = 1000,
    estimate_only: Annotated[
        bool,
        Field(
            description=(
                "True (default): return only a cost estimate (runtime, memory, edges). False: build and store the artifact (seconds to minutes)."
            )
        ),
    ] = True,
    ctx: Context = None,
) -> ToolResult:
    """Estimate or build a spectral-sparsifier graph artifact (estimates for any cached run; builds for the latest).

    Expensive by design: estimate_only defaults true. A built artifact gets an
    artifact_id usable with diagnose_cosmic_graph and get_topological_skeleton
    (surface='artifact'). It never changes the fitted graph.
    """
    tool = "create_graph_artifact"
    session = _get_session(ctx)
    resolved, err = resolve_run_or_error(session, tool, run_id)
    if err is not None:
        return err
    target_run_id = resolved.run_id
    if not target_run_id:
        return model_not_loaded_error(tool)
    if kind != "spectral_sparsifier":
        return mcp_error(
            tool,
            f"Unsupported graph artifact kind: {kind!r}",
            error_code="GRAPH_ARTIFACT_KIND_UNSUPPORTED",
            agent_action="Use kind='spectral_sparsifier'. structural_backbone is planned but not callable.",
        )
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        return mcp_error(
            tool,
            "epsilon must be finite and positive.",
            error_code="INVALID_ARGUMENT",
            agent_action="Pass a finite epsilon > 0 (default 1.0).",
        )

    model = resolved.model
    n_nodes = int(model.cosmic_rust.n)
    edge_count = int(model.cosmic_rust.n_edges)
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
        return mcp_result(
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
            }
        )

    if run_id and not is_latest(session, resolved):
        return stale_write_error(tool, "graph artifact")

    artifact, elapsed, artifact_edge_count = await run_session_worker(
        session,
        _build_sparsifier_artifact_worker,
        model,
        target_run_id,
        kind,
        epsilon,
        seed,
        sketch_dim,
        sample_count,
        pcg_tol,
        max_iter,
        params,
        heavy=True,
        mutates_session=True,
    )
    try:
        # Back on the loop: single synchronous publish (run check + write).
        session.publish_graph_artifact(artifact, expected_run_id=target_run_id)
    except SessionConflictError as e:
        return mcp_error(
            tool,
            str(e),
            error_code=e.error_code,
            agent_action=(
                "The artifact was not published. The live run changed while it "
                "was building; call get_runtime_context and rebuild against the "
                "current run."
            ),
            details={"requested_run_id": target_run_id},
        )
    return mcp_result(
        {
            "status": "ok",
            "estimate_only": False,
            "artifact_id": artifact.artifact_id,
            "kind": kind,
            "run_id": target_run_id,
            "runtime_s": round(elapsed, 3),
            "input_edge_count": edge_count,
            "artifact_edge_count": artifact_edge_count,
            "params": params,
            "staleness": _artifact_staleness(artifact, session.latest_run_id),
        }
    )


_SKELETON_SCHEMA = obj(
    {
        "run_id": STR,
        "dataset_id": nullable(STR),
        "artifact_id": STR,
        "artifact_kind": STR,
        "config_yaml_omitted": BOOL,
        "config_yaml": STR,
        "resolved_construction_threshold": nullable(NUM),
        "graph_surface": STR,
        "artifact_staleness": obj(
            {"stale": BOOL, "artifact_run_id": nullable(STR), "warning": nullable(STR)},
            ["stale"],
            additional=True,
        ),
        "graph": OPEN_OBJ,
        "available_followup_tools": arr(STR),
    },
    [
        "run_id",
        "config_yaml_omitted",
        "graph_surface",
        "graph",
        "available_followup_tools",
    ],
)


@tool_meta(
    read_only=True,
    idempotent=True,
    output_schema=_SKELETON_SCHEMA,
    title="Topological Skeleton",
)
async def get_topological_skeleton(
    run_id: Annotated[
        str,
        Field(
            description=(
                "Run to read, from run_topological_sweep. Omit for the latest run. Reads the on-disk run record, so any stored run works on stateless transports."
            )
        ),
    ] = "",
    surface: Annotated[
        Literal["constructed", "artifact"],
        Field(
            description=(
                "'constructed' (default): the run's persisted cosmic graph. 'artifact': a graph from create_graph_artifact; artifact_id is then required."
            )
        ),
    ] = "constructed",
    artifact_id: Annotated[
        str,
        Field(
            description=(
                "Artifact handle from create_graph_artifact. Required only when "
                "surface='artifact'."
            )
        ),
    ] = "",
    detail: Annotated[
        Literal["summary", "nodes", "edges", "full", "full_nodes"],
        Field(
            description=(
                "'summary' (default): counts and components. 'nodes': per-node summary. 'edges': capped raw edges. 'full_nodes': capped raw nodes. 'full': nodes, edges and config YAML."
            )
        ),
    ] = "summary",
    max_edges: Annotated[
        int,
        Field(description="Cap on edges returned in 'edges'/'full' detail (>= 1)."),
    ] = 100,
    max_nodes: Annotated[
        int,
        Field(
            description="Cap on nodes returned in 'nodes'/'full_nodes'/'full' detail (>= 1)."
        ),
    ] = 100,
    ctx: Context = None,
) -> ToolResult:
    """Connectivity structure (components, nodes, edges) of a run's persisted graph.

    Summary-first: raw nodes and edges are capped by max_nodes / max_edges. Reads
    the stored run record, not the live model. Use diagnose_cosmic_graph for
    risk-oriented measurements instead.
    """
    tool = "get_topological_skeleton"
    try:
        if max_edges < 1:
            return mcp_error(
                tool,
                f"max_edges must be >= 1, got '{max_edges}'",
                error_code="INVALID_ARGUMENT",
                agent_action="Pass max_edges >= 1.",
            )
        if max_nodes < 1:
            return mcp_error(
                tool,
                f"max_nodes must be >= 1, got '{max_nodes}'",
                error_code="INVALID_ARGUMENT",
                agent_action="Pass max_nodes >= 1.",
            )

        session = _get_session(ctx)
        if surface == "artifact":
            if not artifact_id:
                return mcp_error(
                    tool,
                    "artifact_id is required when surface='artifact'.",
                    error_code="ARTIFACT_ID_REQUIRED",
                    agent_action=(
                        "Pass the artifact_id returned by create_graph_artifact, "
                        "or use surface='constructed'."
                    ),
                )
            artifact = session.graph_artifacts.get(artifact_id)
            if artifact is None:
                return unknown_handle_error(tool, "artifact_id", artifact_id)
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
            return mcp_result(payload)

        target_run_id = run_id or session.latest_run_id
        if not target_run_id:
            return model_not_loaded_error(tool)
        record = registry.get_run(target_run_id)
        if record is None:
            return unknown_handle_error(tool, "run_id", target_run_id)
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
        return mcp_result(payload)
    except Exception as e:
        return mcp_error(
            tool,
            str(e),
            error_code="SKELETON_FAILED",
            agent_action="Check get_runtime_context for the session state, then retry.",
        )
