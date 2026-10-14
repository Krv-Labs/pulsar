"""MCP tools for longitudinal (panel) analysis.

Build a longitudinal panel from an ingested long-format table, then interrogate it
through whichever representation answers the question:

- ``TemporalCosmicGraph`` — entity nodes, intra-time edges, tensor aggregations.
- ``CosmicTrajectory`` — observation ``(entity, t)`` nodes, edges that span time.

Artifacts live in the session only, like the fitted model.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import threading
import time
import uuid
from typing import Annotated, Any, Literal

import yaml
from fastmcp import Context
from fastmcp.tools import ToolResult
from pydantic import Field

from pulsar.config import load_config
from pulsar.mcp.config_tools import validate_config_yaml
from pulsar.mcp.errors import mcp_error, path_access_error, unknown_handle_error
from pulsar.mcp.payloads import bounded_list
from pulsar.mcp.results import mcp_result
from pulsar.mcp.toolmeta import (
    ANY,
    default_tool_timeout,
    INT,
    NUM,
    STR,
    arr,
    enum,
    nullable,
    obj,
    tool_meta,
)
from pulsar.mcp.longitudinal import (
    PanelError,
    adapt_config_to_panel,
    archetype_payload,
    build_representations,
    classify_trajectories_payload,
    cross_time_payload,
    estimate_costs,
    guard_representation,
    prepare_panel_frame,
    pivot_panel,
    temporal_diagnosis,
    trajectory_diagnosis,
)
from ..session import (
    LongitudinalArtifact,
    OperationCancelledError,
    _get_session,
    _read_dataset_file,
    _resolve_dataset_path,
    _resolve_longitudinal,
    run_session_worker,
)

logger = logging.getLogger(__name__)


def _json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):  # numpy scalars
        try:
            return item()
        except (TypeError, ValueError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):  # numpy arrays
        return tolist()
    return str(value)


def _json_safe(payload: dict[str, Any]) -> dict[str, Any]:
    """Round-trip through JSON so numpy scalars and timestamps become plain values
    and ``structured_content`` validates against the declared ``output_schema``."""
    return json.loads(json.dumps(payload, default=_json_default))


# ---------------------------------------------------------------------------
# Output schemas (success payloads only; error envelopes are not validated)
# ---------------------------------------------------------------------------
_BOUNDED = obj(
    {"total": INT, "preview": arr(ANY), "omitted": INT},
    ["total", "preview"],
    additional=True,
)
_OPEN = obj({}, additional=True)
_NEXT_TOOLS = arr(STR)
CLASSIFICATION_PREVIEW_LIMIT = 25
_THRESHOLD_SWEEP = obj(
    {
        "basis": STR,
        "note": STR,
        "rows": arr(
            obj(
                {
                    "threshold": NUM,
                    "observation_clusters": INT,
                    "distinct_trajectories": INT,
                    "largest_trajectory_group": INT,
                },
                [
                    "threshold",
                    "observation_clusters",
                    "distinct_trajectories",
                    "largest_trajectory_group",
                ],
            )
        ),
    },
    ["basis", "note", "rows"],
)
_PANEL_REPORT = obj(
    {
        "policy_applied": STR,
        "n_entities_input": INT,
        "n_entities_kept": INT,
        "dropped_entities": _BOUNDED,
        "n_times": INT,
        "n_features": INT,
        "cells_missing_fraction": NUM,
        "alignment": enum(["complete", "ragged"]),
        "observation_count": INT,
    },
    [
        "policy_applied",
        "n_entities_kept",
        "n_times",
        "n_features",
        "alignment",
        "observation_count",
    ],
    additional=True,
)

_BUILD_SCHEMA = obj(
    {
        "status": enum(["ok"]),
        "detail": enum(["summary", "full"]),
        "longitudinal_id": STR,
        "dataset_id": nullable(STR),
        "representation": enum(["trajectory", "temporal", "both"]),
        "panel": obj(
            {
                "entity_column": STR,
                "time_column": STR,
                "n_features": INT,
                "feature_columns_preview": arr(STR),
                **_PANEL_REPORT["properties"],
            },
            ["entity_column", "time_column", "n_features"],
            additional=True,
        ),
        "build_cost": obj(
            {
                "n_observations": INT,
                "n_entities": INT,
                "n_times": INT,
                "temporal_tensor_mb": NUM,
                "temporal_peak_mb": NUM,
                "temporal_tensor_limit_mb": NUM,
                "estimated_ball_maps_trajectory": INT,
                "estimated_ball_maps_temporal": INT,
                "projection_dimensions": arr(INT),
                "epsilons": arr(NUM),
            },
            ["estimated_ball_maps_trajectory", "estimated_ball_maps_temporal"],
            additional=True,
        ),
        "config_adaptation": _OPEN,
        "graph_surface": obj(
            {
                "trajectory": obj(
                    {
                        "n_observations": INT,
                        "n_balls": INT,
                        "n_edges": INT,
                        "cross_time_edges": INT,
                        "cross_time_fraction": NUM,
                    },
                    additional=True,
                ),
                "temporal": obj(
                    {"tensor_shape": arr(INT), "tensor_mb": NUM}, additional=True
                ),
            }
        ),
        "next_tools": _NEXT_TOOLS,
        "config_yaml": STR,
        "times": arr(ANY),
    },
    [
        "status",
        "detail",
        "longitudinal_id",
        "dataset_id",
        "representation",
        "panel",
        "build_cost",
        "config_adaptation",
        "graph_surface",
        "next_tools",
    ],
)

_WEIGHT_DIST = obj(
    {
        "count": INT,
        "p25": nullable(NUM),
        "p50": nullable(NUM),
        "p95": nullable(NUM),
        "max": nullable(NUM),
    },
    additional=True,
)
_TRAJECTORY_SURFACE = obj(
    {
        "representation": enum(["trajectory"]),
        "node_identity": STR,
        "panel": obj(
            {"n_observations": INT, "n_entities": INT, "n_times": INT},
            ["n_observations", "n_entities", "n_times"],
        ),
        "similarity_surface": obj(
            {
                "surface": STR,
                "construction": ANY,
                "similarity_threshold_applied": ANY,
                "n_edges": INT,
                "cross_time_edges": INT,
                "cross_time_fraction": NUM,
                "weight_distribution": _WEIGHT_DIST,
            },
            ["n_edges", "cross_time_edges", "cross_time_fraction"],
            additional=True,
        ),
        "cover": obj(
            {
                "n_balls": INT,
                "balls_by_eps": _OPEN,
                "ball_sizes": _OPEN,
                "memberships_per_observation": _WEIGHT_DIST,
            },
            ["n_balls"],
            additional=True,
        ),
        "threshold_sweep": _THRESHOLD_SWEEP,
        "next_tools": _NEXT_TOOLS,
        "component_morphology": _OPEN,
        "meta": _OPEN,
    },
    [
        "representation",
        "node_identity",
        "panel",
        "similarity_surface",
        "cover",
        "threshold_sweep",
        "next_tools",
    ],
)
_AGGREGATION = obj(
    {
        "aggregation": STR,
        "value_range": STR,
        "measures": STR,
        "why": STR,
        "best_for": arr(STR),
        "avoid_for": arr(STR),
        "distribution": _WEIGHT_DIST,
        "self_scaled_cut": NUM,
        "edges_at_cut": INT,
        "components_at_cut": INT,
        "converging_pairs": INT,
        "diverging_pairs": INT,
    },
    ["aggregation", "distribution", "self_scaled_cut", "edges_at_cut"],
    additional=True,
)
_TEMPORAL_SURFACE = obj(
    {
        "representation": enum(["temporal"]),
        "node_identity": STR,
        "panel": obj({"n_entities": INT, "n_times": INT}, ["n_entities", "n_times"]),
        "tensor": obj({"shape": arr(INT), "bytes": INT, "mb": NUM}, ["shape", "mb"]),
        "cut_basis": STR,
        "aggregations": arr(_AGGREGATION),
        "next_tools": _NEXT_TOOLS,
        "detail": enum(["full"]),
        "threshold": ANY,
    },
    [
        "representation",
        "node_identity",
        "panel",
        "tensor",
        "cut_basis",
        "aggregations",
        "next_tools",
    ],
)
_DIAGNOSE_SCHEMA = obj(
    {
        "status": enum(["ok"]),
        "detail": enum(["summary", "full"]),
        "longitudinal_id": STR,
        "panel": _PANEL_REPORT,
        "surfaces": obj(
            {"trajectory": _TRAJECTORY_SURFACE, "temporal": _TEMPORAL_SURFACE}
        ),
    },
    ["status", "detail", "longitudinal_id", "panel", "surfaces"],
)

_ARCHETYPES_SCHEMA = obj(
    {
        "status": enum(["ok"]),
        "detail": enum(["summary", "full"]),
        "longitudinal_id": STR,
        "representation": enum(["trajectory"]),
        "interpretation_threshold": NUM,
        "threshold_role": STR,
        "n_entities": INT,
        "n_times": INT,
        "distinct_trajectories": INT,
        "archetypes_returned": INT,
        "archetypes_omitted": INT,
        "archetypes": arr(
            obj(
                {
                    "sequence": arr(nullable(INT)),
                    "n_entities": INT,
                    "fraction": NUM,
                    "transitions": INT,
                    "entities": _BOUNDED,
                },
                ["sequence", "n_entities", "fraction", "transitions", "entities"],
            )
        ),
        "threshold_sweep": _THRESHOLD_SWEEP,
        "next_tools": _NEXT_TOOLS,
    },
    [
        "status",
        "detail",
        "longitudinal_id",
        "representation",
        "interpretation_threshold",
        "threshold_role",
        "n_entities",
        "n_times",
        "distinct_trajectories",
        "archetypes_returned",
        "archetypes_omitted",
        "archetypes",
        "threshold_sweep",
        "next_tools",
    ],
)

_NEIGHBOR = obj(
    {
        "obs_id": INT,
        "entity_id": STR,
        "t": INT,
        "time_label": ANY,
        "delta_t": INT,
        "weight": NUM,
    },
    ["obs_id", "entity_id", "t", "delta_t", "weight"],
)
_CROSS_TIME_SCHEMA = obj(
    {
        "status": enum(["ok"]),
        "longitudinal_id": STR,
        "representation": enum(["trajectory"]),
        "surface": STR,
        "interpretation_threshold": NUM,
        "direction": enum(["any", "forward", "backward"]),
        "source": obj(
            {"obs_id": INT, "entity_id": STR, "t": INT, "time_label": ANY},
            ["obs_id", "entity_id", "t"],
        ),
        "neighbors_returned": INT,
        "neighbors_omitted": INT,
        "cross_time_degree": INT,
        "neighbors": arr(_NEIGHBOR),
        "next_tools": _NEXT_TOOLS,
    },
    [
        "status",
        "longitudinal_id",
        "representation",
        "surface",
        "interpretation_threshold",
        "direction",
        "source",
        "neighbors_returned",
        "neighbors_omitted",
        "cross_time_degree",
        "neighbors",
        "next_tools",
    ],
)

_CLASSIFY_SCHEMA = obj(
    {
        "status": enum(["ok"]),
        "longitudinal_id": STR,
        "representation": enum(["trajectory"]),
        "method": enum(["complexity", "transition", "sequence", "levenshtein", "dtw"]),
        "interpretation_threshold": NUM,
        "n_entities": INT,
        "classes_summary": obj({}, additional=True),
        "detail": enum(["summary", "full"]),
        "classification_preview": obj({}, additional=True),
        "entities_returned": INT,
        "entities_omitted": INT,
        # Only present when detail="full" (O(entities)).
        "classification": obj({}, additional=True),
        "next_tools": _NEXT_TOOLS,
    },
    [
        "status",
        "longitudinal_id",
        "representation",
        "method",
        "interpretation_threshold",
        "n_entities",
        "classes_summary",
        "detail",
        "classification_preview",
        "entities_returned",
        "entities_omitted",
        "next_tools",
    ],
)

_LONGITUDINAL_ID_DESC = "From build_longitudinal_graph. Omit for the most recent panel."
_RESPONSE_FORMAT_DESC = (
    "Text view: 'json' or 'markdown' table. structured_content is identical."
)


def _panel_error(tool: str, exc: PanelError) -> ToolResult:
    return mcp_error(
        tool,
        exc.reason,
        error_code=exc.error_code,
        agent_action=exc.agent_action,
        details=exc.details,
    )


def _invalid(tool: str, reason: str) -> ToolResult:
    return mcp_error(
        tool,
        reason,
        error_code="INVALID_ARGUMENT",
        agent_action="Fix the argument named in reason and retry.",
    )


def _internal(tool: str, exc: Exception) -> ToolResult:
    return mcp_error(
        tool,
        str(exc),
        error_code="LONGITUDINAL_QUERY_FAILED",
        agent_action=(
            "Call diagnose_longitudinal_graph to confirm the panel is intact, or "
            "rebuild it with build_longitudinal_graph."
        ),
    )


def _stale_handle(tool: str, longitudinal_id: str) -> ToolResult:
    if not longitudinal_id:
        return mcp_error(
            tool,
            "No longitudinal panel has been built in this session.",
            error_code="LONGITUDINAL_ID_UNKNOWN",
            agent_action=(
                "Call build_longitudinal_graph first, or pass the longitudinal_id "
                "it returned."
            ),
            details={"longitudinal_id": longitudinal_id},
        )
    return unknown_handle_error(tool, "longitudinal_id", longitudinal_id)


def _require_trajectory(tool: str, artifact: LongitudinalArtifact) -> ToolResult | None:
    if artifact.trajectory is None:
        return mcp_error(
            tool,
            "This panel was built without the trajectory representation.",
            error_code="REPRESENTATION_NOT_BUILT",
            agent_action=(
                "Rebuild with build_longitudinal_graph(representation='trajectory') "
                "or representation='both'."
            ),
            details={
                "longitudinal_id": artifact.longitudinal_id,
                "representation": artifact.representation,
            },
        )
    return None


def _check_cancelled(cancel_event: threading.Event | None, stage: str) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise OperationCancelledError(f"Longitudinal build cancelled during {stage}.")


def _build_surface_summary(
    trajectory: Any,
    temporal: Any,
) -> dict[str, Any]:
    surface: dict[str, Any] = {}
    if trajectory is not None:
        cross = trajectory.cross_time()
        total = int(trajectory.similarity.nnz // 2)
        surface["trajectory"] = {
            "n_observations": trajectory.n_observations,
            "n_balls": int(len(trajectory.balls)),
            "n_edges": total,
            "cross_time_edges": int(cross.nnz // 2),
            "cross_time_fraction": (
                round((cross.nnz // 2) / total, 4) if total else 0.0
            ),
        }
    if temporal is not None:
        surface["temporal"] = {
            "tensor_shape": list(temporal.shape),
            "tensor_mb": round(temporal.tensor.nbytes / (1024 * 1024), 2),
        }
    return surface


def _build_longitudinal_worker(
    *,
    dataset_id: str,
    dataset_path: str,
    current_yaml: str,
    entity_column: str,
    time_column: str,
    feature_columns: list[str] | None,
    representation: str,
    on_missing: str,
    similarity_threshold: float,
    detail: str,
    cancel_event: threading.Event | None = None,
) -> tuple[LongitudinalArtifact, dict[str, Any]]:
    """Build the panel/representations. Pure compute: returns the artifact and
    response payload; the caller publishes on the event loop."""
    _check_cancelled(cancel_event, "config load")
    cfg = load_config(yaml.safe_load(current_yaml))

    # Deliberately NOT _bind_session_data: a panel is n*T observation rows and
    # rebinding would invalidate the static cluster and feature-evidence caches.
    df = _read_dataset_file(dataset_path)
    _check_cancelled(cancel_event, "dataset read")
    df = prepare_panel_frame(
        df,
        cfg,
        entity_column,
        time_column,
        feature_columns,
    )
    _check_cancelled(cancel_event, "panel preparation")
    panel = pivot_panel(
        df,
        entity_column,
        time_column,
        feature_columns,
        on_missing,
    )
    guard_representation(panel, representation)
    cfg, config_notes = adapt_config_to_panel(cfg, panel)
    costs = estimate_costs(panel, cfg)
    _check_cancelled(cancel_event, "representation build")
    trajectory, temporal = build_representations(
        panel,
        cfg,
        representation,
        similarity_threshold,
        cancel_event=cancel_event,
    )
    _check_cancelled(cancel_event, "artifact publish")
    longitudinal_id = f"lng_{uuid.uuid4().hex[:12]}"
    artifact = LongitudinalArtifact(
        longitudinal_id=longitudinal_id,
        dataset_id=dataset_id or None,
        representation=representation,
        trajectory=trajectory,
        temporal=temporal,
        panel={
            "entity_column": entity_column,
            "time_column": time_column,
            "feature_columns": panel.feature_columns,
            "times": panel.times,
            "report": panel.report,
        },
        config_yaml=current_yaml,
        created_at=time.time(),
    )

    payload: dict[str, Any] = {
        "status": "ok",
        "detail": detail,
        "longitudinal_id": longitudinal_id,
        "dataset_id": dataset_id or None,
        "representation": representation,
        "panel": {
            "entity_column": entity_column,
            "time_column": time_column,
            "n_features": len(panel.feature_columns),
            "feature_columns_preview": panel.feature_columns[:10],
            **panel.report,
        },
        "build_cost": costs,
        "config_adaptation": config_notes,
        "graph_surface": _build_surface_summary(trajectory, temporal),
        "next_tools": [
            "diagnose_longitudinal_graph",
            "get_trajectory_archetypes",
            "get_cross_time_neighbors",
        ],
    }
    if detail == "full":
        payload["config_yaml"] = current_yaml
        payload["times"] = panel.times
    return artifact, payload


@tool_meta(
    read_only=False,
    idempotent=False,
    timeout=default_tool_timeout(),
    output_schema=_BUILD_SCHEMA,
)
async def build_longitudinal_graph(
    dataset_id: Annotated[
        str,
        Field(
            description=(
                "From ingest_dataset or sample_longitudinal_entities: long-format panel, one row per (entity, time step), numeric features."
            )
        ),
    ] = "",
    entity_column: Annotated[
        str,
        Field(
            description="Column identifying the tracked entity (e.g. an id). Required."
        ),
    ] = "",
    time_column: Annotated[
        str,
        Field(
            description=(
                "Column holding the discrete, orderable time step. Required; "
                "continuous timestamps must be binned before ingest."
            )
        ),
    ] = "",
    feature_columns: Annotated[
        list[str] | None,
        Field(
            description=(
                "Numeric columns to embed. Omit to use every numeric column except "
                "the entity/time keys and any preprocessing.drop_columns."
            )
        ),
    ] = None,
    representation: Annotated[
        Literal["trajectory", "temporal", "both"],
        Field(
            description=(
                "'trajectory' (default): sparse graph over (entity, time) observations; edges may cross time steps. 'temporal': dense (n, n, T) entity tensor, memory-guarded. 'both': build each."
            )
        ),
    ] = "trajectory",
    config_yaml: Annotated[
        str,
        Field(
            description=(
                "Pulsar params YAML from create_config. Omit to reuse the session's active config. Excess projection dimensions are dropped (see config_adaptation)."
            )
        ),
    ] = "",
    on_missing: Annotated[
        Literal["drop_entity", "forward_fill", "allow_ragged"],
        Field(
            description=(
                "Entities missing at some time step: 'drop_entity' removes them, 'forward_fill' carries values forward, 'allow_ragged' keeps them (trajectory only)."
            )
        ),
    ] = "drop_entity",
    similarity_threshold: Annotated[
        float,
        Field(
            description=(
                "Construction-time floor (>= 0) on trajectory CO_SIMILAR edge weights; "
                "0 keeps all edges. Interpretation thresholds are set later."
            )
        ),
    ] = 0.0,
    detail: Annotated[
        Literal["summary", "full"],
        Field(
            description="'full' also returns the normalized config_yaml and time labels."
        ),
    ] = "summary",
    response_format: Annotated[
        Literal["json", "markdown"], Field(description=_RESPONSE_FORMAT_DESC)
    ] = "json",
    ctx: Context = None,
) -> ToolResult:
    """Pivot a long-format dataset into a panel and build longitudinal graphs.

    Returns a longitudinal_id handle, panel shape and alignment report, build cost,
    and graph surface. Does not touch the static sweep/cluster state. Use before
    diagnose_longitudinal_graph, get_trajectory_archetypes, get_cross_time_neighbors
    or classify_trajectories.
    """
    session = _get_session(ctx)

    if detail not in {"summary", "full"}:
        return _invalid(
            "build_longitudinal_graph", "detail must be 'summary' or 'full'."
        )
    if response_format not in {"json", "markdown"}:
        return _invalid(
            "build_longitudinal_graph", "response_format must be 'json' or 'markdown'."
        )
    if not entity_column or not time_column:
        return mcp_error(
            "build_longitudinal_graph",
            "entity_column and time_column are required.",
            error_code="PANEL_KEYS_MISSING",
            agent_action=(
                "Call probe_columns or characterize_dataset to identify the entity "
                "and time columns, then retry."
            ),
        )
    if similarity_threshold < 0.0:
        return _invalid(
            "build_longitudinal_graph", "similarity_threshold must be >= 0."
        )

    current_yaml = config_yaml or session.active_config_yaml or ""
    if not current_yaml:
        return mcp_error(
            "build_longitudinal_graph",
            "No config available for the longitudinal build.",
            error_code="CONFIG_REQUIRED",
            agent_action=(
                "Call create_config(dataset_id) first — it calibrates epsilon and "
                "the projection grid against this dataset — then pass config_yaml."
            ),
            details={"dataset_id": dataset_id},
        )

    try:
        dataset_path = _resolve_dataset_path(dataset_id) if dataset_id else None
    except LookupError:
        return unknown_handle_error(
            "build_longitudinal_graph", "dataset_id", dataset_id
        )
    if dataset_path is None:
        return mcp_error(
            "build_longitudinal_graph",
            "dataset_id is required for a longitudinal build.",
            error_code="DATASET_ID_REQUIRED",
            agent_action="Call ingest_dataset first and pass the returned dataset_id.",
        )

    validation = await asyncio.to_thread(
        validate_config_yaml, current_yaml, dataset_path=dataset_path
    )
    if not validation.ok or validation.normalized_yaml is None:
        return mcp_error(
            "build_longitudinal_graph",
            "Config validation failed.",
            error_code=validation.error_code or "CONFIG_VALIDATION_FAILED",
            agent_action=validation.agent_action,
            details={
                "resolved_dataset_path": validation.resolved_dataset_path,
                "issues": [dataclasses.asdict(issue) for issue in validation.issues],
            },
        )
    current_yaml = validation.normalized_yaml

    async def report(stage: str, fraction: float) -> None:
        if ctx is None:
            return
        await ctx.report_progress(progress=fraction, total=1.0, message=stage)

    cancel_event = threading.Event()
    try:
        await report(f"build {representation}", 0.1)
        artifact, payload = await run_session_worker(
            session,
            _build_longitudinal_worker,
            dataset_id=dataset_id,
            dataset_path=dataset_path,
            current_yaml=current_yaml,
            entity_column=entity_column,
            time_column=time_column,
            feature_columns=feature_columns,
            representation=representation,
            on_missing=on_missing,
            similarity_threshold=similarity_threshold,
            detail=detail,
            cancel_event=cancel_event,
            heavy=True,
            mutates_session=True,
        )
        # Back on the loop: single synchronous publish.
        session.publish_longitudinal(artifact)
        await report("complete", 1.0)

        payload = _json_safe(payload)
        if response_format == "json":
            return mcp_result(payload)
        return mcp_result(payload, text=_build_to_markdown(payload))

    except PanelError as exc:
        return _panel_error("build_longitudinal_graph", exc)
    except FileNotFoundError:
        return path_access_error("build_longitudinal_graph", dataset_path)
    except OperationCancelledError as exc:
        return mcp_error(
            "build_longitudinal_graph",
            str(exc),
            error_code="OPERATION_CANCELLED",
            agent_action="Retry build_longitudinal_graph when ready.",
        )
    except Exception as exc:  # noqa: BLE001 - surface as a structured envelope
        logger.error("build_longitudinal_graph failed: %s", exc, exc_info=True)
        return mcp_error(
            "build_longitudinal_graph",
            str(exc),
            error_code="LONGITUDINAL_BUILD_FAILED",
            agent_action=(
                "Check entity_column, time_column and config_yaml against the "
                "dataset; try representation='trajectory' or a smaller panel."
            ),
        )


def _build_to_markdown(payload: dict[str, Any]) -> str:
    panel = payload["panel"]
    cost = payload["build_cost"]
    lines = [
        f"# Longitudinal Panel `{payload['longitudinal_id']}`",
        "",
        f"- Representation: **{payload['representation']}**",
        f"- Panel: {panel['n_entities_kept']} entities x {panel['n_times']} times "
        f"= {panel['observation_count']} observations",
        f"- Features: {panel['n_features']}",
        f"- Alignment: {panel['alignment']} (policy `{panel['policy_applied']}`, "
        f"{panel['dropped_entities']['total']} entities dropped)",
        f"- Missing cells: {panel['cells_missing_fraction']:.2%}",
        f"- Estimated ball maps: {cost['estimated_ball_maps_trajectory']} "
        f"(trajectory) / {cost['estimated_ball_maps_temporal']} (temporal)",
    ]
    surface = payload.get("graph_surface", {})
    if "trajectory" in surface:
        traj = surface["trajectory"]
        lines += [
            "",
            "## Trajectory surface",
            f"- {traj['n_observations']} observations, {traj['n_balls']} balls, "
            f"{traj['n_edges']} edges",
            f"- Cross-time edges: {traj['cross_time_edges']} "
            f"({traj['cross_time_fraction']:.1%})",
        ]
    if "temporal" in surface:
        temp = surface["temporal"]
        lines += [
            "",
            "## Temporal surface",
            f"- Tensor {temp['tensor_shape']} ({temp['tensor_mb']} MB)",
        ]
    lines += ["", "## Next Tools"] + [f"- `{t}`" for t in payload["next_tools"]]
    return "\n".join(lines)


@tool_meta(read_only=True, idempotent=True, output_schema=_DIAGNOSE_SCHEMA)
async def diagnose_longitudinal_graph(
    longitudinal_id: Annotated[str, Field(description=_LONGITUDINAL_ID_DESC)] = "",
    representation: Annotated[
        Literal["auto", "trajectory", "temporal"],
        Field(
            description=(
                "Surface to measure: 'trajectory' (observation graph), 'temporal' "
                "(entity tensor aggregations), or 'auto' for every surface the panel "
                "was built with."
            )
        ),
    ] = "auto",
    detail: Annotated[
        Literal["summary", "full"],
        Field(description="'full' adds component morphology and build metadata."),
    ] = "summary",
    response_format: Annotated[
        Literal["json", "markdown"], Field(description=_RESPONSE_FORMAT_DESC)
    ] = "json",
    ctx: Context = None,
) -> ToolResult:
    """Measure a built longitudinal panel without prescribing an interpretation.

    Trajectory surface: edge/cover counts, weight distribution, cross-time edge
    fraction, threshold ladder. Temporal surface: the six tensor aggregations side
    by side, each cut at its own q90, with why/best_for/avoid_for.
    """
    session = _get_session(ctx)

    if detail not in {"summary", "full"}:
        return _invalid(
            "diagnose_longitudinal_graph", "detail must be 'summary' or 'full'."
        )
    if response_format not in {"json", "markdown"}:
        return _invalid(
            "diagnose_longitudinal_graph",
            "response_format must be 'json' or 'markdown'.",
        )
    if representation not in {"auto", "trajectory", "temporal"}:
        return _invalid(
            "diagnose_longitudinal_graph",
            "representation must be 'auto', 'trajectory', or 'temporal'.",
        )

    try:
        artifact = _resolve_longitudinal(session, longitudinal_id)
    except LookupError:
        return _stale_handle("diagnose_longitudinal_graph", longitudinal_id)

    try:
        surfaces: dict[str, Any] = {}
        if representation in {"auto", "trajectory"} and artifact.trajectory is not None:
            surfaces["trajectory"] = await asyncio.to_thread(
                trajectory_diagnosis, artifact.trajectory, detail=detail
            )
        if representation in {"auto", "temporal"} and artifact.temporal is not None:
            surfaces["temporal"] = await asyncio.to_thread(
                temporal_diagnosis, artifact.temporal, detail=detail
            )

        if not surfaces:
            return mcp_error(
                "diagnose_longitudinal_graph",
                f"Panel has no '{representation}' surface to diagnose.",
                error_code="REPRESENTATION_NOT_BUILT",
                agent_action=(
                    "Rebuild with build_longitudinal_graph(representation=...) "
                    "including the surface you want to measure."
                ),
                details={
                    "longitudinal_id": artifact.longitudinal_id,
                    "built": artifact.representation,
                    "requested": representation,
                },
            )

        payload = {
            "status": "ok",
            "detail": detail,
            "longitudinal_id": artifact.longitudinal_id,
            "panel": artifact.panel["report"],
            "surfaces": surfaces,
        }
        payload = _json_safe(payload)
        if response_format == "json":
            return mcp_result(payload)
        return mcp_result(payload, text=_diagnosis_to_markdown(payload))

    except Exception as exc:  # noqa: BLE001
        logger.error("diagnose_longitudinal_graph failed: %s", exc, exc_info=True)
        return _internal("diagnose_longitudinal_graph", exc)


def _diagnosis_to_markdown(payload: dict[str, Any]) -> str:
    lines = [f"# Longitudinal Diagnosis `{payload['longitudinal_id']}`"]
    traj = payload["surfaces"].get("trajectory")
    if traj:
        sim = traj["similarity_surface"]
        lines += [
            "",
            "## Trajectory surface (observation nodes)",
            f"- {traj['panel']['n_observations']} observations "
            f"({traj['panel']['n_entities']} entities x {traj['panel']['n_times']} times)",
            f"- {sim['n_edges']} edges; {sim['cross_time_edges']} cross-time "
            f"({sim['cross_time_fraction']:.1%})",
            f"- Weights p25/p50/p95: {sim['weight_distribution']['p25']} / "
            f"{sim['weight_distribution']['p50']} / {sim['weight_distribution']['p95']}",
            f"- Cover: {traj['cover']['n_balls']} balls",
            "",
            "### Threshold ladder",
            "| Threshold | Obs clusters | Distinct trajectories | Largest group |",
            "|---|---|---|---|",
        ]
        for row in traj["threshold_sweep"]["rows"]:
            lines.append(
                f"| {row['threshold']} | {row['observation_clusters']} | "
                f"{row['distinct_trajectories']} | {row['largest_trajectory_group']} |"
            )
        lines += ["", f"_{traj['threshold_sweep']['note']}_"]

    temporal = payload["surfaces"].get("temporal")
    if temporal:
        lines += [
            "",
            "## Temporal surface (entity nodes)",
            f"- Tensor {temporal['tensor']['shape']} ({temporal['tensor']['mb']} MB)",
            f"- _{temporal['cut_basis']}_",
            "",
            "| Aggregation | Range | Edges@cut | Components@cut | Best for | Why |",
            "|---|---|---|---|---|---|",
        ]
        for agg in temporal["aggregations"]:
            lines.append(
                f"| {agg['aggregation']} | {agg['value_range']} | "
                f"{agg['edges_at_cut']} | {agg['components_at_cut']} | "
                f"{', '.join(agg['best_for'])} | {agg['why']} |"
            )
    return "\n".join(lines)


@tool_meta(read_only=True, idempotent=True, output_schema=_ARCHETYPES_SCHEMA)
async def get_trajectory_archetypes(
    longitudinal_id: Annotated[str, Field(description=_LONGITUDINAL_ID_DESC)] = "",
    threshold: Annotated[
        float,
        Field(
            description=(
                "Interpretation filter (>= 0) on CO_SIMILAR edge weights used to "
                "cluster observations; not the construction threshold. 0 uses every "
                "built edge."
            )
        ),
    ] = 0.0,
    max_archetypes: Annotated[
        int, Field(description="Most populous archetypes to return (>= 1).")
    ] = 10,
    max_entities_per_archetype: Annotated[
        int,
        Field(description="Example entity ids listed per archetype (>= 0)."),
    ] = 5,
    detail: Annotated[
        Literal["summary", "full"],
        Field(description="Echoed in the payload; archetype output is always bounded."),
    ] = "summary",
    response_format: Annotated[
        Literal["json", "markdown"], Field(description=_RESPONSE_FORMAT_DESC)
    ] = "markdown",
    ctx: Context = None,
) -> ToolResult:
    """Group entities by the cluster sequence they trace through time.

    An archetype is a set of entities with an identical per-time-step cluster
    sequence. Needs the trajectory representation. Cluster counts move steeply with
    threshold, so the threshold ladder is always included; read it before trusting
    one row.
    """
    session = _get_session(ctx)

    if response_format not in {"json", "markdown"}:
        return _invalid(
            "get_trajectory_archetypes",
            "response_format must be 'json' or 'markdown'.",
        )
    if threshold < 0.0:
        return _invalid("get_trajectory_archetypes", "threshold must be >= 0.")
    if max_archetypes < 1:
        return _invalid("get_trajectory_archetypes", "max_archetypes must be >= 1.")
    if max_entities_per_archetype < 0:
        return _invalid(
            "get_trajectory_archetypes", "max_entities_per_archetype must be >= 0."
        )

    try:
        artifact = _resolve_longitudinal(session, longitudinal_id)
    except LookupError:
        return _stale_handle("get_trajectory_archetypes", longitudinal_id)

    unavailable = _require_trajectory("get_trajectory_archetypes", artifact)
    if unavailable:
        return unavailable

    try:
        payload = await asyncio.to_thread(
            archetype_payload,
            artifact.trajectory,
            threshold=threshold,
            max_archetypes=max_archetypes,
            max_entities_per_archetype=max_entities_per_archetype,
        )
        payload["status"] = "ok"
        payload["detail"] = detail
        payload["longitudinal_id"] = artifact.longitudinal_id
        payload = _json_safe(payload)
        if response_format == "json":
            return mcp_result(payload)
        return mcp_result(payload, text=_archetypes_to_markdown(payload))
    except Exception as exc:  # noqa: BLE001
        logger.error("get_trajectory_archetypes failed: %s", exc, exc_info=True)
        return _internal("get_trajectory_archetypes", exc)


def _archetypes_to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Trajectory Archetypes `{payload['longitudinal_id']}`",
        "",
        f"- Interpretation threshold: {payload['interpretation_threshold']} "
        f"({payload['threshold_role']})",
        f"- {payload['n_entities']} entities over {payload['n_times']} time steps",
        f"- {payload['distinct_trajectories']} distinct trajectories "
        f"({payload['archetypes_omitted']} not shown)",
        "",
        "| Sequence | Entities | Share | Transitions | Example members |",
        "|---|---|---|---|---|",
    ]
    for arch in payload["archetypes"]:
        members = ", ".join(arch["entities"]["preview"])
        lines.append(
            f"| {arch['sequence']} | {arch['n_entities']} | {arch['fraction']:.1%} | "
            f"{arch['transitions']} | {members} |"
        )
    lines += [
        "",
        "## Threshold ladder",
        "| Threshold | Obs clusters | Distinct trajectories | Largest group |",
        "|---|---|---|---|",
    ]
    for row in payload["threshold_sweep"]["rows"]:
        lines.append(
            f"| {row['threshold']} | {row['observation_clusters']} | "
            f"{row['distinct_trajectories']} | {row['largest_trajectory_group']} |"
        )
    lines += ["", f"_{payload['threshold_sweep']['note']}_"]
    lines += ["", "## Next Tools"] + [f"- `{t}`" for t in payload["next_tools"]]
    return "\n".join(lines)


@tool_meta(read_only=True, idempotent=True, output_schema=_CROSS_TIME_SCHEMA)
async def get_cross_time_neighbors(
    longitudinal_id: Annotated[str, Field(description=_LONGITUDINAL_ID_DESC)] = "",
    entity_id: Annotated[
        str,
        Field(
            description=(
                "Entity to look up (as text), used together with t. Ignored when "
                "observation_id is given."
            )
        ),
    ] = "",
    t: Annotated[
        int | float | str | None,
        Field(
            description=(
                "Original label from the panel's time column for the source "
                "observation (not the positional index). Used with entity_id."
            )
        ),
    ] = None,
    observation_id: Annotated[
        int | None,
        Field(
            description=(
                "Positional observation id (an obs_id from an earlier response); "
                "alternative to entity_id plus t."
            )
        ),
    ] = None,
    threshold: Annotated[
        float,
        Field(description="Interpretation filter (>= 0) on edge weight; 0 keeps all."),
    ] = 0.0,
    max_neighbors: Annotated[
        int, Field(description="Highest-weight neighbors to return (>= 1).")
    ] = 20,
    direction: Annotated[
        Literal["any", "forward", "backward"],
        Field(
            description=(
                "'any' time step, 'forward' (later steps only) or 'backward' "
                "(earlier steps only)."
            )
        ),
    ] = "any",
    response_format: Annotated[
        Literal["json", "markdown"], Field(description=_RESPONSE_FORMAT_DESC)
    ] = "markdown",
    ctx: Context = None,
) -> ToolResult:
    """Rank observations from other time steps that resemble one observation.

    Answers "who does this entity at this time look like at a different time?".
    Needs the trajectory representation. Identify the source by observation_id or by
    entity_id plus t; the response gives both time_label and the positional t index.
    """
    session = _get_session(ctx)

    if response_format not in {"json", "markdown"}:
        return _invalid(
            "get_cross_time_neighbors", "response_format must be 'json' or 'markdown'."
        )
    if direction not in {"any", "forward", "backward"}:
        return _invalid(
            "get_cross_time_neighbors",
            "direction must be 'any', 'forward', or 'backward'.",
        )
    if max_neighbors < 1:
        return _invalid("get_cross_time_neighbors", "max_neighbors must be >= 1.")
    if threshold < 0.0:
        return _invalid("get_cross_time_neighbors", "threshold must be >= 0.")

    try:
        artifact = _resolve_longitudinal(session, longitudinal_id)
    except LookupError:
        return _stale_handle("get_cross_time_neighbors", longitudinal_id)

    unavailable = _require_trajectory("get_cross_time_neighbors", artifact)
    if unavailable:
        return unavailable

    trajectory = artifact.trajectory
    obs_id = observation_id
    if obs_id is None:
        if not entity_id or t is None:
            return mcp_error(
                "get_cross_time_neighbors",
                "Provide observation_id, or both entity_id and t.",
                error_code="OBSERVATION_NOT_SPECIFIED",
                agent_action=(
                    "Pass entity_id and t (from the panel's entity/time columns), "
                    "or an observation_id from a previous response."
                ),
            )
        # Entity ids survive the pivot as their original dtype; match on string form
        # so the agent can always pass a plain string.
        panel_times = artifact.panel["times"]
        try:
            time_index = panel_times.index(t)
        except ValueError:
            time_index = next(
                (i for i, value in enumerate(panel_times) if str(value) == str(t)),
                None,
            )
        if time_index is None:
            return mcp_error(
                "get_cross_time_neighbors",
                f"No panel time label matches t={t!r}.",
                error_code="OBSERVATION_NOT_FOUND",
                agent_action="Pass a time label from the panel's time_column.",
                details={
                    "entity_id": entity_id,
                    "t": t,
                    "available_times": bounded_list(panel_times),
                },
            )
        matches = trajectory.obs.index[
            (trajectory.obs["entity_id"].astype(str) == str(entity_id))
            & (trajectory.obs["t"] == time_index)
        ]
        if len(matches) != 1:
            return mcp_error(
                "get_cross_time_neighbors",
                f"No unique observation for entity '{entity_id}' at t={t}.",
                error_code="OBSERVATION_NOT_FOUND",
                agent_action=(
                    "Check entity_id and the original time label against the panel; "
                    "call diagnose_longitudinal_graph for the panel shape."
                ),
                details={"entity_id": entity_id, "t": t, "matches": int(len(matches))},
            )
        obs_id = int(matches[0])

    if obs_id < 0 or obs_id >= trajectory.n_observations:
        return mcp_error(
            "get_cross_time_neighbors",
            f"observation_id {obs_id} is out of range.",
            error_code="OBSERVATION_NOT_FOUND",
            agent_action=(
                f"Valid observation ids are 0..{trajectory.n_observations - 1}."
            ),
            details={"observation_id": obs_id},
        )

    try:
        payload = await asyncio.to_thread(
            cross_time_payload,
            trajectory,
            obs_id=obs_id,
            threshold=threshold,
            max_neighbors=max_neighbors,
            direction=direction,
        )
        payload["status"] = "ok"
        payload["longitudinal_id"] = artifact.longitudinal_id
        payload = _json_safe(payload)
        if response_format == "json":
            return mcp_result(payload)
        return mcp_result(payload, text=_neighbors_to_markdown(payload))
    except Exception as exc:  # noqa: BLE001
        logger.error("get_cross_time_neighbors failed: %s", exc, exc_info=True)
        return _internal("get_cross_time_neighbors", exc)


def _neighbors_to_markdown(payload: dict[str, Any]) -> str:
    source = payload["source"]
    lines = [
        f"# Cross-Time Neighbors of entity `{source['entity_id']}` at time "
        f"`{source['time_label']}` (trajectory index t={source['t']})",
        "",
        f"- Surface: {payload['surface']}",
        f"- Interpretation threshold: {payload['interpretation_threshold']}, "
        f"direction `{payload['direction']}`",
        f"- Cross-time degree: {payload['cross_time_degree']} "
        f"({payload['neighbors_omitted']} not shown)",
        "",
        "| Entity | Time label | t index | delta steps | Weight |",
        "|---|---|---|---|---|",
    ]
    for neighbor in payload["neighbors"]:
        lines.append(
            f"| {neighbor['entity_id']} | {neighbor['time_label']} | {neighbor['t']} | "
            f"{neighbor['delta_t']:+d} | {neighbor['weight']:.4f} |"
        )
    lines += ["", "## Next Tools"] + [f"- `{t}`" for t in payload["next_tools"]]
    return "\n".join(lines)


@tool_meta(read_only=True, idempotent=True, output_schema=_CLASSIFY_SCHEMA)
async def classify_trajectories(
    longitudinal_id: Annotated[str, Field(description=_LONGITUDINAL_ID_DESC)] = "",
    method: Annotated[
        Literal["complexity", "transition", "sequence", "levenshtein", "dtw"],
        Field(
            description=(
                "'complexity': entropy and transition count. 'transition': Markov self-retention. 'sequence': path shape. 'levenshtein'/'dtw': complete-linkage clustering of pairwise distance into 4 cohorts (O(n^2))."
            )
        ),
    ] = "complexity",
    threshold: Annotated[
        float,
        Field(
            description=(
                "Interpretation filter (>= 0) on CO_SIMILAR edges that defines the "
                "clusters whose per-entity sequences are classified."
            )
        ),
    ] = 0.15,
    detail: Annotated[
        Literal["summary", "full"],
        Field(
            description=(
                "'summary' (default) returns the per-class summary plus a "
                f"{CLASSIFICATION_PREVIEW_LIMIT}-entity preview. 'full' adds the "
                "complete per-entity `classification` map (O(entities))."
            )
        ),
    ] = "summary",
    response_format: Annotated[
        Literal["json", "markdown"], Field(description=_RESPONSE_FORMAT_DESC)
    ] = "markdown",
    ctx: Context = None,
) -> ToolResult:
    """Classify each entity's cluster path through time into trajectory classes.

    Returns a per-class summary plus a bounded entity preview (all entities with
    detail="full"). Needs the trajectory representation. Use
    get_trajectory_archetypes to group by exact sequence; this tool gives rule- or
    distance-based classes.
    """
    session = _get_session(ctx)

    if response_format not in {"json", "markdown"}:
        return _invalid(
            "classify_trajectories", "response_format must be 'json' or 'markdown'."
        )
    if method not in {"complexity", "transition", "sequence", "levenshtein", "dtw"}:
        return _invalid(
            "classify_trajectories",
            "method must be 'complexity', 'transition', 'sequence', 'levenshtein', or 'dtw'.",
        )
    if detail not in {"summary", "full"}:
        return _invalid("classify_trajectories", "detail must be 'summary' or 'full'.")
    if threshold < 0.0:
        return _invalid("classify_trajectories", "threshold must be >= 0.")

    try:
        artifact = _resolve_longitudinal(session, longitudinal_id)
    except LookupError:
        return _stale_handle("classify_trajectories", longitudinal_id)

    unavailable = _require_trajectory("classify_trajectories", artifact)
    if unavailable:
        return unavailable

    try:
        payload = await asyncio.to_thread(
            classify_trajectories_payload,
            artifact.trajectory,
            method=method,
            threshold=threshold,
        )
        payload["status"] = "ok"
        payload["longitudinal_id"] = artifact.longitudinal_id
        full = payload.pop("classification")
        preview = dict(list(full.items())[:CLASSIFICATION_PREVIEW_LIMIT])
        payload["detail"] = detail
        payload["classification_preview"] = preview
        payload["entities_returned"] = len(full) if detail == "full" else len(preview)
        payload["entities_omitted"] = (
            0 if detail == "full" else len(full) - len(preview)
        )
        if detail == "full":
            payload["classification"] = full
        payload = _json_safe(payload)
        if response_format == "json":
            return mcp_result(payload)
        return mcp_result(payload, text=_classification_to_markdown(payload))
    except Exception as exc:  # noqa: BLE001
        logger.error("classify_trajectories failed: %s", exc, exc_info=True)
        return _internal("classify_trajectories", exc)


def _classification_to_markdown(payload: dict[str, Any]) -> str:
    method_labels = {
        "complexity": "Shannon Entropy & Transition Counts (Stable vs Volatile)",
        "transition": "Markov Self-Retention Probabilities",
        "sequence": "Structural Path Profile (Singleton vs Multistate)",
        "levenshtein": "Levenshtein Edit Distance Complete-Linkage Clustering",
        "dtw": "Dynamic Time Warping Complete-Linkage Clustering",
    }
    lines = [
        f"# Trajectory Classification `{payload['longitudinal_id']}`",
        "",
        f"- **Method**: `{payload['method']}` ({method_labels[payload['method']]})",
        f"- **Interpretation Threshold**: {payload['interpretation_threshold']}",
        f"- **Entities Classified**: {payload['n_entities']}",
        "",
        "## Cohort Summary",
        "",
        "| Cohort Class | Entity Count | Population Share |",
        "|---|---|---|",
    ]
    for p_class, info in sorted(
        payload["classes_summary"].items(), key=lambda x: -x[1]["count"]
    ):
        lines.append(f"| {p_class} | {info['count']} | {info['fraction']:.1%} |")

    lines += [
        "",
        "## Entity Classification Samples (first 15)",
        "",
        "| Entity ID | Cohort Class | Visits | Sequence (Observed) |",
        "|---|---|---|---|",
    ]
    for p_id, info in list(payload["classification_preview"].items())[:15]:
        lines.append(
            f"| {p_id} | {info['class']} | {info['length']} | {info['sequence']} |"
        )

    shown = min(len(payload["classification_preview"]), 15)
    if payload["n_entities"] > shown:
        lines.append(
            f"| ... | and {payload['n_entities'] - shown} more entities "
            '(detail="full" returns all) | | |'
        )

    lines += ["", "## Next Tools"] + [f"- `{t}`" for t in payload["next_tools"]]
    return "\n".join(lines)
