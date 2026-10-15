from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from typing import Annotated, Any, Literal

import yaml
from fastmcp import Context
from fastmcp.tools import ToolResult
from pydantic import Field

from pulsar.mcp.config_tools import (
    apply_overrides,
    render_validation_report,
    validate_config_yaml,
    _build_initial_config_yaml,
    _build_sparsify_warning,
)
from pulsar.mcp.errors import mcp_error, unknown_handle_error
from pulsar.mcp.results import mcp_result
from pulsar.mcp.toolmeta import INT, NUM, STR, arr, enum, nullable, obj, tool_meta
from pulsar.mcp.preprocessing import _calibrate_processed_space
from pulsar.mcp.preprocessing_plan import (
    draft_plan,
    plan_fingerprint,
    summarize_plan_counts,
)
from pulsar.mcp.session import (
    _get_session,
    _load_session_dataframe,
    _resolve_dataset_path,
)

logger = logging.getLogger(__name__)

_OPEN = obj({}, additional=True)

_PREPROCESSING_PLAN_PROVENANCE_SCHEMA = obj(
    {
        "plan_id": STR,
        "revision": INT,
        "plan_fingerprint": STR,
        "source": enum(["explicit_plan", "baseline_defaults"]),
        "open_questions": INT,
        "decided_by": obj(
            {"default": INT, "agent": INT, "user": INT},
            ["default", "agent", "user"],
        ),
    },
    [
        "plan_id",
        "revision",
        "plan_fingerprint",
        "source",
        "open_questions",
        "decided_by",
    ],
)

_CREATE_WARNING_SCHEMA = obj(
    {
        "code": STR,
        "message": STR,
    },
    ["code", "message"],
)

_CREATE_SCHEMA = obj(
    {
        "status": STR,
        "config_yaml": STR,
        "calibration_space": enum(["processed", "raw"]),
        "preprocessing_plan": _PREPROCESSING_PLAN_PROVENANCE_SCHEMA,
        "warnings": arr(_CREATE_WARNING_SCHEMA),
        "sweep_strategy": _OPEN,
        "processed_feature_count": INT,
        "raw_to_processed_expansion_ratio": NUM,
        "knn_distance_percentiles": obj(
            {k: NUM for k in ("p5", "p25", "p50", "p75", "p95")}, additional=True
        ),
        "calibration_note": STR,
    },
    [
        "status",
        "config_yaml",
        "calibration_space",
        "preprocessing_plan",
        "sweep_strategy",
        "calibration_note",
    ],
)

_REFINE_SCHEMA = obj(
    {
        "status": STR,
        "applied_overrides": arr(STR),
        "diff": arr(_OPEN),
        "config_yaml": STR,
        "warnings": arr(_OPEN),
        "dataset_id": nullable(STR),
    },
    ["status", "applied_overrides", "diff", "config_yaml"],
)

_VALIDATE_SCHEMA = obj(
    {
        "status": STR,
        "resolved_dataset_path": nullable(STR),
        "issues": arr(_OPEN),
        "warnings": arr(_OPEN),
        "normalized_config_yaml": STR,
    },
    ["status", "resolved_dataset_path", "issues"],
)


@tool_meta(title="Create Config", read_only=False, output_schema=_CREATE_SCHEMA)
async def create_config(
    dataset_id: Annotated[
        str,
        Field(description="Dataset handle returned by ingest_dataset."),
    ],
    intent: Annotated[
        str,
        Field(
            description="Optional short run name (e.g. 'baseline') stored as run.name in the config. Defaults to 'initial_sweep'."
        ),
    ] = "",
    plan_id: Annotated[
        str,
        Field(
            description="Optional preprocessing plan handle from draft_preprocessing_plan. If omitted, baseline defaults are drafted and stored automatically."
        ),
    ] = "",
    ctx: Context = None,
) -> ToolResult:
    """Generate a broad baseline Pulsar sweep config (YAML) for an ingested dataset.

    Epsilon and dimensions are calibrated in the processed feature space under the
    preprocessing plan (or baseline defaults if plan_id is omitted, with a warning
    when open questions remain). Stores the YAML as the session's active config."""
    try:
        dataset_path = _resolve_dataset_path(dataset_id)
        from pulsar.analysis.characterization import characterize_dataset as _char

        session = _get_session(ctx)
        if plan_id:
            plan = session.preprocessing_plans.get(plan_id)
            if plan is None:
                return mcp_error(
                    "create_config",
                    f"Unknown plan_id '{plan_id}'.",
                    error_code="INVALID_ARGUMENT",
                    agent_action="Call draft_preprocessing_plan(dataset_id) first and pass the returned plan_id, or omit plan_id to use baseline defaults.",
                    details={"plan_id": plan_id},
                )
            if plan.dataset_id != dataset_id:
                return mcp_error(
                    "create_config",
                    f"Plan '{plan_id}' belongs to dataset '{plan.dataset_id}', not '{dataset_id}'.",
                    error_code="INVALID_ARGUMENT",
                    agent_action="Pass a plan_id created for this dataset_id, or call draft_preprocessing_plan(dataset_id).",
                    details={
                        "plan_id": plan_id,
                        "plan_dataset_id": plan.dataset_id,
                        "dataset_id": dataset_id,
                    },
                )
            plan_source = "explicit_plan"
        else:
            plan = None
            plan_source = "baseline_defaults"

        df, normalized_path = await _load_session_dataframe(
            session,
            dataset_id=dataset_id,
        )
        if plan is None:
            plan = await asyncio.to_thread(
                draft_plan,
                df,
                dataset_id,
                stakes=False,
            )
            session.preprocessing_plans[plan.plan_id] = plan

        result = await asyncio.to_thread(_char, dataset_path, dataframe=df)
        geo = dataclasses.asdict(result)
        run_name = intent.strip() or "initial_sweep"

        # Calibrate against processed feature space using the SAME preprocessing plan
        processed = await asyncio.to_thread(
            _calibrate_processed_space,
            df,
            plan,
            normalized_path,
        )

        config_yaml = _build_initial_config_yaml(
            geo,
            data_path=normalized_path,
            run_name=run_name,
            processed_profile=processed,
            plan=plan,
        )
        session.active_config_yaml = config_yaml
        session.active_config_dataset_id = dataset_id

        # Build response with calibration and plan provenance
        calibration_space = "processed" if processed is not None else "raw"
        resolved_cfg = yaml.safe_load(config_yaml)
        projection = resolved_cfg.get("sweep", {}).get("projection", {})
        projection_dimensions = projection.get("dimensions", {}).get(
            "values"
        ) or resolved_cfg.get("sweep", {}).get("pca", {}).get("dimensions", {}).get(
            "values", []
        )
        projection_seeds = projection.get("seed", {}).get("values") or resolved_cfg.get(
            "sweep", {}
        ).get("pca", {}).get("seed", {}).get("values", [])
        projection_method = str(projection.get("method", "jl")).lower()
        legacy_pca_dims = (
            resolved_cfg.get("sweep", {})
            .get("pca", {})
            .get("dimensions", {})
            .get("values", [])
        )
        epsilon_steps = (
            resolved_cfg.get("sweep", {})
            .get("ball_mapper", {})
            .get("epsilon", {})
            .get("range", {})
            .get("steps", 0)
        )
        counts = summarize_plan_counts(plan)
        open_q = int(counts["open_questions"])
        response: dict[str, Any] = {
            "status": "ok",
            "config_yaml": config_yaml,
            "calibration_space": calibration_space,
            "preprocessing_plan": {
                "plan_id": plan.plan_id,
                "revision": plan.revision,
                "plan_fingerprint": plan_fingerprint(plan),
                "source": plan_source,
                "open_questions": open_q,
                "decided_by": dict(counts["decided_by"]),
            },
            "sweep_strategy": {
                "projection_method": projection_method,
                "projection_dimensions": projection_dimensions,
                "projection_seeds": projection_seeds,
                "epsilon_steps": epsilon_steps,
                "estimated_ball_maps": len(projection_dimensions)
                * max(len(projection_seeds), 1)
                * max(int(epsilon_steps or 0), 1),
                "compatibility_note": (
                    "sweep.pca is a legacy mirror of sweep.projection for older "
                    "configs; sweep.projection is authoritative."
                    if legacy_pca_dims
                    else None
                ),
                "agent_guidance": (
                    "Run this broad baseline first, inspect diagnose_cosmic_graph, "
                    "then use refine_config plus compare_sweeps to shift or "
                    "concentrate the grid around the informative region."
                ),
                "parameter_reasoning": [
                    (
                        "Projection dimensions are selected from the processed "
                        "cumulative-variance structure to span multiple resolutions; "
                        "the default projection method is JL."
                    ),
                    (
                        "Epsilon is calibrated in processed feature space when "
                        "possible, with k-NN percentiles returned as the safe "
                        "domain for refinement."
                    ),
                ],
            },
        }
        if open_q > 0:
            response["warnings"] = [
                {
                    "code": "PREPROCESSING_QUESTIONS_OPEN",
                    "message": (
                        f"{open_q} preprocessing questions are unanswered; "
                        "this config uses labelled defaults for them. "
                        "See draft_preprocessing_plan."
                    ),
                }
            ]
        if processed is not None:
            response["processed_feature_count"] = processed.n_features
            raw_features = geo["n_features"]
            response["raw_to_processed_expansion_ratio"] = round(
                processed.n_features / max(raw_features, 1), 2
            )
            response["knn_distance_percentiles"] = {
                "p5": round(processed.knn_p5, 4),
                "p25": round(processed.knn_p25, 4),
                "p50": round(processed.knn_p50, 4),
                "p75": round(processed.knn_p75, 4),
                "p95": round(processed.knn_p95, 4),
            }
            response["calibration_note"] = (
                "Geometry calibrated under the active preprocessing plan. "
                "Epsilon and projection dimensions reflect the processed feature "
                "space (after drop/impute/encode/scale), not raw columns. Dimensions "
                "are selected from the cumulative-variance structure (frontiers + "
                "elbow, capped at 16 for the KD-tree envelope); the default "
                "projection is JL. knn_distance_percentiles show the valid epsilon "
                "domain — epsilon values outside [p5, p95] will produce degenerate "
                "graphs."
            )
        else:
            response["calibration_note"] = (
                "Processed-space calibration unavailable; epsilon and projection "
                "dimensions are calibrated against raw numeric columns only."
            )

        return mcp_result(response)
    except LookupError:
        return unknown_handle_error("create_config", "dataset_id", dataset_id)
    except Exception as e:
        return mcp_error(
            "create_config",
            str(e),
            error_code="CONFIG_CREATION_FAILED",
            agent_action="Check the dataset_id is valid and the data is readable, then retry.",
        )


@tool_meta(title="Refine Config", read_only=False, output_schema=_REFINE_SCHEMA)
async def refine_config(
    config_yaml: Annotated[
        str,
        Field(
            description="Full Pulsar config YAML to refine. Leave empty to refine the session's active config (set by create_config) and store the result back."
        ),
    ] = "",
    overrides: Annotated[
        dict[str, Any] | None,
        Field(
            description="Nested dict merged into the config, e.g. {'sweep': {'ball_mapper': {'epsilon': {'range': {'min': 0.2, 'max': 1.0, 'steps': 6}}}}}. Delete keys with {'remove_keys': ['dotted.path']}."
        ),
    ] = None,
    response_format: Annotated[
        Literal["json", "markdown", "yaml"],
        Field(
            description="Text view: 'json', 'markdown' (diff plus YAML) or 'yaml' (bare YAML). structured_content is identical."
        ),
    ] = "json",
    ctx: Context = None,
) -> ToolResult:
    """Apply constrained overrides to a Pulsar config and return normalized YAML plus a diff.

    With no config_yaml it edits the session's active config in place. Deletion
    via overrides={"remove_keys": [...]} is idempotent and shows as removed:true
    in the diff. Use between sweeps to concentrate or shift the grid."""
    if overrides is None:
        overrides = {}
    if response_format not in {"json", "markdown", "yaml"}:
        return mcp_error(
            "refine_config",
            f"Unknown response_format '{response_format}'.",
            error_code="UNKNOWN_RESPONSE_FORMAT",
            agent_action="Use response_format='json', 'markdown', or 'yaml'.",
        )
    session = _get_session(ctx)
    use_active = not config_yaml
    if use_active:
        if not session.active_config_yaml:
            return mcp_error(
                "refine_config",
                "No config_yaml provided and no active config in session.",
                error_code="ACTIVE_CONFIG_MISSING",
                agent_action="Pass config_yaml or call create_config(dataset_id) first.",
            )
        source_yaml = session.active_config_yaml
    else:
        source_yaml = config_yaml

    try:
        result = apply_overrides(source_yaml, overrides)
        payload: dict[str, Any] = {
            "status": "ok",
            "applied_overrides": result.applied_overrides,
            "diff": result.diff,
            "config_yaml": result.config_yaml,
        }
        resolved = yaml.safe_load(result.config_yaml) or {}
        if isinstance(resolved, dict) and (resolved.get("cosmic_graph") or {}).get(
            "sparsify"
        ):
            payload["warnings"] = [dataclasses.asdict(_build_sparsify_warning())]
        if use_active:
            session.active_config_yaml = result.config_yaml
            payload["dataset_id"] = session.active_config_dataset_id
        if response_format == "yaml":
            return mcp_result(payload, text=result.config_yaml)
        if response_format == "markdown":
            return mcp_result(payload, text=_refine_config_to_markdown(payload))
        return mcp_result(payload)
    except ValueError as e:
        return mcp_error(
            "refine_config",
            str(e),
            error_code="UNKNOWN_OVERRIDE_KEY",
            agent_action="Use only valid override keys. See error message for valid key list.",
        )
    except Exception as e:
        return mcp_error(
            "refine_config",
            str(e),
            error_code="CONFIG_REFINE_FAILED",
            agent_action="Check config_yaml is valid YAML and overrides is a nested dict, then retry.",
        )


def _refine_config_to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Refined Config",
        "",
        f"- Status: {payload.get('status', 'unknown')}",
    ]
    dataset_id = payload.get("dataset_id")
    if dataset_id:
        lines.append(f"- Dataset ID: `{dataset_id}`")
    applied = payload.get("applied_overrides", [])
    lines.append(f"- Applied overrides: {len(applied)}")
    if applied:
        lines.append("- Paths: " + ", ".join(f"`{path}`" for path in applied))

    warnings = payload.get("warnings", [])
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            path = warning.get("path", "$")
            message = warning.get("message", "")
            suggestion = warning.get("suggestion")
            line = f"- `{path}`: {message}"
            if suggestion:
                line += f" Suggestion: {suggestion}"
            lines.append(line)

    diff = payload.get("diff", [])
    lines.extend(["", "## Diff", ""])
    if diff:
        lines.extend(["| Path | Old | New |", "|---|---|---|"])
        for row in diff:
            new_value = "<removed>" if row.get("removed") else row.get("new")
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row.get('path', '')}`",
                        _format_refine_value(row.get("old")),
                        _format_refine_value(new_value),
                    ]
                )
                + " |"
            )
    else:
        lines.append("- No effective changes.")

    lines.extend(
        [
            "",
            "## Config YAML",
            "",
            "```yaml",
            payload.get("config_yaml", "").strip(),
            "```",
        ]
    )
    return "\n".join(lines).strip()


def _format_refine_value(value: Any, *, limit: int = 120) -> str:
    if value is None:
        return "`null`"
    text = str(value) if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return f"`{text}`"


@tool_meta(
    title="Validate Config",
    read_only=True,
    idempotent=True,
    output_schema=_VALIDATE_SCHEMA,
)
async def validate_config(
    config_yaml: Annotated[
        str,
        Field(description="Full Pulsar config as raw YAML text (no Markdown fences)."),
    ],
    dataset_id: Annotated[
        str,
        Field(
            description="Optional dataset handle from ingest_dataset; run.data is resolved against it and it becomes the session's current dataset."
        ),
    ] = "",
    ctx: Context = None,
) -> ToolResult:
    """Validate a full Pulsar config and normalize it into canonical YAML.

    Returns issues, warnings and normalized_config_yaml on success; a structured
    error with an error_code and agent_action when the config is invalid. Use
    before run_topological_sweep when you hand-edited the YAML."""
    try:
        dataset_path = _resolve_dataset_path(dataset_id) if dataset_id else None
        report = await asyncio.to_thread(
            validate_config_yaml, config_yaml, dataset_path=dataset_path
        )
        if dataset_id:
            _get_session(ctx).dataset_id = dataset_id
        rendered = render_validation_report(report)
        if isinstance(rendered, ToolResult):
            return rendered
        return mcp_result(json.loads(rendered))
    except LookupError:
        return unknown_handle_error("validate_config", "dataset_id", dataset_id)
    except Exception as e:
        return mcp_error(
            "validate_config",
            str(e),
            error_code="CONFIG_VALIDATION_FAILED",
            agent_action="Pass the config as raw YAML text and retry.",
        )
