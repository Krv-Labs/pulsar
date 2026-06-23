from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from typing import Any, Literal

import yaml
from fastmcp import Context

from pulsar.mcp.config_tools import (
    apply_overrides,
    render_validation_report,
    validate_config_yaml,
    _build_initial_config_yaml,
    _build_sparsify_warning,
)
from pulsar.mcp.errors import mcp_error, unknown_handle_error
from pulsar.mcp.preprocessing import _calibrate_processed_space
from pulsar.mcp.session import (
    _get_session,
    _load_session_dataframe,
    _resolve_dataset_path,
)

logger = logging.getLogger(__name__)


async def create_config(dataset_id: str, intent: str = "", ctx: Context = None) -> str:
    """Canonical Pulsar YAML for an ingested dataset. Epsilon and projection
    dimensions are calibrated against the processed feature space (post
    drop/impute/encode/scale), not raw columns. Dimensions are selected from the
    data's cumulative-variance structure; the default projection method is JL."""
    try:
        dataset_path = _resolve_dataset_path(dataset_id)
        from pulsar.analysis.characterization import characterize_dataset as _char

        session = _get_session(ctx)
        df, normalized_path = await _load_session_dataframe(
            session,
            dataset_id=dataset_id,
        )
        result = await asyncio.to_thread(_char, dataset_path, dataframe=df)
        geo = dataclasses.asdict(result)
        run_name = intent.strip() or "initial_sweep"

        # Calibrate against processed feature space
        processed = await asyncio.to_thread(
            _calibrate_processed_space,
            df,
            geo["column_profiles"],
            geo["n_samples"],
            normalized_path,
        )

        config_yaml = _build_initial_config_yaml(
            geo,
            data_path=normalized_path,
            run_name=run_name,
            processed_profile=processed,
        )
        session.active_config_yaml = config_yaml
        session.active_config_dataset_id = dataset_id

        # Build response with calibration provenance
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
        response: dict[str, Any] = {
            "status": "ok",
            "config_yaml": config_yaml,
            "calibration_space": calibration_space,
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
                "Geometry calibrated under recommended initial preprocessing policy. "
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

        return json.dumps(response, indent=2)
    except LookupError:
        return unknown_handle_error("create_config", "dataset_id", dataset_id)
    except Exception as e:
        return mcp_error("create_config", str(e))


async def refine_config(
    config_yaml: str = "",
    overrides: dict[str, Any] | None = None,
    response_format: Literal["json", "markdown", "yaml"] = "json",
    ctx: Context = None,
) -> str:
    """Apply constrained overrides; returns normalized YAML. Omit
    `config_yaml` (or pass "") to refine the session's active config in place.

    Overrides merge into existing config. To DELETE a key (rather than null it),
    pass `overrides={"remove_keys": ["preprocessing.encode.species"]}` — a list of
    dotted paths. Deletion is idempotent and reported in `diff` with `removed: true`.
    """
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
            return result.config_yaml
        if response_format == "markdown":
            return _refine_config_to_markdown(payload)
        return json.dumps(payload, indent=2)
    except ValueError as e:
        return mcp_error(
            "refine_config",
            str(e),
            error_code="UNKNOWN_OVERRIDE_KEY",
            agent_action="Use only valid override keys. See error message for valid key list.",
        )
    except Exception as e:
        return mcp_error("refine_config", str(e))


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


async def validate_config(
    config_yaml: str,
    dataset_id: str = "",
    ctx: Context = None,
) -> str:
    """Validate full Pulsar config and normalize into canonical YAML."""
    try:
        dataset_path = _resolve_dataset_path(dataset_id) if dataset_id else None
        report = validate_config_yaml(config_yaml, dataset_path=dataset_path)
        if dataset_id:
            _get_session(ctx).dataset_id = dataset_id
        return render_validation_report(report)
    except LookupError:
        return unknown_handle_error("validate_config", "dataset_id", dataset_id)
    except Exception as e:
        return mcp_error("validate_config", str(e))
