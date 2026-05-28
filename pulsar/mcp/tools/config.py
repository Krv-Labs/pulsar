from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from typing import Any

import yaml
from fastmcp import Context
from fastmcp.exceptions import ToolError

from pulsar.mcp.config_tools import (
    apply_overrides,
    render_validation_report,
    validate_config_yaml,
    _build_initial_config_yaml,
)
from pulsar.mcp.errors import mcp_error, unknown_handle_error
from pulsar.mcp.preprocessing import _calibrate_processed_space
from pulsar.mcp.session import (
    _get_session,
    _load_session_dataframe,
    _resolve_dataset_path,
)

logger = logging.getLogger(__name__)


def _format_epsilon(cfg: dict) -> str:
    """Format epsilon config as a display string, handling both range and values shapes."""
    eps_node = cfg.get("sweep", {}).get("ball_mapper", {}).get("epsilon", {})
    if "range" in eps_node:
        r = eps_node["range"]
        return f"[{r.get('min', 0):.3f}, {r.get('max', 0):.3f}]"
    elif "values" in eps_node:
        return str(eps_node["values"])
    return "n/a"


async def explain_suggestion(
    config_yaml: str, dataset_geometry: str, ctx: Context
) -> str:
    """Markdown explanation of WHY config parameters were chosen, given
    `dataset_geometry` JSON from `characterize_dataset`."""
    try:
        geo = json.loads(dataset_geometry)
        config_dict = yaml.safe_load(config_yaml)

        pca_dims = (
            config_dict.get("sweep", {})
            .get("pca", {})
            .get("dimensions", {})
            .get("values", [])
        )
        eps_str = _format_epsilon(config_dict)

        n_samples = geo.get("n_samples", "unknown")
        cum_var_map = {int(d): v for d, v in geo.get("pca_cumulative_variance", [])}
        knn_mean = geo.get("knn_k5_mean") or geo.get("knn_mean")

        explanation = "### Parameter Reasoning\n\n"

        # 1. PCA Reasoning
        pca_reasons = []
        for dim in pca_dims:
            var = cum_var_map.get(int(dim))
            if var is not None:
                next_var = cum_var_map.get(int(dim) + 1)
                benefit = f" (+{next_var - var:.1%})" if next_var else ""
                pca_reasons.append(f"Dim {dim} captures {var:.1%} variance{benefit}.")

        if pca_reasons:
            explanation += f"- **PCA Dimensions {pca_dims}**: " + " ".join(pca_reasons)
            explanation += " An array of dimensions is used rather than a single point estimate to prevent dimension collapse and ensure the CosmicGraph captures topology across varying geometric resolutions.\n"
            if isinstance(n_samples, int):
                pts_per_dim = n_samples / max(pca_dims)
                explanation += f" With N={n_samples}, this maintains ~{pts_per_dim:.1f} points per dimension, ensuring sufficient manifold density.\n"
            else:
                explanation += "\n"
        else:
            explanation += f"- **PCA Dimensions {pca_dims}**: Chosen as a multi-scale array around the variance curve elbow to aggregate varying topological resolutions (values not provided in geo summary).\n"

        # 2. Epsilon Reasoning
        if knn_mean:
            eps_node = (
                config_dict.get("sweep", {}).get("ball_mapper", {}).get("epsilon", {})
            )
            if "range" in eps_node:
                r = eps_node["range"]
                e_min, e_max = r.get("min", 0), r.get("max", 0)
                explanation += f"- **Epsilon Range {eps_str}**: Anchored at knn_mean={knn_mean:.4f}. The range spans {e_min / knn_mean:.2f}x to {e_max / knn_mean:.2f}x the mean distance. This filtration sweeps from local neighborhoods to global structures, aggregating multi-scale persistent homology into the final graph.\n"
            else:
                explanation += f"- **Epsilon {eps_str}**: Evaluated relative to knn_mean={knn_mean:.4f}. Note: A single epsilon limits the graph to a single scale. Consider sweeping a range to capture ensemble topology.\n"
        else:
            explanation += f"- **Epsilon {eps_str}**: Search window centered around k-NN mean (knn_mean not provided in summary).\n"

        return explanation
    except Exception as e:
        return mcp_error("explain_suggestion", str(e))


async def create_config(dataset_id: str, intent: str = "", ctx: Context = None) -> str:
    """Canonical Pulsar YAML for an ingested dataset. Epsilon and PCA dims are
    calibrated against the processed feature space (post drop/impute/encode/scale),
    not raw columns."""
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
        pca_dims = (
            resolved_cfg.get("sweep", {})
            .get("pca", {})
            .get("dimensions", {})
            .get("values", [])
        )
        pca_seeds = (
            resolved_cfg.get("sweep", {})
            .get("pca", {})
            .get("seed", {})
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
                "pca_dimensions": pca_dims,
                "pca_seeds": pca_seeds,
                "epsilon_steps": epsilon_steps,
                "estimated_ball_maps": len(pca_dims)
                * max(len(pca_seeds), 1)
                * max(int(epsilon_steps or 0), 1),
                "agent_guidance": (
                    "Run this broad baseline first, inspect diagnose_cosmic_graph, "
                    "then use refine_config plus compare_sweeps to shift or "
                    "concentrate the grid around the informative region."
                ),
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
                "Epsilon and PCA dimensions reflect the processed feature space "
                "(after drop/impute/encode/scale), not raw columns. "
                "knn_distance_percentiles show the valid epsilon domain — "
                "epsilon values outside [p5, p95] will produce degenerate graphs."
            )
        else:
            response["calibration_note"] = (
                "Processed-space calibration unavailable; epsilon and PCA "
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
    ctx: Context = None,
) -> str:
    """Apply constrained overrides; returns normalized YAML. Omit
    `config_yaml` (or pass "") to refine the session's active config in place."""
    if overrides is None:
        overrides = {}
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
        if use_active:
            session.active_config_yaml = result.config_yaml
            payload["dataset_id"] = session.active_config_dataset_id
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
