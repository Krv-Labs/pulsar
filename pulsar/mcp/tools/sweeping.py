from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import time
from typing import Any, Literal

import networkx as nx
import yaml
from fastmcp import Context
from fastmcp.exceptions import ToolError

from pulsar.config import config_to_yaml
from pulsar.mcp.config_tools import validate_config_yaml
from pulsar.mcp.diagnostics import (
    _build_graph_summary,
    _finalization_gate,
    _threshold_stability_summary,
    diagnose_model,
)
from pulsar.mcp.errors import mcp_error, path_access_error, unknown_handle_error
from pulsar.mcp.history import summarize_history
from pulsar.mcp.payloads import (
    build_sweep_summary_payload,
    sweep_payload_to_markdown,
)
from pulsar.mcp.registry import registry
from pulsar.mcp.session import (
    SweepRecord,
    _bind_session_data,
    _dataset_id_for_path,
    _get_session,
    _graph_health_summary,
    _invalidate_feature_evidence_cache,
    _normalize_data_path,
    _pca_cache_status,
    _resolve_dataset_path,
)
from pulsar.pipeline import ThemaRS
from pulsar.runtime.fingerprint import pca_fingerprint

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


def _validate_config_path(path: str) -> None:
    """Validate that config file exists and is a YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    if not path.endswith((".yaml", ".yml")):
        raise ValueError(f"Config must be a YAML file (*.yaml or *.yml), got: {path}")


def _auto_save_config(cfg) -> str:
    """Save resolved config to disk for reproducibility. Returns saved path."""
    name = cfg.run_name or "pulsar"
    if cfg.data and os.path.isfile(cfg.data):
        save_dir = os.path.dirname(os.path.abspath(cfg.data))
    else:
        save_dir = os.getcwd()
    save_path = os.path.join(save_dir, f"{name}_params.yaml")
    with open(save_path, "w") as f:
        f.write(config_to_yaml(cfg))
    logger.info("Config saved to %s", save_path)
    return save_path


async def get_experiment_history(ctx: Context) -> str:
    """Markdown table of all sweeps run in this session."""
    session = _get_session(ctx)
    if not session.sweep_history:
        return "No experiments run yet in this session.\n\n| Run | PCA Dims | Epsilon Range | Nodes | Edges | Components | Giant Fraction |\n|---|---|---|---|---|---|---|"

    lines = [
        "| Run | PCA Dims | Epsilon Range | Nodes | Edges | Components | Giant Fraction |"
    ]
    lines.append("|---|---|---|---|---|---|---|")

    for i, record in enumerate(session.sweep_history):
        cfg = yaml.safe_load(record.config_yaml)
        pca = str(
            cfg.get("sweep", {}).get("pca", {}).get("dimensions", {}).get("values", [])
        )
        eps = _format_epsilon(cfg)

        m = record.metrics
        lines.append(
            f"| {i + 1} | {pca} | {eps} | {m.get('n_nodes')} | {m.get('n_edges')} | {m.get('component_count')} | {m.get('giant_fraction', 0):.2%} |"
        )

    return "\n".join(lines)


async def summarize_sweep_history(ctx: Context = None) -> str:
    """Synthesize patterns across the session's sweeps. Returns
    `{n_runs, observations, rationale}`. Agent owns next-step decision.
    Use `get_experiment_history` for the raw per-run table.
    """
    session = _get_session(ctx)
    summary = summarize_history(list(session.sweep_history))
    return json.dumps(summary, indent=2)


async def compare_sweeps(run_a: str, run_b: str, ctx: Context = None) -> str:
    """Compare two persisted sweeps by config and graph metrics."""
    try:
        record_a = registry.get_run(run_a)
        record_b = registry.get_run(run_b)
        if record_a is None:
            return unknown_handle_error("compare_sweeps", "run_id", run_a)
        if record_b is None:
            return unknown_handle_error("compare_sweeps", "run_id", run_b)

        cfg_a = yaml.safe_load(record_a.config_yaml)
        cfg_b = yaml.safe_load(record_b.config_yaml)
        metrics_a = record_a.metrics
        metrics_b = record_b.metrics

        lines = [
            "### Sweep Comparison",
            "",
            f"- Run A: {record_a.run_id}",
            f"- Run B: {record_b.run_id}",
            f"- Dataset A: {record_a.dataset_id}",
            f"- Dataset B: {record_b.dataset_id}",
            "",
            "| Field | Run A | Run B |",
            "|---|---|---|",
            f"| pca_dims | {cfg_a.get('sweep', {}).get('pca', {}).get('dimensions', {}).get('values', [])} | {cfg_b.get('sweep', {}).get('pca', {}).get('dimensions', {}).get('values', [])} |",
            f"| epsilon | {_format_epsilon(cfg_a)} | {_format_epsilon(cfg_b)} |",
            f"| threshold | {cfg_a.get('cosmic_graph', {}).get('construction_threshold', 'auto')} | {cfg_b.get('cosmic_graph', {}).get('construction_threshold', 'auto')} |",
            f"| nodes | {metrics_a.get('n_nodes')} | {metrics_b.get('n_nodes')} |",
            f"| edges | {metrics_a.get('n_edges')} | {metrics_b.get('n_edges')} |",
            f"| components | {metrics_a.get('component_count')} | {metrics_b.get('component_count')} |",
            f"| giant_fraction | {metrics_a.get('giant_fraction', 0):.2%} | {metrics_b.get('giant_fraction', 0):.2%} |",
            f"| weight_p95 | {metrics_a.get('weight_p95', 0):.4f} | {metrics_b.get('weight_p95', 0):.4f} |",
        ]
        return "\n".join(lines)
    except Exception as e:
        return mcp_error("compare_sweeps", str(e))


async def run_topological_sweep(
    config_path: str = "",
    config_yaml: str = "",
    dataset_id: str = "",
    save_config: bool = False,
    detail: Literal["summary", "full"] = "summary",
    response_format: Literal["markdown", "json"] = "markdown",
    include_config_yaml: bool = False,
    component_limit: int = 20,
    ctx: Context = None,
) -> str:
    """Run the Pulsar topological sweep pipeline. Prefer `config_yaml` inline;
    falls back to active session config if neither is given.

    Args:
        save_config: Persist resolved config to disk for reproducibility.
        component_limit: Component sizes included in summary output.
    """
    if component_limit < 1:
        return mcp_error(
            "run_topological_sweep",
            f"component_limit must be >= 1, got '{component_limit}'",
        )
    try:
        session = _get_session(ctx)
        if config_yaml:
            current_yaml = config_yaml
        elif config_path:
            _validate_config_path(config_path)
            with open(config_path) as f:
                current_yaml = f.read()
        elif session.active_config_yaml:
            current_yaml = session.active_config_yaml
        else:
            raise ToolError(
                "Provide either config_path or config_yaml, or establish an active config first."
            )
        dataset_path = _resolve_dataset_path(dataset_id) if dataset_id else None
        validation = validate_config_yaml(current_yaml, dataset_path=dataset_path)
        if not validation.ok or validation.normalized_yaml is None:
            return mcp_error(
                "run_topological_sweep",
                "Config validation failed before execution.",
                error_code=validation.error_code or "CONFIG_VALIDATION_FAILED",
                agent_action=validation.agent_action,
                details={
                    "resolved_dataset_path": validation.resolved_dataset_path,
                    "issues": [
                        dataclasses.asdict(issue) for issue in validation.issues
                    ],
                },
            )

        current_yaml = validation.normalized_yaml
        config_dict = yaml.safe_load(current_yaml)
        session.active_config_yaml = current_yaml
        resolved_dataset_id = dataset_id or _dataset_id_for_path(
            session.active_config_dataset_id,
            validation.resolved_dataset_path,
        )
        session.active_config_dataset_id = (
            resolved_dataset_id or session.active_config_dataset_id
        )

        logger.info("Starting topological sweep from normalized YAML")
        model = ThemaRS(config_dict)

        cfg = model.config

        precomputed, pca_cache_status = _pca_cache_status(session, cfg)
        if precomputed is not None:
            logger.info("Reusing cached PCA embeddings (fingerprint match)")

        loop = asyncio.get_running_loop()

        def progress_callback(stage: str, fraction: float) -> None:
            if ctx is None:
                return
            try:
                asyncio.run_coroutine_threadsafe(
                    ctx.report_progress(progress=fraction, total=1.0, message=stage),
                    loop,
                )
            except RuntimeError:
                pass

        await asyncio.to_thread(
            model.fit,
            _precomputed_embeddings=precomputed,
            progress_callback=progress_callback,
        )

        session.model = model
        bound_data_path = _normalize_data_path(cfg.data) if cfg.data else None
        bound_dataset_id = dataset_id or _dataset_id_for_path(
            session.dataset_id, bound_data_path
        )
        _bind_session_data(
            session,
            model.data,  # raw pre-preprocessing DataFrame
            dataset_id=bound_dataset_id,
            data_path=bound_data_path,
        )
        _invalidate_feature_evidence_cache(session)

        if precomputed is None:
            session.embeddings = model._embeddings
            session.pca_fingerprint = pca_fingerprint(cfg, len(model.data), model.data)

        saved_path = _auto_save_config(cfg) if save_config else None

        # Calculate metrics for diff
        current_metrics_obj = diagnose_model(model)
        current_metrics = dataclasses.asdict(current_metrics_obj)
        graph_summary = _build_graph_summary(model)

        # Build structured diff
        persisted_dataset_id = bound_dataset_id
        diff: list[dict[str, Any]] = []
        if session.sweep_history:
            prev_record = session.sweep_history[-1]
            prev_cfg = yaml.safe_load(prev_record.config_yaml)
            curr_cfg = yaml.safe_load(current_yaml)

            p_pca = (
                prev_cfg.get("sweep", {})
                .get("pca", {})
                .get("dimensions", {})
                .get("values", [])
            )
            c_pca = (
                curr_cfg.get("sweep", {})
                .get("pca", {})
                .get("dimensions", {})
                .get("values", [])
            )
            if str(p_pca) != str(c_pca):
                diff.append({"field": "pca_dims", "previous": p_pca, "current": c_pca})

            p_eps = _format_epsilon(prev_cfg)
            c_eps = _format_epsilon(curr_cfg)
            if p_eps != c_eps:
                diff.append({"field": "epsilon", "previous": p_eps, "current": c_eps})

            p_dataset = getattr(prev_record, "dataset_id", None)
            c_dataset = persisted_dataset_id
            if p_dataset and c_dataset and p_dataset != c_dataset:
                diff.append(
                    {"field": "dataset_id", "previous": p_dataset, "current": c_dataset}
                )

            pm = prev_record.metrics
            cm = current_metrics
            for key in ("n_edges", "component_count", "giant_fraction"):
                pv, cv = pm.get(key, 0), cm.get(key, 0)
                if pv != cv:
                    diff.append({"field": key, "previous": pv, "current": cv})

        # Record history
        session.sweep_history.append(
            SweepRecord(
                timestamp=time.time(),
                config_yaml=current_yaml,
                metrics=current_metrics,
                dataset_id=persisted_dataset_id,
            )
        )
        run_record = registry.save_run(
            dataset_id=persisted_dataset_id,
            config_yaml=current_yaml,
            metrics=current_metrics,
            resolved_construction_threshold=model.resolved_construction_threshold,
            graph_summary=graph_summary,
        )
        session.latest_run_id = run_record.run_id

        # Build configuration advisory message
        if save_config and saved_path:
            config_advisory = f"Config changes persisted to host at: {saved_path}"
        else:
            config_advisory = "Config changes are in-memory (session-only). Set save_config=True to persist to params.yaml."

        response: dict[str, Any] = {
            "status": "ok",
            "run_id": run_record.run_id,
            "metrics": current_metrics,
            "pca_cached": precomputed is not None,
            "pca_cache_status": pca_cache_status,
            "memory_usage_mb": session.calculate_memory_mb(),
            "diff": diff,
            "config_advisory": config_advisory,
            "config_yaml_normalized": current_yaml,
            "data_shape": list(session.data.shape),
        }
        graph_health, is_connected, recommended_next_action = _graph_health_summary(
            current_metrics
        )
        full_affinity_connected = bool(
            nx.is_connected(nx.from_numpy_array((model.weighted_adjacency > 0)))
        )
        response["is_connected"] = is_connected
        response["constructed_graph_connected"] = is_connected
        response["full_affinity_connected"] = full_affinity_connected
        response["singleton_fraction"] = current_metrics.get("singleton_fraction", 0.0)
        response["spectral_clustering_allowed"] = full_affinity_connected
        response["graph_health"] = graph_health
        response["recommended_next_action"] = recommended_next_action
        response["finalization_gate"] = _finalization_gate(
            current_metrics,
            sweep_count=len(session.sweep_history),
            config_yaml=current_yaml,
        )
        response["construction_threshold"] = float(
            model.resolved_construction_threshold
        )
        stability_summary = _threshold_stability_summary(model, current_metrics)
        if stability_summary is not None:
            response["threshold_stability_summary"] = stability_summary
        if saved_path:
            response["saved_config_path"] = saved_path
        if persisted_dataset_id:
            response["dataset_id"] = persisted_dataset_id

        if detail == "full":
            response["detail"] = "full"
            if response_format == "markdown":
                summary_payload = build_sweep_summary_payload(
                    response,
                    component_limit=component_limit,
                    include_config_yaml=include_config_yaml,
                )
                return sweep_payload_to_markdown(summary_payload)
            return json.dumps(response, indent=2)

        summary_payload = build_sweep_summary_payload(
            response,
            component_limit=component_limit,
            include_config_yaml=include_config_yaml,
        )
        if response_format == "markdown":
            return sweep_payload_to_markdown(summary_payload)
        return json.dumps(summary_payload, indent=2)

    except FileNotFoundError:
        return path_access_error(
            "run_topological_sweep",
            config_path,
            missing_code="CONFIG_FILE_NOT_VISIBLE",
            missing_reason="Config file does not exist on the MCP host filesystem.",
            missing_action=(
                "Use config_yaml directly, or provide a host-visible absolute config path."
            ),
        )
    except LookupError:
        return unknown_handle_error("run_topological_sweep", "dataset_id", dataset_id)
    except Exception as e:
        logger.error(f"Error running sweep: {e}")
        return mcp_error("run_topological_sweep", str(e))
