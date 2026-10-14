from __future__ import annotations

import asyncio
from collections.abc import Callable
import dataclasses
import logging
import os
import threading
import time
from typing import Annotated, Any, Literal

import yaml
from fastmcp import Context
from fastmcp.tools import ToolResult
from pydantic import Field

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
from pulsar.mcp.results import mcp_result
from pulsar.mcp.toolmeta import BOOL, INT, NUM, STR, arr, nullable, obj, tool_meta
from pulsar.mcp.payloads import (
    build_sweep_summary_payload,
    sweep_payload_to_markdown,
)
from pulsar.mcp.registry import StagedWrite, registry
from pulsar.mcp.session import (
    OperationCancelledError,
    SweepRecord,
    _dataset_id_for_path,
    _get_session,
    _graph_health_summary,
    _normalize_data_path,
    _projection_cache_status,
    _resolve_dataset_path,
    run_session_worker,
)
from pulsar.pipeline import ThemaRS
from pulsar.runtime.fingerprint import projection_fingerprint

logger = logging.getLogger(__name__)

_ANY: dict = {}  # unconstrained JSON value
_OPEN = obj({}, additional=True)


def _tool_timeout_s() -> float:
    try:
        return float(os.environ.get("PULSAR_MCP_TOOL_TIMEOUT_S", "1800"))
    except ValueError:
        return 1800.0


_HISTORY_SCHEMA = obj(
    {
        "status": STR,
        "detail": STR,
        "n_runs": INT,
        "runs": arr(_OPEN),
        "summary": _OPEN,
    },
    ["status", "detail", "n_runs"],
)

_RUN_SIDE = obj(
    {
        "run_id": STR,
        "dataset_id": nullable(STR),
        "projection_dimensions": arr(_ANY),
        "epsilon": STR,
        "threshold": _ANY,
        "metrics": obj(
            {
                "n_nodes": _ANY,
                "n_edges": _ANY,
                "component_count": _ANY,
                "giant_fraction": _ANY,
                "weight_p95": _ANY,
            },
            additional=True,
        ),
    },
    ["run_id", "dataset_id", "metrics"],
)
_COMPARE_SCHEMA = obj(
    {"status": STR, "run_a": _RUN_SIDE, "run_b": _RUN_SIDE},
    ["status", "run_a", "run_b"],
)

# One schema covers detail="summary" (compact packet) and detail="full" (raw
# response); the keys of both are declared, only a subset is present per call.
_SWEEP_SCHEMA = obj(
    {
        "status": STR,
        "detail": STR,
        "run_id": STR,
        "dataset_id": nullable(STR),
        "data_shape": arr(INT),
        "construction_threshold": NUM,
        "graph_health": _ANY,
        "analysis_status": STR,
        "next_required_check": STR,
        "finalization_gate": _ANY,
        "constructed_graph_connected": BOOL,
        "full_affinity_connected": BOOL,
        "spectral_clustering_allowed": BOOL,
        "is_connected": BOOL,
        "singleton_fraction": NUM,
        "metrics": _OPEN,
        "key_metrics": _OPEN,
        "component_sizes_preview": arr(_ANY),
        "component_sizes_omitted": INT,
        "diff": arr(_ANY),
        "diff_summary": _OPEN,
        "threshold_stability_summary": nullable(_OPEN),
        "projection_method": STR,
        "projection_cached": BOOL,
        "projection_cache_status": _ANY,
        "pca_cached": BOOL,
        "pca_cache_status": _ANY,
        "memory_usage_mb": NUM,
        "config_advisory": STR,
        "config_yaml_included": BOOL,
        "config_yaml_available_via": STR,
        "config_yaml_normalized": STR,
        "saved_config_path": STR,
        "next_tools": arr(STR),
    },
    [
        "status",
        "detail",
        "run_id",
        "dataset_id",
        "data_shape",
        "construction_threshold",
        "analysis_status",
        "next_required_check",
    ],
)


def _projection_dimensions_from_config(cfg: dict) -> list[Any]:
    sweep = cfg.get("sweep", {})
    return sweep.get("projection", {}).get("dimensions", {}).get("values") or sweep.get(
        "pca", {}
    ).get("dimensions", {}).get("values", [])


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


def _sweep_history_rows(history: list[SweepRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, record in enumerate(history):
        cfg = yaml.safe_load(record.config_yaml)
        metrics = record.metrics
        rows.append(
            {
                "run_index": i + 1,
                "dataset_id": record.dataset_id,
                "projection_dimensions": _projection_dimensions_from_config(cfg),
                "epsilon": _format_epsilon(cfg),
                "nodes": metrics.get("n_nodes"),
                "edges": metrics.get("n_edges"),
                "components": metrics.get("component_count"),
                "giant_fraction": metrics.get("giant_fraction", 0),
            }
        )
    return rows


def _sweep_history_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Run | Projection Dims | Epsilon Range | Nodes | Edges | Components | Giant Fraction |",
        "|---|---|---|---|---|---|---|",
    ]
    if not rows:
        return "\n".join(lines)

    for row in rows:
        lines.append(
            f"| {row['run_index']} | {row['projection_dimensions']} | {row['epsilon']} | "
            f"{row['nodes']} | {row['edges']} | {row['components']} | "
            f"{float(row['giant_fraction']):.2%} |"
        )
    return "\n".join(lines)


def _history_markdown(
    rows: list[dict[str, Any]], summary: dict[str, Any], detail: str
) -> str:
    if not rows:
        return "No sweeps run yet in this session.\n\n" + _sweep_history_table(rows)
    if detail == "table":
        return _sweep_history_table(rows)
    observations = summary.get("observations", [])
    raw_rationale = summary.get("rationale", "")
    rationale = (
        raw_rationale
        if isinstance(raw_rationale, list)
        else [raw_rationale]
        if raw_rationale
        else []
    )
    lines = ["# Sweep History Summary", "", f"- Runs: {len(rows)}"]
    if observations:
        lines.extend(["", "## Observations"])
        lines.extend(f"- {item}" for item in observations)
    if rationale:
        lines.extend(["", "## Rationale"])
        lines.extend(f"- {item}" for item in rationale)
    summary_md = "\n".join(lines).strip()
    if detail == "summary":
        return summary_md
    return "\n\n".join(
        ["# Sweep History", _sweep_history_table(rows), summary_md]
    ).strip()


@tool_meta(read_only=True, idempotent=True, output_schema=_HISTORY_SCHEMA)
async def get_sweep_history(
    detail: Annotated[
        Literal["table", "summary", "full"],
        Field(
            description="'table' = compact per-run rows; 'summary' = synthesized patterns across runs; 'full' = both."
        ),
    ] = "table",
    response_format: Annotated[
        Literal["markdown", "json"],
        Field(
            description="Text rendering only: 'markdown' or 'json'. structured_content is identical in both."
        ),
    ] = "markdown",
    ctx: Context = None,
) -> ToolResult:
    """List the sweeps run in this session with graph metrics per run.

    Use after several run_topological_sweep calls to see how epsilon and
    projection dimensions moved nodes, edges, components and giant fraction.
    Session-scoped; use compare_sweeps for two persisted run_ids."""
    session = _get_session(ctx)
    if detail not in {"table", "summary", "full"}:
        return mcp_error(
            "get_sweep_history",
            "detail must be 'table', 'summary', or 'full'.",
            error_code="INVALID_DETAIL",
            agent_action="Use detail='table', 'summary', or 'full'.",
        )
    if response_format not in {"markdown", "json"}:
        return mcp_error(
            "get_sweep_history",
            "response_format must be 'markdown' or 'json'.",
            error_code="UNKNOWN_RESPONSE_FORMAT",
            agent_action="Use response_format='markdown' or 'json'.",
        )

    rows = _sweep_history_rows(list(session.sweep_history))
    summary = summarize_history(list(session.sweep_history))
    payload: dict[str, Any] = {
        "status": "ok",
        "detail": detail,
        "n_runs": len(rows),
    }
    if detail in {"table", "full"}:
        payload["runs"] = rows
    if detail in {"summary", "full"}:
        payload["summary"] = summary

    if response_format == "json":
        return mcp_result(payload)
    return mcp_result(payload, text=_history_markdown(rows, summary, detail))


def _compare_side(record: Any, cfg: dict) -> dict[str, Any]:
    metrics = record.metrics
    return {
        "run_id": record.run_id,
        "dataset_id": record.dataset_id,
        "projection_dimensions": _projection_dimensions_from_config(cfg),
        "epsilon": _format_epsilon(cfg),
        "threshold": cfg.get("cosmic_graph", {}).get("construction_threshold", "auto"),
        "metrics": {
            "n_nodes": metrics.get("n_nodes"),
            "n_edges": metrics.get("n_edges"),
            "component_count": metrics.get("component_count"),
            "giant_fraction": metrics.get("giant_fraction", 0),
            "weight_p95": metrics.get("weight_p95", 0),
        },
    }


@tool_meta(read_only=True, idempotent=True, output_schema=_COMPARE_SCHEMA)
async def compare_sweeps(
    run_a: Annotated[
        str,
        Field(
            description="First run_id, as returned by run_topological_sweep (persisted runs only)."
        ),
    ],
    run_b: Annotated[
        str,
        Field(description="Second run_id to compare against run_a."),
    ],
    ctx: Context = None,
) -> ToolResult:
    """Compare two persisted sweeps side by side by config and graph metrics.

    Returns each run's projection dimensions, epsilon, construction threshold,
    nodes, edges, components, giant fraction and weight p95; the text view is a
    markdown table. Measurement only: it does not rank the runs."""
    try:
        record_a = registry.get_run(run_a)
        record_b = registry.get_run(run_b)
        if record_a is None:
            return unknown_handle_error("compare_sweeps", "run_id", run_a)
        if record_b is None:
            return unknown_handle_error("compare_sweeps", "run_id", run_b)

        cfg_a = yaml.safe_load(record_a.config_yaml)
        cfg_b = yaml.safe_load(record_b.config_yaml)
        side_a = _compare_side(record_a, cfg_a)
        side_b = _compare_side(record_b, cfg_b)
        ma, mb = side_a["metrics"], side_b["metrics"]

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
            f"| projection_dimensions | {side_a['projection_dimensions']} | {side_b['projection_dimensions']} |",
            f"| epsilon | {side_a['epsilon']} | {side_b['epsilon']} |",
            f"| threshold | {side_a['threshold']} | {side_b['threshold']} |",
            f"| nodes | {ma['n_nodes']} | {mb['n_nodes']} |",
            f"| edges | {ma['n_edges']} | {mb['n_edges']} |",
            f"| components | {ma['component_count']} | {mb['component_count']} |",
            f"| giant_fraction | {ma['giant_fraction']:.2%} | {mb['giant_fraction']:.2%} |",
            f"| weight_p95 | {ma['weight_p95']:.4f} | {mb['weight_p95']:.4f} |",
        ]
        return mcp_result(
            {"status": "ok", "run_a": side_a, "run_b": side_b},
            text="\n".join(lines),
        )
    except Exception as e:
        return mcp_error(
            "compare_sweeps",
            str(e),
            error_code="COMPARE_FAILED",
            agent_action="Check both run_ids came from run_topological_sweep and retry.",
        )


def _compute_sweep_diff(
    prev_record: SweepRecord | None,
    current_yaml: str,
    persisted_dataset_id: str | None,
    current_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    if prev_record is None:
        return []
    diff: list[dict[str, Any]] = []
    prev_cfg = yaml.safe_load(prev_record.config_yaml)
    curr_cfg = yaml.safe_load(current_yaml)

    p_projection = _projection_dimensions_from_config(prev_cfg)
    c_projection = _projection_dimensions_from_config(curr_cfg)
    if str(p_projection) != str(c_projection):
        diff.append(
            {
                "field": "projection_dimensions",
                "previous": p_projection,
                "current": c_projection,
            }
        )

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
    return diff


_POST_FIT_PROGRESS_STAGES = frozenset(
    {
        "finalize session",
        "diagnostics",
        "threshold stability summary",
        "build response",
        "persist run",
    }
)


@dataclasses.dataclass
class _SweepComputation:
    """Pure result of a sweep worker; published later on the event loop."""

    staged_run: StagedWrite
    model: ThemaRS
    current_metrics: dict[str, Any]
    graph_summary: dict[str, Any]
    threshold_curve_summary: dict[str, Any]
    bound_dataset_id: str | None
    bound_data_path: str | None
    embeddings: list[Any] | None
    projection_fingerprint: str | None
    pca_fingerprint: str | None


def _execute_sweep_worker(
    session: Any,
    model: ThemaRS,
    cfg: Any,
    current_yaml: str,
    dataset_id: str,
    session_dataset_id: str | None,
    precomputed: list[Any] | None,
    cached_fingerprints: tuple[str | None, str | None],
    progress_callback: Callable[[str, float], None],
    cancel_event: threading.Event,
) -> _SweepComputation:
    """Compute a sweep. Never mutates ``session``; only stages a run file.

    Publication (session state, saved config, run-file commit) happens on the
    loop in ``run_topological_sweep`` so loop-side readers never see
    half-published state. The run record is only *staged* here (serialized and
    fsynced to a temp file); it becomes visible when the loop commits it.
    """
    from pulsar.mcp.tools.diagnostics import _threshold_curve_payload

    model.fit(
        _precomputed_embeddings=precomputed,
        progress_callback=progress_callback,
    )
    progress_callback("finalize session", 0.91)

    bound_data_path = _normalize_data_path(cfg.data) if cfg.data else None
    bound_dataset_id = dataset_id or _dataset_id_for_path(
        session_dataset_id, bound_data_path
    )
    if precomputed is None:
        new_embeddings = model._embeddings
        new_fp = projection_fingerprint(cfg, len(model.data), model.data)
        new_pca_fp = new_fp
    else:
        new_embeddings = precomputed
        new_fp, new_pca_fp = cached_fingerprints

    progress_callback("diagnostics", 0.92)
    current_metrics_obj = diagnose_model(model)
    current_metrics = dataclasses.asdict(current_metrics_obj)
    graph_summary = _build_graph_summary(model)
    progress_callback("threshold stability summary", 0.95)
    threshold_curve_summary = _threshold_curve_payload(
        model,
        detail="summary",
        threshold_candidate_policy="balanced",
    )
    progress_callback("build response", 0.97)
    progress_callback("persist run", 0.98)
    staged_run = registry.stage_run(
        dataset_id=bound_dataset_id,
        config_yaml=current_yaml,
        metrics=current_metrics,
        resolved_construction_threshold=model.resolved_construction_threshold,
        graph_summary=graph_summary,
        threshold_stability_summary=threshold_curve_summary,
    )
    if cancel_event.is_set():
        registry.discard_staged(staged_run)
        raise OperationCancelledError("Topological sweep cancelled before publication.")
    return _SweepComputation(
        staged_run=staged_run,
        model=model,
        current_metrics=current_metrics,
        graph_summary=graph_summary,
        threshold_curve_summary=threshold_curve_summary,
        bound_dataset_id=bound_dataset_id,
        bound_data_path=bound_data_path,
        embeddings=new_embeddings,
        projection_fingerprint=new_fp,
        pca_fingerprint=new_pca_fp,
    )


def _publish_sweep_on_loop(
    session: Any,
    computation: _SweepComputation,
    *,
    current_yaml: str,
    resolved_config_dataset_id: str | None,
    cfg: Any,
    save_config: bool,
) -> tuple[Any, str | None]:
    """Commit the staged run and publish a finished sweep, synchronously.

    Sweeps are serialized by the session mutation lock and each result is
    self-contained (own config, fingerprint, run record), so there is no
    sweep-vs-sweep conflict: the last sweep to finish wins. The commit
    (``os.replace``) and ``publish_sweep`` have no ``await`` between them; if
    anything fails before the commit the staged file is discarded, so a run is
    visible only if it was published.
    """
    metrics = computation.current_metrics
    model = computation.model
    record = SweepRecord(
        timestamp=time.time(),
        config_yaml=current_yaml,
        metrics=metrics,
        dataset_id=computation.bound_dataset_id,
    )
    try:
        run_record = registry.commit_staged(computation.staged_run)
    except BaseException:
        registry.discard_staged(computation.staged_run)
        raise
    session.publish_sweep(
        config_yaml=current_yaml,
        config_dataset_id=resolved_config_dataset_id,
        model=model,
        data=model.data,
        dataset_id=computation.bound_dataset_id,
        data_path=computation.bound_data_path,
        embeddings=computation.embeddings,
        projection_fingerprint=computation.projection_fingerprint,
        pca_fingerprint=computation.pca_fingerprint,
        record=record,
        run_id=run_record.run_id,
    )
    saved_path = _auto_save_config(cfg) if save_config else None
    return run_record, saved_path


@tool_meta(
    read_only=False,
    output_schema=_SWEEP_SCHEMA,
    timeout=_tool_timeout_s(),
)
async def run_topological_sweep(
    config_path: Annotated[
        str,
        Field(
            description="Host-visible absolute path to a *.yaml/*.yml config file. Used only when config_yaml is empty."
        ),
    ] = "",
    config_yaml: Annotated[
        str,
        Field(
            description="Full Pulsar config as raw YAML text. Preferred. If this and config_path are empty, the session's active config (from create_config/refine_config) is used."
        ),
    ] = "",
    dataset_id: Annotated[
        str,
        Field(
            description="Dataset handle from ingest_dataset; overrides run.data in the config. Empty uses the config's run.data."
        ),
    ] = "",
    save_config: Annotated[
        bool,
        Field(
            description="If true, also write the resolved config to <run name>_params.yaml next to the data for reproducibility."
        ),
    ] = False,
    detail: Annotated[
        Literal["summary", "full"],
        Field(
            description="'summary' = compact decision packet (key metrics, bounded component preview); 'full' = the complete raw response including all metrics."
        ),
    ] = "summary",
    response_format: Annotated[
        Literal["markdown", "json"],
        Field(
            description="Text rendering only: 'markdown' or 'json'. structured_content is identical in both."
        ),
    ] = "markdown",
    include_config_yaml: Annotated[
        bool,
        Field(
            description="Summary detail only: include the normalized config YAML in the payload."
        ),
    ] = False,
    component_limit: Annotated[
        int,
        Field(
            description="Summary detail only: how many component sizes to preview (largest first); the rest are counted as omitted. Must be >= 1."
        ),
    ] = 20,
    ctx: Context = None,
) -> ToolResult:
    """Run the Pulsar topological sweep (project, ball-map grid, cosmic graph) and publish the run.

    Returns run_id, key graph metrics, threshold and a finalization gate; follow
    with diagnose_cosmic_graph. Prefer inline config_yaml; otherwise the active
    session config is used.

    Background task mode: with PULSAR_MCP_ENABLE_TASKS=1 (requires fastmcp[tasks])
    this is the only task-capable tool; a task-capable client may run it in the
    background, and the result and session handles (run_id, dataset_id) are
    identical to a synchronous call."""
    if component_limit < 1:
        return mcp_error(
            "run_topological_sweep",
            f"component_limit must be >= 1, got '{component_limit}'",
            error_code="INVALID_COMPONENT_LIMIT",
            agent_action="Pass component_limit >= 1.",
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
            return mcp_error(
                "run_topological_sweep",
                "Provide either config_path or config_yaml, or establish an active config first.",
                error_code="CONFIG_NOT_PROVIDED",
                agent_action="Pass config_yaml, or call create_config(dataset_id) first to set an active config.",
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
        resolved_dataset_id = dataset_id or _dataset_id_for_path(
            session.active_config_dataset_id,
            validation.resolved_dataset_path,
        )
        resolved_config_dataset_id = (
            resolved_dataset_id or session.active_config_dataset_id
        )
        cached_fingerprints = (
            session.projection_fingerprint,
            session.pca_fingerprint,
        )
        prev_record = session.sweep_history[-1] if session.sweep_history else None
        session_dataset_id = session.dataset_id

        logger.info("Starting topological sweep from normalized YAML")
        model = ThemaRS(config_dict)

        cfg = model.config

        projection_method = getattr(cfg.projection, "method", "jl")
        precomputed, projection_cache_status = _projection_cache_status(session, cfg)
        if precomputed is not None:
            logger.info(
                "Reusing cached %s projection embeddings (fingerprint match)",
                projection_method,
            )

        loop = asyncio.get_running_loop()
        progress_futures = []
        cancel_event = threading.Event()

        async def report_progress(stage: str, fraction: float) -> None:
            if ctx is None:
                return
            await ctx.report_progress(progress=fraction, total=1.0, message=stage)

        def progress_callback(stage: str, fraction: float) -> None:
            if cancel_event.is_set():
                raise OperationCancelledError(
                    f"Topological sweep cancelled at stage '{stage}'."
                )
            if ctx is None:
                return
            scaled_fraction = (
                float(fraction)
                if stage in _POST_FIT_PROGRESS_STAGES
                else min(float(fraction), 1.0) * 0.9
            )
            try:
                progress_futures.append(
                    asyncio.run_coroutine_threadsafe(
                        ctx.report_progress(
                            progress=scaled_fraction,
                            total=1.0,
                            message=stage,
                        ),
                        loop,
                    )
                )
            except RuntimeError:
                pass

        computation = await run_session_worker(
            session,
            _execute_sweep_worker,
            session,
            model,
            cfg,
            current_yaml,
            dataset_id,
            session_dataset_id,
            precomputed,
            cached_fingerprints,
            progress_callback,
            cancel_event,
            heavy=True,
            mutates_session=True,
            cancel_event=cancel_event,
        )
        # Back on the loop: persist + publish atomically (no await inside).
        run_record, saved_path = _publish_sweep_on_loop(
            session,
            computation,
            current_yaml=current_yaml,
            resolved_config_dataset_id=resolved_config_dataset_id,
            cfg=cfg,
            save_config=save_config,
        )
        current_metrics = computation.current_metrics
        threshold_curve_summary = computation.threshold_curve_summary
        persisted_dataset_id = computation.bound_dataset_id
        if progress_futures:
            await asyncio.gather(
                *(asyncio.wrap_future(future) for future in progress_futures),
                return_exceptions=True,
            )
        await report_progress("format response", 0.99)

        diff = _compute_sweep_diff(
            prev_record,
            current_yaml,
            persisted_dataset_id,
            current_metrics,
        )

        # Build configuration advisory message
        if save_config and saved_path:
            config_advisory = f"Config changes persisted to host at: {saved_path}"
        else:
            config_advisory = "Config changes are in-memory (session-only). Set save_config=True to persist to params.yaml."

        response: dict[str, Any] = {
            "status": "ok",
            "run_id": run_record.run_id,
            "metrics": current_metrics,
            "projection_method": projection_method,
            "projection_cached": precomputed is not None,
            "projection_cache_status": projection_cache_status,
            "pca_cached": precomputed is not None,
            "pca_cache_status": projection_cache_status,
            "memory_usage_mb": session.calculate_memory_mb(),
            "diff": diff,
            "config_advisory": config_advisory,
            "config_yaml_normalized": current_yaml,
            "data_shape": list(model.data.shape),
        }
        graph_health, is_connected, _ = _graph_health_summary(current_metrics)
        unthresholded_component_count = threshold_curve_summary.get(
            "unthresholded_component_count"
        )
        full_affinity_connected = bool(
            int(model.cosmic_rust.n) > 0
            and unthresholded_component_count is not None
            and int(unthresholded_component_count) <= 1
        )
        response["is_connected"] = is_connected
        response["constructed_graph_connected"] = is_connected
        response["full_affinity_connected"] = full_affinity_connected
        response["singleton_fraction"] = current_metrics.get("singleton_fraction", 0.0)
        response["spectral_clustering_allowed"] = full_affinity_connected
        response["graph_health"] = graph_health
        response["analysis_status"] = "diagnostics_required"
        response["next_required_check"] = "diagnose_cosmic_graph"
        response["finalization_gate"] = _finalization_gate(
            current_metrics,
            sweep_count=len(session.sweep_history),
            config_yaml=current_yaml,
            threshold_curve_summary=threshold_curve_summary,
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

        summary_payload = build_sweep_summary_payload(
            response,
            component_limit=component_limit,
            include_config_yaml=include_config_yaml,
        )
        if detail == "full":
            response["detail"] = "full"
            structured = response
        else:
            structured = summary_payload
        await report_progress("complete", 1.0)
        if response_format == "markdown":
            return mcp_result(
                structured, text=sweep_payload_to_markdown(summary_payload)
            )
        return mcp_result(structured)

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
        return mcp_error(
            "run_topological_sweep",
            str(e),
            error_code="SWEEP_FAILED",
            agent_action="Inspect the reason, fix the config or dataset, and re-run; call validate_config first if unsure.",
        )
