"""
FastMCP Server for Pulsar.

Exposes "Thick Tools" for topological data analysis and interpretation.
"""

from __future__ import annotations

import asyncio
import base64
from collections import OrderedDict
import dataclasses
from dataclasses import dataclass, field
import gc
import hashlib
import json
import logging
import networkx as nx
import os
from pathlib import Path
import time
from typing import Any

import pandas as pd
import yaml
from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError

import numpy as np
from sklearn.preprocessing import StandardScaler as SkScaler

from pulsar.analysis.characterization import NumericProfile, profile_numeric_matrix
from pulsar.config import config_to_yaml, load_config
from pulsar.preprocessing import preprocess_dataframe
from pulsar.runtime.fingerprint import pca_fingerprint
from pulsar.pipeline import ThemaRS
from pulsar.mcp.interpreter import (
    resolve_clusters,
    build_feature_evidence_index,
    build_dossier,
    cluster_profile_payload,
    dossier_to_markdown,
    dossier_to_html,
    compare_clusters,
    comparison_to_markdown,
    feature_signal_payload,
    signal_matrix_payload,
    FeatureEvidenceIndex,
)
from pulsar.mcp.config_tools import (
    apply_overrides,
    render_validation_report,
    validate_config_yaml,
)
from pulsar.mcp.errors import mcp_error, path_access_error, unknown_handle_error
from pulsar.mcp.preprocessing import (
    _preprocessing_block_to_yaml,
    _rationale_table,
    _recommend_preprocessing_block,
    repair_config,
)
from pulsar.mcp.registry import MCPRegistry

logger = logging.getLogger(__name__)
registry = MCPRegistry()


# ---------------------------------------------------------------------------
# Defensive patch: strip unknown kwargs from non-compliant MCP clients.
# Some clients (e.g. Gemini CLI) inject orchestration fields like
# ``wait_for_previous`` into tool calls.  FastMCP's Pydantic validation
# rejects these.  Patching FunctionTool.run at the class level filters
# known orchestration keys *before* validation — one patch protects every tool.
# Unknown keys outside the allowlist are logged at WARNING to surface caller bugs.
# ---------------------------------------------------------------------------
from fastmcp.tools.function_tool import FunctionTool  # noqa: E402

_original_function_tool_run = FunctionTool.run

# Known non-compliant orchestration keys injected by MCP clients.
_KNOWN_ORCHESTRATION_KEYS = frozenset({"wait_for_previous"})


async def _lenient_function_tool_run(self, arguments):
    if isinstance(arguments, dict) and arguments:
        valid_keys = set(self.parameters.get("properties", {}).keys())
        unknown_keys = set(arguments.keys()) - valid_keys
        if unknown_keys:
            unexpected = unknown_keys - _KNOWN_ORCHESTRATION_KEYS
            if unexpected:
                logger.warning(
                    "Stripped unexpected argument(s) %s from tool %s",
                    sorted(unexpected),
                    getattr(self, "name", "unknown"),
                )
            arguments = {k: v for k, v in arguments.items() if k in valid_keys}
    return await _original_function_tool_run(self, arguments)


FunctionTool.run = _lenient_function_tool_run


def _build_initial_config_yaml(
    geo: dict[str, Any],
    *,
    data_path: str,
    run_name: str = "initial_sweep",
    processed_profile: NumericProfile | None = None,
) -> str:
    """Construct a canonical initial config from dataset geometry.

    When *processed_profile* is provided, epsilon and PCA dimensions are
    calibrated against the processed feature space (after preprocessing +
    scaling).  Otherwise falls back to raw-space geometry.
    """
    # Use processed-space geometry for calibration when available;
    # fall back to raw-space otherwise.
    if processed_profile is not None:
        knn_mean = processed_profile.knn_k5_mean or 0.5
        pca_cum_var = processed_profile.pca_cumulative_variance
        # Use actual distance percentiles for epsilon bounds when available.
        # p25-p75 captures the core of the distance distribution — wide enough
        # to aggregate diverse topology, narrow enough to avoid blob/shatter.
        knn_p25 = processed_profile.knn_p25
        knn_p75 = processed_profile.knn_p75
    else:
        knn_mean = geo.get("knn_k5_mean") or geo.get("knn_mean") or 0.5
        pca_cum_var = geo.get("pca_cumulative_variance", [])
        knn_p25 = knn_p75 = 0.0

    pca_knee = 3
    if pca_cum_var:
        for dim, var in pca_cum_var:
            if var >= 0.90:
                pca_knee = dim
                break

    # PCA dims: wider array for multi-scale aggregation.
    # More dimensions = more topological evidence fused into the cosmic graph.
    pca_dims = sorted(
        {
            max(2, pca_knee - 2),
            max(2, pca_knee - 1),
            pca_knee,
            pca_knee + 1,
            pca_knee + 2,
        }
    )

    # Epsilon range: use percentile bounds when available, fall back to
    # knn_mean multipliers.  Wider range = better multi-scale fusion.
    if knn_p25 > 0 and knn_p75 > 0:
        eps_min = knn_p25 * 0.8
        eps_max = knn_p75 * 1.3
    else:
        eps_min = knn_mean * 0.8
        eps_max = knn_mean * 1.5

    n_samples = geo.get("n_samples", 0)
    column_profiles = geo.get("column_profiles", [])
    drop, impute, encode, _ = _recommend_preprocessing_block(column_profiles, n_samples)
    preprocessing_block = _preprocessing_block_to_yaml(drop, impute, encode)

    return f"""run:
  name: {run_name}
  data: {data_path}
{preprocessing_block}
sweep:
  pca:
    dimensions:
      values: {pca_dims}
    seed:
      values: [42, 7]
  ball_mapper:
    epsilon:
      range:
        min: {eps_min:.4f}
        max: {eps_max:.4f}
        steps: 20
cosmic_graph:
  threshold: auto
output:
  n_reps: 4
"""


# ---------------------------------------------------------------------------
# Initialize FastMCP
mcp = FastMCP(
    "Pulsar",
    instructions=(
        "Reveal the dataset's topology; do not force convenient clusters.\n\n"
        "### PHASE I: INGEST & CALIBRATE\n"
        "1. Ingest: Use ingest_dataset handles. Prefer dataset_id everywhere.\n"
        "2. Characterize: characterize_dataset(dataset_id) returns a SPARSE schema — dtype, n_unique, missingness — for ALL columns. Numeric stats and top_values are intentionally omitted to keep payload small for wide datasets. Use probe_columns(dataset_id, ['col_name']) for deep per-column inspection (sample values, distributions). Max 20 columns per probe call.\n"
        "3. Calibrate: create_config(dataset_id) is mandatory. It returns the [p5, p95] epsilon domain. EPSILON OUTSIDE THIS RANGE PRODUCES DEGENERATE GRAPHS.\n"
        "4. Validate Config: validate_config(config_yaml, dataset_id).\n\n"
        "### PHASE II: EXECUTE & VALIDATE\n"
        "4. Run: run_topological_sweep.\n"
        "5. Validate: diagnose_cosmic_graph.\n"
        "   - GATE: If density > 0.8 or < 0.1, STOP. Refine config (Step 2).\n"
        "   - GATE: component_count=1 is normal; do not force separation by narrowing epsilon.\n\n"
        "### PHASE III: CONTRASTIVE INTERPRETATION\n"
        "6. Cluster: generate_cluster_dossier.\n"
        "7. Contrast: Perform comparative analysis. Identify the 'Pivot Feature'—the variable that most cleanly separates Cluster A from its topological neighbors. Do not name in isolation.\n"
        "8. Report: export_html_report. CRITICAL: YOU MUST pass synthesized, highly informative 'cluster_names' based on Step 7. Names must be descriptive (e.g., 'Male Gentoos w/ Large Flippers'). Passing 'cluster_names' is the difference between a raw Data Dump and a high-impact Research Paper.\n\n"
        "### PHILOSOPHY\n"
        "- Pulsar is a multi-scale aggregator, not a tuner. More grid points = more topological evidence. ALL ball maps are fused into ONE cosmic graph.\n"
        "- Wide PCA arrays and epsilon ranges are always superior to single points.\n"
        "- The cosmic graph is evidence, not a score to maximize.\n"
        "- Performance & Isolation: Claude Desktop sandboxes are isolated. DO NOT use chunked/base64 uploads for local files. Use the 'Cache-Bridge' pattern: 1. Call `get_runtime_context` to find `cache_dir`. 2. Use a shell command to `cp` your file into that `cache_dir`. 3. Call `ingest_dataset(path)` on the new path. This is 100x faster and avoids protocol overhead.\n\n"
        "### SEMANTIC SAFETY\n"
        "- Preprocessing blindspots: `recommend_preprocessing` is purely structural. It may suggest dropping high-cardinality geographic (lat/lon), temporal, or ID columns. You MUST manually review these against the user's domain goal. Do not drop critical signal just because the schema looks sparse.\n"
    ),
)


@dataclass
class SweepRecord:
    timestamp: float
    config_yaml: str
    metrics: dict


# Session state: stores (model, data, clusters) per session
@dataclass
class _PulsarSession:
    """Session state for a single MCP client."""

    model: ThemaRS | None = None
    data: pd.DataFrame | None = None
    clusters: pd.Series | None = None
    embeddings: list | None = None  # cached PCA output from last fit
    pca_fingerprint: str | None = None  # SHA256 of (data_path, dims, seeds, n_rows)
    sweep_history: list[SweepRecord] = field(default_factory=list)
    dataset_id: str | None = None
    latest_run_id: str | None = None
    report_exclude_columns: list[str] | None = None
    data_dataset_id: str | None = None
    data_path: str | None = None
    feature_evidence_index: FeatureEvidenceIndex | None = None
    feature_evidence_fingerprint: str | None = None
    feature_evidence_cluster_meta: dict[str, Any] | None = None
    active_config_yaml: str | None = None
    active_config_dataset_id: str | None = None

    def calculate_memory_mb(self) -> float:
        """Estimate current session memory footprint in MB."""
        bytes_total = 0
        if self.data is not None:
            # Deep memory usage for Pandas (captures string objects etc)
            bytes_total += self.data.memory_usage(deep=True).sum()

        if self.model is not None and self.model._weighted_adjacency is not None:
            bytes_total += self.model._weighted_adjacency.nbytes

        if self.embeddings is not None:
            for emb in self.embeddings:
                if hasattr(emb, "nbytes"):
                    bytes_total += emb.nbytes

        return round(float(bytes_total) / (1024 * 1024), 2)


# Global session storage, keyed by session_id (or "default" for STDIO)
# Using OrderedDict for LRU (Least Recently Used) eviction policy.
_sessions: OrderedDict[str, _PulsarSession] = OrderedDict()
_MAX_SESSIONS = int(os.environ.get("PULSAR_MAX_SESSIONS", "3"))
_MAX_EDGES_IN_SUMMARY = 500


def _session_key(ctx: Context | None) -> str:
    """Get the session key from context (session_id or 'default' for STDIO)."""
    if ctx is None:
        return "default"
    return ctx.session_id or "default"


def _get_session(ctx: Context) -> _PulsarSession:
    """Get or create session state for the current client with LRU eviction."""
    key = _session_key(ctx)

    if key in _sessions:
        # Move to end (mark as recently used)
        _sessions.move_to_end(key)
        return _sessions[key]

    # Create new session
    if len(_sessions) >= _MAX_SESSIONS:
        # Evict oldest session
        evicted_key, evicted_session = _sessions.popitem(last=False)
        logger.info(
            "Evicting oldest session '%s' to free memory (Max sessions: %d)",
            evicted_key,
            _MAX_SESSIONS,
        )
        # Cleanup
        del evicted_session
        gc.collect()

    session = _PulsarSession()
    _sessions[key] = session
    return session


def _resolve_dataset_record(dataset_id: str):
    record = registry.get_dataset(dataset_id)
    if record is None:
        raise LookupError(dataset_id)
    return record


def _resolve_dataset_path(dataset_id: str) -> str:
    return _resolve_dataset_record(dataset_id).path


def _normalize_data_path(path: str) -> str:
    expanded = Path(path).expanduser()
    return str(expanded.resolve(strict=False))


def _read_dataset_file(path: str) -> pd.DataFrame:
    normalized_path = _normalize_data_path(path)
    if normalized_path.lower().endswith(".parquet"):
        return pd.read_parquet(normalized_path)
    return pd.read_csv(normalized_path)


def _bind_session_data(
    session: _PulsarSession,
    df: pd.DataFrame,
    *,
    dataset_id: str | None = None,
    data_path: str | None = None,
) -> None:
    session.data = df
    session.data_dataset_id = dataset_id
    session.data_path = _normalize_data_path(data_path) if data_path else None
    if dataset_id:
        session.dataset_id = dataset_id


def _dataset_id_for_path(dataset_id: str | None, data_path: str | None) -> str | None:
    if not dataset_id or not data_path:
        return None

    try:
        dataset_path = _normalize_data_path(_resolve_dataset_path(dataset_id))
    except LookupError:
        return None

    normalized_path = _normalize_data_path(data_path)
    return dataset_id if dataset_path == normalized_path else None


async def _load_session_dataframe(
    session: _PulsarSession,
    *,
    dataset_id: str = "",
    data_path: str = "",
) -> tuple[pd.DataFrame, str]:
    if dataset_id:
        resolved_path = _resolve_dataset_path(dataset_id)
        normalized_path = _normalize_data_path(resolved_path)
        if (
            session.data is not None
            and session.data_dataset_id == dataset_id
            and session.data_path == normalized_path
        ):
            session.dataset_id = dataset_id
            return session.data, normalized_path

        df = await asyncio.to_thread(_read_dataset_file, normalized_path)
        _bind_session_data(
            session,
            df,
            dataset_id=dataset_id,
            data_path=normalized_path,
        )
        return df, normalized_path

    if not data_path:
        raise ToolError("Provide either data_path or dataset_id.")

    normalized_path = _normalize_data_path(data_path)
    if (
        session.data is not None
        and session.data_dataset_id is None
        and session.data_path == normalized_path
    ):
        return session.data, normalized_path

    df = await asyncio.to_thread(_read_dataset_file, normalized_path)
    _bind_session_data(session, df, data_path=normalized_path)
    return df, normalized_path


def _invalidate_feature_evidence_cache(session: _PulsarSession) -> None:
    session.feature_evidence_index = None
    session.feature_evidence_fingerprint = None
    session.feature_evidence_cluster_meta = None
    session.clusters = None


def _feature_evidence_fingerprint(
    *,
    run_id: str | None,
    labels: pd.Series,
    method: str,
    edge_weight_threshold: float,
    exclude_columns: list[str] | None,
) -> str:
    payload = {
        "run_id": run_id,
        "labels": labels.astype(int).tolist(),
        "method": method,
        "edge_weight_threshold": edge_weight_threshold,
        "exclude_columns": sorted(exclude_columns or []),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _graph_health_summary(metrics: dict[str, Any]) -> tuple[str, bool, str]:
    density = float(metrics.get("density", 0.0))
    component_count = int(metrics.get("component_count", 0))
    singleton_fraction = float(metrics.get("singleton_fraction", 0.0))
    is_connected = component_count <= 1
    if density > 0.8:
        return "hairball", is_connected, "Refine epsilon downward or raise threshold before clustering."
    if singleton_fraction > 0.25 or component_count > max(3, int(metrics.get("n_nodes", 0) * 0.1)):
        return "fragmented", is_connected, "Increase epsilon support or fall back to component-based clustering."
    if density < 0.05:
        return "sparse", is_connected, "Broaden the sweep or inspect threshold stability before spectral clustering."
    return "connected", is_connected, "Proceed to clustering; graph structure is suitable for interpretive analysis."


def _pca_cache_status(
    session: _PulsarSession,
    cfg: Any,
) -> tuple[list | None, dict[str, Any]]:
    """Return reusable PCA embeddings and an operational status payload."""
    status: dict[str, Any] = {
        "scope": "session",
        "status": "miss",
        "reason": "no_cached_embeddings",
    }
    if session.embeddings is None:
        return None, status
    if session.data is None:
        status["reason"] = "no_session_data"
        return None, status
    if session.pca_fingerprint is None:
        status["reason"] = "no_cached_fingerprint"
        return None, status

    fingerprint = pca_fingerprint(cfg, len(session.data), session.data)
    if fingerprint != session.pca_fingerprint:
        status["reason"] = "fingerprint_mismatch"
        return None, status

    return (
        session.embeddings,
        {
            "scope": "session",
            "status": "hit",
            "reason": "fingerprint_match",
        },
    )


def _cluster_result_payload(result: Any) -> dict[str, Any]:
    return {
        "method_used": result.method_used,
        "n_clusters": result.n_clusters,
        "silhouette_score": result.silhouette_score,
        "edge_weight_threshold_applied": result.edge_weight_threshold_applied,
        "stability_plateaus": result.stability_plateaus,
        "failure_reason": result.failure_reason,
    }


def _cluster_payload_from_dossier(dossier: Any) -> list[dict[str, Any]]:
    return [
        {
            "cluster_id": profile.cluster_id,
            "size": profile.size,
            "size_pct": profile.size_pct,
            "semantic_name": profile.semantic_name,
            "topological_neighbors": profile.topological_neighbors,
            "numeric_features": profile.numeric_features,
            "categorical_features": profile.categorical_features,
            "numeric_tiers": profile.numeric_tiers,
            "categorical_tiers": profile.categorical_tiers,
            "central_rows": profile.central_rows,
        }
        for profile in dossier.clusters
    ]


def _resolve_response_format(
    *,
    response_format: str,
    legacy_format: str,
    detail: str,
) -> tuple[str, str]:
    resolved_detail = detail or "summary"
    resolved_format = response_format or "json"
    if legacy_format:
        if legacy_format == "full":
            resolved_detail = "full"
            resolved_format = "json"
        elif legacy_format in {"json", "markdown"}:
            resolved_format = legacy_format
        else:
            raise ToolError(
                f"format must be 'json', 'markdown', or 'full', got '{legacy_format}'"
            )
    if resolved_detail not in {"summary", "standard", "full"}:
        raise ToolError(
            f"detail must be 'summary', 'standard', or 'full', got '{resolved_detail}'"
        )
    if resolved_format not in {"json", "markdown"}:
        raise ToolError(
            f"response_format must be 'json' or 'markdown', got '{resolved_format}'"
        )
    return resolved_detail, resolved_format


def _build_evidence_payload(
    dossier: Any,
    cluster_meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "ok",
        "cluster_result": cluster_meta,
        "detail": dossier.global_stats.get("detail", "standard"),
        "evidence_metadata": dossier.global_stats.get("evidence_metadata", {}),
        "graph_metrics": dossier.global_stats.get("graph_metrics", {}),
        "signal_matrix": dossier.global_stats.get("signal_matrix", {}),
        "numeric_global_ranking": dossier.global_stats.get("numeric_global_ranking", []),
        "categorical_global_ranking": dossier.global_stats.get(
            "categorical_global_ranking", []
        ),
        "clusters": _cluster_payload_from_dossier(dossier),
    }


def _get_or_build_evidence_index(
    session: _PulsarSession,
    *,
    cluster_result: Any,
    exclude_columns: list[str] | None,
) -> FeatureEvidenceIndex:
    fingerprint = _feature_evidence_fingerprint(
        run_id=session.latest_run_id,
        labels=cluster_result.labels,
        method=cluster_result.method_used,
        edge_weight_threshold=cluster_result.edge_weight_threshold_applied,
        exclude_columns=exclude_columns,
    )
    if (
        session.feature_evidence_index is not None
        and session.feature_evidence_fingerprint == fingerprint
    ):
        return session.feature_evidence_index

    evidence_index = build_feature_evidence_index(
        session.model,
        session.data,
        cluster_result.labels,
        exclude_columns=exclude_columns,
    )
    session.feature_evidence_index = evidence_index
    session.feature_evidence_fingerprint = fingerprint
    session.feature_evidence_cluster_meta = _cluster_result_payload(cluster_result)
    return evidence_index


def _require_cluster_state(session: _PulsarSession) -> tuple[FeatureEvidenceIndex, dict[str, Any]]:
    if session.model is None or session.data is None:
        raise ToolError("No model found. Run run_topological_sweep() first.")
    if session.clusters is None or session.feature_evidence_index is None:
        raise ToolError("No cluster evidence found. Run generate_cluster_dossier() first.")
    return (
        session.feature_evidence_index,
        session.feature_evidence_cluster_meta or {},
    )


def _build_graph_summary(model: ThemaRS) -> dict[str, Any]:
    graph = model.cosmic_graph
    components = list(nx.connected_components(graph))
    component_sizes = sorted((len(component) for component in components), reverse=True)

    component_by_node: dict[int, int] = {}
    for component_id, component in enumerate(components):
        for node in component:
            component_by_node[int(node)] = component_id

    nodes = []
    for node in sorted(graph.nodes()):
        nodes.append(
            {
                "node": int(node),
                "component_id": component_by_node[int(node)],
                "degree": int(graph.degree(node)),
                "weighted_degree": float(graph.degree(node, weight="weight")),
            }
        )

    edges = []
    for source, target, data in graph.edges(data=True):
        edges.append(
            {
                "source": int(source),
                "target": int(target),
                "weight": float(data.get("weight", 0.0)),
            }
        )
    edges.sort(key=lambda edge: (-edge["weight"], edge["source"], edge["target"]))

    total_edges = len(edges)
    truncated = total_edges > _MAX_EDGES_IN_SUMMARY
    edges = edges[:_MAX_EDGES_IN_SUMMARY]

    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": total_edges,
        "resolved_threshold": float(model.resolved_threshold),
        "component_count": len(components),
        "component_sizes": component_sizes,
        "nodes": nodes,
        "edges": edges,
        "edges_truncated": truncated,
        "edges_shown": len(edges),
    }


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


@mcp.tool()
async def explain_suggestion(
    config_yaml: str, dataset_geometry: str, ctx: Context
) -> str:
    """
    Explains the mathematical reasoning behind a specific parameter suggestion based on raw geometry.

    Args:
        config_yaml: The YAML config to explain.
        dataset_geometry: JSON string of the dataset geometry summary.

    Returns:
        A Markdown explanation of WHY these parameters were chosen.
    """
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


@mcp.tool()
async def get_experiment_history(ctx: Context) -> str:
    """
    Returns a markdown table of all topological sweeps run in the current session.
    Use this to reason about your trajectory across multiple iterations.

    Returns:
        Markdown table of history. Returns an empty table if no sweeps have been run.
    """
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


@mcp.tool()
async def get_runtime_context(ctx: Context = None) -> str:
    """
    Return the MCP server runtime context so agents can reason about path visibility
    and handle lifecycle before attempting file-based operations.
    """
    session = _get_session(ctx)
    payload = {
        "cwd": os.getcwd(),
        "cache_dir": str(registry.cache_dir),
        "temp_dir": os.getenv("TMPDIR", "/tmp"),
        "transport_assumption": "stdio-single-client",
        "session_id": _session_key(ctx),
        "dataset_handle_persistence": "on-disk registry under cache_dir",
        "run_handle_persistence": "on-disk registry under cache_dir/runs",
        "latest_dataset_id": session.dataset_id,
        "latest_run_id": session.latest_run_id,
        "path_guidance": [
            "Use ingest_dataset(path) for host-visible absolute paths.",
            "SANDBOX ISOLATION: If your file is in a sandbox (e.g. /home/claude), use the 'Cache-Bridge' pattern: Copy the file to the `cache_dir` shown above, then call `ingest_dataset(path)` on the destination.",
            "Chunked/Base64 uploads are a last-resort legacy fallback for remote-only servers. DO NOT use them for local files.",
            "config_path must be server-visible; config_yaml should be raw YAML (no Markdown).",
        ],
    }
    return json.dumps(payload, indent=2)


@mcp.tool()
async def ingest_dataset(path: str, ctx: Context = None) -> str:
    """
    Register a host-visible absolute dataset path and return a stable dataset_id handle.
    Use this only when the MCP server can read the path directly.
    """
    try:
        record = registry.register_dataset(path)
        session = _get_session(ctx)
        session.dataset_id = record.dataset_id
        return json.dumps(dataclasses.asdict(record), indent=2)
    except FileNotFoundError:
        return path_access_error(
            "ingest_dataset",
            path,
            missing_action=(
                "Ask the user for a host-visible absolute dataset path, then call "
                "ingest_dataset again."
            ),
            sandbox_action=(
                "Your file is isolated in a sandbox. DO NOT use base64 or chunked uploads. "
                "Run a bash script to copy the file to the `cache_dir` (call `get_runtime_context` "
                "to find it), then retry `ingest_dataset(path)` with the new path."
            ),
        )
    except PermissionError:
        return mcp_error(
            "ingest_dataset",
            "Dataset path exists but is not readable by the MCP server.",
            error_code="FILE_PERMISSION_DENIED",
            agent_action="Provide a readable host-visible dataset path.",
            details={"path_context": {"attempted_path": path}},
        )
    except Exception as e:
        return mcp_error("ingest_dataset", str(e))


@mcp.tool()
async def ingest_dataset_base64(
    filename: str,
    content_base64: str,
    media_type: str = "text/csv",
    ctx: Context = None,
) -> str:
    """
    Persist a small or medium uploaded dataset sent as base64 and return dataset_id.
    Prefer this over raw text content for one-shot uploads. Use staged upload for larger files.
    """
    try:
        if media_type not in {"text/csv", "application/octet-stream"}:
            return mcp_error(
                "ingest_dataset_base64",
                f"Unsupported media_type '{media_type}'.",
                error_code="UPLOAD_MEDIA_TYPE_UNSUPPORTED",
                agent_action=(
                    "Use text/csv for CSV uploads, or application/octet-stream if the "
                    "client cannot provide a specific text media type."
                ),
            )
        try:
            content_bytes = base64.b64decode(content_base64, validate=True)
        except Exception:
            return mcp_error(
                "ingest_dataset_base64",
                "Base64 payload could not be decoded.",
                error_code="UPLOAD_DECODE_FAILED",
                agent_action="Provide valid base64 content for one-shot upload, or use staged upload for large files.",
            )
        record = registry.register_dataset_bytes(
            filename,
            content_bytes,
            source="base64",
        )
        session = _get_session(ctx)
        session.dataset_id = record.dataset_id
        return json.dumps(dataclasses.asdict(record), indent=2)
    except Exception as e:
        return mcp_error("ingest_dataset_base64", str(e))


@mcp.tool()
async def begin_dataset_upload(
    filename: str,
    media_type: str = "text/csv",
    ctx: Context = None,
) -> str:
    """
    Begin a staged server-side upload for a dataset that is not reachable by path.
    Use this for larger sandboxed uploads, then append chunks and finalize to get dataset_id.
    """
    try:
        record = registry.begin_upload(filename, media_type=media_type)
        return json.dumps(dataclasses.asdict(record), indent=2)
    except Exception as e:
        return mcp_error("begin_dataset_upload", str(e))


@mcp.tool()
async def append_dataset_chunk(
    upload_id: str,
    chunk: str,
    encoding: str = "base64",
    ctx: Context = None,
) -> str:
    """
    Append one chunk to a staged dataset upload.
    Use base64 encoding by default to avoid newline and control-character corruption.
    """
    try:
        if encoding == "base64":
            try:
                chunk_bytes = base64.b64decode(chunk, validate=True)
            except Exception:
                return mcp_error(
                    "append_dataset_chunk",
                    "Chunk payload could not be decoded from base64.",
                    error_code="UPLOAD_DECODE_FAILED",
                    agent_action="Retry with valid base64 chunk data.",
                )
        elif encoding == "utf-8":
            chunk_bytes = chunk.encode("utf-8")
        else:
            return mcp_error(
                "append_dataset_chunk",
                f"Unsupported chunk encoding '{encoding}'.",
                error_code="UPLOAD_ENCODING_UNSUPPORTED",
                agent_action="Use encoding='base64' for binary-safe chunk transport.",
            )

        record = registry.append_upload_chunk(upload_id, chunk_bytes)
        if record is None:
            return unknown_handle_error("append_dataset_chunk", "upload_id", upload_id)
        return json.dumps(dataclasses.asdict(record), indent=2)
    except Exception as e:
        return mcp_error("append_dataset_chunk", str(e))


@mcp.tool()
async def finalize_dataset_upload(upload_id: str, ctx: Context = None) -> str:
    """
    Finalize a staged upload and register it as a dataset_id for downstream tools.
    """
    try:
        record = registry.finalize_upload(upload_id)
        if record is None:
            return unknown_handle_error(
                "finalize_dataset_upload", "upload_id", upload_id
            )
        session = _get_session(ctx)
        session.dataset_id = record.dataset_id
        return json.dumps(dataclasses.asdict(record), indent=2)
    except Exception as e:
        return mcp_error("finalize_dataset_upload", str(e))


@mcp.tool()
async def ingest_dataset_content(
    filename: str,
    content: str,
    ctx: Context = None,
) -> str:
    """
    Persist uploaded or sandbox-local dataset content into the MCP server cache and
    return a stable dataset_id handle. This is a legacy text-only fallback.
    Prefer ingest_dataset_base64 for one-shot uploads and staged upload for larger files.
    """
    try:
        record = registry.register_dataset_content(filename, content)
        session = _get_session(ctx)
        session.dataset_id = record.dataset_id
        return json.dumps(dataclasses.asdict(record), indent=2)
    except Exception as e:
        return mcp_error("ingest_dataset_content", str(e))


def _calibrate_processed_space(
    df: pd.DataFrame,
    column_profiles: list[Any],
    n_samples: int,
    data_path: str,
) -> NumericProfile | None:
    """Run recommended preprocessing + scaling, then profile the result.

    Returns ``None`` if preprocessing fails (e.g. too few numeric columns
    after encoding).  The caller should fall back to raw-space calibration.
    """
    drop, impute_dict, encode_dict, _ = _recommend_preprocessing_block(
        column_profiles, n_samples
    )

    # Build a minimal config just for preprocessing (sweep/cosmic values
    # are irrelevant — we only need preprocessing fields + data path).
    from pulsar.config import (
        BallMapperSpec,
        CosmicGraphSpec,
        EncodeSpec,
        ImputeSpec,
        PCASpec,
        PulsarConfig,
    )

    impute_specs = {
        col: ImputeSpec(method=spec["method"], seed=spec.get("seed", 42))
        for col, spec in impute_dict.items()
    }
    encode_specs = {
        col: EncodeSpec(
            method=spec["method"], max_categories=spec.get("max_categories")
        )
        for col, spec in encode_dict.items()
    }
    temp_cfg = PulsarConfig(
        data=data_path,
        impute=impute_specs,
        encode=encode_specs,
        drop_columns=drop,
        pca=PCASpec(),
        ball_mapper=BallMapperSpec(),
        cosmic_graph=CosmicGraphSpec(),
    )

    try:
        df_processed, layout = preprocess_dataframe(df, temp_cfg)
    except (ValueError, TypeError):
        logger.info("Processed-space calibration failed; falling back to raw geometry")
        return None

    X = df_processed.to_numpy(dtype=np.float64)
    if X.shape[1] < 2:
        return None
    X_scaled = SkScaler().fit_transform(X)
    return profile_numeric_matrix(X_scaled)


@mcp.tool()
async def create_config(dataset_id: str, intent: str = "", ctx: Context = None) -> str:
    """
    Generate canonical Pulsar YAML for an ingested dataset_id.

    Calibrates epsilon and PCA dimensions against the processed feature
    space (after recommended preprocessing + scaling), not raw columns.
    Supports ingested CSV and Parquet datasets.
    """
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
        response: dict[str, Any] = {
            "status": "ok",
            "config_yaml": config_yaml,
            "calibration_space": calibration_space,
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


@mcp.tool()
async def refine_config(config_yaml: str, overrides: dict[str, Any]) -> str:
    """
    Apply constrained overrides to canonical Pulsar YAML and return normalized YAML.
    """
    try:
        result = apply_overrides(config_yaml, overrides)
        payload = {
            "status": "ok",
            "applied_overrides": result.applied_overrides,
            "diff": result.diff,
            "config_yaml": result.config_yaml,
        }
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


@mcp.tool()
async def get_active_config(ctx: Context = None) -> str:
    """Return the active in-session config, if any."""
    session = _get_session(ctx)
    if not session.active_config_yaml:
        return mcp_error(
            "get_active_config",
            "No active config in session. Run create_config or refine_active_config first.",
            error_code="ACTIVE_CONFIG_MISSING",
            agent_action="Create or refine a config before requesting the active session config.",
        )
    return json.dumps(
        {
            "status": "ok",
            "config_yaml": session.active_config_yaml,
            "dataset_id": session.active_config_dataset_id,
        },
        indent=2,
    )


@mcp.tool()
async def refine_active_config(overrides: dict[str, Any], ctx: Context = None) -> str:
    """Apply constrained overrides to the session's active config."""
    session = _get_session(ctx)
    if not session.active_config_yaml:
        return mcp_error(
            "refine_active_config",
            "No active config in session. Run create_config first.",
            error_code="ACTIVE_CONFIG_MISSING",
            agent_action="Call create_config(dataset_id) before refining the session config.",
        )
    try:
        result = apply_overrides(session.active_config_yaml, overrides)
        session.active_config_yaml = result.config_yaml
        payload = {
            "status": "ok",
            "applied_overrides": result.applied_overrides,
            "diff": result.diff,
            "config_yaml": result.config_yaml,
            "dataset_id": session.active_config_dataset_id,
        }
        return json.dumps(payload, indent=2)
    except ValueError as e:
        return mcp_error(
            "refine_active_config",
            str(e),
            error_code="UNKNOWN_OVERRIDE_KEY",
            agent_action="Use only valid override keys. See error message for valid key list.",
        )
    except Exception as e:
        return mcp_error("refine_active_config", str(e))


@mcp.tool()
async def validate_config(
    config_yaml: str,
    dataset_id: str = "",
    ctx: Context = None,
) -> str:
    """
    Validate full Pulsar config shape and normalize it into canonical YAML.
    Prefer dataset_id once data has been ingested.
    """
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


@mcp.tool()
async def run_topological_sweep(
    config_path: str = "",
    config_yaml: str = "",
    dataset_id: str = "",
    save_config: bool = False,
    ctx: Context = None,
) -> str:
    """
    Run the Pulsar topological sweep pipeline on a dataset.

    Returns a markdown diff of parameter and metric changes compared to your previous run,
    followed by the full execution summary.

    Args:
        config_path: Path to a params.yaml file on disk.
        config_yaml: Inline YAML string (preferred).
        dataset_id: Preferred dataset handle when data has already been ingested.
        save_config: If True, persist the resolved config YAML to disk.
    """
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
        bound_dataset_id = (
            dataset_id
            or _dataset_id_for_path(session.dataset_id, bound_data_path)
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
        from pulsar.mcp.diagnostics import diagnose_model

        current_metrics_obj = diagnose_model(model)
        current_metrics = dataclasses.asdict(current_metrics_obj)
        graph_summary = _build_graph_summary(model)

        # Build structured diff
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

            pm = prev_record.metrics
            cm = current_metrics
            for key in ("n_edges", "component_count", "giant_fraction"):
                pv, cv = pm.get(key, 0), cm.get(key, 0)
                if pv != cv:
                    diff.append({"field": key, "previous": pv, "current": cv})

        # Record history
        session.sweep_history.append(
            SweepRecord(time.time(), current_yaml, current_metrics)
        )
        persisted_dataset_id = (
            dataset_id
            or _dataset_id_for_path(session.dataset_id, cfg.data)
        )
        run_record = registry.save_run(
            dataset_id=persisted_dataset_id,
            config_yaml=current_yaml,
            metrics=current_metrics,
            resolved_threshold=model.resolved_threshold,
            graph_summary=graph_summary,
        )
        session.latest_run_id = run_record.run_id
        response: dict[str, Any] = {
            "status": "ok",
            "run_id": run_record.run_id,
            "metrics": current_metrics,
            "pca_cached": precomputed is not None,
            "pca_cache_status": pca_cache_status,
            "memory_usage_mb": session.calculate_memory_mb(),
            "diff": diff,
            "config_yaml_normalized": current_yaml,
            "data_shape": list(session.data.shape),
        }
        graph_health, is_connected, recommended_next_action = _graph_health_summary(
            current_metrics
        )
        response["is_connected"] = is_connected
        response["singleton_fraction"] = current_metrics.get("singleton_fraction", 0.0)
        response["spectral_clustering_allowed"] = bool(is_connected)
        response["graph_health"] = graph_health
        response["recommended_next_action"] = recommended_next_action
        if saved_path:
            response["saved_config_path"] = saved_path
        if persisted_dataset_id:
            response["dataset_id"] = persisted_dataset_id

        return json.dumps(response, indent=2)

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


@mcp.tool()
async def generate_cluster_dossier(
    method: str = "auto",
    max_k: int = 15,
    edge_weight_threshold: float = 0.0,
    detail: str = "summary",
    response_format: str = "json",
    format: str = "",
    exclude_columns: list[str] | None = None,
    ctx: Context = None,
) -> str:
    """
    Generate a statistical dossier of the topological clusters.

    Args:
        method: Clustering method ("auto", "spectral", "components").
        max_k: Maximum k for spectral clustering search.
        edge_weight_threshold: Drop edges with weight <= this value before
            clustering.  Edge weights are the fraction of ball maps that
            placed two points together.  Use weight percentiles from
            diagnose_cosmic_graph to choose a value (e.g. weight_p50 to
            keep only the stronger half of edges).
        detail: Payload richness. "summary" is compact, "standard" retains
            full evidence for core/supporting signals, and "full" returns the
            full evidence matrix.
        response_format: "json" for structured payloads or "markdown" for
            narrative output only.
        format: Legacy alias. "full" maps to detail="full"; "json" and
            "markdown" map to response_format.
        exclude_columns: Optional list of columns to exclude from the report.
            This hides columns from the final dossier but does NOT affect the
            topological modeling itself.
    """
    session = _get_session(ctx)

    if session.model is None or session.data is None:
        raise ToolError("No model found. Run run_topological_sweep() first.")

    if method not in ("auto", "spectral", "components"):
        raise ToolError(
            f"method must be 'auto', 'spectral', or 'components', got '{method}'"
        )

    try:
        detail, response_format = _resolve_response_format(
            response_format=response_format,
            legacy_format=format,
            detail=detail,
        )
        result = resolve_clusters(
            session.model,
            method=method,
            max_k=max_k,
            edge_weight_threshold=edge_weight_threshold,
        )
        session.clusters = result.labels
        session.report_exclude_columns = exclude_columns
        session.feature_evidence_cluster_meta = _cluster_result_payload(result)

        evidence_index = _get_or_build_evidence_index(
            session,
            cluster_result=result,
            exclude_columns=exclude_columns,
        )

        dossier = build_dossier(
            session.model,
            session.data,
            result.labels,
            exclude_columns=exclude_columns,
            detail="standard" if response_format == "markdown" and detail == "summary" else detail,
            evidence_index=evidence_index,
        )
        cluster_meta = _cluster_result_payload(result)
        if response_format == "markdown":
            return dossier_to_markdown(dossier)

        payload = _build_evidence_payload(dossier, cluster_meta)
        return json.dumps(
            payload,
            separators=(",", ":") if len(dossier.clusters) > 10 else None,
            indent=None if len(dossier.clusters) > 10 else 2,
        )

    except ValueError as e:
        if "Graph is disconnected" in str(e):
            return mcp_error(
                "generate_cluster_dossier",
                "Spectral clustering requires a connected affinity graph. Your current threshold has disconnected the manifold.",
                error_code="GRAPH_DISCONNECTED",
                agent_action="Decrease edge_weight_threshold or increase epsilon sweep range to connect the graph.",
            )
        logger.error(f"Error generating dossier: {e}")
        return mcp_error("generate_cluster_dossier", str(e))
    except Exception as e:
        logger.error(f"Error generating dossier: {e}")
        return mcp_error("generate_cluster_dossier", str(e))


@mcp.tool()
async def get_cluster_profile(
    cluster_id: int,
    detail: str = "standard",
    response_format: str = "json",
    ctx: Context = None,
) -> str:
    """Return targeted evidence for one cluster from the cached dossier state."""
    try:
        if detail not in {"summary", "standard", "full"}:
            raise ToolError(
                f"detail must be 'summary', 'standard', or 'full', got '{detail}'"
            )
        if response_format not in {"json", "markdown"}:
            raise ToolError(
                f"response_format must be 'json' or 'markdown', got '{response_format}'"
            )

        session = _get_session(ctx)
        evidence_index, cluster_meta = _require_cluster_state(session)
        payload = {
            "status": "ok",
            "cluster_result": cluster_meta,
            "detail": detail,
            "cluster": cluster_profile_payload(
                evidence_index,
                cluster_id,
                detail=detail,
            ),
        }
        if response_format == "markdown":
            cluster = payload["cluster"]
            lines = [
                f"## Cluster {cluster['cluster_id']}: {cluster['semantic_name']}",
                "",
                f"- Size: {cluster['size']} ({cluster['size_pct']:.1f}%)",
                f"- Topological neighbors: {', '.join(str(neighbor['cluster_id']) for neighbor in cluster['topological_neighbors']) or 'None'}",
                "",
                "### Numeric signals",
            ]
            for row in cluster["numeric_features"]:
                lines.append(
                    f"- {row['column']}: {row.get('direction', 'mixed')} signal ({row['signal_tier']}, score={row['aggregate_score']:.3f})"
                )
            lines.append("")
            lines.append("### Categorical signals")
            for row in cluster["categorical_features"]:
                lines.append(
                    f"- {row['column']}={row['value']}: {row['signal_tier']} signal (lift={row.get('lift', 0.0):.2f}, score={row['aggregate_score']:.3f})"
                )
            return "\n".join(lines)
        return json.dumps(payload, indent=2)
    except KeyError:
        return mcp_error(
            "get_cluster_profile",
            f"Unknown cluster_id '{cluster_id}'.",
            error_code="CLUSTER_ID_UNKNOWN",
            agent_action="Use generate_cluster_dossier first and inspect available cluster IDs.",
        )
    except Exception as e:
        return mcp_error("get_cluster_profile", str(e))


@mcp.tool()
async def get_feature_signal(
    feature_names: list[str],
    cluster_ids: list[int] | None = None,
    ctx: Context = None,
) -> str:
    """Return cross-cluster evidence for specific feature columns."""
    try:
        session = _get_session(ctx)
        evidence_index, cluster_meta = _require_cluster_state(session)
        payload = {
            "status": "ok",
            "cluster_result": cluster_meta,
            "signals": feature_signal_payload(
                evidence_index,
                feature_names,
                cluster_ids=cluster_ids,
            ),
        }
        return json.dumps(payload, indent=2)
    except Exception as e:
        return mcp_error("get_feature_signal", str(e))


@mcp.tool()
async def get_cluster_signal_matrix(
    cluster_ids: list[int] | None = None,
    include_context: bool = False,
    ctx: Context = None,
) -> str:
    """Return the cached cross-cluster signal matrix without regenerating the full dossier."""
    try:
        session = _get_session(ctx)
        evidence_index, cluster_meta = _require_cluster_state(session)
        payload = {
            "status": "ok",
            "cluster_result": cluster_meta,
            "signal_matrix": signal_matrix_payload(
                evidence_index,
                cluster_ids=cluster_ids,
                include_context=include_context,
            ),
        }
        return json.dumps(payload, indent=2)
    except Exception as e:
        return mcp_error("get_cluster_signal_matrix", str(e))


@mcp.tool()
async def compare_clusters_tool(
    cluster_a: int, cluster_b: int, ctx: Context = None
) -> str:
    """
    Perform pairwise statistical tests between two clusters.
    """
    session = _get_session(ctx)

    if session.data is None or session.clusters is None:
        raise ToolError(
            "No data or clusters found. Run generate_cluster_dossier() first."
        )

    try:
        results = compare_clusters(session.data, session.clusters, cluster_a, cluster_b)
        if not results:
            return mcp_error(
                "compare_clusters_tool",
                f"No results for comparison between clusters {cluster_a} and {cluster_b}.",
            )

        markdown = comparison_to_markdown(cluster_a, cluster_b, results)
        return markdown

    except Exception as e:
        logger.error(f"Error comparing clusters: {e}")
        return mcp_error("compare_clusters_tool", str(e))


@mcp.tool()
async def export_labeled_data(
    cluster_names: dict[int, str], output_path: str, ctx: Context
) -> str:
    """
    Assign semantic names to clusters and export the labeled dataset to CSV.
    """
    session = _get_session(ctx)

    if session.data is None or session.clusters is None:
        raise ToolError(
            "No data or clusters found. Run generate_cluster_dossier() first."
        )

    actual_ids = set(session.clusters.unique().tolist())
    provided_ids = set(int(str(k).lstrip("_")) for k in cluster_names.keys())
    missing_ids = actual_ids - provided_ids
    if missing_ids:
        raise ToolError(
            f"cluster IDs {sorted(missing_ids)} have no name mapping. Provide all {len(actual_ids)} cluster IDs."
        )

    try:
        df = session.data.copy()
        df["topological_cluster_id"] = session.clusters
        names_map = {int(str(k).lstrip("_")): v for k, v in cluster_names.items()}
        df["topological_cluster_name"] = df["topological_cluster_id"].map(names_map)
        df.to_csv(output_path, index=False)
        return f"Successfully exported labeled data to {output_path}"

    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return mcp_error("export_labeled_data", str(e))


@mcp.tool()
async def characterize_dataset(
    csv_path: str = "",
    dataset_id: str = "",
    ctx: Context = None,
) -> str:
    """
    Probes dataset geometry to return raw facts (N, features, variance curve, k-NN mean).
    Prefer dataset_id after ingest. Use csv_path only for host-visible CSV or Parquet files.
    """
    try:
        from pulsar.analysis.characterization import characterize_dataset as _char

        if dataset_id:
            csv_path = _resolve_dataset_path(dataset_id)
        if not csv_path:
            raise ToolError("Provide either csv_path or dataset_id.")

        session = _get_session(ctx)
        if dataset_id:
            df, normalized_path = await _load_session_dataframe(
                session,
                dataset_id=dataset_id,
            )
        else:
            df, normalized_path = await _load_session_dataframe(
                session,
                data_path=csv_path,
            )

        result = await asyncio.to_thread(_char, normalized_path, dataframe=df)
        return json.dumps(dataclasses.asdict(result), indent=2)
    except FileNotFoundError:
        return path_access_error(
            "characterize_dataset",
            csv_path,
            missing_action=(
                "Ask the user for a host-visible absolute dataset path, or use "
                "ingest_dataset first and pass dataset_id."
            ),
            sandbox_action=(
                "Your file is isolated in a sandbox. DO NOT use base64 or chunked uploads. "
                "Run a bash script to copy the file to the `cache_dir` (call `get_runtime_context` "
                "to find it), use `ingest_dataset(path)` to get a dataset_id, and then "
                "retry this tool."
            ),
        )
    except LookupError:
        return unknown_handle_error("characterize_dataset", "dataset_id", dataset_id)
    except Exception as e:
        logger.error(f"Error analyzing dataset: {e}")
        return mcp_error("characterize_dataset", str(e))


@mcp.tool()
async def diagnose_cosmic_graph(ctx: Context) -> str:
    """
    Diagnose the fitted cosmic graph quality by returning pure GraphMetrics.
    Interpret these metrics (e.g. density, component distribution) given N.
    """
    session = _get_session(ctx)

    if session.model is None:
        raise ToolError("No model found. Run run_topological_sweep() first.")

    try:
        from pulsar.mcp.diagnostics import diagnose_model

        result = diagnose_model(session.model)
        return json.dumps(dataclasses.asdict(result), indent=2)
    except Exception as e:
        logger.error(f"Error diagnosing graph: {e}")
        return mcp_error("diagnose_cosmic_graph", str(e))


@mcp.tool()
async def get_threshold_stability_curve(ctx: Context) -> str:
    """
    Return the full component-count-vs-edge-weight-threshold curve.

    Uses H0 persistent homology on the cosmic graph's weighted adjacency
    to show how many connected components exist at each edge weight threshold.
    Use this to reason about alternative clustering thresholds after the
    initial auto-clustering.

    Returns:
        JSON with thresholds, component_counts, top plateaus, and the
        auto-selected threshold (midpoint of longest valid plateau).
    """
    session = _get_session(ctx)

    if session.model is None:
        raise ToolError("No model found. Run run_topological_sweep() first.")

    try:
        from pulsar._pulsar import find_stable_thresholds

        adj = session.model.weighted_adjacency
        stability = await asyncio.to_thread(find_stable_thresholds, adj)

        plateaus = [
            {
                "start": float(p.start_threshold),
                "end": float(p.end_threshold),
                "component_count": int(p.component_count),
                "length": float(p.length),
                "midpoint": float(p.midpoint),
            }
            for p in stability.top_k_plateaus(10)
        ]

        return json.dumps(
            {
                "status": "ok",
                "optimal_threshold": float(stability.optimal_threshold),
                "plateaus": plateaus,
                "thresholds": [float(t) for t in stability.thresholds],
                "component_counts": [int(c) for c in stability.component_counts],
            },
            indent=2,
        )
    except Exception as e:
        logger.error(f"Error computing stability curve: {e}")
        return mcp_error("get_threshold_stability_curve", str(e))


@mcp.tool()
async def get_topological_skeleton(run_id: str = "", ctx: Context = None) -> str:
    """
    Return structured graph connectivity for the latest run or an explicit run_id.

    The edge list is capped at 500 entries (top edges by weight). If the graph
    has more edges, edges_truncated=true and edge_count shows the true total.
    Use edge_count and component_sizes for density reasoning — do not rely on
    len(edges) when edges_truncated is true.
    """
    try:
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
            "config_yaml": record.config_yaml,
            "resolved_threshold": record.resolved_threshold,
            "graph": record.graph_summary,
        }
        return json.dumps(payload, indent=2)
    except Exception as e:
        return mcp_error("get_topological_skeleton", str(e))


@mcp.tool()
async def compare_sweeps(run_a: str, run_b: str, ctx: Context = None) -> str:
    """
    Compare two persisted sweep runs by config and graph metrics.
    """
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
            f"| threshold | {cfg_a.get('cosmic_graph', {}).get('threshold', 'auto')} | {cfg_b.get('cosmic_graph', {}).get('threshold', 'auto')} |",
            f"| nodes | {metrics_a.get('n_nodes')} | {metrics_b.get('n_nodes')} |",
            f"| edges | {metrics_a.get('n_edges')} | {metrics_b.get('n_edges')} |",
            f"| components | {metrics_a.get('component_count')} | {metrics_b.get('component_count')} |",
            f"| giant_fraction | {metrics_a.get('giant_fraction', 0):.2%} | {metrics_b.get('giant_fraction', 0):.2%} |",
            f"| weight_p95 | {metrics_a.get('weight_p95', 0):.4f} | {metrics_b.get('weight_p95', 0):.4f} |",
        ]
        return "\n".join(lines)
    except Exception as e:
        return mcp_error("compare_sweeps", str(e))


@mcp.tool()
async def recommend_preprocessing(
    dataset_geometry: str = "",
    dataset_id: str = "",
    ctx: Context = None,
) -> str:
    """
    Analyze column profiles and return preprocessing recommendations.
    Prefer dataset_id after ingest; accepts dataset_geometry as fallback.

    Args:
        dataset_geometry: The raw JSON string from characterize_dataset.
        dataset_id: Preferred dataset handle. When provided, characterizes
            the dataset automatically (dataset_geometry is ignored).

    Returns:
        JSON with preprocessing_yaml, per-column rationale, and expansion estimate.
    """
    try:
        if dataset_id:
            from pulsar.analysis.characterization import characterize_dataset as _char

            dataset_path = _resolve_dataset_path(dataset_id)
            result = await asyncio.to_thread(_char, dataset_path)
            geo = dataclasses.asdict(result)
        elif dataset_geometry:
            geo = json.loads(dataset_geometry)
        else:
            return mcp_error(
                "recommend_preprocessing",
                "Provide either dataset_id or dataset_geometry.",
                error_code="MISSING_INPUT",
                agent_action="Pass dataset_id after ingest, or dataset_geometry JSON from characterize_dataset.",
            )

        n_samples = geo.get("n_samples", 0)
        column_profiles = geo.get("column_profiles", [])

        if not column_profiles:
            return mcp_error(
                "recommend_preprocessing",
                "No column_profiles found in dataset_geometry. Pass the full JSON from characterize_dataset.",
            )

        drop, impute, encode, rationale = _recommend_preprocessing_block(
            column_profiles, n_samples
        )

        preprocessing_yaml = _preprocessing_block_to_yaml(drop, impute, encode)

        # Estimate expansion: each encoded column adds ~n_unique dummy columns
        expansion_estimate = 0
        for raw_cp in column_profiles:
            cp = (
                raw_cp
                if isinstance(raw_cp, dict)
                else {"name": raw_cp.name, "n_unique": raw_cp.n_unique}
            )
            name = cp["name"]
            if name in encode:
                expansion_estimate += cp.get("n_unique", 2)

        return json.dumps(
            {
                "status": "ok",
                "preprocessing_yaml": preprocessing_yaml,
                "rationale": [
                    {"column": col, "decision": dec, "reason": reason}
                    for col, dec, reason in rationale
                ],
                "expansion_estimate": expansion_estimate,
                "markdown_summary": _rationale_table(rationale),
            },
            indent=2,
        )

    except Exception as e:
        logger.error(f"Error in recommend_preprocessing: {e}")
        return mcp_error("recommend_preprocessing", str(e))


@mcp.tool()
async def repair_preprocessing_config(
    error_message: str,
    config_yaml: str,
    dataset_geometry: str,
    ctx: Context,
) -> str:
    """
    Given a preprocessing error from run_topological_sweep, produce a corrected
    config_yaml with a change log of what was fixed and why.

    Handles: NaN remaining, non-numeric columns, coercion failure, all-missing
    columns, and cardinality violations.

    Args:
        error_message: The full error text from the failed sweep.
        config_yaml: The config_yaml that caused the error.
        dataset_geometry: The raw JSON string from characterize_dataset.

    Returns:
        Markdown with error classification, change log table, and patched config_yaml.
    """
    try:
        geo = json.loads(dataset_geometry)
        if not geo.get("column_profiles"):
            return mcp_error(
                "repair_preprocessing_config",
                "No column_profiles found in dataset_geometry. Pass the full JSON from characterize_dataset.",
                error_code="MISSING_COLUMN_PROFILES",
                agent_action="Call characterize_dataset first, then pass its full JSON output.",
            )
        profiles_by_name: dict[str, Any] = {}
        for cp in geo.get("column_profiles", []):
            pname = cp["name"] if isinstance(cp, dict) else cp.name
            profiles_by_name[pname] = cp
        return repair_config(config_yaml, error_message, profiles_by_name)
    except Exception as e:
        logger.error(f"Error in repair_preprocessing_config: {e}")
        return mcp_error("repair_preprocessing_config", str(e))


@mcp.tool()
async def validate_preprocessing_config(config_yaml: str, ctx: Context) -> str:
    """
    Dry-run the preprocessing stage only against session data — no PCA, no BallMapper,
    no sweep cost. Use this to confirm a config is valid before run_topological_sweep.

    Requires a prior run_topological_sweep call (to populate session data).

    Args:
        config_yaml: Inline YAML config string to validate.

    Returns:
        PASS with schema summary, or a structured error matching repair_preprocessing_config input format.
    """
    session = _get_session(ctx)

    if session.data is None:
        return mcp_error(
            "validate_preprocessing_config",
            "No data in session. Run run_topological_sweep (even with a minimal config) or characterize_dataset first to load data into the session.",
        )

    try:
        config_dict = yaml.safe_load(config_yaml)
        if not isinstance(config_dict, dict):
            return mcp_error(
                "validate_preprocessing_config",
                "config_yaml must be a valid YAML mapping.",
            )

        cfg = load_config(config_dict)
        df_out, layout = await asyncio.to_thread(
            preprocess_dataframe, session.data, cfg
        )

        # Compute expansion diagnostics
        input_cols = set(session.data.columns)
        output_names = layout.feature_names
        dummy_count = sum(
            1 for name in output_names if "_" in name and name not in input_cols
        )
        missingness_flag_count = sum(
            1 for name in output_names if name.endswith("_was_missing")
        )
        high_cardinality_encoded = [
            col for col, cats in layout.vocab.items() if len(cats) > 20
        ]

        col_preview = list(output_names[:8])
        if len(output_names) > 8:
            col_preview.append(f"... +{len(output_names) - 8} more")

        return json.dumps(
            {
                "status": "ok",
                "valid": True,
                "input_rows": len(session.data),
                "output_rows": layout.n_rows,
                "output_feature_count": len(output_names),
                "dummy_expansion_count": dummy_count,
                "missingness_flag_count": missingness_flag_count,
                "high_cardinality_encoded_columns": high_cardinality_encoded,
                "feature_names_preview": list(col_preview),
                "nan_remaining": 0,
            },
            indent=2,
        )

    except (ValueError, TypeError) as e:
        return json.dumps(
            {
                "status": "error",
                "valid": False,
                "error": str(e),
                "agent_action": (
                    "Call repair_preprocessing_config(error_message=..., "
                    "config_yaml=..., dataset_geometry=...) to fix this automatically."
                ),
            },
            indent=2,
        )
    except Exception as e:
        logger.error(f"Error in validate_preprocessing_config: {e}")
        return mcp_error("validate_preprocessing_config", str(e))


@mcp.tool()
async def export_html_report(
    dataset_id: str = "",
    exclude_columns: list[str] | None = None,
    cluster_names: list[dict[str, str]] | dict[str, str] | None = None,
    ctx: Context = None,
) -> str:
    """
    Generate a rich HTML report of the latest topological analysis.

    Args:
        dataset_id: The dataset to report on.
        exclude_columns: Optional list of columns to exclude from the report.
            Defaults to the exclusion list used in generate_cluster_dossier.
        cluster_names: Optional dictionary or list of objects mapping string cluster IDs
            to semantic names.
            Preferred format (List of Objects): [{"id": "0", "name": "Metabolic Cohort"}]
            Legacy format (Dictionary): {"0": "Metabolic Cohort"}
            CRITICAL: The system will generate robotic fallback names (e.g. '[Auto] High X, Low Y')
            if you do not provide this. It is YOUR responsibility as the analytical agent
            to provide human-tractable, scientific cohort names.
    """
    session = _get_session(ctx)
    if session.model is None or session.data is None:
        return mcp_error(
            "export_html_report", "No model found. Run run_topological_sweep first."
        )

    if session.clusters is None:
        return mcp_error(
            "export_html_report",
            "No clusters found. Run generate_cluster_dossier first.",
        )

    # Use explicitly passed columns, or fallback to session defaults
    cols_to_exclude = (
        exclude_columns
        if exclude_columns is not None
        else session.report_exclude_columns
    )

    try:
        evidence_index = None
        if (
            session.feature_evidence_index is not None
            and cols_to_exclude == session.report_exclude_columns
        ):
            evidence_index = session.feature_evidence_index

        dossier = build_dossier(
            session.model,
            session.data,
            session.clusters,
            exclude_columns=cols_to_exclude,
            detail="full",
            evidence_index=evidence_index,
        )

        # Apply LLM-provided cluster names if available
        if cluster_names:
            import re

            def normalize_key(k: str) -> str:
                # Extract digits to handle "u0", "cluster_0", "_0", etc.
                match = re.search(r"(\d+)", str(k))
                return match.group(1) if match else str(k)

            # Handle both formats: List[dict] and dict
            if isinstance(cluster_names, list):
                raw_names = {}
                for item in cluster_names:
                    # Support multiple key names for flexibility
                    cid = item.get("id") or item.get("cluster_id") or item.get("cid")
                    name = item.get("name") or item.get("semantic_name") or item.get("label")
                    if cid is not None and name is not None:
                        raw_names[str(cid)] = name
            else:
                raw_names = cluster_names

            normalized_names = {normalize_key(k): v for k, v in raw_names.items()}
            
            # Check for total mismatch to provide better feedback
            valid_ids = {str(p.cluster_id) for p in dossier.clusters}
            provided_ids = set(normalized_names.keys())
            if not (provided_ids & valid_ids) and raw_names:
                logger.warning(
                    "export_html_report: Provided cluster_names keys %s do not match any cluster IDs %s",
                    list(provided_ids),
                    list(valid_ids),
                )

            for p in dossier.clusters:
                str_id = str(p.cluster_id)
                if str_id in normalized_names:
                    p.semantic_name = normalized_names[str_id]

        config_yaml = (
            session.sweep_history[-1].config_yaml if session.sweep_history else None
        )
        html = dossier_to_html(
            dossier,
            model=session.model,
            config_yaml=config_yaml,
        )

        # Save to a file near the dataset if possible, or current dir
        dataset_path = (
            _resolve_dataset_path(dataset_id) if dataset_id else "pulsar_report.html"
        )
        if os.path.isfile(dataset_path):
            report_path = os.path.splitext(dataset_path)[0] + "_report.html"
        else:
            report_path = os.path.join(os.getcwd(), "pulsar_report.html")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

        return json.dumps(
            {
                "status": "ok",
                "report_path": report_path,
                "report_url": f"file://{os.path.abspath(report_path)}",
                "message": "HTML report exported successfully. Open the report_url in your browser.",
            },
            indent=2,
        )
    except Exception as e:
        return mcp_error("export_html_report", str(e))


@mcp.tool()
async def probe_columns(
    dataset_id: str,
    columns: list[str],
    ctx: Context = None,
) -> str:
    """
    Generate rich, detailed profiles for specific columns (Magnifying Glass).
    Use this when you need sample values or distributions for specific columns
    after seeing the global sparse schema from characterize_dataset.

    Args:
        dataset_id: Handle for the ingested dataset.
        columns: List of column names to probe (max 20).
    """
    from pulsar.analysis.characterization import profile_column_details

    if len(columns) > 20:
        return mcp_error(
            "probe_columns", "Too many columns requested. Max 20 at a time."
        )

    session = _get_session(ctx)
    try:
        df, _ = await _load_session_dataframe(session, dataset_id=dataset_id)
    except LookupError:
        return unknown_handle_error("probe_columns", "dataset_id", dataset_id)
    except Exception as e:
        return mcp_error("probe_columns", f"Could not load data: {e}")

    try:
        profiles = []
        for col in columns:
            if col not in df.columns:
                continue
            profiles.append(
                dataclasses.asdict(profile_column_details(df, col))
            )

        return json.dumps({"status": "ok", "column_profiles": profiles}, indent=2)
    except Exception as e:
        return mcp_error("probe_columns", str(e))


def main():
    mcp.run()


if __name__ == "__main__":
    main()
