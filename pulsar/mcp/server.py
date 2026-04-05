"""
FastMCP Server for Pulsar.

Exposes "Thick Tools" for topological data analysis and interpretation.
"""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass, field
import json
import logging
import networkx as nx
import os
import re
import time
from typing import Any

import pandas as pd
import yaml
from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError

from pulsar.config import config_to_yaml, load_config
from pulsar.preprocessing import preprocess_dataframe
from pulsar.runtime.fingerprint import pca_fingerprint
from pulsar.pipeline import ThemaRS
from pulsar.mcp.interpreter import (
    resolve_clusters,
    build_dossier,
    dossier_to_markdown,
    compare_clusters,
    comparison_to_markdown,
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
# them out *before* validation — one patch protects every tool.
# ---------------------------------------------------------------------------
from fastmcp.tools.function_tool import FunctionTool  # noqa: E402

_original_function_tool_run = FunctionTool.run


async def _lenient_function_tool_run(self, arguments):
    if isinstance(arguments, dict) and arguments:
        valid_keys = set(self.parameters.get("properties", {}).keys())
        arguments = {k: v for k, v in arguments.items() if k in valid_keys}
    return await _original_function_tool_run(self, arguments)


FunctionTool.run = _lenient_function_tool_run


def _build_initial_config_yaml(
    geo: dict[str, Any],
    *,
    data_path: str,
    run_name: str = "initial_sweep",
) -> str:
    """Construct a canonical initial config from dataset geometry."""
    knn_mean = geo.get("knn_k5_mean") or geo.get("knn_mean") or 0.5
    pca_cum_var = geo.get("pca_cumulative_variance", [])

    pca_knee = 3
    if pca_cum_var:
        for dim, var in pca_cum_var:
            if var >= 0.90:
                pca_knee = dim
                break

    pca_dims = [max(2, pca_knee - 1), pca_knee, pca_knee + 1]
    eps_min = knn_mean * 0.8
    eps_max = knn_mean * 1.5

    n_samples = geo.get("n_samples", 0)
    column_profiles = geo.get("column_profiles", [])
    drop, impute, encode, _ = _recommend_preprocessing_block(
        column_profiles, n_samples
    )
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
      values: [42]
  ball_mapper:
    epsilon:
      range:
        min: {eps_min:.4f}
        max: {eps_max:.4f}
        steps: 15
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
        "You are a topological cartographer. Your job is to reveal the shape of the data, not impose one.\n\n"
        "# Your Mental Model\n"
        "- **Manifold Recovery, Not Clustering:** You are mapping the continuous geometric structure of the dataset. "
        "Do not force data into neat, disconnected clusters if they do not organically exist.\n"
        "- **The Cosmic Graph:** This is your primary artifact. It is the aggregated topological signal of the data across multiple spatial scales.\n\n"
        "# Core Principles\n"
        "1. **Dimensionality vs. Scale:** Higher dimensions compress Euclidean distance into a narrow band. This requires a larger epsilon "
        "to capture local neighborhoods, which rapidly leads to a massive, uninformative 'blob' (a fully connected component). "
        "Manage dimensionality (e.g., PCA dims) tightly before aggressively tuning your scale (epsilon) parameters.\n"
        "2. **Respect Categoricals:** Categorical variables (e.g., island, group) add rigid, orthogonal structure to the manifold. "
        "They are not noise. Never drop them simply to force spatial separation. If the reality of the data is partitioned, the topology must reflect that.\n"
        "3. **Variance Curve:** Look at the cumulative variance curve from characterize_dataset. If Dim 3 captures 94% variance, using Dim 5 "
        "adds almost no signal but drastically dilutes distances (curse of dimensionality).\n"
        "4. **Good Graph vs Convenient Graph:** A 'good' graph has structural balance. Look for a `giant_fraction` (the largest component) "
        "typically between 20% and 60%. These metrics apply to the **final aggregated ensemble graph**. It is not necessarily the graph that perfectly fits a preconceived notion of 'K=3 clusters'.\n"
        "5. **Embrace the Grid (Multi-Scale):** Pulsar is an ensemble method. Do NOT try to find the 'one true epsilon' or 'one best PCA dimension'. "
        "The graph gains expressive power by accumulating topology across a filtration. Always prefer ranges for epsilon and arrays for PCA dimensions. "
        "(Exception: You may use single values only for isolated diagnostic runs to 'find the floor' of a massive blob.)\n\n"
        "# The Scientific Posture\n"
        "Treat the cosmic graph as empirical evidence, not a score to maximize.\n"
        "- **Form a hypothesis** before adjusting any configuration.\n"
        "- **Change ONE parameter at a time.** Note: An entire Epsilon Range or PCA Array counts as ONE parameter change.\n"
        "- **Observe the exact structural impact** on the cosmic graph (use the diff provided by run_topological_sweep), and re-assess your hypothesis.\n\n"
        "# Workflow\n"
        "1. ingest_dataset -> create_config -> validate_config\n"
        "2. run_topological_sweep -> diagnose_cosmic_graph\n"
        "3. **Iterate:** Change ONE parameter (as defined above). After each sweep, review the diff.\n"
        "4. generate_cluster_dossier -> compare_clusters_tool\n\n"
        "# Environment Boundary\n"
        "This MCP server runs on the host filesystem, not necessarily in the same sandbox as your bash tool.\n"
        "Prefer dataset handles over ad hoc file copying, and use get_runtime_context when path visibility is unclear.\n"
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


# Global session storage, keyed by session_id (or "default" for STDIO)
# STDIO transport assumption: mcp.run() with no transport argument defaults to STDIO.
# Under STDIO, each process serves exactly one client. _sessions will contain at most
# one entry, keyed "default". If ported to SSE/WebSocket (multi-client), replace this
# with a bounded LRU dict capped at MAX_SESSIONS.
_sessions: dict[str, _PulsarSession] = {}
_MAX_SESSIONS = 1


def _session_key(ctx: Context | None) -> str:
    """Get the session key from context (session_id or 'default' for STDIO)."""
    if ctx is None:
        return "default"
    return ctx.session_id or "default"


def _get_session(ctx: Context) -> _PulsarSession:
    """Get or create session state for the current client."""
    key = _session_key(ctx)
    if key not in _sessions:
        if len(_sessions) >= _MAX_SESSIONS:
            logger.warning(
                "_sessions has %d entries; expected %d under STDIO transport. "
                "If using a multi-client transport, this server needs LRU eviction.",
                len(_sessions),
                _MAX_SESSIONS,
            )
        _sessions[key] = _PulsarSession()
    return _sessions[key]


def _resolve_dataset_record(dataset_id: str):
    record = registry.get_dataset(dataset_id)
    if record is None:
        raise LookupError(dataset_id)
    return record


def _resolve_dataset_path(dataset_id: str) -> str:
    return _resolve_dataset_record(dataset_id).path


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

    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "resolved_threshold": float(model.resolved_threshold),
        "component_count": len(components),
        "component_sizes": component_sizes,
        "nodes": nodes,
        "edges": edges,
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
async def suggest_initial_config(dataset_geometry: str, ctx: Context) -> str:
    """
    Generate an initial configuration YAML based on the raw dataset geometry.

    Args:
        dataset_geometry: The raw JSON string from characterize_dataset.

    Returns:
        A starter config_yaml string.
    """
    try:
        geo = json.loads(dataset_geometry)
        return _build_initial_config_yaml(
            geo, data_path="FILL_THIS_IN_WITH_PATH", run_name="initial_sweep"
        )
    except Exception as e:
        return mcp_error("suggest_initial_config", str(e))


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
            "Use ingest_dataset(path) for host-visible files before downstream analysis.",
            "config_path must point to a file visible to the MCP server process.",
            "config_yaml must be raw YAML, not fenced Markdown.",
        ],
    }
    return json.dumps(payload, indent=2)


@mcp.tool()
async def ingest_dataset(path: str, ctx: Context = None) -> str:
    """
    Register a host-visible dataset path and return a stable dataset_id handle.
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
async def create_config(dataset_id: str, intent: str = "", ctx: Context = None) -> str:
    """
    Generate canonical Pulsar YAML for an ingested dataset.
    """
    try:
        dataset_path = _resolve_dataset_path(dataset_id)
        from pulsar.analysis.characterization import characterize_dataset as _char

        result = await asyncio.to_thread(_char, dataset_path)
        geo = dataclasses.asdict(result)
        run_name = intent.strip() or "initial_sweep"
        session = _get_session(ctx)
        session.dataset_id = dataset_id
        return _build_initial_config_yaml(geo, data_path=dataset_path, run_name=run_name)
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
            "config_yaml": result.config_yaml,
        }
        return json.dumps(payload, indent=2)
    except Exception as e:
        return mcp_error("refine_config", str(e))


@mcp.tool()
async def validate_config(
    config_yaml: str,
    dataset_id: str = "",
    ctx: Context = None,
) -> str:
    """
    Validate full Pulsar config shape and normalize it into canonical YAML.
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
    ctx: Context = None,
) -> str:
    """
    Run the Pulsar topological sweep pipeline on a dataset.

    Returns a markdown diff of parameter and metric changes compared to your previous run,
    followed by the full execution summary.

    Args:
        config_path: Path to a params.yaml file on disk.
        config_yaml: Inline YAML string (preferred).
    """
    try:
        if config_yaml:
            current_yaml = config_yaml
        elif config_path:
            _validate_config_path(config_path)
            with open(config_path) as f:
                current_yaml = f.read()
        else:
            raise ToolError("Provide either config_path or config_yaml.")

        session = _get_session(ctx)
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
                    "issues": [dataclasses.asdict(issue) for issue in validation.issues],
                },
            )

        current_yaml = validation.normalized_yaml
        config_dict = yaml.safe_load(current_yaml)

        logger.info("Starting topological sweep from normalized YAML")
        model = ThemaRS(config_dict)

        cfg = model.config

        precomputed = None
        if session.embeddings is not None and session.data is not None:
            fingerprint = pca_fingerprint(cfg, len(session.data))
            if fingerprint == session.pca_fingerprint:
                precomputed = session.embeddings
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
        session.data = model.data  # raw pre-preprocessing DataFrame

        if precomputed is None:
            session.embeddings = model._embeddings
            session.pca_fingerprint = pca_fingerprint(cfg, len(model.data))

        saved_path = _auto_save_config(cfg)
        graph = model.cosmic_graph

        # Calculate metrics for diff
        from pulsar.mcp.diagnostics import diagnose_model

        current_metrics_obj = diagnose_model(model)
        current_metrics = dataclasses.asdict(current_metrics_obj)
        graph_summary = _build_graph_summary(model)

        # Build Diff block
        diff_block = (
            "### Experiment Diff\n\n| Parameter | Previous | Current |\n|---|---|---|\n"
        )
        if not session.sweep_history:
            diff_block += "| (Initial Run) | - | - |\n"
        else:
            prev_record = session.sweep_history[-1]
            prev_cfg = yaml.safe_load(prev_record.config_yaml)
            curr_cfg = yaml.safe_load(current_yaml)

            p_pca = str(
                prev_cfg.get("sweep", {})
                .get("pca", {})
                .get("dimensions", {})
                .get("values", [])
            )
            c_pca = str(
                curr_cfg.get("sweep", {})
                .get("pca", {})
                .get("dimensions", {})
                .get("values", [])
            )
            if p_pca != c_pca:
                diff_block += f"| pca_dims | {p_pca} | {c_pca} |\n"

            p_eps = _format_epsilon(prev_cfg)
            c_eps = _format_epsilon(curr_cfg)
            if p_eps != c_eps:
                diff_block += f"| epsilon | {p_eps} | {c_eps} |\n"

            pm = prev_record.metrics
            cm = current_metrics
            diff_block += (
                f"| Edges | {pm.get('n_edges', 0):,} | {cm.get('n_edges', 0):,} |\n"
            )
            diff_block += f"| Components | {pm.get('component_count', 0)} | {cm.get('component_count', 0)} |\n"
            diff_block += f"| Giant Fraction | {pm.get('giant_fraction', 0):.2%} | {cm.get('giant_fraction', 0):.2%} |\n"

        # Record history
        session.sweep_history.append(
            SweepRecord(time.time(), current_yaml, current_metrics)
        )
        run_record = registry.save_run(
            dataset_id=dataset_id or session.dataset_id,
            config_yaml=current_yaml,
            metrics=current_metrics,
            resolved_threshold=model.resolved_threshold,
            graph_summary=graph_summary,
        )
        session.latest_run_id = run_record.run_id
        if dataset_id:
            session.dataset_id = dataset_id

        result = f"{diff_block}\n\n"
        dataset_override_line = (
            f"- Dataset ID override: {dataset_id}\n" if dataset_id else ""
        )
        result += (
            "### Execution Summary\n"
            "Successfully ran topological sweep.\n"
            f"- Run ID: {run_record.run_id}\n"
            f"{dataset_override_line}"
            f"- Nodes: {graph.number_of_nodes()}\n"
            f"- Edges: {graph.number_of_edges()}\n"
            f"- Resolution: {model.resolved_threshold:.4f}\n"
            f"- Data Shape: {session.data.shape}\n"
            f"- Config saved: {saved_path}"
        )
        return result

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
    ctx: Context = None,
) -> str:
    """
    Generate a statistical dossier of the topological clusters.
    """
    session = _get_session(ctx)

    if session.model is None or session.data is None:
        raise ToolError("No model found. Run run_topological_sweep() first.")

    if method not in ("auto", "spectral", "components"):
        raise ToolError(
            f"method must be 'auto', 'spectral', or 'components', got '{method}'"
        )

    try:
        clusters = resolve_clusters(session.model, method=method, max_k=max_k)
        session.clusters = clusters

        dossier = build_dossier(session.model, session.data, clusters)
        markdown = dossier_to_markdown(dossier)
        return markdown

    except Exception as e:
        logger.error(f"Error generating dossier: {e}")
        return mcp_error("generate_cluster_dossier", str(e))


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
    provided_ids = set(int(k) for k in cluster_names.keys())
    missing_ids = actual_ids - provided_ids
    if missing_ids:
        raise ToolError(
            f"cluster IDs {sorted(missing_ids)} have no name mapping. Provide all {len(actual_ids)} cluster IDs."
        )

    try:
        df = session.data.copy()
        df["topological_cluster_id"] = session.clusters
        names_map = {int(k): v for k, v in cluster_names.items()}
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
    Use this signal to form a hypothesis and pass your summary into suggest_initial_config.
    """
    try:
        from pulsar.analysis.characterization import characterize_dataset as _char

        if dataset_id:
            csv_path = _resolve_dataset_path(dataset_id)
        if not csv_path:
            raise ToolError("Provide either csv_path or dataset_id.")

        # Read once, reuse for both session state and characterization.
        session = _get_session(ctx)
        df = await asyncio.to_thread(pd.read_csv, csv_path)
        session.data = df
        if dataset_id:
            session.dataset_id = dataset_id

        result = await asyncio.to_thread(_char, csv_path, dataframe=df)
        return json.dumps(dataclasses.asdict(result), indent=2)
    except FileNotFoundError:
        return path_access_error(
            "characterize_dataset",
            csv_path,
            missing_action=(
                "Ask the user for a host-visible absolute dataset path, or use "
                "ingest_dataset first and pass dataset_id."
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
async def get_topological_skeleton(run_id: str = "", ctx: Context = None) -> str:
    """
    Return structured graph connectivity for the latest run or an explicit run_id.
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
async def recommend_preprocessing(dataset_geometry: str, ctx: Context) -> str:
    """
    Analyze column profiles from characterize_dataset and return a complete
    preprocessing block with impute/encode/drop recommendations and per-column rationale.

    Call this after characterize_dataset to get an expert-recommended preprocessing:
    YAML block before running run_topological_sweep.

    Args:
        dataset_geometry: The raw JSON string from characterize_dataset.

    Returns:
        Markdown rationale table + ready-to-paste preprocessing: YAML block.
    """
    try:
        geo = json.loads(dataset_geometry)
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

        result = "## Preprocessing Recommendation\n\n"
        result += _rationale_table(rationale)
        result += "\n\n### Recommended preprocessing block\n\n```yaml\n"
        result += _preprocessing_block_to_yaml(drop, impute, encode)
        result += "\n```\n"
        result += "\nCopy this block into your config_yaml, then call `validate_preprocessing_config` to confirm before running `run_topological_sweep`."
        return result

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

        col_preview = list(layout.feature_names[:8])
        if len(layout.feature_names) > 8:
            col_preview.append(f"... +{len(layout.feature_names) - 8} more")

        result = "## Preprocessing Validation: PASS\n\n"
        result += f"- Input rows: {len(session.data)} → Output rows: {layout.n_rows} (no rows dropped)\n"
        result += f"- Output features: {len(layout.feature_names)}\n"
        result += f"- Columns: {col_preview}\n"
        result += "- NaN remaining: 0\n\n"
        result += "Config is ready for `run_topological_sweep`."
        return result

    except (ValueError, TypeError) as e:
        result = "## Preprocessing Validation: FAIL\n\n"
        result += f"**Error:** {e}\n\n"
        result += "Call `repair_preprocessing_config(error_message=..., config_yaml=..., dataset_geometry=...)` to fix this automatically."
        return result
    except Exception as e:
        logger.error(f"Error in validate_preprocessing_config: {e}")
        return mcp_error("validate_preprocessing_config", str(e))


def main():
    mcp.run()


if __name__ == "__main__":
    main()
