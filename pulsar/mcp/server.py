"""
FastMCP Server for Pulsar.

Exposes "Thick Tools" for topological data analysis and interpretation.
"""

import asyncio
import dataclasses
from dataclasses import dataclass
import json
import logging
import os
import time
from typing import Dict, Optional

import pandas as pd
import yaml
from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError

from pulsar.config import config_to_yaml
from pulsar.runtime.fingerprint import pca_fingerprint
from pulsar.pipeline import ThemaRS
from pulsar.mcp.interpreter import (
    resolve_clusters,
    build_dossier,
    dossier_to_markdown,
    compare_clusters,
    comparison_to_markdown,
)
from pulsar.mcp.errors import mcp_error

logger = logging.getLogger(__name__)

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
        "typically between 20% and 60%, accompanied by multiple smaller structural components. It is not necessarily the graph that perfectly fits a preconceived notion of 'K=3 clusters'.\n\n"
        "# The Scientific Posture\n"
        "Treat the cosmic graph as empirical evidence, not a score to maximize.\n"
        "- **Form a hypothesis** before adjusting any configuration.\n"
        "- **Change ONE parameter at a time** (e.g., epsilon bounds OR PCA dims OR categorical encoding).\n"
        "- **Observe the exact structural impact** on the cosmic graph (use the diff provided by run_topological_sweep), and re-assess your hypothesis.\n\n"
        "# Workflow\n"
        "1. characterize_dataset -> suggest_initial_config -> explain_suggestion\n"
        "2. run_topological_sweep -> diagnose_cosmic_graph\n"
        "3. **Iterate:** Change ONE parameter. After each sweep, review the diff. Call `get_experiment_history` when reasoning across multiple iterations.\n"
        "4. **Blob Recovery:** If run 1 produces a massive blob (`giant_fraction` > 80%), call `explain_suggestion` to understand the initial reasoning *before* blindly deviating. "
        "Usually, you must decrease epsilon OR reduce PCA dims to break the blob.\n"
        "5. **Shatter Recovery:** If the graph is fragmented (`component_count` > 20 and `giant_fraction` < 5%), your epsilon is too small or threshold too aggressive. "
        "Increase epsilon modestly (10-20%) or reduce the threshold before any other change. After calling `explain_suggestion`, return to step 2 (`run_topological_sweep`) with ONE adjusted parameter.\n"
        "6. generate_cluster_dossier -> compare_clusters\n"
    ),
)


@dataclass
class SweepRecord:
    timestamp: float
    config_yaml: str
    metrics: dict


# Session state: stores (model, data, clusters) per session
@dataclasses.dataclass
class _PulsarSession:
    """Session state for a single MCP client."""

    model: Optional[ThemaRS] = None
    data: Optional[pd.DataFrame] = None
    clusters: Optional[pd.Series] = None
    embeddings: Optional[list] = None  # cached PCA output from last fit
    pca_fingerprint: Optional[str] = None  # SHA256 of (data_path, dims, seeds, n_rows)
    sweep_history: list[SweepRecord] = dataclasses.field(default_factory=list)


# Global session storage, keyed by session_id (or "default" for STDIO)
# STDIO transport assumption: mcp.run() with no transport argument defaults to STDIO.
# Under STDIO, each process serves exactly one client. _sessions will contain at most
# one entry, keyed "default". If ported to SSE/WebSocket (multi-client), replace this
# with a bounded LRU dict capped at MAX_SESSIONS.
_sessions: Dict[str, _PulsarSession] = {}
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
    Generate an initial configuration YAML based on your interpretation of the raw dataset geometry.

    You MUST call characterize_dataset first, analyze the variance curve and column missingness/cardinality,
    and then formulate a JSON-encoded summary of the geometry to pass into this tool.

    Args:
        dataset_geometry: A JSON string containing your summary of the geometry (e.g., {"N": 344, "pca_knee_dim": 3, "knn_mean": 0.6}).

    Returns:
        A starter config_yaml string.
    """
    try:
        geo = json.loads(dataset_geometry)
        pca_dims = geo.get("pca_knee_dim", 3)
        if isinstance(pca_dims, int):
            pca_dims = [pca_dims]

        eps_min = geo.get("knn_mean", 0.5) * 0.8
        eps_max = geo.get("knn_mean", 0.5) * 1.5

        yaml_str = f"""run:
  name: initial_sweep
  data: "FILL_THIS_IN_WITH_PATH"
preprocessing:
  drop_columns: []
  encode: {{}}
  impute: {{}}
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
        return yaml_str
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
        knn_mean = geo.get("knn_k5_mean")

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
            if isinstance(n_samples, int):
                pts_per_dim = n_samples / max(pca_dims)
                explanation += f" With N={n_samples}, this maintains ~{pts_per_dim:.1f} points per dimension, ensuring sufficient manifold density.\n"
            else:
                explanation += "\n"
        else:
            explanation += f"- **PCA Dimensions {pca_dims}**: Chosen based on the variance curve elbow (values not provided in geo summary).\n"

        # 2. Epsilon Reasoning
        if knn_mean:
            eps_node = (
                config_dict.get("sweep", {}).get("ball_mapper", {}).get("epsilon", {})
            )
            if "range" in eps_node:
                r = eps_node["range"]
                e_min, e_max = r.get("min", 0), r.get("max", 0)
                explanation += f"- **Epsilon Range {eps_str}**: Anchored at knn_k5_mean={knn_mean:.4f}. The range spans {e_min/knn_mean:.2f}x to {e_max/knn_mean:.2f}x the mean distance, transitioning from local neighborhoods to global structure.\n"
            else:
                explanation += f"- **Epsilon {eps_str}**: Evaluated relative to knn_k5_mean={knn_mean:.4f}.\n"
        else:
            explanation += f"- **Epsilon {eps_str}**: Search window centered around k-NN mean (knn_k5_mean not provided in summary).\n"

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
            f"| {i+1} | {pca} | {eps} | {m.get('n_nodes')} | {m.get('n_edges')} | {m.get('component_count')} | {m.get('giant_fraction', 0):.2%} |"
        )

    return "\n".join(lines)


@mcp.tool()
async def run_topological_sweep(
    config_path: str = "",
    config_yaml: str = "",
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
    if config_yaml:
        config_dict = yaml.safe_load(config_yaml)
        if not isinstance(config_dict, dict):
            raise ToolError("config_yaml must be a valid YAML mapping")
        current_yaml = config_yaml
    elif config_path:
        _validate_config_path(config_path)
        with open(config_path) as f:
            current_yaml = f.read()
    else:
        raise ToolError("Provide either config_path or config_yaml.")

    session = _get_session(ctx)

    try:
        if config_yaml:
            logger.info("Starting topological sweep from inline YAML")
            model = ThemaRS(config_dict)
        else:
            logger.info("Starting topological sweep for: %s", config_path)
            model = ThemaRS(config_path)

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
        session.data = model.preprocessed_data

        if precomputed is None:
            session.embeddings = model._embeddings
            session.pca_fingerprint = pca_fingerprint(cfg, len(model.data))

        saved_path = _auto_save_config(cfg)
        graph = model.cosmic_graph

        # Calculate metrics for diff
        from pulsar.mcp.diagnostics import diagnose_model

        current_metrics_obj = diagnose_model(model)
        current_metrics = dataclasses.asdict(current_metrics_obj)

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

        result = f"{diff_block}\n\n"
        result += (
            "### Execution Summary\n"
            "Successfully ran topological sweep.\n"
            f"- Nodes: {graph.number_of_nodes()}\n"
            f"- Edges: {graph.number_of_edges()}\n"
            f"- Resolution: {model.resolved_threshold:.4f}\n"
            f"- Data Shape: {session.data.shape}\n"
            f"- Config saved: {saved_path}"
        )
        return result

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
    cluster_names: Dict[int, str], output_path: str, ctx: Context
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
async def characterize_dataset(csv_path: str, ctx: Context) -> str:
    """
    Probes dataset geometry to return raw facts (N, features, variance curve, k-NN mean).
    Use this signal to form a hypothesis and pass your summary into suggest_initial_config.
    """
    try:
        from pulsar.analysis.characterization import characterize_dataset as _char

        result = await asyncio.to_thread(_char, csv_path)
        return json.dumps(dataclasses.asdict(result), indent=2)
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


def main():
    mcp.run()


if __name__ == "__main__":
    main()
