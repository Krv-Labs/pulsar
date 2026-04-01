"""
FastMCP Server for Pulsar.

Exposes "Thick Tools" for topological data analysis and interpretation.
"""

import asyncio
import dataclasses
import json
import logging
import os
from typing import Dict, Optional

import pandas as pd
import yaml
from fastmcp import FastMCP, Context

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

logger = logging.getLogger(__name__)

# Initialize FastMCP
mcp = FastMCP(
    "Pulsar",
    instructions=(
        "Pulsar is a geometric deep learning toolkit for discovering structure "
        "in complex datasets. Follow this workflow:\n"
        "1. characterize_dataset(csv_path) → get column_profiles and a YAML template\n"
        "2. Review column_profiles and the suggested YAML:\n"
        "   - Low-cardinality strings (≤10 unique) are automatically added to "
        "the 'encode' block. This preserves their geometric context.\n"
        "   - High-cardinality or unique strings (like Names/IDs) are added to 'drop_columns'.\n"
        "   - Edit the YAML if you want to encode a column that was dropped, or vice-versa.\n"
        "3. run_topological_sweep(config_yaml=<your edited YAML>) → fit pipeline\n"
        "4. diagnose_cosmic_graph() → check quality and balance.\n"
        "   - Review 'component_sizes'. If one component contains >80% of data, "
        "the graph is a 'blob'.\n"
        "5. generate_cluster_dossier(method, max_k) → get cluster statistics.\n"
        "   - If diagnose showed a 'blob', use method='spectral' to find sub-structure.\n"
        "   - If diagnose showed balanced components, use method='auto' or 'components'.\n"
        "6. compare_clusters(cluster_a, cluster_b) → use this for academic rigor to "
        "prove why two clusters are statistically distinct (p-values, Cohen's d).\n"
        "7. export_labeled_data(cluster_names, output_path) → save results\n"
        "IMPORTANT: Pass YAML strings directly between tools. Never ask the user "
        "to write files manually — the tools handle file I/O automatically."
    ),
)


# Session state: stores (model, data, clusters) per session
@dataclasses.dataclass
class _PulsarSession:
    """Session state for a single MCP client."""

    model: Optional[ThemaRS] = None
    data: Optional[pd.DataFrame] = None
    clusters: Optional[pd.Series] = None
    embeddings: Optional[list] = None  # cached PCA output from last fit
    pca_fingerprint: Optional[str] = None  # SHA256 of (data_path, dims, seeds, n_rows)
    sweep_history: list = dataclasses.field(
        default_factory=list
    )  # list[SweepHistoryEntry]


# Global session storage, keyed by session_id (or "default" for STDIO)
_sessions: Dict[str, _PulsarSession] = {}


def _session_key(ctx: Context) -> str:
    """Get the session key from context (session_id or 'default' for STDIO)."""
    return ctx.session_id or "default"


def _get_session(ctx: Context) -> _PulsarSession:
    """Get or create session state for the current client."""
    key = _session_key(ctx)
    if key not in _sessions:
        _sessions[key] = _PulsarSession()
    return _sessions[key]


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
async def run_topological_sweep(
    config_path: str = "",
    config_yaml: str = "",
    ctx: Context = None,
) -> str:
    """
    Run the Pulsar topological sweep pipeline on a dataset.

    PREFERRED WORKFLOW:
    1. Call characterize_dataset(csv_path) first to get a suggested config.
    2. Extract the "suggested_params_yaml" string from the JSON response.
    3. Pass it directly here as config_yaml — NO file writing needed.
    4. If diagnose_cosmic_graph() later returns quality != "good", take its
       "suggested_config_yaml" and pass it here as config_yaml to retry.

    The resolved config is automatically saved to disk for reproducibility.

    Args:
        config_path: Path to a params.yaml file on disk (use this OR config_yaml).
        config_yaml: Inline YAML string — pass the suggested_params_yaml from
            characterize_dataset() or suggested_config_yaml from
            diagnose_cosmic_graph() directly. No file writing needed.

    Returns:
        Summary with node/edge counts, resolution, data shape, and saved config path.
    """
    session = _get_session(ctx)

    try:
        # Resolve config from either source
        if config_yaml:
            config_dict = yaml.safe_load(config_yaml)
            if not isinstance(config_dict, dict):
                raise ValueError("config_yaml must be a valid YAML mapping")
            logger.info("Starting topological sweep from inline YAML")
            model = ThemaRS(config_dict)
        elif config_path:
            _validate_config_path(config_path)
            logger.info("Starting topological sweep for: %s", config_path)
            model = ThemaRS(config_path)
        else:
            return "Error: Provide either config_path or config_yaml."

        cfg = model.config

        # Check if cached PCA embeddings can be reused (same data + PCA params)
        precomputed = None
        if session.embeddings is not None and session.data is not None:
            fingerprint = pca_fingerprint(cfg, len(session.data))
            if fingerprint == session.pca_fingerprint:
                precomputed = session.embeddings
                logger.info("Reusing cached PCA embeddings (fingerprint match)")

        # Build progress callback that fires ctx.report_progress() from the fit thread.
        loop = asyncio.get_running_loop()

        def progress_callback(stage: str, fraction: float) -> None:
            try:
                asyncio.run_coroutine_threadsafe(
                    ctx.report_progress(progress=fraction, total=1.0, message=stage),
                    loop,
                )
            except RuntimeError:
                pass  # Event loop closed; progress reporting is best-effort

        # Run blocking fit() in a threadpool — avoids starving the event loop.
        await asyncio.to_thread(
            model.fit,
            _precomputed_embeddings=precomputed,
            progress_callback=progress_callback,
        )

        session.model = model
        session.data = model.preprocessed_data  # row-aligned with graph nodes

        # Update session cache with fresh embeddings if not reused
        if precomputed is None:
            session.embeddings = model._embeddings
            session.pca_fingerprint = pca_fingerprint(cfg, len(model.data))

        # Auto-save resolved config for reproducibility
        saved_path = _auto_save_config(cfg)

        graph = model.cosmic_graph
        result = (
            "Successfully ran topological sweep.\n"
            f"- Nodes: {graph.number_of_nodes()}\n"
            f"- Edges: {graph.number_of_edges()}\n"
            f"- Resolution: {model.resolved_threshold:.4f}\n"
            f"- Data Shape: {session.data.shape}\n"
            f"- Config saved: {saved_path}"
        )
        logger.info(
            "Topological sweep complete: %d nodes, %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return result

    except FileNotFoundError as e:
        error_msg = f"Error: {e}"
        logger.error(error_msg)
        return error_msg
    except ValueError as e:
        error_msg = f"Error: {e}"
        logger.error(error_msg)
        return error_msg
    except (RuntimeError, OSError) as e:
        error_msg = f"Error running sweep: {e}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
async def generate_cluster_dossier(
    method: str = "auto",
    max_k: int = 15,
    ctx: Context = None,
) -> str:
    """
    Generate a statistical dossier of the topological clusters.

    Call this AFTER diagnose_cosmic_graph() returns quality == "good".

    CHOOSING A METHOD: Review component_sizes and clustering_notes from the
    diagnose results to decide:
    - method="auto" (default): Uses connected components if balanced, falls
      back to spectral. Works well when the graph has natural separation.
    - method="spectral": Forces spectral clustering with silhouette-optimized k.
      Use this when component_sizes shows one dominant component (e.g. [162, 7, 2, 1, 1]).
    - method="components": Forces connected components. Use when components
      are naturally balanced.

    Args:
        method: Clustering strategy — "auto", "spectral", or "components".
        max_k: Maximum k for spectral clustering search (default 15).
            Increase for large datasets (e.g. max_k=30 for 5000+ nodes).

    Returns:
        Markdown-formatted cluster dossier.
    """
    session = _get_session(ctx)

    if session.model is None or session.data is None:
        return "Error: No model found. Run run_topological_sweep() first."

    if method not in ("auto", "spectral", "components"):
        return (
            f"Error: method must be 'auto', 'spectral', or 'components', got '{method}'"
        )

    try:
        logger.info("Generating cluster dossier (method=%s, max_k=%d)", method, max_k)

        # 1. Resolve Clusters
        clusters = resolve_clusters(session.model, method=method, max_k=max_k)
        session.clusters = clusters

        # 2. Build Statistical Dossier
        dossier = build_dossier(session.model, session.data, clusters)

        # 3. Convert to Markdown
        markdown = dossier_to_markdown(dossier)
        logger.info("Cluster dossier generated: %d clusters", dossier.n_clusters)
        return markdown

    except ValueError as e:
        error_msg = f"Error generating dossier: {e}"
        logger.error(error_msg)
        return error_msg
    except RuntimeError as e:
        error_msg = f"Error generating dossier: {e}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
async def compare_clusters_tool(
    cluster_a: int, cluster_b: int, ctx: Context = None
) -> str:
    """
    Perform pairwise statistical tests between two clusters.

    Returns a Markdown report with Welch's T-test, KS-test, and Cohen's d
    effect size for all numeric features. Features are sorted by the magnitude
    of the difference (Cohen's d).

    Args:
        cluster_a: ID of the first cluster.
        cluster_b: ID of the second cluster.

    Returns:
        Markdown-formatted statistical comparison.
    """
    session = _get_session(ctx)

    if session.data is None or session.clusters is None:
        return "Error: No data or clusters found. Run generate_cluster_dossier() first."

    try:
        logger.info("Comparing clusters %d and %d", cluster_a, cluster_b)

        # 1. Perform Comparison
        results = compare_clusters(session.data, session.clusters, cluster_a, cluster_b)

        if not results:
            return f"Error: No results for comparison between clusters {cluster_a} and {cluster_b}. Ensure clusters have at least 2 points."

        # 2. Convert to Markdown
        markdown = comparison_to_markdown(cluster_a, cluster_b, results)
        return markdown

    except (ValueError, KeyError, IndexError) as e:
        error_msg = f"Error comparing clusters: {e}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
async def export_labeled_data(
    cluster_names: Dict[int, str], output_path: str, ctx: Context
) -> str:
    """
    Assign semantic names to clusters and export the labeled dataset to CSV.

    Call this AFTER generate_cluster_dossier(). Maps cluster IDs to
    human-readable names chosen by the agent based on the dossier analysis.

    Args:
        cluster_names: Dict mapping cluster IDs (from the dossier) to semantic
            names. Example: {0: "High Emission Plants", 1: "Clean Energy"}
        output_path: Absolute path where the labeled CSV will be saved.

    Returns:
        Confirmation message with export path.
    """
    session = _get_session(ctx)

    if session.data is None or session.clusters is None:
        return "Error: No data or clusters found. Run generate_cluster_dossier() first."

    try:
        # Validate cluster_names coverage
        actual_ids = set(session.clusters.unique().tolist())
        provided_ids = set(int(k) for k in cluster_names.keys())
        missing_ids = actual_ids - provided_ids
        if missing_ids:
            missing_str = sorted(missing_ids)
            return (
                f"Error: cluster IDs {missing_str} have no name mapping. "
                f"Provide all {len(actual_ids)} cluster IDs."
            )

        logger.info("Exporting labeled data to: %s", output_path)

        df = session.data.copy()
        df["topological_cluster_id"] = session.clusters

        # Map IDs to names (ensure keys are ints, as LLMs sometimes send strings)
        names_map = {int(k): v for k, v in cluster_names.items()}
        df["topological_cluster_name"] = df["topological_cluster_id"].map(names_map)

        df.to_csv(output_path, index=False)
        logger.info("Successfully exported %d rows to %s", len(df), output_path)
        return f"Successfully exported labeled data to {output_path}"

    except ValueError as e:
        error_msg = f"Error exporting data: {e}"
        logger.error(error_msg)
        return error_msg
    except OSError as e:
        error_msg = f"Error writing to {output_path}: {e}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
async def characterize_dataset(csv_path: str, ctx: Context) -> str:
    """
    Probe a dataset's geometry and return column metadata + a YAML config template.

    ALWAYS call this FIRST before run_topological_sweep(). Returns JSON with:

    - profile.column_profiles: per-column metadata (dtype, uniqueness, missingness).
    - recommendations.suggested_params_yaml: a complete YAML template.
      - Non-numeric columns with <=10 unique values are auto-added to the 'encode' block.
      - High-cardinality strings are auto-added to 'drop_columns'.
      - PCA dims, epsilon range, and imputation methods are suggested based on geometry.
    - recommendations.warnings: factual observations about data quality and dimensionality.

    Args:
        csv_path: Absolute path to a CSV file (must have >=2 numeric columns).

    Returns:
        JSON with profile, recommendations, and suggested_params_yaml.
    """
    try:
        from pulsar.analysis.characterization import characterize_dataset as _char

        result = await asyncio.to_thread(_char, csv_path)
        return json.dumps(dataclasses.asdict(result), indent=2)
    except (ValueError, FileNotFoundError) as e:
        error_msg = f"Error: {e}"
        logger.error(error_msg)
        return error_msg
    except (RuntimeError, OSError) as e:
        error_msg = f"Error analyzing dataset: {e}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
async def diagnose_cosmic_graph(ctx: Context) -> str:
    """
    Diagnose the fitted cosmic graph quality and topological balance.

    Call this AFTER run_topological_sweep(). Returns JSON with:
    - quality: "good", "hairball", "singletons", etc.
    - metrics.component_sizes: sorted list of connected component sizes.
      Use this to identify a 'blob' (one massive component) vs. a balanced graph.
    - clustering_notes: advice on which clustering method to use in the dossier step.
    - suggested_config_yaml: corrected YAML if quality != "good".

    RETRY WORKFLOW: If quality != "good", pass the "suggested_config_yaml" string
    DIRECTLY to run_topological_sweep(config_yaml=...).

    Returns:
        JSON with quality classification, metrics, diagnosis, and corrected YAML.
    """
    session = _get_session(ctx)

    if session.model is None:
        return "Error: No model found. Run run_topological_sweep() first."

    try:
        from pulsar.mcp.diagnostics import diagnose_model, SweepHistoryEntry

        result = diagnose_model(session.model, history=session.sweep_history)

        # Record this attempt for future binary search
        current_epsilons = [bm.eps for bm in session.model.ball_maps]
        session.sweep_history.append(
            SweepHistoryEntry(
                quality=result.quality,
                epsilon_min=min(current_epsilons) if current_epsilons else 0.5,
                epsilon_max=max(current_epsilons) if current_epsilons else 1.5,
                pca_dims=list(session.model.config.pca.dimensions),
            )
        )

        return json.dumps(dataclasses.asdict(result), indent=2)
    except (ValueError, RuntimeError) as e:
        error_msg = f"Error diagnosing graph: {e}"
        logger.error(error_msg)
        return error_msg


def main():
    """Entry point for the pulsar-mcp CLI."""
    mcp.run()


if __name__ == "__main__":
    main()
