"""
FastMCP Server for Pulsar.

Exposes "Thick Tools" for topological data analysis and interpretation.
"""

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
from typing import Dict, Optional

import pandas as pd
from fastmcp import FastMCP, Context

from pulsar.pipeline import ThemaRS
from pulsar.mcp.interpreter import resolve_clusters, build_dossier, dossier_to_markdown

logger = logging.getLogger(__name__)

# Initialize FastMCP
mcp = FastMCP("Pulsar")


# Session state: stores (model, data, clusters) per session
@dataclasses.dataclass
class _PulsarSession:
    """Session state for a single MCP client."""
    model: Optional[ThemaRS] = None
    data: Optional[pd.DataFrame] = None
    clusters: Optional[pd.Series] = None
    embeddings: Optional[list] = None                # cached PCA output from last fit
    pca_fingerprint: Optional[str] = None            # SHA256 of (data_path, dims, seeds, n_rows)


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


def _pca_fingerprint(cfg, n_rows: int) -> str:
    """
    Compute a fingerprint for PCA configuration and data shape.

    Used to detect when cached PCA embeddings can be reused (same data + PCA params).
    Includes data_path, n_rows, and mtime to invalidate cache on data changes.
    """
    # Use 0 if data path doesn't exist (unlikely at this stage)
    mtime = os.path.getmtime(cfg.data) if os.path.exists(cfg.data) else 0
    payload = json.dumps({
        "data_path": cfg.data,
        "mtime": mtime,
        "dimensions": list(cfg.pca.dimensions),
        "seeds": list(cfg.pca.seeds),
        "n_rows": n_rows,
    })
    return hashlib.sha256(payload.encode()).hexdigest()


@mcp.tool()
async def run_topological_sweep(config_path: str, ctx: Context) -> str:
    """
    Runs the ThemaRS topological sweep pipeline on a dataset.

    Args:
        config_path: Path to the params.yaml configuration file.
        ctx: FastMCP context (auto-injected).

    Returns:
        A summary of the generated Cosmic Graph.
    """
    session = _get_session(ctx)

    try:
        _validate_config_path(config_path)
        logger.info("Starting topological sweep for: %s", config_path)

        model = ThemaRS(config_path)
        cfg = model.config

        # Check if cached PCA embeddings can be reused (same data + PCA params)
        precomputed = None
        if session.embeddings is not None and session.data is not None:
            fingerprint = _pca_fingerprint(cfg, len(session.data))
            if fingerprint == session.pca_fingerprint:
                precomputed = session.embeddings
                logger.info("Reusing cached PCA embeddings (fingerprint match)")

        # Build progress callback that fires ctx.report_progress() from the fit thread.
        # ctx.report_progress() is async; use run_coroutine_threadsafe() to schedule it
        # from the threadpool without blocking the fit() call.
        loop = asyncio.get_running_loop()

        def progress_callback(stage: str, fraction: float) -> None:
            asyncio.run_coroutine_threadsafe(
                ctx.report_progress(progress=fraction, total=1.0, message=stage),
                loop,
            )

        # Run blocking fit() in a threadpool — avoids starving the event loop.
        await asyncio.to_thread(
            model.fit,
            _precomputed_embeddings=precomputed,
            progress_callback=progress_callback,
        )

        session.model = model
        session.data = model.data  # Use public property

        # Update session cache with fresh embeddings if not reused
        if precomputed is None:
            session.embeddings = model._embeddings
            session.pca_fingerprint = _pca_fingerprint(cfg, len(model.data))

        graph = model.cosmic_graph
        result = (
            "Successfully ran topological sweep.\n"
            f"- Nodes: {graph.number_of_nodes()}\n"
            f"- Edges: {graph.number_of_edges()}\n"
            f"- Resolution: {model.resolved_threshold:.4f}\n"
            f"- Data Shape: {session.data.shape}"
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
async def generate_cluster_dossier(ctx: Context) -> str:
    """
    Analyzes the topological graph, finds stable clusters, and generates
    a statistical dossier for semantic interpretation.

    Args:
        ctx: FastMCP context (auto-injected).

    Returns:
        A Markdown-formatted dossier describing the relative shifts,
        homogeneity, and defining features of each cluster.
    """
    session = _get_session(ctx)

    if session.model is None or session.data is None:
        return "Error: No model found. Run run_topological_sweep() first."

    try:
        logger.info("Generating cluster dossier")

        # 1. Resolve Clusters
        clusters = resolve_clusters(session.model)
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
async def export_labeled_data(
    cluster_names: Dict[int, str], output_path: str, ctx: Context
) -> str:
    """
    Assigns human-readable names to clusters and exports the labeled dataset to CSV.

    Args:
        cluster_names: Dict mapping cluster IDs (ints) to semantic names (strings).
        output_path: Path where the labeled CSV should be saved.
        ctx: FastMCP context (auto-injected).

    Returns:
        Confirmation message or error description.
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
        df['topological_cluster_id'] = session.clusters

        # Map IDs to names (ensure keys are ints, as LLMs sometimes send strings)
        names_map = {int(k): v for k, v in cluster_names.items()}
        df['topological_cluster_name'] = df['topological_cluster_id'].map(names_map)

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
    Probes dataset geometry before fitting to suggest PCA dims and epsilon range.

    Returns JSON with profile (measurements), recommendations (derived outputs),
    and a suggested params.yaml template for use with run_topological_sweep().

    Args:
        csv_path: Path to CSV file (must have >=2 numeric columns).
        ctx: FastMCP context (auto-injected).

    Returns:
        JSON string with CharacterizationResult or error message.
    """
    try:
        from pulsar.characterization import characterize_dataset as _char

        result = _char(csv_path)
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
    Classifies the fitted cosmic graph and suggests epsilon corrections.

    Analyzes graph structure (density, connectivity, components) to classify
    quality as good/hairball/singletons/fragmented/sparse_connected and returns
    concrete suggested epsilon range for parameter retry loops.

    Args:
        ctx: FastMCP context (auto-injected).

    Returns:
        JSON string with DiagnosisResult or error message.
    """
    session = _get_session(ctx)

    if session.model is None:
        return "Error: No model found. Run run_topological_sweep() first."

    try:
        from pulsar.mcp.diagnostics import diagnose_model

        result = diagnose_model(session.model)
        return json.dumps(dataclasses.asdict(result), indent=2)
    except RuntimeError as e:
        error_msg = f"Error: {e}"
        logger.error(error_msg)
        return error_msg
    except (ValueError, Exception) as e:
        error_msg = f"Error diagnosing graph: {e}"
        logger.error(error_msg)
        return error_msg


def main():
    """Entry point for the pulsar-mcp CLI."""
    mcp.run()


if __name__ == "__main__":
    main()
