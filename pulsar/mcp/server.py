"""
FastMCP Server for Pulsar.

Exposes "Thick Tools" for topological data analysis and interpretation.
"""

import dataclasses
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
        model.fit()

        session.model = model
        session.data = model.data  # Use public property

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


def main():
    """Entry point for the pulsar-mcp CLI."""
    mcp.run()


if __name__ == "__main__":
    main()
