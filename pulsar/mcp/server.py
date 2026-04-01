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
import yaml
from fastmcp import FastMCP, Context

from pulsar.config import config_to_yaml
from pulsar.pipeline import ThemaRS
from pulsar.mcp.interpreter import resolve_clusters, build_dossier, dossier_to_markdown

logger = logging.getLogger(__name__)

# Initialize FastMCP
mcp = FastMCP(
    "Pulsar",
    instructions=(
        "Pulsar is a geometric deep learning toolkit for discovering structure "
        "in complex datasets. Follow this workflow:\n"
        "1. characterize_dataset(csv_path) → get column_profiles and a YAML template\n"
        "2. Review column_profiles to decide preprocessing:\n"
        "   - Non-numeric columns: decide drop vs encode based on column name, "
        "cardinality, and sample values in the profiles\n"
        "   - Edit the suggested_params_yaml: add columns to drop_columns, "
        "adjust impute methods, or tweak PCA/epsilon as needed\n"
        "3. run_topological_sweep(config_yaml=<your edited YAML>) → fit pipeline\n"
        "4. diagnose_cosmic_graph() → if quality != 'good', retry with suggested_config_yaml\n"
        "5. generate_cluster_dossier() → get cluster statistics\n"
        "6. export_labeled_data(cluster_names, output_path) → save results\n"
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
    embeddings: Optional[list] = None                # cached PCA output from last fit
    pca_fingerprint: Optional[str] = None            # SHA256 of (data_path, dims, seeds, n_rows)
    sweep_history: list = dataclasses.field(default_factory=list)  # list[SweepHistoryEntry]


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
            fingerprint = _pca_fingerprint(cfg, len(session.data))
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
        session.data = model.data

        # Update session cache with fresh embeddings if not reused
        if precomputed is None:
            session.embeddings = model._embeddings
            session.pca_fingerprint = _pca_fingerprint(cfg, len(model.data))

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
async def generate_cluster_dossier(ctx: Context) -> str:
    """
    Generate a statistical dossier of the topological clusters.

    Call this AFTER diagnose_cosmic_graph() returns quality == "good".
    Finds stable clusters in the cosmic graph and produces a Markdown report
    with per-cluster statistics: size, defining features, relative shifts
    from the global mean, and homogeneity scores.

    NEXT STEP: Present the dossier to the user, then call export_labeled_data()
    with semantic cluster names to save the labeled dataset.

    Returns:
        Markdown-formatted cluster dossier.
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
    Probe a dataset's geometry and return column metadata + a YAML config template.

    ALWAYS call this FIRST before run_topological_sweep(). Returns JSON with:

    - profile.column_profiles: per-column metadata. For each column you get:
      name, dtype, is_numeric, n_unique, missing_pct, sample_values, and
      top_values (with frequency counts for non-numeric columns).
      Use this to decide which columns to drop, encode, or impute.
    - recommendations.suggested_params_yaml: a YAML template with geometry-based
      PCA dims, epsilon range, seeds, and an impute block for NaN columns.
      drop_columns starts EMPTY — you must populate it based on your analysis
      of column_profiles. Non-numeric columns that are not encoded must be added
      to drop_columns (the pipeline requires float64 input).
    - recommendations.warnings: factual observations about the dataset.

    NEXT STEPS:
    1. Review column_profiles. For each non-numeric column, decide: drop or encode.
    2. Edit the suggested_params_yaml: add columns to drop_columns as needed.
    3. Pass the edited YAML to run_topological_sweep(config_yaml=<your YAML>).
       Do NOT write it to a file — the sweep tool auto-saves for reproducibility.

    Args:
        csv_path: Absolute path to a CSV file (must have >=2 numeric columns).

    Returns:
        JSON with profile (column_profiles), recommendations, and suggested_params_yaml.
    """
    try:
        from pulsar.characterization import characterize_dataset as _char

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
    Diagnose the fitted cosmic graph and get a corrected config if needed.

    Call this AFTER run_topological_sweep(). Returns JSON with:
    - quality: "good", "hairball", "singletons", "fragmented", or "sparse_connected"
    - suggested_config_yaml: a complete corrected YAML config with adjusted epsilon
      (and possibly adjusted PCA dims). If quality != "good", pass this string
      DIRECTLY to run_topological_sweep(config_yaml=...) to retry.
    - diagnosis: human-readable explanation of the graph's structure.
    - suggestions: list of corrective actions.

    HISTORY-AWARE RETRY: This tool tracks the quality and epsilon range of every
    prior call within this session. On repeated retries it narrows the epsilon
    search via binary search between the last known "too sparse" and "too dense"
    bounds. If oscillation is detected (2+ direction changes), it will also
    reduce PCA dimensions in the suggested config to escape the curse of
    dimensionality.

    RETRY WORKFLOW: If quality != "good", take the "suggested_config_yaml" value
    and call run_topological_sweep(config_yaml=<that string>). Repeat until
    quality == "good" or you've tried 3 times.

    When quality == "good", proceed to generate_cluster_dossier().

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
        session.sweep_history.append(SweepHistoryEntry(
            quality=result.quality,
            epsilon_min=min(current_epsilons) if current_epsilons else 0.5,
            epsilon_max=max(current_epsilons) if current_epsilons else 1.5,
            pca_dims=list(session.model.config.pca.dimensions),
        ))

        return json.dumps(dataclasses.asdict(result), indent=2)
    except RuntimeError as e:
        error_msg = f"Error: {e}"
        logger.error(error_msg)
        return error_msg
    except (ValueError, RuntimeError) as e:
        error_msg = f"Error diagnosing graph: {e}"
        logger.error(error_msg)
        return error_msg


def main():
    """Entry point for the pulsar-mcp CLI."""
    mcp.run()


if __name__ == "__main__":
    main()
