from __future__ import annotations

from collections.abc import Callable
import os
from typing import Any

from fastmcp.utilities.tasks import TaskConfig

from pulsar.mcp.tools.ingestion import (
    ingest_dataset,
    sample_longitudinal_entities,
    begin_dataset_upload,
    append_dataset_chunk,
    finalize_dataset_upload,
)
from pulsar.mcp.tools.config import (
    create_config,
    refine_config,
    validate_config,
)
from pulsar.mcp.tools.sweeping import (
    get_sweep_history,
    compare_sweeps,
    run_topological_sweep,
)
from pulsar.mcp.tools.diagnostics import (
    diagnose_cosmic_graph,
    create_graph_artifact,
    get_threshold_stability_curve,
    get_topological_skeleton,
)
from pulsar.mcp.tools.clustering import (
    generate_cluster_dossier,
    get_cluster_profile,
    get_feature_signal,
    get_cluster_signal_matrix,
    compare_clusters,
)
from pulsar.mcp.tools.reporting import (
    export_labeled_data,
    export_html_report,
    probe_columns,
    export_dataset_bundle,
)
from pulsar.mcp.tools.preprocessing import (
    draft_preprocessing_plan,
    edit_preprocessing_plan,
    validate_preprocessing_config,
)
from pulsar.mcp.tools.longitudinal import (
    build_longitudinal_graph,
    diagnose_longitudinal_graph,
    get_trajectory_archetypes,
    get_cross_time_neighbors,
    classify_trajectories,
)
from pulsar.mcp.tools.meta import (
    get_workflow_guide,
    get_runtime_context,
    characterize_dataset,
)

# Chunked-upload trio is a sandboxed-client escape hatch; opt in via env.
_ENABLE_UPLOAD = os.environ.get("PULSAR_MCP_ENABLE_UPLOAD") == "1"


def _tasks_enabled() -> bool:
    return os.environ.get("PULSAR_MCP_ENABLE_TASKS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _sweep_task_config(tasks_available: bool = True) -> bool | TaskConfig:
    if tasks_available and _tasks_enabled():
        return TaskConfig(mode="optional")
    return False


def get_tool_task_config(
    tool_fn: Callable[..., Any], tasks_available: bool = True
) -> bool | TaskConfig:
    if (
        tool_fn is run_topological_sweep
        or getattr(tool_fn, "__name__", "") == "run_topological_sweep"
    ):
        return _sweep_task_config(tasks_available)
    return False


ALL_TOOLS_LIST = [
    # Ingestion
    ingest_dataset,
    sample_longitudinal_entities,
    *(
        [begin_dataset_upload, append_dataset_chunk, finalize_dataset_upload]
        if _ENABLE_UPLOAD
        else []
    ),
    # Config
    create_config,
    refine_config,
    validate_config,
    # Sweeping
    get_sweep_history,
    compare_sweeps,
    run_topological_sweep,
    # Diagnostics
    diagnose_cosmic_graph,
    create_graph_artifact,
    get_threshold_stability_curve,
    get_topological_skeleton,
    # Clustering
    generate_cluster_dossier,
    get_cluster_profile,
    get_feature_signal,
    get_cluster_signal_matrix,
    compare_clusters,
    # Reporting
    export_labeled_data,
    export_html_report,
    probe_columns,
    export_dataset_bundle,
    # Preprocessing
    draft_preprocessing_plan,
    edit_preprocessing_plan,
    validate_preprocessing_config,
    # Longitudinal
    build_longitudinal_graph,
    diagnose_longitudinal_graph,
    get_trajectory_archetypes,
    get_cross_time_neighbors,
    classify_trajectories,
    # Meta
    get_workflow_guide,
    get_runtime_context,
    characterize_dataset,
]

__all__ = [
    "ALL_TOOLS_LIST",
    "_tasks_enabled",
    "_sweep_task_config",
    "get_tool_task_config",
    "ingest_dataset",
    "sample_longitudinal_entities",
    "begin_dataset_upload",
    "append_dataset_chunk",
    "finalize_dataset_upload",
    "create_config",
    "refine_config",
    "validate_config",
    "get_sweep_history",
    "compare_sweeps",
    "run_topological_sweep",
    "diagnose_cosmic_graph",
    "create_graph_artifact",
    "get_threshold_stability_curve",
    "get_topological_skeleton",
    "generate_cluster_dossier",
    "get_cluster_profile",
    "get_feature_signal",
    "get_cluster_signal_matrix",
    "compare_clusters",
    "export_labeled_data",
    "export_html_report",
    "probe_columns",
    "export_dataset_bundle",
    "draft_preprocessing_plan",
    "edit_preprocessing_plan",
    "validate_preprocessing_config",
    "build_longitudinal_graph",
    "diagnose_longitudinal_graph",
    "get_trajectory_archetypes",
    "get_cross_time_neighbors",
    "classify_trajectories",
    "get_workflow_guide",
    "get_runtime_context",
    "characterize_dataset",
]
