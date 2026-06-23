from __future__ import annotations

import os

from pulsar.mcp.tools.ingestion import (
    ingest_dataset,
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
)
from pulsar.mcp.tools.preprocessing import (
    recommend_preprocessing,
    repair_preprocessing_config,
    validate_preprocessing_config,
)
from pulsar.mcp.tools.meta import (
    get_workflow_guide,
    get_runtime_context,
    characterize_dataset,
)

# Chunked-upload trio is a sandboxed-client escape hatch; opt in via env.
_ENABLE_UPLOAD = os.environ.get("PULSAR_MCP_ENABLE_UPLOAD") == "1"

ALL_TOOLS_LIST = [
    # Ingestion
    ingest_dataset,
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
    # Preprocessing
    recommend_preprocessing,
    repair_preprocessing_config,
    validate_preprocessing_config,
    # Meta
    get_workflow_guide,
    get_runtime_context,
    characterize_dataset,
]

__all__ = [
    "ALL_TOOLS_LIST",
    "ingest_dataset",
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
    "recommend_preprocessing",
    "repair_preprocessing_config",
    "validate_preprocessing_config",
    "get_workflow_guide",
    "get_runtime_context",
    "characterize_dataset",
]
