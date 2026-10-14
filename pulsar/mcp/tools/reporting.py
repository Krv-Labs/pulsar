from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Literal

from fastmcp import Context
from fastmcp.exceptions import ToolError

from pulsar.mcp.characterization import (
    probe_columns_payload,
    probe_columns_payload_to_markdown,
)
from pulsar.mcp.errors import mcp_error, unknown_handle_error
from pulsar.mcp.interpreter import build_dossier, build_feature_evidence_index
from pulsar.mcp.report import dossier_to_html
from pulsar.mcp.registry import registry
from pulsar.runtime.utils import default_file_mode
from pulsar.mcp.session import (
    _get_session,
    _load_session_dataframe,
    _read_dataset_file,
    _resolve_dataset_path,
)

logger = logging.getLogger(__name__)


def _atomic_write_text_or_df(
    dest_path: Path,
    write_fn: Callable[[Path], None],
) -> None:
    """Write via a sibling temporary file in dest_path.parent and atomically replace."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=dest_path.parent,
        prefix=f".{dest_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        write_fn(tmp_path)
        os.chmod(tmp_path, default_file_mode())
        os.replace(tmp_path, dest_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _missing_session_state_error(tool: str, session: Any) -> str:
    latest_run_id = getattr(session, "latest_run_id", None)
    latest_dataset_id = getattr(session, "dataset_id", None)
    return mcp_error(
        tool,
        "No fitted model is loaded in this MCP session.",
        error_code="SESSION_STATE_MISSING",
        agent_action=(
            "Call get_runtime_context to inspect latest_dataset_id/latest_run_id. "
            "Run records persist under cache_dir/runs, but fitted model state is "
            "currently in-memory only; rerun run_topological_sweep before retrying "
            "tools that need session.model."
        ),
        details={
            "latest_run_id": latest_run_id,
            "latest_dataset_id": latest_dataset_id,
            "cache_dir": str(registry.cache_dir),
            "run_handle_persistence": "metadata/config/metrics only",
            "model_state_persistence": "in-memory only",
        },
    )


def _normalize_cluster_names(
    cluster_names: dict[str, str] | None,
    *,
    valid_cluster_ids: set[int],
    require_all: bool,
) -> dict[int, str]:
    """Validate flat cluster-name mappings from MCP JSON arguments."""
    if not cluster_names:
        return {}
    if not isinstance(cluster_names, dict):
        raise ToolError(
            "cluster_names must be a flat object mapping cluster ID strings to "
            'names, e.g. {"0": "Moderate Desert", "1": "Baseline"}.'
        )

    normalized: dict[int, str] = {}
    for raw_id, raw_name in cluster_names.items():
        key = str(raw_id).strip()
        if not key:
            raise ToolError("cluster_names contains an empty cluster ID key.")
        try:
            cluster_id = int(key)
        except ValueError as exc:
            raise ToolError(
                "cluster_names keys must be integer cluster IDs encoded as "
                f"strings; got {raw_id!r}."
            ) from exc
        if cluster_id in normalized:
            raise ToolError(
                f"cluster_names contains duplicate cluster ID {cluster_id}."
            )
        if cluster_id not in valid_cluster_ids:
            raise ToolError(
                f"cluster_names contains unknown cluster ID {cluster_id}. "
                f"Available IDs: {sorted(valid_cluster_ids)}."
            )
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ToolError(f"cluster_names[{key!r}] must be a non-empty string name.")
        normalized[cluster_id] = raw_name.strip()

    if require_all:
        missing_ids = valid_cluster_ids - set(normalized)
        if missing_ids:
            raise ToolError(
                f"cluster IDs {sorted(missing_ids)} have no name mapping. "
                f"Provide all {len(valid_cluster_ids)} cluster IDs."
            )

    return normalized


def _write_labeled_csv_worker(
    df: Any,
    labels: Any,
    names_map: dict[int, str],
    output_path: str,
) -> int:
    dest_path = Path(output_path)
    out_df = df.copy()
    out_df["topological_cluster_id"] = labels
    out_df["topological_cluster_name"] = out_df["topological_cluster_id"].map(names_map)
    _atomic_write_text_or_df(
        dest_path,
        lambda tmp_path: out_df.to_csv(tmp_path, index=False),
    )
    return len(out_df)


async def export_labeled_data(
    cluster_assignment_id: str,
    cluster_names: dict[str, str],
    output_path: str,
    ctx: Context = None,
) -> str:
    """Assign semantic names to a persisted cluster assignment and export CSV."""
    session = _get_session(ctx)
    assignment = registry.get_cluster_assignment(cluster_assignment_id)
    if assignment is None:
        return unknown_handle_error(
            "export_labeled_data",
            "cluster_assignment_id",
            cluster_assignment_id,
        )

    if assignment.dataset_id:
        dataset = registry.get_dataset(assignment.dataset_id)
        if dataset is None or not registry.dataset_matches_disk(dataset):
            return mcp_error(
                "export_labeled_data",
                "The dataset backing this cluster assignment is gone or has "
                "changed since the labels were computed, so the labels no longer "
                "align with its rows.",
                error_code="CLUSTER_ASSIGNMENT_DATASET_STALE",
                agent_action=(
                    "Re-ingest the dataset and rerun generate_cluster_dossier, "
                    "then export with the new cluster_assignment_id."
                ),
                details={
                    "cluster_assignment_id": cluster_assignment_id,
                    "dataset_id": assignment.dataset_id,
                },
            )
        # Read-only: export must never rebind session.data, which would
        # invalidate the active clusters/feature-evidence caches. Reuse the
        # in-memory frame only when it is already this exact dataset.
        if (
            session.data is not None
            and session.data_dataset_id == assignment.dataset_id
        ):
            df = session.data
        else:
            df = await asyncio.to_thread(_read_dataset_file, dataset.path)
    elif (
        session.data is not None
        and session.latest_run_id == assignment.run_id
        and len(session.data) == assignment.label_count
    ):
        df = session.data
    else:
        return mcp_error(
            "export_labeled_data",
            "Cluster assignment has no durable dataset_id and the matching data is "
            "not loaded in this session.",
            error_code="CLUSTER_ASSIGNMENT_DATA_UNAVAILABLE",
            agent_action=(
                "Use a cluster_assignment_id emitted from a dossier generated after "
                "ingest_dataset/run_topological_sweep(dataset_id=...), or regenerate "
                "the dossier with the source data loaded in this MCP session."
            ),
            details={
                "cluster_assignment_id": cluster_assignment_id,
                "run_id": assignment.run_id,
                "dataset_id": assignment.dataset_id,
                "label_count": assignment.label_count,
            },
        )

    if len(df) != assignment.label_count:
        return mcp_error(
            "export_labeled_data",
            "Cluster assignment label count does not match the dataset row count.",
            error_code="CLUSTER_ASSIGNMENT_ROW_MISMATCH",
            agent_action=(
                "Regenerate generate_cluster_dossier for the dataset you want to "
                "export and use the new cluster_assignment_id."
            ),
            details={
                "cluster_assignment_id": cluster_assignment_id,
                "dataset_id": assignment.dataset_id,
                "label_count": assignment.label_count,
                "row_count": len(df),
            },
        )

    actual_ids = {int(cluster_id) for cluster_id in assignment.cluster_ids}
    names_map = _normalize_cluster_names(
        cluster_names,
        valid_cluster_ids=actual_ids,
        require_all=True,
    )

    try:
        row_count = await asyncio.to_thread(
            _write_labeled_csv_worker,
            df,
            assignment.labels,
            names_map,
            output_path,
        )
        return json.dumps(
            {
                "status": "ok",
                "output_path": output_path,
                "cluster_assignment_id": cluster_assignment_id,
                "run_id": assignment.run_id,
                "dataset_id": assignment.dataset_id,
                "row_count": row_count,
                "cluster_ids": assignment.cluster_ids,
            },
            indent=2,
        )

    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return mcp_error("export_labeled_data", str(e))


def _export_html_report_worker(
    *,
    model: Any,
    data: Any,
    clusters: Any,
    cached_evidence_index: Any,
    cols_to_exclude: list[str] | None,
    report_exclude_columns: list[str] | None,
    cluster_names: dict[str, str] | None,
    config_yaml: str | None,
    dataset_id: str,
) -> str:
    if cached_evidence_index is not None and cols_to_exclude == report_exclude_columns:
        evidence_index = cached_evidence_index
    else:
        evidence_index = build_feature_evidence_index(
            model,
            data,
            clusters,
            exclude_columns=cols_to_exclude,
        )

    dossier = build_dossier(
        model,
        data,
        clusters,
        detail="full",
        evidence_index=evidence_index,
    )

    valid_ids = {int(profile.cluster_id) for profile in dossier.clusters}
    normalized_names = _normalize_cluster_names(
        cluster_names,
        valid_cluster_ids=valid_ids,
        require_all=False,
    )
    for profile in dossier.clusters:
        if profile.cluster_id in normalized_names:
            profile.semantic_name = normalized_names[profile.cluster_id]

    html = dossier_to_html(
        dossier,
        model=model,
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

    _atomic_write_text_or_df(
        Path(report_path),
        lambda tmp_path: tmp_path.write_text(html, encoding="utf-8"),
    )
    return report_path


async def export_html_report(
    dataset_id: str = "",
    exclude_columns: list[str] | None = None,
    cluster_names: dict[str, str] | None = None,
    ctx: Context = None,
) -> str:
    """Rich HTML report of the latest topological analysis.

    Args:
        cluster_names: `{"0": "Metabolic Cohort", "1": "Baseline", ...}`. When
            omitted, robotic auto-names are used and the response includes a
            warning prompting the agent to provide semantic names.
    """
    session = _get_session(ctx)
    if session.model is None or session.data is None:
        return _missing_session_state_error("export_html_report", session)

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
        config_yaml = (
            session.sweep_history[-1].config_yaml if session.sweep_history else None
        )
        report_path = await asyncio.to_thread(
            _export_html_report_worker,
            model=session.model,
            data=session.data,
            clusters=session.clusters,
            cached_evidence_index=session.feature_evidence_index,
            cols_to_exclude=cols_to_exclude,
            report_exclude_columns=session.report_exclude_columns,
            cluster_names=cluster_names,
            config_yaml=config_yaml,
            dataset_id=dataset_id,
        )

        response: dict[str, Any] = {
            "status": "ok",
            "report_path": report_path,
            "report_url": f"file://{os.path.abspath(report_path)}",
            "message": "HTML report exported successfully. Open the report_url in your browser.",
        }
        if not cluster_names:
            response["warning"] = (
                "cluster_names was not provided — clusters are labeled with "
                "robotic auto-names (e.g. '[Auto] High X, Low Y'). For a "
                "human-tractable report, re-run with "
                'cluster_names={"0": "<scientific cohort name>", ...}.'
            )
        return json.dumps(response, indent=2)
    except Exception as e:
        return mcp_error("export_html_report", str(e))


def _probe_columns_worker(df: Any, columns: list[str]) -> dict[str, Any]:
    from pulsar.analysis.characterization import profile_column_details

    profiles = []
    missing_columns = []
    for col in columns:
        if col not in df.columns:
            missing_columns.append(col)
            continue
        profiles.append(dataclasses.asdict(profile_column_details(df, col)))

    return probe_columns_payload(profiles, columns, missing_columns)


async def probe_columns(
    dataset_id: str,
    columns: list[str],
    response_format: Literal["markdown", "json"] = "markdown",
    ctx: Context = None,
) -> str:
    """Targeted column drilldown: sample values, missingness, and distributions.
    Use after `characterize_dataset`; max 20 columns.
    """
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
        payload = await asyncio.to_thread(_probe_columns_worker, df, columns)
        if response_format == "markdown":
            return probe_columns_payload_to_markdown(payload)
        return json.dumps(payload, indent=2)
    except Exception as e:
        return mcp_error("probe_columns", str(e))


def _normalize_cluster_metadata(
    metadata: dict[str, str] | None,
    *,
    valid_cluster_ids: set[int],
    name: str,
) -> dict[int, str] | None:
    """Validate flat metadata (names or descriptions) from MCP JSON arguments."""
    if not metadata:
        return None
    if not isinstance(metadata, dict):
        raise ToolError(
            f"{name} must be a flat object mapping cluster ID strings to values, e.g. "
            '{"0": "Cohort A", "1": "Cohort B"}.'
        )

    normalized: dict[int, str] = {}
    for raw_id, val in metadata.items():
        key = str(raw_id).strip()
        if not key:
            raise ToolError(f"{name} contains an empty cluster ID key.")
        try:
            cluster_id = int(key)
        except ValueError as exc:
            raise ToolError(
                f"{name} keys must be integer cluster IDs encoded as strings; got {raw_id!r}."
            ) from exc
        if cluster_id not in valid_cluster_ids:
            raise ToolError(
                f"{name} contains unknown cluster ID {cluster_id}. "
                f"Available IDs: {sorted(valid_cluster_ids)}."
            )
        normalized[cluster_id] = str(val).strip()
    return normalized


async def export_dataset_bundle(
    output_dir: str,
    slug: str,
    cluster_assignment_id: str | None = None,
    cluster_names: dict[str, str] | None = None,
    cluster_descriptions: dict[str, str] | None = None,
    edges_threshold: float | None = None,
    layout: Literal["projection", "spectral", "zeros"] = "projection",
    include_clean: bool = False,
    write_manifest: bool = True,
    ctx: Context = None,
) -> str:
    """Export the complete dataset bundle to a directory.

    Writes a standard bundle under `{output_dir}/{slug}/` containing tabular/raw, tabular/clean
    (optional), graph/nodes, graph/edges, graph/cosmic, graph/groups Parquet files,
    and a manifest JSON file.

    Args:
        output_dir: The directory where the exported bundle should be written.
        slug: The subdirectory name (e.g. 'project_alpha') under output_dir.
        cluster_assignment_id: Optional ID of a saved cluster assignment. If omitted,
            the active session's clusters are used.
        cluster_names: Optional dictionary mapping string cluster IDs to semantic names.
        cluster_descriptions: Optional dictionary mapping string cluster IDs to semantic descriptions.
        edges_threshold: Optional threshold for graph/edges.parquet (snapshot edges).
            Omit to inherit the model's resolved construction threshold.
        layout: Layout coordinates mode for graph/nodes.parquet ('projection', 'spectral', or 'zeros').
        include_clean: If True, include preprocessed (imputed) clean data under tabular/clean.parquet.
        write_manifest: If True, write an export_manifest.json with metadata and counts.
    """
    import numpy as np
    from pathlib import Path

    session = _get_session(ctx)
    if session.model is None or session.data is None:
        return _missing_session_state_error("export_dataset_bundle", session)

    if edges_threshold is not None:
        try:
            edges_threshold = float(edges_threshold)
            if (
                not np.isfinite(edges_threshold)
                or edges_threshold < 0.0
                or edges_threshold > 1.0
            ):
                return mcp_error(
                    "export_dataset_bundle",
                    "edges_threshold must be a float between 0.0 and 1.0",
                )
        except ValueError:
            return mcp_error(
                "export_dataset_bundle",
                "edges_threshold must be a float between 0.0 and 1.0",
            )

    # Resolve cluster labels
    if cluster_assignment_id is not None:
        assignment = registry.get_cluster_assignment(cluster_assignment_id)
        if assignment is None:
            return unknown_handle_error(
                "export_dataset_bundle",
                "cluster_assignment_id",
                cluster_assignment_id,
            )
        if assignment.run_id != session.latest_run_id:
            return mcp_error(
                "export_dataset_bundle",
                "Cluster assignment was computed from a different run than the active fitted model.",
                error_code="CLUSTER_ASSIGNMENT_PROVENANCE_MISMATCH",
                agent_action=(
                    "Regenerate generate_cluster_dossier for the active model and use "
                    "the new cluster_assignment_id."
                ),
                details={
                    "cluster_assignment_id": cluster_assignment_id,
                    "assignment_run_id": assignment.run_id,
                    "current_run_id": session.latest_run_id,
                },
            )
        if assignment.dataset_id:
            dataset = registry.get_dataset(assignment.dataset_id)
            if dataset is None or not registry.dataset_matches_disk(dataset):
                return mcp_error(
                    "export_dataset_bundle",
                    "The dataset backing this cluster assignment is gone or has "
                    "changed since the labels were computed, so the labels no longer "
                    "align with its rows.",
                    error_code="CLUSTER_ASSIGNMENT_DATASET_STALE",
                    agent_action=(
                        "Re-ingest the dataset and rerun generate_cluster_dossier, "
                        "then export with the new cluster_assignment_id."
                    ),
                    details={
                        "cluster_assignment_id": cluster_assignment_id,
                        "dataset_id": assignment.dataset_id,
                    },
                )
            if (
                session.data_dataset_id is not None
                and assignment.dataset_id != session.data_dataset_id
            ):
                return mcp_error(
                    "export_dataset_bundle",
                    "Cluster assignment dataset does not match the active fitted model's dataset.",
                    error_code="CLUSTER_ASSIGNMENT_PROVENANCE_MISMATCH",
                    agent_action=(
                        "Export using a cluster_assignment_id generated for the "
                        "active fitted model."
                    ),
                    details={
                        "cluster_assignment_id": cluster_assignment_id,
                        "assignment_dataset_id": assignment.dataset_id,
                        "current_dataset_id": session.data_dataset_id,
                    },
                )
        if assignment.label_count != len(session.data):
            return mcp_error(
                "export_dataset_bundle",
                "Cluster assignment label count does not match the active model's dataset row count.",
                error_code="CLUSTER_ASSIGNMENT_ROW_MISMATCH",
                agent_action=(
                    "Regenerate generate_cluster_dossier for this dataset and use "
                    "the new cluster_assignment_id."
                ),
                details={
                    "cluster_assignment_id": cluster_assignment_id,
                    "label_count": assignment.label_count,
                    "row_count": len(session.data),
                },
            )
        cluster_labels = np.array(assignment.labels, dtype=np.uint32)
        valid_cluster_ids = set(assignment.cluster_ids)
    else:
        if session.clusters is None:
            return mcp_error(
                "export_dataset_bundle",
                "No cluster assignment ID was provided and no clusters are active in this session.",
                error_code="CLUSTERS_MISSING",
                agent_action="Run generate_cluster_dossier first or provide a cluster_assignment_id.",
            )
        if session.clusters_run_id != session.latest_run_id:
            return mcp_error(
                "export_dataset_bundle",
                "Cached cluster state is stale. It was computed from run "
                f"'{session.clusters_run_id}', but the active fitted model is run "
                f"'{session.latest_run_id}'.",
                error_code="CLUSTER_CACHE_STALE",
                agent_action="Re-run generate_cluster_dossier() to recompute clusters before exporting.",
                details={
                    "cached_run_id": session.clusters_run_id,
                    "current_run_id": session.latest_run_id,
                },
            )
        cluster_labels = session.clusters
        valid_cluster_ids = set(int(gid) for gid in cluster_labels.unique())

    try:
        norm_names = _normalize_cluster_metadata(
            cluster_names,
            valid_cluster_ids=valid_cluster_ids,
            name="cluster_names",
        )
        norm_descs = _normalize_cluster_metadata(
            cluster_descriptions,
            valid_cluster_ids=valid_cluster_ids,
            name="cluster_descriptions",
        )
    except ToolError as e:
        return mcp_error("export_dataset_bundle", str(e))

    try:
        # Use asyncio.to_thread to avoid blocking FastMCP's event loop on large exports.
        manifest_data = await asyncio.to_thread(
            session.model.export_dataset_bundle,
            output_dir=output_dir,
            slug=slug,
            cluster_labels=cluster_labels,
            cluster_names=norm_names,
            cluster_descriptions=norm_descs,
            edges_threshold=edges_threshold,
            layout=layout,
            include_clean=include_clean,
            write_manifest=write_manifest,
        )

        response = {
            "status": "ok",
            "output_dir": str(Path(output_dir).resolve()),
            "slug": slug,
            "bundle_path": str((Path(output_dir) / slug).resolve()),
            "manifest": manifest_data,
        }
        return json.dumps(response, indent=2)

    except Exception as e:
        logger.error(f"Error exporting dataset bundle: {e}")
        return mcp_error("export_dataset_bundle", str(e))
