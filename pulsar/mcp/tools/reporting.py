from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
from pathlib import Path
import tempfile
from typing import Annotated, Any, Callable, Literal

from fastmcp import Context
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from pydantic import Field

from pulsar.mcp.characterization import (
    probe_columns_payload,
    probe_columns_payload_to_markdown,
)
from pulsar.mcp.errors import mcp_error, unknown_handle_error
from pulsar.mcp.interpreter import build_dossier, build_feature_evidence_index
from pulsar.mcp.report import dossier_to_html
from pulsar.mcp.registry import registry
from pulsar.mcp.results import mcp_result
from pulsar.mcp.toolmeta import (
    default_tool_timeout,
    INT,
    STR,
    arr,
    enum,
    nullable,
    obj,
    tool_meta,
)
from pulsar.runtime.utils import default_file_mode
from pulsar.mcp.session import (
    _get_session,
    _load_session_dataframe,
    _read_dataset_file,
    _resolve_dataset_path,
    resolve_run,
)

logger = logging.getLogger(__name__)


_OPEN = obj({}, additional=True)

_LABELED_SCHEMA = obj(
    {
        "status": enum(["ok"]),
        "output_path": STR,
        "cluster_assignment_id": STR,
        "run_id": nullable(STR),
        "dataset_id": nullable(STR),
        "row_count": INT,
        "cluster_ids": arr(INT),
    },
    [
        "status",
        "output_path",
        "cluster_assignment_id",
        "run_id",
        "dataset_id",
        "row_count",
        "cluster_ids",
    ],
)
_HTML_SCHEMA = obj(
    {
        "status": enum(["ok"]),
        "report_path": STR,
        "report_url": STR,
        "message": STR,
        "warning": STR,
    },
    ["status", "report_path", "report_url", "message"],
)
_PROBE_SCHEMA = obj(
    {
        "status": enum(["ok"]),
        "columns_requested": INT,
        "columns_returned": INT,
        "missing_columns": arr(STR),
        "column_profiles": arr(
            obj(
                {
                    "name": STR,
                    "dtype": STR,
                    "sample_values": arr(STR),
                    "truncation": _OPEN,
                },
                ["name"],
                additional=True,
            )
        ),
    },
    [
        "status",
        "columns_requested",
        "columns_returned",
        "missing_columns",
        "column_profiles",
    ],
)
_BUNDLE_SCHEMA = obj(
    {
        "status": enum(["ok"]),
        "output_dir": STR,
        "slug": STR,
        "bundle_path": STR,
        "manifest": _OPEN,
    },
    ["status", "output_dir", "slug", "bundle_path", "manifest"],
)

_RUN_ID_DESC = (
    "Run from run_topological_sweep. Omit for the latest run. Must be the run the "
    "session's cluster labels were built from (a newer sweep discards them), so "
    "pass it mainly on stateless transports."
)


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


def _missing_session_state_error(tool: str, session: Any) -> ToolResult:
    latest_run_id = getattr(session, "latest_run_id", None)
    latest_dataset_id = getattr(session, "dataset_id", None)
    return mcp_error(
        tool,
        "No fitted model is loaded in this MCP session.",
        error_code="MODEL_NOT_LOADED",
        agent_action=(
            "Run run_topological_sweep first, or pass run_id of a cached run. "
            "Call get_runtime_context to inspect latest_dataset_id/latest_run_id; "
            "run records persist under cache_dir/runs, but fitted model state is "
            "in-memory only."
        ),
        details={
            "latest_run_id": latest_run_id,
            "latest_dataset_id": latest_dataset_id,
            "cache_dir": str(registry.cache_dir),
            "run_handle_persistence": "metadata/config/metrics only",
            "model_state_persistence": "in-memory only",
        },
    )


def _resolve_for_export(
    tool: str, session: Any, run_id: str
) -> tuple[Any, ToolResult | None]:
    """Resolve the fitted run for an export; return ``(resolved, error)``."""
    try:
        resolved = resolve_run(session, run_id)
    except LookupError:
        if run_id:
            return None, unknown_handle_error(tool, "run_id", run_id)
        return None, _missing_session_state_error(tool, session)
    if resolved.data is None:
        return None, _missing_session_state_error(tool, session)
    return resolved, None


def _clusters_stale_error(tool: str, session: Any, run_id: str) -> ToolResult:
    return mcp_error(
        tool,
        f"Session cluster labels were computed from run '{session.clusters_run_id}', "
        f"not the requested run '{run_id}'.",
        error_code="CLUSTERS_STALE",
        agent_action=(
            "Rebuild the dossier for this run with "
            f"generate_cluster_dossier(run_id='{run_id}') (it must be the latest "
            "run), then retry."
        ),
        details={
            "requested_run_id": run_id,
            "clusters_run_id": session.clusters_run_id,
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


@tool_meta(
    read_only=False,
    idempotent=True,
    output_schema=_LABELED_SCHEMA,
    title="Export Labeled Data",
)
async def export_labeled_data(
    cluster_assignment_id: Annotated[
        str,
        Field(
            description=(
                "Handle of a saved cluster assignment, emitted by "
                "generate_cluster_dossier."
            )
        ),
    ],
    cluster_names: Annotated[
        dict[str, str],
        Field(
            description=(
                'Names for EVERY cluster: integer cluster ids as string keys, non-empty values, e.g. {"0": "Cohort A"}. Pulsar does not infer domain meaning.'
            )
        ),
    ],
    output_path: Annotated[
        str,
        Field(
            description=(
                "Absolute server-visible path of the CSV to write; an existing file is replaced."
            )
        ),
    ],
    run_id: Annotated[
        str,
        Field(
            description=(
                "Optional run handle: labels must come from that run (else CLUSTER_ASSIGNMENT_PROVENANCE_MISMATCH); its in-memory data is used."
            )
        ),
    ] = "",
    ctx: Context = None,
) -> ToolResult:
    """Write the dataset to CSV with a topological cluster id and your cluster name per row.

    Returns the output path, row count and cluster ids. Use after
    generate_cluster_dossier once you have named the clusters; the original
    columns are preserved and two columns are appended.
    """
    session = _get_session(ctx)
    assignment = registry.get_cluster_assignment(cluster_assignment_id)
    if assignment is None:
        return unknown_handle_error(
            "export_labeled_data",
            "cluster_assignment_id",
            cluster_assignment_id,
        )

    resolved = None
    if run_id:
        try:
            resolved = resolve_run(session, run_id)
        except LookupError:
            return unknown_handle_error("export_labeled_data", "run_id", run_id)
        if assignment.run_id != resolved.run_id:
            return mcp_error(
                "export_labeled_data",
                "The cluster assignment was computed from a different run than run_id.",
                error_code="CLUSTER_ASSIGNMENT_PROVENANCE_MISMATCH",
                agent_action=(
                    "Pass the run_id the assignment came from, or omit run_id, or "
                    "regenerate the dossier for the run you want."
                ),
                details={
                    "cluster_assignment_id": cluster_assignment_id,
                    "assignment_run_id": assignment.run_id,
                    "requested_run_id": run_id,
                },
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
        if resolved is not None and resolved.data is not None:
            df = resolved.data
        elif (
            session.data is not None
            and session.data_dataset_id == assignment.dataset_id
        ):
            df = session.data
        else:
            df = await asyncio.to_thread(_read_dataset_file, dataset.path)
    elif (
        resolved is not None
        and resolved.data is not None
        and len(resolved.data) == assignment.label_count
    ):
        df = resolved.data
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
    try:
        names_map = _normalize_cluster_names(
            cluster_names,
            valid_cluster_ids=actual_ids,
            require_all=True,
        )
    except ToolError as e:
        return mcp_error(
            "export_labeled_data",
            str(e),
            error_code="CLUSTER_NAMES_INVALID",
            agent_action=(
                "Pass cluster_names with every cluster id as a string key "
                f"({sorted(actual_ids)}) and a non-empty name for each."
            ),
            details={"valid_cluster_ids": sorted(actual_ids)},
        )

    try:
        row_count = await asyncio.to_thread(
            _write_labeled_csv_worker,
            df,
            assignment.labels,
            names_map,
            output_path,
        )
        return mcp_result(
            {
                "status": "ok",
                "output_path": output_path,
                "cluster_assignment_id": cluster_assignment_id,
                "run_id": assignment.run_id,
                "dataset_id": assignment.dataset_id,
                "row_count": row_count,
                "cluster_ids": [int(c) for c in assignment.cluster_ids],
            }
        )

    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return mcp_error(
            "export_labeled_data",
            str(e),
            error_code="EXPORT_WRITE_FAILED",
            agent_action="Check that output_path is writable on the host, then retry.",
            details={"output_path": output_path},
        )


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


@tool_meta(
    title="Export HTML Report",
    read_only=False,
    idempotent=True,
    output_schema=_HTML_SCHEMA,
    timeout=default_tool_timeout(),
)
async def export_html_report(
    dataset_id: Annotated[
        str,
        Field(
            description=(
                "Optional dataset handle; the report is written next to that file as <name>_report.html, else pulsar_report.html in the working directory."
            )
        ),
    ] = "",
    exclude_columns: Annotated[
        list[str] | None,
        Field(
            description=(
                "Column names to leave out of the feature evidence shown in the "
                "report. Omit to reuse the session's dossier exclusions."
            )
        ),
    ] = None,
    cluster_names: Annotated[
        dict[str, str] | None,
        Field(
            description=(
                'Optional cluster names: integer cluster ids as string keys, e.g. {"0": "Cohort A"}. Unnamed clusters get auto-names plus a warning.'
            )
        ),
    ] = None,
    run_id: Annotated[str, Field(description=_RUN_ID_DESC)] = "",
    ctx: Context = None,
) -> ToolResult:
    """Write a self-contained HTML report of the fitted topology and its clusters.

    Returns `report_path` and a file:// `report_url`. Requires clusters from
    generate_cluster_dossier for the same run (else CLUSTERS_STALE or
    CLUSTERS_MISSING). Name clusters via cluster_names for a readable report.
    """
    session = _get_session(ctx)
    resolved, err = _resolve_for_export("export_html_report", session, run_id)
    if err is not None:
        return err

    if session.clusters is None:
        return mcp_error(
            "export_html_report",
            "No clusters found. Run generate_cluster_dossier first.",
            error_code="CLUSTERS_MISSING",
            agent_action="Run generate_cluster_dossier for this run, then retry.",
        )
    if run_id and session.clusters_run_id != resolved.run_id:
        return _clusters_stale_error("export_html_report", session, run_id)

    # Use explicitly passed columns, or fallback to session defaults
    cols_to_exclude = (
        exclude_columns
        if exclude_columns is not None
        else session.report_exclude_columns
    )

    try:
        config_yaml = None
        if not run_id or resolved.run_id == session.latest_run_id:
            config_yaml = (
                session.sweep_history[-1].config_yaml if session.sweep_history else None
            )
        else:
            record = registry.get_run(resolved.run_id)
            config_yaml = record.config_yaml if record else None
        report_path = await asyncio.to_thread(
            _export_html_report_worker,
            model=resolved.model,
            data=resolved.data,
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
        return mcp_result(response)
    except ToolError as e:
        return mcp_error(
            "export_html_report",
            str(e),
            error_code="CLUSTER_NAMES_INVALID",
            agent_action=(
                "Pass cluster_names as a flat object whose keys are existing "
                "integer cluster ids encoded as strings and whose values are "
                "non-empty names."
            ),
        )
    except LookupError:
        return unknown_handle_error("export_html_report", "dataset_id", dataset_id)
    except Exception as e:
        return mcp_error(
            "export_html_report",
            str(e),
            error_code="EXPORT_REPORT_FAILED",
            agent_action="Check the reason, confirm the output location is writable, and retry.",
        )


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


@tool_meta(
    read_only=True, idempotent=True, output_schema=_PROBE_SCHEMA, title="Probe Columns"
)
async def probe_columns(
    dataset_id: Annotated[
        str, Field(description="Dataset handle returned by ingest_dataset.")
    ],
    columns: Annotated[
        list[str],
        Field(
            description=(
                "Names of columns of the raw dataset to inspect (at most 20); "
                "unknown names are listed under missing_columns."
            )
        ),
    ],
    response_format: Annotated[
        Literal["markdown", "json"],
        Field(
            description=(
                "Text view: 'markdown' or 'json'. structured_content is identical."
            )
        ),
    ] = "markdown",
    ctx: Context = None,
) -> ToolResult:
    """Drill into specific columns: sample values, missingness, ranges and top values.

    Reads the raw dataset (not the fitted model). Use after characterize_dataset
    or recommend_preprocessing flags columns to review; values are truncated and
    at most 20 columns per call.
    """
    if len(columns) > 20:
        return mcp_error(
            "probe_columns",
            "Too many columns requested. Max 20 at a time.",
            error_code="TOO_MANY_COLUMNS",
            agent_action="Split the request into batches of at most 20 columns.",
            details={"requested": len(columns), "max": 20},
        )

    session = _get_session(ctx)
    try:
        df, _ = await _load_session_dataframe(session, dataset_id=dataset_id)
    except LookupError:
        return unknown_handle_error("probe_columns", "dataset_id", dataset_id)
    except Exception as e:
        return mcp_error(
            "probe_columns",
            f"Could not load data: {e}",
            error_code="DATASET_LOAD_FAILED",
            agent_action="Check the dataset file is still readable, or re-ingest it.",
            details={"dataset_id": dataset_id},
        )

    try:
        payload = await asyncio.to_thread(_probe_columns_worker, df, columns)
        if response_format == "markdown":
            return mcp_result(payload, text=probe_columns_payload_to_markdown(payload))
        return mcp_result(payload)
    except Exception as e:
        return mcp_error(
            "probe_columns",
            str(e),
            error_code="PROBE_FAILED",
            agent_action="Check the column names and retry with fewer columns.",
        )


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


@tool_meta(
    title="Export Dataset Bundle",
    read_only=False,
    output_schema=_BUNDLE_SCHEMA,
    timeout=default_tool_timeout(),
)
async def export_dataset_bundle(
    output_dir: Annotated[
        str,
        Field(
            description=(
                "Absolute host-visible directory (on the MCP server machine) "
                "under which the bundle folder is created."
            )
        ),
    ],
    slug: Annotated[
        str,
        Field(
            description="Bundle folder name created under output_dir, e.g. project_alpha."
        ),
    ],
    cluster_assignment_id: Annotated[
        str | None,
        Field(
            description=(
                "Optional cluster_assignment_id from generate_cluster_dossier. Omit to use the session's clusters (must be the same run)."
            )
        ),
    ] = None,
    cluster_names: Annotated[
        dict[str, str] | None,
        Field(
            description=(
                "Optional cluster names: keys are integer cluster ids as strings, "
                'e.g. {"0": "Cohort A"}; unnamed clusters keep default labels.'
            )
        ),
    ] = None,
    cluster_descriptions: Annotated[
        dict[str, str] | None,
        Field(
            description=(
                "Optional free-text cluster descriptions keyed like cluster_names "
                "(integer cluster ids as strings)."
            )
        ),
    ] = None,
    edges_threshold: Annotated[
        float | None,
        Field(
            description=(
                "Edge weight cut-off in [0, 1] for graph/edges.parquet. Omit to "
                "use the model's resolved construction threshold."
            )
        ),
    ] = None,
    layout: Annotated[
        Literal["projection", "spectral", "zeros"],
        Field(
            description=(
                "Node coordinates in graph/nodes.parquet: projection (from the "
                "fitted embedding), spectral (graph eigenvectors), or zeros."
            )
        ),
    ] = "projection",
    include_clean: Annotated[
        bool,
        Field(
            description="Also write the preprocessed (imputed) data to tabular/clean.parquet."
        ),
    ] = False,
    write_manifest: Annotated[
        bool,
        Field(description="Write export_manifest.json with metadata and counts."),
    ] = True,
    run_id: Annotated[str, Field(description=_RUN_ID_DESC)] = "",
    ctx: Context = None,
) -> ToolResult:
    """Export the fitted graph, node/edge tables, clusters and data as a Parquet bundle.

    Writes `{output_dir}/{slug}/` (tables, graph, groups, manifest) and returns the
    paths and manifest. Needs clusters for the same run (generate_cluster_dossier).
    For one labeled CSV use export_labeled_data.
    """
    import numpy as np
    from pathlib import Path

    session = _get_session(ctx)
    resolved, err = _resolve_for_export("export_dataset_bundle", session, run_id)
    if err is not None:
        return err
    run_model = resolved.model
    run_data = resolved.data
    current_run_id = resolved.run_id or None
    current_dataset_id = resolved.dataset_id if run_id else session.data_dataset_id

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
                    error_code="INVALID_EDGES_THRESHOLD",
                    agent_action="Pass edges_threshold in [0, 1], or omit it.",
                )
        except (ValueError, TypeError):
            return mcp_error(
                "export_dataset_bundle",
                "edges_threshold must be a float between 0.0 and 1.0",
                error_code="INVALID_EDGES_THRESHOLD",
                agent_action="Pass edges_threshold in [0, 1], or omit it.",
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
        if assignment.run_id != current_run_id:
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
                    "current_run_id": current_run_id,
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
                current_dataset_id is not None
                and assignment.dataset_id != current_dataset_id
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
                        "current_dataset_id": current_dataset_id,
                    },
                )
        if assignment.label_count != len(run_data):
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
                    "row_count": len(run_data),
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
        if run_id and session.clusters_run_id != resolved.run_id:
            return _clusters_stale_error("export_dataset_bundle", session, run_id)
        if session.clusters_run_id != current_run_id:
            return mcp_error(
                "export_dataset_bundle",
                "Cached cluster state is stale. It was computed from run "
                f"'{session.clusters_run_id}', but the active fitted model is run "
                f"'{current_run_id}'.",
                error_code="CLUSTER_CACHE_STALE",
                agent_action="Re-run generate_cluster_dossier() to recompute clusters before exporting.",
                details={
                    "cached_run_id": session.clusters_run_id,
                    "current_run_id": current_run_id,
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
        return mcp_error(
            "export_dataset_bundle",
            str(e),
            error_code="CLUSTER_NAMES_INVALID",
            agent_action=(
                "Pass cluster_names / cluster_descriptions as flat objects keyed by "
                f"existing integer cluster ids as strings ({sorted(valid_cluster_ids)})."
            ),
            details={"valid_cluster_ids": sorted(valid_cluster_ids)},
        )

    try:
        # Use asyncio.to_thread to avoid blocking FastMCP's event loop on large exports.
        manifest_data = await asyncio.to_thread(
            run_model.export_dataset_bundle,
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
        return mcp_result(response)

    except Exception as e:
        logger.error(f"Error exporting dataset bundle: {e}")
        return mcp_error(
            "export_dataset_bundle",
            str(e),
            error_code="EXPORT_BUNDLE_FAILED",
            agent_action="Check that output_dir is writable on the host and retry.",
        )
