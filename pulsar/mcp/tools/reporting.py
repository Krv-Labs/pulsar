from __future__ import annotations

import dataclasses
import json
import logging
import os
from typing import Any, Literal

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
from pulsar.mcp.session import (
    _get_session,
    _load_session_dataframe,
    _resolve_dataset_path,
)

logger = logging.getLogger(__name__)


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


async def export_labeled_data(
    cluster_names: dict[str, str], output_path: str, ctx: Context
) -> str:
    """Assign semantic names to clusters and export labeled dataset to CSV."""
    session = _get_session(ctx)

    if session.data is None or session.clusters is None:
        raise ToolError(
            "No data or clusters found. Run generate_cluster_dossier() first."
        )

    actual_ids = {int(cluster_id) for cluster_id in session.clusters.unique().tolist()}
    names_map = _normalize_cluster_names(
        cluster_names,
        valid_cluster_ids=actual_ids,
        require_all=True,
    )

    try:
        df = session.data.copy()
        df["topological_cluster_id"] = session.clusters
        df["topological_cluster_name"] = df["topological_cluster_id"].map(names_map)
        df.to_csv(output_path, index=False)
        return f"Successfully exported labeled data to {output_path}"

    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return mcp_error("export_labeled_data", str(e))


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
        if (
            session.feature_evidence_index is not None
            and cols_to_exclude == session.report_exclude_columns
        ):
            evidence_index = session.feature_evidence_index
        else:
            evidence_index = build_feature_evidence_index(
                session.model,
                session.data,
                session.clusters,
                exclude_columns=cols_to_exclude,
            )

        dossier = build_dossier(
            session.model,
            session.data,
            session.clusters,
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

        config_yaml = (
            session.sweep_history[-1].config_yaml if session.sweep_history else None
        )
        html = dossier_to_html(
            dossier,
            model=session.model,
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

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

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


async def probe_columns(
    dataset_id: str,
    columns: list[str],
    response_format: Literal["markdown", "json"] = "markdown",
    ctx: Context = None,
) -> str:
    """Detailed per-column profiles (sample values, distributions). Use after
    the compact preview from `characterize_dataset`. Max 20 columns.
    """
    from pulsar.analysis.characterization import profile_column_details

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
        profiles = []
        missing_columns = []
        for col in columns:
            if col not in df.columns:
                missing_columns.append(col)
                continue
            profiles.append(dataclasses.asdict(profile_column_details(df, col)))

        payload = probe_columns_payload(profiles, columns, missing_columns)
        if response_format == "markdown":
            return probe_columns_payload_to_markdown(payload)
        return json.dumps(payload, indent=2)
    except Exception as e:
        return mcp_error("probe_columns", str(e))
