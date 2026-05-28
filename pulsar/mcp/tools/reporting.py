from __future__ import annotations

import dataclasses
import json
import logging
import os
from typing import Any

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


async def export_labeled_data(
    cluster_names: dict[int, str], output_path: str, ctx: Context
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
    provided_ids = set(int(str(k).lstrip("_")) for k in cluster_names.keys())
    missing_ids = actual_ids - provided_ids
    if missing_ids:
        raise ToolError(
            f"cluster IDs {sorted(missing_ids)} have no name mapping. Provide all {len(actual_ids)} cluster IDs."
        )

    try:
        df = session.data.copy()
        df["topological_cluster_id"] = session.clusters
        names_map = {int(str(k).lstrip("_")): v for k, v in cluster_names.items()}
        df["topological_cluster_name"] = df["topological_cluster_id"].map(names_map)
        df.to_csv(output_path, index=False)
        return f"Successfully exported labeled data to {output_path}"

    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return mcp_error("export_labeled_data", str(e))


async def export_html_report(
    dataset_id: str = "",
    exclude_columns: list[str] | None = None,
    cluster_names: list[dict[str, str]] | dict[str, str] | None = None,
    ctx: Context = None,
) -> str:
    """
    Generate a rich HTML report of the latest topological analysis.

    Args:
        dataset_id: The dataset to report on.
        exclude_columns: Optional list of columns to exclude from the report.
            Defaults to the exclusion list used in generate_cluster_dossier.
        cluster_names: Optional dictionary or list of objects mapping string cluster IDs
            to semantic names.
            Preferred format (List of Objects): [{"id": "0", "name": "Metabolic Cohort"}]
            Legacy format (Dictionary): {"0": "Metabolic Cohort"}
            CRITICAL: The system will generate robotic fallback names (e.g. '[Auto] High X, Low Y')
            if you do not provide this. It is YOUR responsibility as the analytical agent
            to provide human-tractable, scientific cohort names.
    """
    session = _get_session(ctx)
    if session.model is None or session.data is None:
        return mcp_error(
            "export_html_report", "No model found. Run run_topological_sweep first."
        )

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

        # Apply LLM-provided cluster names if available
        if cluster_names:
            import re

            def normalize_key(k: str) -> str:
                # Extract digits to handle "u0", "cluster_0", "_0", etc.
                match = re.search(r"(\d+)", str(k))
                return match.group(1) if match else str(k)

            # Handle both formats: List[dict] and dict
            if isinstance(cluster_names, list):
                raw_names = {}
                for item in cluster_names:
                    # Support multiple key names for flexibility
                    cid = item.get("id") or item.get("cluster_id") or item.get("cid")
                    name = (
                        item.get("name")
                        or item.get("semantic_name")
                        or item.get("label")
                    )
                    if cid is not None and name is not None:
                        raw_names[str(cid)] = name
            else:
                raw_names = cluster_names

            normalized_names = {normalize_key(k): v for k, v in raw_names.items()}

            # Check for total mismatch to provide better feedback
            valid_ids = {str(p.cluster_id) for p in dossier.clusters}
            provided_ids = set(normalized_names.keys())
            if not (provided_ids & valid_ids) and raw_names:
                logger.warning(
                    "export_html_report: Provided cluster_names keys %s do not match any cluster IDs %s",
                    list(provided_ids),
                    list(valid_ids),
                )

            for p in dossier.clusters:
                str_id = str(p.cluster_id)
                if str_id in normalized_names:
                    p.semantic_name = normalized_names[str_id]

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

        return json.dumps(
            {
                "status": "ok",
                "report_path": report_path,
                "report_url": f"file://{os.path.abspath(report_path)}",
                "message": "HTML report exported successfully. Open the report_url in your browser.",
            },
            indent=2,
        )
    except Exception as e:
        return mcp_error("export_html_report", str(e))


async def probe_columns(
    dataset_id: str,
    columns: list[str],
    response_format: str = "markdown",
    ctx: Context = None,
) -> str:
    """
    Generate rich, detailed profiles for specific columns (Magnifying Glass).
    Use this when you need sample values or distributions for specific columns
    after seeing the compact preview from characterize_dataset.

    Args:
        dataset_id: Handle for the ingested dataset.
        columns: List of column names to probe (max 20).
    """
    from pulsar.analysis.characterization import profile_column_details

    if response_format not in {"json", "markdown"}:
        return mcp_error(
            "probe_columns",
            f"response_format must be 'json' or 'markdown', got '{response_format}'",
        )
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
