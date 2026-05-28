from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastmcp import Context
from fastmcp.exceptions import ToolError

from pulsar.config import THRESHOLD_RANGE_MESSAGE
from pulsar.mcp.diagnostics import diagnose_model
from pulsar.mcp.errors import mcp_error, unknown_handle_error
from pulsar.mcp.interpreter import (
    FeatureEvidenceIndex,
    build_dossier,
    build_feature_evidence_index,
    cluster_profile_payload,
    compare_clusters,
    comparison_to_markdown,
    dossier_to_markdown,
    feature_signal_payload,
    resolve_clusters,
    signal_matrix_payload,
)
from pulsar.mcp.payloads import (
    _prepend_threshold_markdown,
    _threshold_surface_payload,
    build_evidence_payload,
    build_summary_evidence_payload,
    cluster_profile_payload_to_markdown,
    cluster_result_payload,
    feature_signal_payload_to_markdown,
    summary_evidence_payload_to_markdown,
)
from pulsar.mcp.session import _get_session

logger = logging.getLogger(__name__)


def _validate_unit_threshold(value: float, *, name: str) -> float:
    threshold = float(value)
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ToolError(f"{name}: {THRESHOLD_RANGE_MESSAGE}")
    return threshold




def _feature_evidence_fingerprint(
    *,
    run_id: str | None,
    labels: pd.Series,
    method: str,
    interpretation_edge_weight_threshold: float,
    exclude_columns: list[str] | None,
) -> str:
    import hashlib

    payload = {
        "run_id": run_id,
        "labels": labels.astype(int).tolist(),
        "method": method,
        "interpretation_edge_weight_threshold": interpretation_edge_weight_threshold,
        "exclude_columns": sorted(exclude_columns or []),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _get_or_build_evidence_index(
    session: Any,
    *,
    cluster_result: Any,
    exclude_columns: list[str] | None,
) -> FeatureEvidenceIndex:
    fingerprint = _feature_evidence_fingerprint(
        run_id=session.latest_run_id,
        labels=cluster_result.labels,
        method=cluster_result.method_used,
        interpretation_edge_weight_threshold=cluster_result.interpretation_edge_weight_threshold_applied,
        exclude_columns=exclude_columns,
    )
    if (
        session.feature_evidence_index is not None
        and session.feature_evidence_fingerprint == fingerprint
    ):
        return session.feature_evidence_index

    evidence_index = build_feature_evidence_index(
        session.model,
        session.data,
        cluster_result.labels,
        exclude_columns=exclude_columns,
    )
    session.feature_evidence_index = evidence_index
    session.feature_evidence_fingerprint = fingerprint
    session.feature_evidence_cluster_meta = cluster_result_payload(cluster_result)
    return evidence_index


def _graph_metrics_payload(model: Any) -> dict[str, Any]:
    try:
        return dataclasses.asdict(diagnose_model(model))
    except Exception as exc:
        import dataclasses

        try:
            return dataclasses.asdict(diagnose_model(model))
        except (RuntimeError, AttributeError, NameError) as e:
            logger.warning("diagnose_model failed: %s", e)
            return {}


def _require_cluster_state(
    session: Any,
) -> tuple[FeatureEvidenceIndex, dict[str, Any]]:
    if session.model is None or session.data is None:
        raise ToolError("No model found. Run run_topological_sweep() first.")
    if session.clusters is None or session.feature_evidence_index is None:
        raise ToolError(
            "No cluster evidence found. Run generate_cluster_dossier() first."
        )
    return (
        session.feature_evidence_index,
        session.feature_evidence_cluster_meta or {},
    )


async def generate_cluster_dossier(
    method: Literal["auto", "spectral", "components"] = "auto",
    max_k: int = 15,
    interpretation_edge_weight_threshold: float | None = None,
    detail: Literal["summary", "standard", "full"] = "summary",
    max_clusters: int = 8,
    feature_preview_limit: int = 5,
    response_format: Literal["markdown", "json"] = "markdown",
    exclude_columns: list[str] | None = None,
    ctx: Context = None,
) -> str:
    """Statistical dossier of topological clusters. Use default `summary` for
    routine loops; `standard`/`full` are heavy — only after narrowing focus.

    Args:
        interpretation_edge_weight_threshold: Drop edges with weight <= this
            before clustering. Omit to inherit `resolved_construction_threshold`
            (spectral default is 0.0 / full affinity).
        max_clusters: Truncate to top-N by size; `clusters_omitted` reports drops.
        feature_preview_limit: Feature hints per cluster in summary mode.
        exclude_columns: Hide columns from report (does NOT affect modeling).
    """
    session = _get_session(ctx)

    if session.model is None or session.data is None:
        raise ToolError("No model found. Run run_topological_sweep() first.")
    if max_clusters < 1:
        raise ToolError(f"max_clusters must be >= 1, got '{max_clusters}'")
    if feature_preview_limit < 0:
        raise ToolError(
            f"feature_preview_limit must be >= 0, got '{feature_preview_limit}'"
        )

    try:
        construction_threshold = _validate_unit_threshold(
            session.model.resolved_construction_threshold,
            name="resolved_construction_threshold",
        )
        threshold_inherited = interpretation_edge_weight_threshold is None
        if interpretation_edge_weight_threshold is None and method == "spectral":
            resolved_interpretation_threshold = 0.0
            threshold_source = "spectral_default_full_affinity"
            threshold_inherited = False
        elif interpretation_edge_weight_threshold is None:
            resolved_interpretation_threshold = construction_threshold
            threshold_source = "inherited_construction"
        else:
            resolved_interpretation_threshold = _validate_unit_threshold(
                interpretation_edge_weight_threshold,
                name="interpretation_edge_weight_threshold",
            )
            threshold_source = "explicit"
        threshold_surface = _threshold_surface_payload(
            construction_threshold=construction_threshold,
            interpretation_threshold=resolved_interpretation_threshold,
            threshold_inherited=threshold_inherited,
            threshold_source=threshold_source,
        )
        result = resolve_clusters(
            session.model,
            method=method,
            max_k=max_k,
            interpretation_edge_weight_threshold=resolved_interpretation_threshold,
        )
        session.clusters = result.labels
        session.report_exclude_columns = exclude_columns

        session.feature_evidence_cluster_meta = cluster_result_payload(result)

        evidence_index = _get_or_build_evidence_index(
            session,
            cluster_result=result,
            exclude_columns=exclude_columns,
        )

        cluster_meta = cluster_result_payload(result)
        if detail == "summary":
            payload = build_summary_evidence_payload(
                evidence_index,
                cluster_meta,
                _graph_metrics_payload(session.model),
                max_clusters=max_clusters,
                feature_preview_limit=feature_preview_limit,
            )
            payload["construction_threshold"] = construction_threshold
            payload["interpretation_edge_weight_threshold"] = (
                resolved_interpretation_threshold
            )
            payload["threshold_inherited"] = threshold_inherited
            payload["threshold_source"] = threshold_source
            payload["threshold_surface"] = threshold_surface
            payload["recommended_next_tools"] = [
                "get_cluster_profile",
                "get_feature_signal",
                "get_cluster_signal_matrix",
                "compare_clusters_tool",
            ]
            payload["payload_guidance"] = (
                "Default summary output is a compact cluster map. Use "
                "response_format='json' for structured fields, or targeted tools "
                "for cluster/feature evidence. Set detail='standard' or 'full' "
                "only after narrowing to specific clusters/features or for "
                "explicit audit/debugging."
            )
            if response_format == "markdown":
                return _prepend_threshold_markdown(
                    summary_evidence_payload_to_markdown(payload),
                    threshold_surface,
                )
            return json.dumps(
                payload,
                separators=(",", ":") if len(payload["clusters"]) > 10 else None,
                indent=None if len(payload["clusters"]) > 10 else 2,
            )

        dossier = build_dossier(
            session.model,
            session.data,
            result.labels,
            detail=detail,
            evidence_index=evidence_index,
        )
        if response_format == "markdown":
            return _prepend_threshold_markdown(
                dossier_to_markdown(dossier),
                threshold_surface,
            )

        payload = build_evidence_payload(
            dossier,
            cluster_meta,
            detail=detail,
            max_clusters=max_clusters,
            feature_preview_limit=feature_preview_limit,
        )
        payload["construction_threshold"] = construction_threshold
        payload["interpretation_edge_weight_threshold"] = (
            resolved_interpretation_threshold
        )
        payload["threshold_inherited"] = threshold_inherited
        payload["threshold_source"] = threshold_source
        payload["threshold_surface"] = threshold_surface
        payload["recommended_next_tools"] = (
            [
                "get_cluster_profile",
                "get_feature_signal",
                "get_cluster_signal_matrix",
                "compare_clusters_tool",
            ]
            if detail == "summary"
            else ["get_feature_signal", "compare_clusters_tool", "export_html_report"]
        )
        payload["payload_guidance"] = (
            "For routine agent loops, omit detail or use detail='summary'. Summary "
            "is a compact cluster map with capped feature previews and counts only. "
            "Use get_cluster_profile, get_feature_signal, or get_cluster_signal_matrix "
            "for targeted evidence. Set detail='standard' or 'full' only after "
            "narrowing to specific clusters/features or for explicit audit/debugging."
        )
        return json.dumps(
            payload,
            separators=(",", ":") if len(dossier.clusters) > 10 else None,
            indent=None if len(dossier.clusters) > 10 else 2,
        )

    except ValueError as e:
        if "Graph is disconnected" in str(e):
            return mcp_error(
                "generate_cluster_dossier",
                "Spectral clustering requires a connected affinity graph. Your current threshold has disconnected the manifold.",
                error_code="GRAPH_DISCONNECTED",
                agent_action="Decrease interpretation_edge_weight_threshold or increase epsilon sweep range to connect the graph.",
            )
        logger.error(f"Error generating dossier: {e}")
        return mcp_error("generate_cluster_dossier", str(e))
    except Exception as e:
        logger.error(f"Error generating dossier: {e}")
        return mcp_error("generate_cluster_dossier", str(e))


async def get_cluster_profile(
    cluster_id: int,
    detail: Literal["summary", "standard", "full"] = "standard",
    max_features: int = 16,
    response_format: Literal["markdown", "json"] = "markdown",
    ctx: Context = None,
) -> str:
    """Targeted evidence for one cluster from cached dossier state."""
    try:
        if max_features < 1:
            raise ToolError(f"max_features must be >= 1, got '{max_features}'")

        session = _get_session(ctx)
        evidence_index, cluster_meta = _require_cluster_state(session)
        payload = {
            "status": "ok",
            "cluster_result": cluster_meta,
            "detail": detail,
            "max_features": max_features,
            "cluster": cluster_profile_payload(
                evidence_index,
                cluster_id,
                detail=detail,
                max_features=max_features,
            ),
        }
        if response_format == "markdown":
            return cluster_profile_payload_to_markdown(payload)
        return json.dumps(payload, indent=2)
    except KeyError:
        return mcp_error(
            "get_cluster_profile",
            f"Unknown cluster_id '{cluster_id}'.",
            error_code="CLUSTER_ID_UNKNOWN",
            agent_action="Use generate_cluster_dossier first and inspect available cluster IDs.",
        )
    except Exception as e:
        return mcp_error("get_cluster_profile", str(e))


async def get_feature_signal(
    feature_names: list[str],
    cluster_ids: list[int] | None = None,
    detail: Literal["summary", "standard", "full"] = "summary",
    max_clusters: int = 8,
    response_format: Literal["markdown", "json"] = "markdown",
    ctx: Context = None,
) -> str:
    """Cross-cluster evidence for specific feature columns.

    Args:
        feature_names: Numeric columns or `column=value` pairs for categoricals.
        cluster_ids: Restrict to specific clusters; omit to use top `max_clusters` by signal.
    """
    if max_clusters < 1:
        return mcp_error(
            "get_feature_signal",
            f"max_clusters must be >= 1, got '{max_clusters}'",
        )
    try:
        session = _get_session(ctx)
        evidence_index, cluster_meta = _require_cluster_state(session)
        payload = {
            "status": "ok",
            "cluster_result": cluster_meta,
            "signals": feature_signal_payload(
                evidence_index,
                feature_names,
                cluster_ids=cluster_ids,
                detail=detail,
                max_clusters=max_clusters,
            ),
        }
        if response_format == "markdown":
            return feature_signal_payload_to_markdown(payload["signals"])
        return json.dumps(payload, indent=2)
    except Exception as e:
        return mcp_error("get_feature_signal", str(e))


async def get_cluster_signal_matrix(
    cluster_ids: list[int] | None = None,
    include_context_tier: bool = False,
    max_clusters: int = 8,
    return_markdown: bool = True,
    ctx: Context = None,
) -> str:
    """Cross-cluster signal matrix. `max_clusters` is presentation-only
    truncation (ranks by signal). `include_context_tier` adds context-tier rows
    beyond core/supporting. `return_markdown=False` returns flattened JSON.
    """
    if max_clusters < 1:
        return mcp_error(
            "get_cluster_signal_matrix",
            f"max_clusters must be >= 1, got '{max_clusters}'",
        )
    try:
        session = _get_session(ctx)
        evidence_index, cluster_meta = _require_cluster_state(session)
        payload = {
            "status": "ok",
            "cluster_result": cluster_meta,
            "signal_matrix": signal_matrix_payload(
                evidence_index,
                cluster_ids=cluster_ids,
                include_context_tier=include_context_tier,
                max_clusters=max_clusters,
                return_markdown=return_markdown,
            ),
        }
        if return_markdown:
            return payload["signal_matrix"]["markdown_report"]
        return json.dumps(payload, indent=2)
    except Exception as e:
        return mcp_error("get_cluster_signal_matrix", str(e))


async def compare_clusters_tool(
    cluster_a: int, cluster_b: int, ctx: Context = None
) -> str:
    """Pairwise statistical tests between two clusters."""
    session = _get_session(ctx)

    if session.data is None or session.clusters is None:
        raise ToolError(
            "No data or clusters found. Run generate_cluster_dossier() first."
        )

    try:
        results = compare_clusters(session.data, session.clusters, cluster_a, cluster_b)
        if not results:
            return mcp_error(
                "compare_clusters_tool",
                f"No results for comparison between clusters {cluster_a} and {cluster_b}.",
            )

        markdown = comparison_to_markdown(cluster_a, cluster_b, results)
        return markdown

    except Exception as e:
        logger.error(f"Error comparing clusters: {e}")
        return mcp_error("compare_clusters_tool", str(e))
