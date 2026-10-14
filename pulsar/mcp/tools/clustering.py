from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastmcp import Context
from fastmcp.exceptions import ToolError

from pulsar.config import THRESHOLD_RANGE_MESSAGE
from pulsar.mcp.diagnostics import diagnose_model
from pulsar.mcp.errors import mcp_error
from pulsar.mcp.interpreter import (
    FeatureEvidenceIndex,
    SpectralClusterCutError,
    build_dossier,
    build_feature_evidence_index,
    cluster_profile_payload,
    compare_clusters as _compare_clusters,
    comparison_to_markdown,
    dossier_to_markdown,
    feature_signal_payload,
    resolve_clusters,
    signal_matrix_payload,
)
from pulsar.mcp.payloads import (
    _prepend_threshold_markdown,
    _threshold_surface_payload,
    build_cluster_selection,
    build_evidence_payload,
    build_summary_evidence_payload,
    bounded_list,
    cluster_profile_payload_to_markdown,
    cluster_result_payload,
    cluster_selection_to_markdown,
    compact_cluster_result_payload,
    feature_signal_payload_to_markdown,
    size_summary,
    summary_evidence_payload_to_markdown,
)
from pulsar.mcp.registry import registry
from pulsar.mcp.session import SessionConflictError, _get_session, run_session_worker

logger = logging.getLogger(__name__)

# B3: explicit labels mapping each payload section to the graph/threshold lever
# it is computed at, so agents cannot conflate construction-threshold graph
# stats with the interpretation-threshold clustering slice.
_SURFACE_LABELS = {
    "graph_metrics": (
        "Computed on the persisted cosmic graph @ construction_threshold "
        "(the fitted graph). Components, density, and singletons describe this graph."
    ),
    "clusters": (
        "Computed on the interpretation slice @ interpretation_edge_weight_threshold "
        "of the [0,1]-normalized weighted adjacency (sparsified only when enabled). "
        "Cluster counts and membership reflect this slice, not the persisted graph."
    ),
}


def _two_lever_markdown_note(
    construction_threshold: float,
    interpretation_threshold: float,
) -> str:
    """Concise two-lever clarification prepended to dossier markdown (B3)."""
    return "\n".join(
        [
            "## Two Levers (do not conflate)",
            "",
            f"- Graph-level stats (components, density, singletons) describe the "
            f"persisted cosmic graph @ construction_threshold "
            f"({construction_threshold:.4f}).",
            f"- Cluster tables and per-cluster evidence describe the interpretation "
            f"slice @ interpretation_edge_weight_threshold "
            f"({interpretation_threshold:.4f}) of the [0,1]-normalized weighted "
            "adjacency (sparsified only when enabled) — not a dense affinity matrix.",
            "",
        ]
    )


def _cluster_assignment_summary(record: Any) -> dict[str, Any]:
    return {
        "cluster_assignment_id": record.cluster_assignment_id,
        "run_id": record.run_id,
        "dataset_id": record.dataset_id,
        "method": record.method,
        "interpretation_edge_weight_threshold": (
            record.interpretation_edge_weight_threshold
        ),
        "threshold_source": record.threshold_source,
        "construction_threshold": record.construction_threshold,
        "label_count": record.label_count,
        "cluster_ids": bounded_list(record.cluster_ids),
    }


def _cluster_assignment_markdown(record: Any) -> str:
    return "\n".join(
        [
            "## Cluster Assignment Artifact",
            "",
            f"- `cluster_assignment_id`: `{record.cluster_assignment_id}`",
            f"- `run_id`: `{record.run_id}`",
            f"- `dataset_id`: `{record.dataset_id}`",
            (
                "- Use this handle with `export_labeled_data` to export this "
                "exact clustering, including its method and interpretation threshold."
            ),
            "",
        ]
    )


def _validate_unit_threshold(value: float, *, name: str) -> float:
    threshold = float(value)
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ToolError(f"{name}: {THRESHOLD_RANGE_MESSAGE}")
    return threshold


def _spectral_cut_error_to_markdown(error: SpectralClusterCutError) -> str:
    payload = error.payload(detail="summary")
    details = payload["details"]
    best_score = details.get("best_silhouette_score")
    best = (
        f"k={details.get('best_k')}, silhouette={best_score:.3f}"
        if isinstance(best_score, (int, float))
        else "none"
    )
    acceptance_floor = float(details.get("accepted_silhouette_min", 0.0))
    return "\n".join(
        [
            "# No Stable Spectral Cut",
            "",
            "- Method: `spectral`",
            "- Interpretation threshold: "
            f"`{details.get('interpretation_edge_weight_threshold', 0.0):.4f}`",
            f"- Affinity components: {details.get('affinity_component_count')}",
            "- Giant component: "
            f"{details.get('giant_component_size')} "
            f"({float(details.get('giant_component_fraction', 0.0)):.1%})",
            f"- Residual nodes: {details.get('residual_node_count')}",
            f"- k evaluated: {details.get('k_min')}-{details.get('max_k')}",
            f"- Best candidate: {best}",
            f"- Acceptance floor: {acceptance_floor:.3f}",
            "",
            "Spectral search ran, but no evaluated cut exceeded the acceptance floor.",
        ]
    ).strip()


def _feature_evidence_fingerprint(
    *,
    run_id: str | None,
    labels: pd.Series,
    method: str,
    interpretation_edge_weight_threshold: float,
    exclude_columns: list[str] | None,
    max_clusters_to_characterize: int | None = None,
) -> str:
    import hashlib

    payload = {
        "run_id": run_id,
        "labels": labels.astype(int).tolist(),
        "method": method,
        "interpretation_edge_weight_threshold": interpretation_edge_weight_threshold,
        "exclude_columns": sorted(exclude_columns or []),
        "max_clusters_to_characterize": max_clusters_to_characterize,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _compute_evidence_index(
    model: Any,
    data: pd.DataFrame,
    latest_run_id: str | None,
    cached_index: FeatureEvidenceIndex | None,
    cached_fingerprint: str | None,
    *,
    cluster_result: Any,
    exclude_columns: list[str] | None,
    max_clusters_to_characterize: int | None = None,
) -> tuple[FeatureEvidenceIndex, str]:
    fingerprint = _feature_evidence_fingerprint(
        run_id=latest_run_id,
        labels=cluster_result.labels,
        method=cluster_result.method_used,
        interpretation_edge_weight_threshold=cluster_result.interpretation_edge_weight_threshold_applied,
        exclude_columns=exclude_columns,
        max_clusters_to_characterize=max_clusters_to_characterize,
    )
    if cached_index is not None and cached_fingerprint == fingerprint:
        return cached_index, fingerprint

    evidence_index = build_feature_evidence_index(
        model,
        data,
        cluster_result.labels,
        exclude_columns=exclude_columns,
        max_clusters_to_characterize=max_clusters_to_characterize,
    )
    return evidence_index, fingerprint


def _get_or_build_evidence_index(
    session: Any,
    *,
    cluster_result: Any,
    exclude_columns: list[str] | None,
    max_clusters_to_characterize: int | None = None,
) -> FeatureEvidenceIndex:
    evidence_index, fingerprint = _compute_evidence_index(
        session.model,
        session.data,
        session.latest_run_id,
        session.feature_evidence_index,
        session.feature_evidence_fingerprint,
        cluster_result=cluster_result,
        exclude_columns=exclude_columns,
        max_clusters_to_characterize=max_clusters_to_characterize,
    )
    session.feature_evidence_index = evidence_index
    session.feature_evidence_fingerprint = fingerprint
    # cluster_meta is owned by generate_cluster_dossier (built once per call and
    # assigned to the session), so it is not (re)built here. Building it on the
    # cache-miss path only would also leave it stale on a cache hit.
    return evidence_index


def _compute_dossier_worker(
    session: Any,
    model: Any,
    data: pd.DataFrame,
    expected_run_id: str | None,
    data_dataset_id: str | None,
    cached_index: FeatureEvidenceIndex | None,
    cached_fingerprint: str | None,
    method: str,
    max_k: int,
    resolved_interpretation_threshold: float,
    threshold_source: str,
    construction_threshold: float,
    exclude_columns: list[str] | None,
    max_char_cap: int,
    detail: Literal["summary", "standard", "full"],
) -> tuple[Any, dict[str, Any], FeatureEvidenceIndex, str, Any]:
    """Compute clusters/evidence/dossier. Never mutates ``session`` or persists.

    The caller publishes on the event loop (``_publish_dossier_on_loop``).
    """
    if expected_run_id is not None and session.latest_run_id != expected_run_id:
        raise SessionConflictError(
            "Session run changed before dossier computation started; "
            "refusing to publish stale cluster dossier."
        )
    result = resolve_clusters(
        model,
        method=method,
        max_k=max_k,
        interpretation_edge_weight_threshold=resolved_interpretation_threshold,
    )
    cluster_meta = cluster_result_payload(result)
    evidence_index, fingerprint = _compute_evidence_index(
        model,
        data,
        expected_run_id,
        cached_index,
        cached_fingerprint,
        cluster_result=result,
        exclude_columns=exclude_columns,
        max_clusters_to_characterize=max_char_cap,
    )
    dossier = (
        None
        if detail == "summary"
        else build_dossier(
            model,
            data,
            result.labels,
            detail=detail,
            evidence_index=evidence_index,
        )
    )
    return result, cluster_meta, evidence_index, fingerprint, dossier


def _publish_dossier_on_loop(
    session: Any,
    *,
    result: Any,
    cluster_meta: dict[str, Any],
    evidence_index: FeatureEvidenceIndex,
    fingerprint: str,
    expected_run_id: str | None,
    data_dataset_id: str | None,
    resolved_interpretation_threshold: float,
    threshold_source: str,
    construction_threshold: float,
    exclude_columns: list[str] | None,
) -> Any:
    """Persist the assignment and publish dossier state in one synchronous block.

    The run check precedes ``save_cluster_assignment`` and no ``await`` separates
    them from ``publish_dossier``, so a refused dossier persists nothing.
    """
    if session.latest_run_id != expected_run_id:
        raise SessionConflictError(
            "Active run changed while computing cluster dossier; "
            "refusing to publish stale clusters."
        )
    assignment = registry.save_cluster_assignment(
        run_id=expected_run_id,
        dataset_id=data_dataset_id,
        method=str(cluster_meta["method_used"]),
        interpretation_edge_weight_threshold=resolved_interpretation_threshold,
        threshold_source=threshold_source,
        construction_threshold=construction_threshold,
        labels=result.labels.astype(int).tolist(),
    )
    session.publish_dossier(
        expected_run_id=expected_run_id,
        clusters=result.labels,
        report_exclude_columns=exclude_columns,
        feature_evidence_cluster_meta=cluster_meta,
        cluster_assignment_id=assignment.cluster_assignment_id,
        feature_evidence_index=evidence_index,
        feature_evidence_fingerprint=fingerprint,
    )
    return assignment


def _graph_metrics_payload(model: Any) -> dict[str, Any]:
    try:
        payload = dataclasses.asdict(diagnose_model(model))
    except (RuntimeError, AttributeError, NameError, Exception) as e:
        logger.warning("diagnose_model failed: %s", e)
        return {"component_sizes_summary": size_summary([])}
    component_sizes = list(payload.get("component_sizes", []) or [])
    payload["component_sizes_summary"] = size_summary(component_sizes)
    payload.pop("component_sizes", None)
    return payload


class StaleClusterCacheError(Exception):
    """Cached cluster state was computed from a prior run than the active model.

    Raised when a newer sweep has advanced ``session.latest_run_id`` but the
    cached clusters/feature evidence were computed from an earlier run. Serving
    them would mix latest-run graph stats with stale cluster tables.
    """

    def __init__(self, *, cached_run_id: str | None, current_run_id: str | None):
        self.cached_run_id = cached_run_id
        self.current_run_id = current_run_id
        super().__init__(
            "Cached cluster state is stale: it was computed from run "
            f"'{cached_run_id}', but the active fitted model is run "
            f"'{current_run_id}'. A newer sweep has run since the last dossier."
        )


def _stale_cluster_cache_error(tool: str, exc: StaleClusterCacheError) -> str:
    return mcp_error(
        tool,
        str(exc),
        error_code="CLUSTER_CACHE_STALE",
        agent_action=(
            "Re-run generate_cluster_dossier() to recompute clusters against the "
            "current fitted model before reading cluster evidence."
        ),
        details={
            "cached_run_id": exc.cached_run_id,
            "current_run_id": exc.current_run_id,
        },
    )


def _require_cluster_state(
    session: Any,
) -> tuple[FeatureEvidenceIndex, dict[str, Any]]:
    if session.model is None or session.data is None:
        raise ToolError("No model found. Run run_topological_sweep() first.")
    if session.clusters is None or session.feature_evidence_index is None:
        raise ToolError(
            "No cluster evidence found. Run generate_cluster_dossier() first."
        )
    # Run-state consistency: reject cluster caches stamped from a prior run.
    # generate_cluster_dossier recomputes and refreshes the stamp, so this only
    # fires when a sweep ran after the last dossier (latest_run_id advanced).
    if session.clusters_run_id != session.latest_run_id:
        raise StaleClusterCacheError(
            cached_run_id=session.clusters_run_id,
            current_run_id=session.latest_run_id,
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
        model = session.model
        data = session.data
        expected_run_id = session.latest_run_id
        data_dataset_id = session.data_dataset_id
        cached_index = session.feature_evidence_index
        cached_fingerprint = session.feature_evidence_fingerprint

        construction_threshold = _validate_unit_threshold(
            model.resolved_construction_threshold,
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

        # Safety cap: only characterize top N clusters to prevent O(K*F) hang
        # on shattered graphs. Small singletons/noise are omitted from profiling.
        max_char_cap = max(50, max_clusters * 2)

        (
            result,
            cluster_meta,
            evidence_index,
            fingerprint,
            dossier,
        ) = await run_session_worker(
            session,
            _compute_dossier_worker,
            session,
            model,
            data,
            expected_run_id,
            data_dataset_id,
            cached_index,
            cached_fingerprint,
            method,
            max_k,
            resolved_interpretation_threshold,
            threshold_source,
            construction_threshold,
            exclude_columns,
            max_char_cap,
            detail,
            heavy=True,
            mutates_session=True,
        )
        assignment = _publish_dossier_on_loop(
            session,
            result=result,
            cluster_meta=cluster_meta,
            evidence_index=evidence_index,
            fingerprint=fingerprint,
            expected_run_id=expected_run_id,
            data_dataset_id=data_dataset_id,
            resolved_interpretation_threshold=resolved_interpretation_threshold,
            threshold_source=threshold_source,
            construction_threshold=construction_threshold,
            exclude_columns=exclude_columns,
        )
        assignment_summary = _cluster_assignment_summary(assignment)
        if detail == "summary":
            summary_feature_preview_limit = (
                min(feature_preview_limit, 3)
                if response_format == "json"
                else feature_preview_limit
            )
            payload = build_summary_evidence_payload(
                evidence_index,
                cluster_meta,
                _graph_metrics_payload(model),
                max_clusters=max_clusters,
                feature_preview_limit=summary_feature_preview_limit,
            )
            payload["construction_threshold"] = construction_threshold
            payload["interpretation_edge_weight_threshold"] = (
                resolved_interpretation_threshold
            )
            payload["threshold_inherited"] = threshold_inherited
            payload["threshold_source"] = threshold_source
            payload["threshold_surface"] = threshold_surface
            payload["cluster_assignment"] = assignment_summary
            payload["cluster_assignment_id"] = assignment.cluster_assignment_id
            payload["cluster_selection"] = build_cluster_selection(
                cluster_meta,
                [c["cluster_id"] for c in payload["clusters"]],
                ranked_by="size",
            )
            payload["recommended_next_tools"] = [
                "get_cluster_profile",
                "get_feature_signal",
                "get_cluster_signal_matrix",
                "compare_clusters",
            ]
            payload["detail_hint"] = (
                "Use get_cluster_profile/get_feature_signal for evidence; "
                "detail='full' returns lossless arrays for audit."
            )
            if response_format == "markdown":
                body = (
                    _two_lever_markdown_note(
                        construction_threshold, resolved_interpretation_threshold
                    )
                    + _cluster_assignment_markdown(assignment)
                    + cluster_selection_to_markdown(payload["cluster_selection"])
                    + summary_evidence_payload_to_markdown(payload)
                )
                return _prepend_threshold_markdown(body, threshold_surface)
            return json.dumps(
                payload,
                separators=(",", ":") if len(payload["clusters"]) > 10 else None,
                indent=None if len(payload["clusters"]) > 10 else 2,
            )
        if response_format == "markdown":
            body = (
                _two_lever_markdown_note(
                    construction_threshold, resolved_interpretation_threshold
                )
                + _cluster_assignment_markdown(assignment)
                + dossier_to_markdown(dossier)
            )
            return _prepend_threshold_markdown(body, threshold_surface)

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
        payload["cluster_assignment"] = assignment_summary
        payload["cluster_assignment_id"] = assignment.cluster_assignment_id
        if detail != "full":
            payload["surface_labels_ref"] = "get_workflow_guide"
        else:
            payload["surface_labels"] = _SURFACE_LABELS
        payload["cluster_selection"] = build_cluster_selection(
            cluster_meta,
            [c["cluster_id"] for c in payload["clusters"]],
            ranked_by="size",
        )
        payload["recommended_next_tools"] = (
            [
                "get_cluster_profile",
                "get_feature_signal",
                "get_cluster_signal_matrix",
                "compare_clusters",
            ]
            if detail == "summary"
            else ["get_feature_signal", "compare_clusters", "export_html_report"]
        )
        payload["detail_hint"] = (
            "Use targeted evidence tools for routine loops; detail='full' is the "
            "lossless audit payload."
        )
        return json.dumps(
            payload,
            separators=(",", ":") if len(dossier.clusters) > 10 else None,
            indent=None if len(dossier.clusters) > 10 else 2,
        )

    except SpectralClusterCutError as e:
        logger.info("No stable spectral cluster cut: %s", e.diagnostics)
        if response_format == "markdown":
            return _spectral_cut_error_to_markdown(e)
        return json.dumps(e.payload(detail=detail), indent=2)
    except SessionConflictError as e:
        logger.info("Dossier refused: %s", e)
        return mcp_error(
            "generate_cluster_dossier",
            str(e),
            error_code=e.error_code,
            agent_action=(
                "Nothing was persisted or published. The live run changed while "
                "the dossier was computing; re-run generate_cluster_dossier "
                "against the current run."
            ),
        )
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
            "cluster_result": compact_cluster_result_payload(
                cluster_meta,
                include_full_sizes=detail == "full",
            ),
            "detail": detail,
            "max_features": max_features,
            "cluster": cluster_profile_payload(
                evidence_index,
                cluster_id,
                detail=detail,
                max_features=max_features,
            ),
        }
        if detail == "full":
            payload["surface_labels"] = _SURFACE_LABELS
        else:
            payload["surface_labels_ref"] = "get_workflow_guide"
        if response_format == "markdown":
            applied = cluster_meta.get("interpretation_edge_weight_threshold_applied")
            note = (
                "_Cluster membership and evidence are computed on the interpretation "
                f"slice @ interpretation_edge_weight_threshold ({applied}) of the "
                "[0,1]-normalized weighted adjacency (sparsified only when enabled), "
                "not the persisted graph @ construction_threshold._\n\n"
            )
            return note + cluster_profile_payload_to_markdown(payload)
        return json.dumps(payload, indent=2)
    except StaleClusterCacheError as e:
        return _stale_cluster_cache_error("get_cluster_profile", e)
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
        signals = feature_signal_payload(
            evidence_index,
            feature_names,
            cluster_ids=cluster_ids,
            detail=detail,
            max_clusters=max_clusters,
        )
        payload = {
            "status": "ok",
            "cluster_result": compact_cluster_result_payload(
                cluster_meta,
                include_full_sizes=detail == "full",
            ),
            "cluster_selection": build_cluster_selection(
                cluster_meta,
                [c["cluster_id"] for c in signals.get("clusters", [])],
                ranked_by="signal",
            ),
            "signals": signals,
        }
        if detail == "full":
            payload["surface_labels"] = _SURFACE_LABELS
        else:
            payload["surface_labels_ref"] = "get_workflow_guide"
        if response_format == "markdown":
            return feature_signal_payload_to_markdown(payload["signals"])
        return json.dumps(payload, indent=2)
    except StaleClusterCacheError as e:
        return _stale_cluster_cache_error("get_feature_signal", e)
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
        signal_matrix = signal_matrix_payload(
            evidence_index,
            cluster_ids=cluster_ids,
            include_context_tier=include_context_tier,
            max_clusters=max_clusters,
            return_markdown=return_markdown,
        )
        omitted_ids = {
            int(item["cluster_id"])
            for item in signal_matrix.get("omitted_clusters", [])
        }
        universe = (
            [int(c) for c in cluster_ids]
            if cluster_ids is not None
            else [int(c["cluster_id"]) for c in cluster_meta.get("cluster_sizes", [])]
        )
        cluster_selection = build_cluster_selection(
            cluster_meta,
            [cid for cid in universe if cid not in omitted_ids],
            ranked_by="signal",
        )
        payload = {
            "status": "ok",
            "cluster_result": compact_cluster_result_payload(cluster_meta),
            "surface_labels_ref": "get_workflow_guide",
            "cluster_selection": cluster_selection,
            "signal_matrix": signal_matrix,
        }
        if return_markdown:
            return (
                cluster_selection_to_markdown(cluster_selection)
                + payload["signal_matrix"]["markdown_report"]
            )
        return json.dumps(payload, indent=2)
    except StaleClusterCacheError as e:
        return _stale_cluster_cache_error("get_cluster_signal_matrix", e)
    except Exception as e:
        return mcp_error("get_cluster_signal_matrix", str(e))


async def compare_clusters(cluster_a: int, cluster_b: int, ctx: Context = None) -> str:
    """Pairwise statistical tests between two clusters."""
    session = _get_session(ctx)

    if session.data is None or session.clusters is None:
        raise ToolError(
            "No data or clusters found. Run generate_cluster_dossier() first."
        )
    if session.clusters_run_id != session.latest_run_id:
        return _stale_cluster_cache_error(
            "compare_clusters",
            StaleClusterCacheError(
                cached_run_id=session.clusters_run_id,
                current_run_id=session.latest_run_id,
            ),
        )

    try:
        results = await asyncio.to_thread(
            _compare_clusters, session.data, session.clusters, cluster_a, cluster_b
        )
        if not results:
            return mcp_error(
                "compare_clusters",
                f"No results for comparison between clusters {cluster_a} and {cluster_b}.",
            )

        markdown = comparison_to_markdown(cluster_a, cluster_b, results)
        return markdown

    except Exception as e:
        logger.error(f"Error comparing clusters: {e}")
        return mcp_error("compare_clusters", str(e))
