from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import math
import threading
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from fastmcp import Context
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from mcp.types import TextContent
from pydantic import Field

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
from pulsar.mcp.registry import StagedWrite, registry
from pulsar.mcp.results import mcp_result
from pulsar.mcp.session import (
    OperationCancelledError,
    SessionConflictError,
    _get_session,
    is_latest,
    run_session_worker,
)
from pulsar.mcp.toolmeta import (
    BOOL,
    INT,
    NUM,
    STR,
    arr,
    enum,
    nullable,
    obj,
    tool_meta,
)
from pulsar.mcp.tools.diagnostics import (
    OPEN_OBJ,
    _tool_timeout_s,
    model_not_loaded_error,
    resolve_run_or_error,
    stale_write_error,
)

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
    cancel_event: threading.Event | None = None,
) -> tuple[Any, dict[str, Any], FeatureEvidenceIndex, str, Any, StagedWrite]:
    """Compute clusters/evidence/dossier. Never mutates ``session``.

    The cluster assignment is only *staged* (serialized + fsynced to a temp
    file; O(rows)). The caller commits it and publishes on the event loop
    (``_publish_dossier_on_loop``), or discards it if the publish is refused.
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
    staged_assignment = registry.stage_cluster_assignment(
        run_id=expected_run_id,
        dataset_id=data_dataset_id,
        method=str(cluster_meta["method_used"]),
        interpretation_edge_weight_threshold=resolved_interpretation_threshold,
        threshold_source=threshold_source,
        construction_threshold=construction_threshold,
        labels=result.labels.astype(int).tolist(),
    )
    # A cancel that arrived during staging would otherwise orphan the temp
    # file: the awaiting coroutine is gone, so nobody would discard it.
    if cancel_event is not None and cancel_event.is_set():
        registry.discard_staged(staged_assignment)
        raise OperationCancelledError("Cluster dossier cancelled before publication.")
    return result, cluster_meta, evidence_index, fingerprint, dossier, staged_assignment


def _publish_dossier_on_loop(
    session: Any,
    *,
    result: Any,
    cluster_meta: dict[str, Any],
    evidence_index: FeatureEvidenceIndex,
    fingerprint: str,
    staged_assignment: StagedWrite,
    expected_run_id: str | None,
    expected_data: Any,
    exclude_columns: list[str] | None,
) -> Any:
    """Commit the staged assignment and publish dossier state synchronously.

    The run and data checks precede the commit and no ``await`` separates them
    from ``publish_dossier``. A refused dossier discards its staged file, so
    nothing is persisted unless it is published.
    """
    try:
        if session.latest_run_id != expected_run_id:
            raise SessionConflictError(
                "Active run changed while computing cluster dossier; "
                "refusing to publish stale clusters."
            )
        # A rebind replaces session.data without changing the run id, so the
        # run check alone would publish labels computed on the old rows.
        if session.data is not expected_data:
            raise SessionConflictError(
                "Session data was rebound while computing cluster dossier; "
                "refusing to publish clusters computed on the previous dataset."
            )
        assignment = registry.commit_staged(staged_assignment)
    except BaseException:
        registry.discard_staged(staged_assignment)
        raise
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


def _stale_cluster_cache_error(tool: str, exc: StaleClusterCacheError) -> ToolResult:
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


def _clusters_not_built_error(tool: str) -> ToolResult:
    return mcp_error(
        tool,
        "No cluster evidence found in this session.",
        error_code="CLUSTERS_NOT_BUILT",
        agent_action="Run generate_cluster_dossier first (detail='summary' is enough).",
    )


def _clusters_stale_for_run_error(
    tool: str, session: Any, requested_run_id: str
) -> ToolResult:
    return mcp_error(
        tool,
        f"Cluster labels in this session were built for run "
        f"'{session.clusters_run_id}', not the requested run '{requested_run_id}'.",
        error_code="CLUSTERS_STALE",
        agent_action=(
            "Rebuild the dossier for that run: omit run_id here and call "
            "generate_cluster_dossier on the session's latest run, or pass "
            "run_id equal to clusters_run_id."
        ),
        details={
            "requested_run_id": requested_run_id,
            "clusters_run_id": session.clusters_run_id,
            "latest_run_id": session.latest_run_id,
        },
    )


def _require_cluster_state(
    session: Any, tool: str, run_id: str = ""
) -> tuple[tuple[FeatureEvidenceIndex, dict[str, Any]] | None, ToolResult | None]:
    """Return ``((evidence_index, cluster_meta), None)`` or ``(None, error)``.

    Cluster labels are session-scoped. With ``run_id`` given, they must have been
    built for that run (else CLUSTERS_STALE). Without it, they must match the
    session's latest run (else CLUSTER_CACHE_STALE).
    """
    if not run_id and session.model is None:
        return None, model_not_loaded_error(tool)
    if session.clusters is None or session.feature_evidence_index is None:
        return None, _clusters_not_built_error(tool)
    if run_id:
        if run_id != session.clusters_run_id:
            return None, _clusters_stale_for_run_error(tool, session, run_id)
    elif session.clusters_run_id != session.latest_run_id:
        # generate_cluster_dossier recomputes and refreshes the stamp, so this only
        # fires when a sweep ran after the last dossier (latest_run_id advanced).
        return None, _stale_cluster_cache_error(
            tool,
            StaleClusterCacheError(
                cached_run_id=session.clusters_run_id,
                current_run_id=session.latest_run_id,
            ),
        )
    return (
        session.feature_evidence_index,
        session.feature_evidence_cluster_meta or {},
    ), None


_CLUSTER_RESULT = OPEN_OBJ
_SELECTION = OPEN_OBJ
_CLUSTERS_RUN_ID = nullable(STR)
_RUN_ID_FOR_CLUSTERS = (
    "Optional guard: the run the cluster labels must have been built for. Cluster "
    "labels are session-scoped (they come from the last generate_cluster_dossier); "
    "if run_id differs from the dossier's run you get CLUSTERS_STALE. Omit to use "
    "the session's current clusters."
)

_DOSSIER_SCHEMA = obj(
    {
        "status": STR,
        "run_id": STR,
        "clusters_run_id": _CLUSTERS_RUN_ID,
        "construction_threshold": NUM,
        "interpretation_edge_weight_threshold": NUM,
        "threshold_inherited": BOOL,
        "threshold_source": STR,
        "threshold_surface": OPEN_OBJ,
        "cluster_assignment": OPEN_OBJ,
        "cluster_assignment_id": STR,
        "cluster_selection": _SELECTION,
        "clusters": arr(OPEN_OBJ),
        "recommended_next_tools": arr(STR),
        "detail_hint": STR,
        "surface_labels_ref": STR,
        "surface_labels": OPEN_OBJ,
    },
    [
        "status",
        "clusters_run_id",
        "interpretation_edge_weight_threshold",
        "threshold_surface",
        "cluster_assignment_id",
        "clusters",
        "recommended_next_tools",
    ],
    additional=True,  # summary/standard/full add differently shaped evidence blocks
)


@tool_meta(
    read_only=False,
    idempotent=False,
    output_schema=_DOSSIER_SCHEMA,
    timeout=_tool_timeout_s(),
)
async def generate_cluster_dossier(
    method: Annotated[
        Literal["auto", "spectral", "components"],
        Field(
            description=(
                "Clustering method. 'components' = connected components of the graph "
                "after dropping edges at or below the interpretation threshold. "
                "'spectral' = spectral cut of the weighted affinity (finds structure "
                "inside a connected graph; may return NO_STABLE_SPECTRAL_CUT). "
                "'auto' (default) uses components when a positive threshold applies, "
                "else spectral."
            )
        ),
    ] = "auto",
    max_k: Annotated[
        int,
        Field(
            description="Upper bound on the number of clusters searched by spectral. Default 15."
        ),
    ] = 15,
    interpretation_edge_weight_threshold: Annotated[
        float | None,
        Field(
            description=(
                "Edge-weight cutoff in [0, 1]; edges with weight <= this are dropped "
                "before clustering. Omit to inherit the run's construction threshold "
                "(spectral defaults to 0.0, the full affinity)."
            )
        ),
    ] = None,
    detail: Annotated[
        Literal["summary", "standard", "full"],
        Field(
            description=(
                "'summary' (default): compact per-cluster headline evidence, cheap, "
                "for routine loops. 'standard': adds per-cluster feature evidence "
                "(heavy). 'full': lossless audit payload (heaviest). Use the narrow "
                "cluster tools before escalating."
            )
        ),
    ] = "summary",
    max_clusters: Annotated[
        int,
        Field(
            description=(
                "Show at most this many clusters, largest first (>= 1); "
                "clusters_omitted reports the rest. Default 8."
            )
        ),
    ] = 8,
    feature_preview_limit: Annotated[
        int,
        Field(description="Feature hints shown per cluster (>= 0). Default 5."),
    ] = 5,
    response_format: Annotated[
        Literal["markdown", "json"],
        Field(
            description=(
                "Text rendering: 'markdown' (default) or 'json'. structured_content "
                "carries the same payload either way."
            )
        ),
    ] = "markdown",
    exclude_columns: Annotated[
        list[str] | None,
        Field(
            description=(
                "Column names to hide from the report. Display only: it does not "
                "change the fitted model or the clustering."
            )
        ),
    ] = None,
    run_id: Annotated[
        str,
        Field(
            description=(
                "Run to cluster, from run_topological_sweep. Omit for the latest run. "
                "This publishes cluster labels into the session and persists an "
                "assignment, so it must be the session's latest run; an older run_id "
                "is refused with SESSION_REVISION_CONFLICT."
            )
        ),
    ] = "",
    ctx: Context = None,
) -> ToolResult:
    """Cluster the fitted graph and build a statistical evidence dossier.

    Publishes the cluster labels used by get_cluster_profile, get_feature_signal,
    get_cluster_signal_matrix, compare_clusters and export_labeled_data, and
    persists a cluster_assignment_id. Start with detail='summary'; escalate only
    after narrowing focus.
    """
    tool = "generate_cluster_dossier"
    session = _get_session(ctx)

    resolved, err = resolve_run_or_error(session, tool, run_id)
    if err is not None:
        return err
    if run_id and not is_latest(session, resolved):
        return stale_write_error(tool, "cluster dossier")
    if resolved.data is None:
        return model_not_loaded_error(tool)
    if max_clusters < 1:
        return mcp_error(
            tool,
            f"max_clusters must be >= 1, got '{max_clusters}'",
            error_code="INVALID_ARGUMENT",
            agent_action="Pass max_clusters >= 1.",
        )
    if feature_preview_limit < 0:
        return mcp_error(
            tool,
            f"feature_preview_limit must be >= 0, got '{feature_preview_limit}'",
            error_code="INVALID_ARGUMENT",
            agent_action="Pass feature_preview_limit >= 0.",
        )

    try:
        model = resolved.model
        data = resolved.data
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

        cancel_event = threading.Event()
        (
            result,
            cluster_meta,
            evidence_index,
            fingerprint,
            dossier,
            staged_assignment,
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
            cancel_event,
            heavy=True,
            mutates_session=True,
            cancel_event=cancel_event,
        )
        assignment = _publish_dossier_on_loop(
            session,
            result=result,
            cluster_meta=cluster_meta,
            evidence_index=evidence_index,
            fingerprint=fingerprint,
            staged_assignment=staged_assignment,
            expected_run_id=expected_run_id,
            expected_data=data,
            exclude_columns=exclude_columns,
        )
        assignment_summary = _cluster_assignment_summary(assignment)
        header_note = _two_lever_markdown_note(
            construction_threshold, resolved_interpretation_threshold
        ) + _cluster_assignment_markdown(assignment)
        if detail == "summary":
            # JSON callers get a tighter per-cluster preview (<= 3); the markdown
            # view honours feature_preview_limit. The payload shape is identical.
            summary_preview_limit = (
                min(feature_preview_limit, 3)
                if response_format == "json"
                else feature_preview_limit
            )
            payload = build_summary_evidence_payload(
                evidence_index,
                cluster_meta,
                _graph_metrics_payload(model),
                max_clusters=max_clusters,
                feature_preview_limit=summary_preview_limit,
            )
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
            text = (
                header_note
                + cluster_selection_to_markdown(payload["cluster_selection"])
                + summary_evidence_payload_to_markdown(payload)
            )
        else:
            payload = build_evidence_payload(
                dossier,
                cluster_meta,
                detail=detail,
                max_clusters=max_clusters,
                feature_preview_limit=feature_preview_limit,
            )
            if detail != "full":
                payload["surface_labels_ref"] = "get_workflow_guide"
            else:
                payload["surface_labels"] = _SURFACE_LABELS
            payload["cluster_selection"] = build_cluster_selection(
                cluster_meta,
                [c["cluster_id"] for c in payload["clusters"]],
                ranked_by="size",
            )
            payload["recommended_next_tools"] = [
                "get_feature_signal",
                "compare_clusters",
                "export_html_report",
            ]
            payload["detail_hint"] = (
                "Use targeted evidence tools for routine loops; detail='full' is the "
                "lossless audit payload."
            )
            text = header_note + dossier_to_markdown(dossier)
        payload["run_id"] = resolved.run_id
        payload["clusters_run_id"] = session.clusters_run_id
        payload["construction_threshold"] = construction_threshold
        payload["interpretation_edge_weight_threshold"] = (
            resolved_interpretation_threshold
        )
        payload["threshold_inherited"] = threshold_inherited
        payload["threshold_source"] = threshold_source
        payload["threshold_surface"] = threshold_surface
        payload["cluster_assignment"] = assignment_summary
        payload["cluster_assignment_id"] = assignment.cluster_assignment_id
        if response_format == "markdown":
            return mcp_result(
                payload, text=_prepend_threshold_markdown(text, threshold_surface)
            )
        return mcp_result(payload)

    except SpectralClusterCutError as e:
        logger.info("No stable spectral cluster cut: %s", e.diagnostics)
        cut = e.payload(detail=detail)
        error = mcp_error(
            tool,
            cut["reason"],
            error_code=cut["error_code"],
            agent_action=(
                "Try method='components' with an explicit "
                "interpretation_edge_weight_threshold (see "
                "get_threshold_stability_curve), or treat the graph as one "
                "continuous structure."
            ),
            details=cut["details"],
        )
        if response_format == "markdown":
            return ToolResult(
                content=[
                    TextContent(type="text", text=_spectral_cut_error_to_markdown(e))
                ],
                structured_content=error.structured_content,
                is_error=True,
            )
        return error
    except SessionConflictError as e:
        logger.info("Dossier refused: %s", e)
        return mcp_error(
            tool,
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
        return mcp_error(
            tool,
            str(e),
            error_code="DOSSIER_FAILED",
            agent_action=(
                "Check the reason, adjust method / interpretation threshold, and retry."
            ),
        )


_PROFILE_SCHEMA = obj(
    {
        "status": enum(["ok"]),
        "clusters_run_id": _CLUSTERS_RUN_ID,
        "cluster_result": _CLUSTER_RESULT,
        "detail": enum(["summary", "standard", "full"]),
        "max_features": INT,
        "cluster": OPEN_OBJ,
        "surface_labels_ref": STR,
        "surface_labels": OPEN_OBJ,
    },
    ["status", "clusters_run_id", "cluster_result", "detail", "cluster"],
)


@tool_meta(read_only=True, idempotent=True, output_schema=_PROFILE_SCHEMA)
async def get_cluster_profile(
    cluster_id: Annotated[
        int,
        Field(
            description=(
                "Integer cluster ID as listed by generate_cluster_dossier "
                "(cluster_selection.shown_ids). IDs are stable within one dossier."
            )
        ),
    ],
    detail: Annotated[
        Literal["summary", "standard", "full"],
        Field(
            description=(
                "'summary': headline features only. 'standard' (default): ranked "
                "numeric and categorical evidence up to max_features. 'full': "
                "lossless evidence plus full cluster-size telemetry."
            )
        ),
    ] = "standard",
    max_features: Annotated[
        int,
        Field(
            description="Maximum feature rows returned for this cluster (>= 1). Default 16."
        ),
    ] = 16,
    response_format: Annotated[
        Literal["markdown", "json"],
        Field(
            description=(
                "Text rendering: 'markdown' (default) or 'json'. structured_content "
                "carries the same payload either way."
            )
        ),
    ] = "markdown",
    run_id: Annotated[str, Field(description=_RUN_ID_FOR_CLUSTERS)] = "",
    ctx: Context = None,
) -> ToolResult:
    """Evidence for one cluster (feature shifts versus the rest) from the cached dossier.

    Reads the session's cluster labels; no recomputation. Use after
    generate_cluster_dossier to drill into one cluster; use get_feature_signal to
    follow one feature across clusters.
    """
    tool = "get_cluster_profile"
    try:
        if max_features < 1:
            return mcp_error(
                tool,
                f"max_features must be >= 1, got '{max_features}'",
                error_code="INVALID_ARGUMENT",
                agent_action="Pass max_features >= 1.",
            )

        session = _get_session(ctx)
        state, err = _require_cluster_state(session, tool, run_id)
        if err is not None:
            return err
        evidence_index, cluster_meta = state
        payload = {
            "status": "ok",
            "clusters_run_id": session.clusters_run_id,
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
            return mcp_result(
                payload, text=note + cluster_profile_payload_to_markdown(payload)
            )
        return mcp_result(payload)
    except KeyError:
        return mcp_error(
            tool,
            f"Unknown cluster_id '{cluster_id}'.",
            error_code="CLUSTER_ID_UNKNOWN",
            agent_action="Use generate_cluster_dossier first and inspect available cluster IDs.",
        )
    except Exception as e:
        return mcp_error(
            tool,
            str(e),
            error_code="CLUSTER_PROFILE_FAILED",
            agent_action="Re-run generate_cluster_dossier and retry with a listed cluster_id.",
        )


_SIGNAL_SCHEMA = obj(
    {
        "status": enum(["ok"]),
        "clusters_run_id": _CLUSTERS_RUN_ID,
        "cluster_result": _CLUSTER_RESULT,
        "cluster_selection": _SELECTION,
        "signals": OPEN_OBJ,
        "surface_labels_ref": STR,
        "surface_labels": OPEN_OBJ,
    },
    ["status", "clusters_run_id", "cluster_result", "cluster_selection", "signals"],
)


@tool_meta(read_only=True, idempotent=True, output_schema=_SIGNAL_SCHEMA)
async def get_feature_signal(
    feature_names: Annotated[
        list[str],
        Field(
            description=(
                "Features to follow across clusters: numeric column names, or "
                "'column=value' pairs for categorical levels (e.g. 'segment=alpha')."
            )
        ),
    ],
    cluster_ids: Annotated[
        list[int] | None,
        Field(
            description=(
                "Restrict to these cluster IDs. Omit to use the top max_clusters "
                "clusters ranked by signal."
            )
        ),
    ] = None,
    detail: Annotated[
        Literal["summary", "standard", "full"],
        Field(
            description=(
                "'summary' (default): per-cluster signal headline for each feature. "
                "'standard'/'full': progressively more per-cluster statistics; "
                "'full' also returns full cluster-size telemetry."
            )
        ),
    ] = "summary",
    max_clusters: Annotated[
        int,
        Field(
            description="Show at most this many clusters, ranked by signal (>= 1). Default 8."
        ),
    ] = 8,
    response_format: Annotated[
        Literal["markdown", "json"],
        Field(
            description=(
                "Text rendering: 'markdown' (default) or 'json'. structured_content "
                "carries the same payload either way."
            )
        ),
    ] = "markdown",
    run_id: Annotated[str, Field(description=_RUN_ID_FOR_CLUSTERS)] = "",
    ctx: Context = None,
) -> ToolResult:
    """Cross-cluster evidence for specific features from the cached dossier.

    Returns, per requested feature, how strongly each cluster departs from the
    rest. Use to test a hypothesis about particular columns; use
    get_cluster_signal_matrix for an all-features overview.
    """
    tool = "get_feature_signal"
    if max_clusters < 1:
        return mcp_error(
            tool,
            f"max_clusters must be >= 1, got '{max_clusters}'",
            error_code="INVALID_ARGUMENT",
            agent_action="Pass max_clusters >= 1.",
        )
    try:
        session = _get_session(ctx)
        state, err = _require_cluster_state(session, tool, run_id)
        if err is not None:
            return err
        evidence_index, cluster_meta = state
        signals = feature_signal_payload(
            evidence_index,
            feature_names,
            cluster_ids=cluster_ids,
            detail=detail,
            max_clusters=max_clusters,
        )
        payload = {
            "status": "ok",
            "clusters_run_id": session.clusters_run_id,
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
            return mcp_result(
                payload, text=feature_signal_payload_to_markdown(payload["signals"])
            )
        return mcp_result(payload)
    except Exception as e:
        return mcp_error(
            tool,
            str(e),
            error_code="FEATURE_SIGNAL_FAILED",
            agent_action=(
                "Check feature_names against the dataset columns (use "
                "probe_columns) and cluster_ids against the dossier, then retry."
            ),
        )


_MATRIX_SCHEMA = obj(
    {
        "status": enum(["ok"]),
        "clusters_run_id": _CLUSTERS_RUN_ID,
        "cluster_result": _CLUSTER_RESULT,
        "surface_labels_ref": STR,
        "cluster_selection": _SELECTION,
        "signal_matrix": obj(
            {
                "numeric_columns": arr(STR),
                "categorical_values": arr(OPEN_OBJ),
                "clusters_returned": INT,
                "clusters_omitted": INT,
                "omitted_clusters": arr(OPEN_OBJ),
                "numeric_features_omitted": INT,
                "categorical_features_omitted": INT,
                "numeric_rows": arr(OPEN_OBJ),
                "categorical_rows": arr(OPEN_OBJ),
            },
            ["clusters_returned", "clusters_omitted"],
            additional=True,
        ),
    },
    [
        "status",
        "clusters_run_id",
        "cluster_result",
        "cluster_selection",
        "signal_matrix",
    ],
)


@tool_meta(read_only=True, idempotent=True, output_schema=_MATRIX_SCHEMA)
async def get_cluster_signal_matrix(
    cluster_ids: Annotated[
        list[int] | None,
        Field(
            description=(
                "Restrict the matrix to these cluster IDs. Omit for all clusters, "
                "truncated to max_clusters by signal."
            )
        ),
    ] = None,
    include_context_tier: Annotated[
        bool,
        Field(
            description=(
                "False (default) shows only core and supporting signal tiers. True "
                "also includes weaker 'context' tier rows."
            )
        ),
    ] = False,
    max_clusters: Annotated[
        int,
        Field(
            description=(
                "Show at most this many clusters, ranked by signal (>= 1). "
                "Presentation-only truncation. Default 8."
            )
        ),
    ] = 8,
    return_markdown: Annotated[
        bool,
        Field(
            description=(
                "True (default): text is a compact markdown matrix (cells are "
                "z-scores for numeric features, prevalence percent for categoricals). "
                "False: text is JSON. structured_content is the same either way."
            )
        ),
    ] = True,
    run_id: Annotated[str, Field(description=_RUN_ID_FOR_CLUSTERS)] = "",
    ctx: Context = None,
) -> ToolResult:
    """Cluster-by-feature signal matrix from the cached dossier.

    Shows which features distinguish which clusters in one dense table, ranked by
    signal. Use for a cross-cluster overview; use get_feature_signal for chosen
    features and get_cluster_profile for one cluster.
    """
    tool = "get_cluster_signal_matrix"
    if max_clusters < 1:
        return mcp_error(
            tool,
            f"max_clusters must be >= 1, got '{max_clusters}'",
            error_code="INVALID_ARGUMENT",
            agent_action="Pass max_clusters >= 1.",
        )
    try:
        session = _get_session(ctx)
        state, err = _require_cluster_state(session, tool, run_id)
        if err is not None:
            return err
        evidence_index, cluster_meta = state
        signal_matrix = signal_matrix_payload(
            evidence_index,
            cluster_ids=cluster_ids,
            include_context_tier=include_context_tier,
            max_clusters=max_clusters,
            return_markdown=False,
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
            "clusters_run_id": session.clusters_run_id,
            "cluster_result": compact_cluster_result_payload(cluster_meta),
            "surface_labels_ref": "get_workflow_guide",
            "cluster_selection": cluster_selection,
            "signal_matrix": signal_matrix,
        }
        if return_markdown:
            report = signal_matrix_payload(
                evidence_index,
                cluster_ids=cluster_ids,
                include_context_tier=include_context_tier,
                max_clusters=max_clusters,
                return_markdown=True,
            )["markdown_report"]
            return mcp_result(
                payload,
                text=cluster_selection_to_markdown(cluster_selection) + report,
            )
        return mcp_result(payload)
    except Exception as e:
        return mcp_error(
            tool,
            str(e),
            error_code="SIGNAL_MATRIX_FAILED",
            agent_action="Re-run generate_cluster_dossier and retry with listed cluster_ids.",
        )


def _finite_or_none(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


_COMPARE_SCHEMA = obj(
    {
        "status": enum(["ok"]),
        "clusters_run_id": _CLUSTERS_RUN_ID,
        "cluster_a": INT,
        "cluster_b": INT,
        "features_compared": INT,
        "features_omitted": INT,
        "results": arr(
            obj(
                {
                    "column": STR,
                    "mean_a": nullable(NUM),
                    "mean_b": nullable(NUM),
                    "p_val_t": nullable(NUM),
                    "p_val_ks": nullable(NUM),
                    "cohens_d": nullable(NUM),
                },
                ["column"],
            )
        ),
    },
    [
        "status",
        "clusters_run_id",
        "cluster_a",
        "cluster_b",
        "features_compared",
        "features_omitted",
        "results",
    ],
)


@tool_meta(read_only=True, idempotent=True, output_schema=_COMPARE_SCHEMA)
async def compare_clusters(
    cluster_a: Annotated[
        int,
        Field(description="First cluster ID, as listed by generate_cluster_dossier."),
    ],
    cluster_b: Annotated[
        int,
        Field(description="Second cluster ID, as listed by generate_cluster_dossier."),
    ],
    max_features: Annotated[
        int,
        Field(
            description=(
                "Return at most this many numeric features, ranked by |Cohen's d| "
                "(>= 1); features_omitted reports the rest. Default 20."
            )
        ),
    ] = 20,
    run_id: Annotated[str, Field(description=_RUN_ID_FOR_CLUSTERS)] = "",
    ctx: Context = None,
) -> ToolResult:
    """Pairwise statistical comparison of two clusters over numeric features.

    Per feature: both means, Welch t-test and KS-test p-values, and Cohen's d,
    ranked by effect size. Use after generate_cluster_dossier to contrast two
    specific clusters; p-values are uncorrected for multiple comparisons.
    """
    tool = "compare_clusters"
    session = _get_session(ctx)
    if max_features < 1:
        return mcp_error(
            tool,
            f"max_features must be >= 1, got '{max_features}'",
            error_code="INVALID_ARGUMENT",
            agent_action="Pass max_features >= 1.",
        )
    if not run_id and session.model is None:
        return model_not_loaded_error(tool)
    if session.clusters is None:
        return _clusters_not_built_error(tool)
    if run_id:
        if run_id != session.clusters_run_id:
            return _clusters_stale_for_run_error(tool, session, run_id)
        resolved, err = resolve_run_or_error(session, tool, run_id)
        if err is not None:
            return err
        data = resolved.data
    else:
        if session.clusters_run_id != session.latest_run_id:
            return _stale_cluster_cache_error(
                tool,
                StaleClusterCacheError(
                    cached_run_id=session.clusters_run_id,
                    current_run_id=session.latest_run_id,
                ),
            )
        data = session.data
    if data is None:
        return model_not_loaded_error(tool)

    try:
        results = await asyncio.to_thread(
            _compare_clusters, data, session.clusters, cluster_a, cluster_b
        )
        if not results:
            return mcp_error(
                tool,
                f"No results for comparison between clusters {cluster_a} and {cluster_b}.",
                error_code="COMPARISON_EMPTY",
                agent_action=(
                    "Both clusters need at least 2 rows and numeric columns with "
                    "at least 2 non-missing values; check IDs with "
                    "generate_cluster_dossier."
                ),
                details={"cluster_a": cluster_a, "cluster_b": cluster_b},
            )

        shown = results[:max_features]
        rows = [
            {
                "column": str(r["column"]),
                **{
                    key: _finite_or_none(r[key])
                    for key in ("mean_a", "mean_b", "p_val_t", "p_val_ks", "cohens_d")
                },
            }
            for r in shown
        ]
        payload = {
            "status": "ok",
            "clusters_run_id": session.clusters_run_id,
            "cluster_a": int(cluster_a),
            "cluster_b": int(cluster_b),
            "features_compared": len(results),
            "features_omitted": len(results) - len(shown),
            "results": rows,
        }
        text = comparison_to_markdown(cluster_a, cluster_b, shown)
        if len(results) > len(shown):
            text += (
                f"\n\n_{len(results) - len(shown)} further features omitted; "
                "raise max_features to see them._"
            )
        return mcp_result(payload, text=text)

    except Exception as e:
        logger.error(f"Error comparing clusters: {e}")
        return mcp_error(
            tool,
            str(e),
            error_code="COMPARE_FAILED",
            agent_action="Check cluster IDs with generate_cluster_dossier and retry.",
        )
