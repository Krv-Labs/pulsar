from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any, TypeVar

import pandas as pd
from fastmcp import Context
from fastmcp.exceptions import ToolError

from pulsar.pipeline import ThemaRS
from pulsar.representations import CosmicTrajectory, TemporalCosmicGraph
from pulsar.mcp.registry import registry
from pulsar.mcp.interpreter import FeatureEvidenceIndex

logger = logging.getLogger(__name__)

T = TypeVar("T")
_MAX_CONCURRENT_HEAVY = max(
    1, int(os.environ.get("PULSAR_MCP_MAX_CONCURRENT_HEAVY", "1"))
)
_HEAVY_COMPUTE_SEMAPHORE = threading.BoundedSemaphore(_MAX_CONCURRENT_HEAVY)


class OperationCancelledError(RuntimeError):
    """Raised when a cooperative cancellation checkpoint fires in a worker thread."""


@dataclass
class SweepRecord:
    timestamp: float
    config_yaml: str
    metrics: dict
    dataset_id: str | None = None


@dataclass
class GraphArtifact:
    artifact_id: str
    run_id: str
    kind: str
    created_at: float
    params: dict[str, Any]
    graph_summary: dict[str, Any]
    metrics: dict[str, Any]


@dataclass
class LongitudinalArtifact:
    """A built longitudinal panel plus whichever representations were requested.

    Lives in the session only, like the fitted ``ThemaRS`` model — the sparse
    matrices and frames do not round-trip through the registry cheaply.
    """

    longitudinal_id: str
    dataset_id: str | None
    representation: str  # "trajectory" | "temporal" | "both"
    trajectory: CosmicTrajectory | None
    temporal: TemporalCosmicGraph | None
    panel: dict[str, Any]
    config_yaml: str
    created_at: float


@dataclass
class _PulsarSession:
    """Session state for a single MCP client."""

    model: ThemaRS | None = None
    data: pd.DataFrame | None = None
    clusters: pd.Series | None = None
    embeddings: list | None = None  # cached projection output from last fit
    projection_fingerprint: str | None = None
    pca_fingerprint: str | None = None  # Legacy alias for projection_fingerprint.
    sweep_history: list[SweepRecord] = field(default_factory=list)
    dataset_id: str | None = None
    latest_run_id: str | None = None
    report_exclude_columns: list[str] | None = None
    data_dataset_id: str | None = None
    data_path: str | None = None
    feature_evidence_index: FeatureEvidenceIndex | None = None
    feature_evidence_fingerprint: str | None = None
    feature_evidence_cluster_meta: dict[str, Any] | None = None
    # run_id of the fitted model the cached cluster state was computed from.
    # A newer sweep advances `latest_run_id` but does NOT recompute clusters,
    # so this stamp lets reads detect (and reject) stale cluster caches.
    clusters_run_id: str | None = None
    cluster_assignment_id: str | None = None
    active_config_yaml: str | None = None
    active_config_dataset_id: str | None = None
    graph_artifacts: dict[str, GraphArtifact] = field(default_factory=dict)
    longitudinal: dict[str, LongitudinalArtifact] = field(default_factory=dict)
    revision: int = 0
    active_ops: int = 0
    _worker_mutation_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def publish_sweep(
        self,
        *,
        expected_revision: int,
        config_yaml: str | None = None,
        config_dataset_id: str | None = None,
        model: ThemaRS,
        data: pd.DataFrame,
        dataset_id: str | None = None,
        data_path: str | None = None,
        dataset_path: str | None = None,
        dataset_fingerprint: str | None = None,
        embeddings: list[Any] | None = None,
        projection_fingerprint: str | None = None,
        pca_fingerprint: str | None = None,
        record: SweepRecord,
        run_id: str,
    ) -> None:
        if self.revision != expected_revision:
            raise RuntimeError(
                "Session state changed during sweep; refusing to publish stale sweep revision."
            )
        if config_yaml is not None:
            self.active_config_yaml = config_yaml
            self.active_config_dataset_id = config_dataset_id
        self.model = model
        _bind_session_data(
            self,
            data,
            dataset_id=dataset_id,
            data_path=data_path if data_path is not None else dataset_path,
        )
        self.embeddings = embeddings
        self.projection_fingerprint = projection_fingerprint
        self.pca_fingerprint = (
            pca_fingerprint if pca_fingerprint is not None else projection_fingerprint
        )
        self.sweep_history.append(record)
        self.latest_run_id = run_id
        self.revision += 1

    def publish_dossier(
        self,
        *,
        expected_run_id: str | None,
        clusters: pd.Series,
        report_exclude_columns: list[str] | None,
        feature_evidence_cluster_meta: dict[int, dict[str, Any]] | dict[str, Any],
        cluster_assignment_id: str | None = None,
        feature_evidence_index: FeatureEvidenceIndex | None = None,
        feature_evidence_fingerprint: str | None = None,
    ) -> None:
        if self.latest_run_id != expected_run_id:
            raise RuntimeError(
                "Active run changed while computing cluster dossier; refusing to publish stale clusters."
            )
        self.clusters = clusters
        self.clusters_run_id = expected_run_id
        self.report_exclude_columns = (
            list(report_exclude_columns)
            if report_exclude_columns is not None
            else None
        )
        self.feature_evidence_cluster_meta = dict(feature_evidence_cluster_meta)
        if cluster_assignment_id is not None:
            self.cluster_assignment_id = cluster_assignment_id
        if feature_evidence_index is not None:
            self.feature_evidence_index = feature_evidence_index
        if feature_evidence_fingerprint is not None:
            self.feature_evidence_fingerprint = feature_evidence_fingerprint
        self.revision += 1

    def publish_graph_artifact(
        self,
        artifact: GraphArtifact,
        *,
        expected_run_id: str | None,
    ) -> None:
        if self.latest_run_id != expected_run_id:
            raise RuntimeError(
                "Active run changed while building graph artifact; refusing to publish stale artifact."
            )
        self.graph_artifacts[artifact.artifact_id] = artifact
        self.revision += 1

    def publish_longitudinal(self, artifact: LongitudinalArtifact) -> None:
        self.longitudinal[artifact.longitudinal_id] = artifact
        self.revision += 1

    def calculate_memory_mb(self) -> float:
        """Estimate current session memory footprint in MB."""
        bytes_total = (
            int(self.data.memory_usage(deep=True).sum())
            if self.data is not None
            else 0
        )
        bytes_total += _model_memory_bytes(self.model)
        bytes_total += _embeddings_memory_bytes(self.embeddings)
        bytes_total += _longitudinal_memory_bytes(self.longitudinal)
        return round(float(bytes_total) / (1024 * 1024), 2)


def _model_memory_bytes(model: ThemaRS | None) -> int:
    if model is None:
        return 0
    if model._weighted_adjacency is not None:
        return int(model._weighted_adjacency.nbytes)
    if model._cosmic_rust is not None:
        return int(model._cosmic_rust.n_edges * 24)
    return 0


def _embeddings_memory_bytes(embeddings: list | None) -> int:
    if not embeddings:
        return 0
    return sum(int(getattr(emb, "nbytes", 0)) for emb in embeddings)


def _longitudinal_memory_bytes(
    longitudinal: dict[str, LongitudinalArtifact],
) -> int:
    total = 0
    for artifact in longitudinal.values():
        if artifact.trajectory is not None:
            for matrix in (
                artifact.trajectory.similarity,
                artifact.trajectory.incidence,
            ):
                total += int(matrix.data.nbytes + matrix.indices.nbytes)
            total += int(artifact.trajectory.obs.memory_usage(deep=True).sum())
        if artifact.temporal is not None:
            total += int(artifact.temporal.tensor.nbytes)
    return total


_sessions: OrderedDict[str, _PulsarSession] = OrderedDict()
_MAX_SESSIONS = int(os.environ.get("PULSAR_MAX_SESSIONS", "3"))


def _execute_in_session_worker(
    session: _PulsarSession,
    fn: Callable[..., T],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    heavy: bool,
    mutates_session: bool,
    cancel_event: threading.Event | None,
) -> T:
    try:
        with ExitStack() as stack:
            if mutates_session:
                stack.enter_context(session._worker_mutation_lock)
            if heavy:
                stack.enter_context(_HEAVY_COMPUTE_SEMAPHORE)
            if cancel_event is not None and cancel_event.is_set():
                raise OperationCancelledError(
                    "Operation cancelled before start."
                )
            return fn(*args, **kwargs)
    finally:
        session.active_ops = max(0, session.active_ops - 1)


async def run_session_worker(
    session: _PulsarSession,
    fn: Callable[..., T],
    *args: Any,
    heavy: bool = True,
    mutates_session: bool = False,
    cancel_event: threading.Event | None = None,
    **kwargs: Any,
) -> T:
    """Run blocking work in a worker thread with session/admission guards.

    Locking and semaphore acquisition happen inside the worker thread so the
    asyncio event loop is never blocked by ``threading.Lock`` or semaphore waits.
    """
    session.active_ops += 1
    try:
        return await asyncio.to_thread(
            _execute_in_session_worker,
            session,
            fn,
            args,
            kwargs,
            heavy=heavy,
            mutates_session=mutates_session,
            cancel_event=cancel_event,
        )
    except asyncio.CancelledError:
        if cancel_event is not None:
            cancel_event.set()
        raise


def _extract_context_session_id(ctx: Context) -> str:
    from fastmcp.server.dependencies import get_http_headers

    rc = ctx.request_context
    sess = getattr(rc, "session", None) or getattr(ctx, "_session", None)
    sid = getattr(getattr(sess, "_connection", None), "session_id", None)
    if isinstance(sid, str) and sid:
        return sid
    headers = getattr(getattr(rc, "request", None), "headers", None)
    header_sid = (
        headers.get("mcp-session-id")
        if headers is not None
        else get_http_headers(include={"mcp-session-id"}).get("mcp-session-id")
    )
    return header_sid if isinstance(header_sid, str) and header_sid else "default"


def _session_key(ctx: Context | None) -> str:
    """Get the session key from context (session_id or 'default' for STDIO)."""
    if ctx is None:
        return "default"
    if isinstance(ctx, Context):
        return _extract_context_session_id(ctx)
    sid = getattr(ctx, "session_id", None)
    return sid if isinstance(sid, str) and sid else "default"


def _get_session(ctx: Context) -> _PulsarSession:
    """Get or create session state for the current client with LRU eviction."""
    key = _session_key(ctx)

    if key in _sessions:
        # Move to end (mark as recently used)
        _sessions.move_to_end(key)
        return _sessions[key]

    # Create new session
    if len(_sessions) >= _MAX_SESSIONS:
        evict_key = next((k for k, s in _sessions.items() if s.active_ops == 0), None)
        if evict_key is not None:
            evicted_session = _sessions.pop(evict_key)
            logger.info(
                "Evicting oldest idle session '%s' to free memory (Max sessions: %d)",
                evict_key,
                _MAX_SESSIONS,
            )
            # Cleanup
            del evicted_session
            gc.collect()

    session = _PulsarSession()
    _sessions[key] = session
    return session


def _resolve_dataset_record(dataset_id: str):
    record = registry.get_dataset(dataset_id)
    if record is None:
        raise LookupError(dataset_id)
    return record


def _resolve_longitudinal(
    session: _PulsarSession, longitudinal_id: str
) -> LongitudinalArtifact:
    """Resolve a longitudinal handle, or raise LookupError for the error envelope.

    Falls back to the most recent artifact when the caller omits the id, so a
    single-panel session does not have to thread the id through every call.
    """
    if not longitudinal_id:
        if not session.longitudinal:
            raise LookupError("")
        return next(reversed(session.longitudinal.values()))
    artifact = session.longitudinal.get(longitudinal_id)
    if artifact is None:
        raise LookupError(longitudinal_id)
    return artifact


def _resolve_dataset_path(dataset_id: str) -> str:
    return _resolve_dataset_record(dataset_id).path


def _normalize_data_path(path: str) -> str:
    expanded = Path(path).expanduser()
    return str(expanded.resolve(strict=False))


def _read_dataset_file(path: str) -> pd.DataFrame:
    normalized_path = _normalize_data_path(path)
    if normalized_path.lower().endswith(".parquet"):
        return pd.read_parquet(normalized_path)
    return pd.read_csv(normalized_path)


def _bind_session_data(
    session: _PulsarSession,
    df: pd.DataFrame,
    *,
    dataset_id: str | None = None,
    data_path: str | None = None,
) -> None:
    session.data = df
    session.data_dataset_id = dataset_id
    session.data_path = _normalize_data_path(data_path) if data_path else None
    if dataset_id:
        session.dataset_id = dataset_id
    # Canonical invalidation point: rebinding the underlying data invalidates
    # any cached feature evidence, which was computed against the previous
    # DataFrame (the evidence fingerprint does not cover the data itself).
    _invalidate_feature_evidence_cache(session)


def _dataset_id_for_path(dataset_id: str | None, data_path: str | None) -> str | None:
    if not dataset_id or not data_path:
        return None

    try:
        dataset_path = _normalize_data_path(_resolve_dataset_path(dataset_id))
    except LookupError:
        return None

    normalized_path = _normalize_data_path(data_path)
    return dataset_id if dataset_path == normalized_path else None


def _resolve_dataframe_source(
    dataset_id: str, data_path: str
) -> tuple[str | None, str]:
    if dataset_id:
        resolved_path = _resolve_dataset_path(dataset_id)
        return dataset_id, _normalize_data_path(resolved_path)
    if not data_path:
        raise ToolError("Provide either data_path or dataset_id.")
    return None, _normalize_data_path(data_path)


def _try_cached_session_dataframe(
    session: _PulsarSession,
    target_dataset_id: str | None,
    normalized_path: str,
) -> pd.DataFrame | None:
    if (
        session.data is None
        or session.data_dataset_id != target_dataset_id
        or session.data_path != normalized_path
    ):
        return None
    if target_dataset_id is not None:
        session.dataset_id = target_dataset_id
    return session.data


async def _load_session_dataframe(
    session: _PulsarSession,
    *,
    dataset_id: str = "",
    data_path: str = "",
) -> tuple[pd.DataFrame, str]:
    target_dataset_id, normalized_path = _resolve_dataframe_source(
        dataset_id, data_path
    )
    cached_df = _try_cached_session_dataframe(
        session, target_dataset_id, normalized_path
    )
    if cached_df is not None:
        return cached_df, normalized_path

    df = await run_session_worker(
        session,
        _read_dataset_file,
        normalized_path,
        heavy=False,
        mutates_session=False,
    )
    with session._worker_mutation_lock:
        _bind_session_data(
            session,
            df,
            dataset_id=target_dataset_id,
            data_path=normalized_path,
        )
    return df, normalized_path


def _invalidate_feature_evidence_cache(session: _PulsarSession) -> None:
    session.feature_evidence_index = None
    session.feature_evidence_fingerprint = None
    session.feature_evidence_cluster_meta = None
    session.clusters = None
    session.clusters_run_id = None
    session.cluster_assignment_id = None


def _feature_evidence_fingerprint(
    *,
    run_id: str | None,
    labels: pd.Series,
    method: str,
    interpretation_edge_weight_threshold: float,
    exclude_columns: list[str] | None,
) -> str:
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


def _graph_health_summary(metrics: dict[str, Any]) -> tuple[str, bool, str]:
    density = float(metrics.get("density", 0.0))
    component_count = int(metrics.get("component_count", 0))
    singleton_fraction = float(metrics.get("singleton_fraction", 0.0))
    is_connected = component_count <= 1
    if density > 0.8:
        return (
            "hairball",
            is_connected,
            "Refine epsilon downward or raise threshold before clustering.",
        )
    if singleton_fraction > 0.25 or component_count > max(
        3, int(metrics.get("n_nodes", 0) * 0.1)
    ):
        return (
            "fragmented",
            is_connected,
            "Increase epsilon support or fall back to component-based clustering.",
        )
    return (
        "connected",
        is_connected,
        "Proceed to clustering; graph structure is suitable for interpretive analysis.",
    )


def _projection_cache_status(
    session: _PulsarSession,
    cfg: Any,
) -> tuple[list | None, dict[str, Any]]:
    """Return reusable projection embeddings and an operational status payload."""
    from pulsar.runtime.fingerprint import projection_fingerprint

    method = getattr(getattr(cfg, "projection", None), "method", "jl")
    status: dict[str, Any] = {
        "scope": "session",
        "artifact": "projection_embeddings",
        "method": method,
        "status": "miss",
        "reason": "no_cached_embeddings",
    }
    if session.embeddings is None:
        return None, status
    if session.data is None:
        status["reason"] = "no_session_data"
        return None, status
    cached_fingerprint = session.projection_fingerprint or session.pca_fingerprint
    if cached_fingerprint is None:
        status["reason"] = "no_cached_fingerprint"
        return None, status

    fingerprint = projection_fingerprint(cfg, len(session.data), session.data)
    if fingerprint != cached_fingerprint:
        status["reason"] = "fingerprint_mismatch"
        return None, status

    return (
        session.embeddings,
        {
            "scope": "session",
            "artifact": "projection_embeddings",
            "method": method,
            "status": "hit",
            "reason": "fingerprint_match",
        },
    )


def _pca_cache_status(
    session: _PulsarSession,
    cfg: Any,
) -> tuple[list | None, dict[str, Any]]:
    """Compatibility alias for the projection embedding cache status."""
    return _projection_cache_status(session, cfg)
