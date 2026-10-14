from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Callable
import contextvars
from dataclasses import dataclass, field
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any, TypeVar
import weakref

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
# Heavy-compute admission is an asyncio.Semaphore owned by the event loop, so a
# queued request waits as a coroutine instead of pinning an executor thread.
# asyncio primitives bind to a loop, hence one semaphore per running loop.
_heavy_semaphores: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, asyncio.Semaphore
] = weakref.WeakKeyDictionary()


def _heavy_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _heavy_semaphores.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(_MAX_CONCURRENT_HEAVY)
        _heavy_semaphores[loop] = sem
    return sem


class OperationCancelledError(RuntimeError):
    """Raised when a cooperative cancellation checkpoint fires in a worker thread."""


class SessionConflictError(RuntimeError):
    """A loop-side publish was refused because the session moved on.

    Carries a stable ``error_code`` and ``agent_action`` so tools can return the
    standard MCP error envelope instead of a bare string.
    """

    error_code = "SESSION_REVISION_CONFLICT"
    agent_action = (
        "The session state this result was computed against changed while it ran. "
        "Nothing was published or persisted. Check get_runtime_context for the "
        "current run, then re-run the tool if its result is still wanted."
    )


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
class ResolvedRun:
    """Everything a tool needs to read one fitted run, live or cached."""

    run_id: str
    model: ThemaRS
    data: pd.DataFrame | None
    dataset_id: str | None
    embeddings: list | None
    projection_fingerprint: str | None
    pca_fingerprint: str | None = None  # Legacy alias; kept consistent on put().


def _model_cache_capacity() -> int:
    try:
        return max(1, int(os.environ.get("PULSAR_MCP_MODEL_CACHE_SIZE", "3")))
    except ValueError:
        return 3


class ModelCache:
    """Process-wide bounded LRU of fitted runs keyed by ``run_id``.

    Only touched on the event-loop thread (``publish_sweep`` and tool-side
    ``resolve_run`` both run there, never in workers), so it deliberately has no
    threading lock. Session eviction does NOT evict entries: the cache is
    process-wide so a ``run_id`` stays addressable on stateless transports.

    Memory: a cached run pins its model and DataFrame (and embeddings) until it
    is evicted, even after the owning session is gone. Capacity comes from
    ``PULSAR_MCP_MODEL_CACHE_SIZE`` (default 3, minimum 1).
    """

    def __init__(self, capacity: int | None = None) -> None:
        self.capacity = (
            max(1, capacity) if capacity is not None else _model_cache_capacity()
        )
        self._entries: OrderedDict[str, ResolvedRun] = OrderedDict()

    def put(self, resolved: ResolvedRun) -> None:
        if resolved.pca_fingerprint is None:
            resolved.pca_fingerprint = resolved.projection_fingerprint
        self._entries[resolved.run_id] = resolved
        self._entries.move_to_end(resolved.run_id)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def get(self, run_id: str) -> ResolvedRun | None:
        entry = self._entries.get(run_id)
        if entry is not None:
            self._entries.move_to_end(run_id)
        return entry

    def evict(self, run_id: str) -> None:
        self._entries.pop(run_id, None)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def _entry_bytes(entry: ResolvedRun) -> int:
        """Approximate bytes: DataFrame + embeddings only (model size unknown)."""
        total = 0
        if entry.data is not None:
            try:
                total += int(entry.data.memory_usage(deep=True).sum())
            except Exception:  # noqa: BLE001 - estimate only
                pass
        for emb in entry.embeddings or []:
            total += int(getattr(emb, "nbytes", 0) or 0)
        return total

    def stats(self) -> dict[str, Any]:
        per_entry = {rid: self._entry_bytes(e) for rid, e in self._entries.items()}
        return {
            "size": len(self._entries),
            "capacity": self.capacity,
            "run_ids": list(self._entries),  # least to most recently used
            "memory_bytes": sum(per_entry.values()),
            "entries": [
                {"run_id": rid, "memory_bytes": b} for rid, b in per_entry.items()
            ],
            "model_bytes_estimated": False,  # fitted model size is not included
        }


model_cache = ModelCache()


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
    # Bumped by every publish (audit counter). Do NOT use it as a conflict token:
    # dossier / artifact / longitudinal publishes bump it too.
    revision: int = 0
    # Bumped only by publish_sweep. Sweeps are serialized by the mutation lock and
    # are last-writer-wins, so this is NOT a sweep conflict token. It exists so
    # loop-side readers (``_load_session_dataframe``) can tell that a sweep
    # rebound the session's data while they were reading.
    sweep_revision: int = 0
    # Mutated only on the event-loop thread (incremented/decremented by
    # run_session_worker), never inside a worker thread.
    active_ops: int = 0
    # Serializes session-mutating workers (sweeps, dossiers, longitudinal builds).
    # An asyncio.Lock acquired on the loop: the loop never blocks on it and no
    # executor thread is pinned waiting for it.
    _worker_mutation_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, repr=False, compare=False
    )

    # All publish_* methods mutate session state and MUST be called on the event
    # loop thread with no ``await`` between the call and its neighbouring reads;
    # a single synchronous block is atomic w.r.t. other coroutines. Workers
    # compute and return results; they never call these.

    def publish_sweep(
        self,
        *,
        config_yaml: str | None = None,
        config_dataset_id: str | None = None,
        model: ThemaRS,
        data: pd.DataFrame,
        dataset_id: str | None = None,
        data_path: str | None = None,
        dataset_fingerprint: str | None = None,
        embeddings: list[Any] | None = None,
        projection_fingerprint: str | None = None,
        pca_fingerprint: str | None = None,
        record: SweepRecord,
        run_id: str,
    ) -> None:
        if config_yaml is not None:
            self.active_config_yaml = config_yaml
            self.active_config_dataset_id = config_dataset_id
        self.model = model
        _bind_session_data(
            self,
            data,
            dataset_id=dataset_id,
            data_path=data_path,
        )
        self.embeddings = embeddings
        self.projection_fingerprint = projection_fingerprint
        self.pca_fingerprint = (
            pca_fingerprint if pca_fingerprint is not None else projection_fingerprint
        )
        self.sweep_history.append(record)
        self.latest_run_id = run_id
        self.sweep_revision += 1
        self.revision += 1
        # Same objects as the session's live state (no copy). The cache is
        # process-wide and outlives session eviction; it pins model and data.
        model_cache.put(
            ResolvedRun(
                run_id=run_id,
                model=model,
                data=data,
                dataset_id=dataset_id,
                embeddings=embeddings,
                projection_fingerprint=self.projection_fingerprint,
                pca_fingerprint=self.pca_fingerprint,
            )
        )

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
            raise SessionConflictError(
                "Active run changed while computing cluster dossier; refusing to publish stale clusters."
            )
        self.clusters = clusters
        self.clusters_run_id = expected_run_id
        self.report_exclude_columns = (
            list(report_exclude_columns) if report_exclude_columns is not None else None
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
            raise SessionConflictError(
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
            int(self.data.memory_usage(deep=True).sum()) if self.data is not None else 0
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


def _run_worker_job(
    cancel_event: threading.Event,
    fn: Callable[..., T],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> T:
    # A job cancelled while still queued in the executor must not start.
    if cancel_event.is_set():
        raise OperationCancelledError("Operation cancelled before start.")
    return fn(*args, **kwargs)


async def run_session_worker(
    session: _PulsarSession,
    fn: Callable[..., T],
    *args: Any,
    heavy: bool = True,
    mutates_session: bool = False,
    cancel_event: threading.Event | None = None,
    **kwargs: Any,
) -> T:
    """Run blocking ``fn`` in an executor thread behind loop-side admission.

    The per-session mutation lock and the heavy-compute semaphore are asyncio
    primitives acquired on the loop, so queued requests never occupy executor
    threads and the loop never blocks on a threading lock. ``fn`` must be a
    pure compute function: it returns results and must not mutate ``session``;
    callers publish after the ``await`` in a single synchronous block.

    ``active_ops``, the lock and the semaphore are released on the loop exactly
    once, when the work is finished or was never submitted. If the awaiting task
    is cancelled while the thread is still running, release is deferred until
    the thread actually ends so admission still bounds real concurrency.
    """
    loop = asyncio.get_running_loop()
    cancel_event = cancel_event if cancel_event is not None else threading.Event()
    heavy_sem = _heavy_semaphore() if heavy else None
    held_lock = False
    held_sem = False
    released = False

    def release() -> None:
        nonlocal released
        if released:
            return
        released = True
        if held_sem and heavy_sem is not None:
            heavy_sem.release()
        if held_lock:
            session._worker_mutation_lock.release()
        session.active_ops = max(0, session.active_ops - 1)

    session.active_ops += 1
    try:
        if mutates_session:
            await session._worker_mutation_lock.acquire()
            held_lock = True
        if heavy_sem is not None:
            await heavy_sem.acquire()
            held_sem = True
        if cancel_event.is_set():
            raise OperationCancelledError("Operation cancelled before start.")
        # Unlike asyncio.to_thread, run_in_executor does not copy contextvars.
        # Without them FastMCP's request context is unset in the worker and
        # ctx.report_progress silently no-ops.
        worker_ctx = contextvars.copy_context()
        fut = loop.run_in_executor(
            None, worker_ctx.run, _run_worker_job, cancel_event, fn, args, kwargs
        )
    except BaseException as exc:
        if isinstance(exc, asyncio.CancelledError):
            cancel_event.set()
        release()
        raise

    def _on_done(done: asyncio.Future[Any]) -> None:
        if not done.cancelled():
            done.exception()  # mark retrieved; the awaiter may be gone
        release()

    fut.add_done_callback(_on_done)
    try:
        return await asyncio.shield(fut)
    except asyncio.CancelledError:
        cancel_event.set()
        raise


def _is_http_context(ctx: Context) -> bool:
    """True for SSE / Streamable HTTP, including background-task workers.

    fastmcp does not restore ``ctx.transport`` inside a task worker, but it does
    snapshot the submitting request's HTTP headers; a non-empty snapshot marks an
    HTTP submitter.
    """
    from fastmcp.server.dependencies import _background_task_headers

    if ctx.transport in ("sse", "streamable-http"):
        return True
    return bool(ctx.is_background_task and _background_task_headers.get())


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
    if isinstance(header_sid, str) and header_sid:
        return header_sid
    # Shared "default" state is only safe for a single-client transport
    # (stdio, or in-memory where ctx.transport is None). On HTTP with no
    # session id, isolate clients by fastmcp's per-request id instead of
    # merging them into one session.
    if _is_http_context(ctx):
        request_sid = getattr(ctx, "session_id", None)
        if isinstance(request_sid, str) and request_sid:
            return request_sid
    return "default"


def _session_key(ctx: Context | None) -> str:
    """Get the session key from context.

    ``"default"`` is used only for stdio / in-memory transports. HTTP without a
    session id falls back to the per-request ``ctx.session_id``.
    """
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


def resolve_run(session: _PulsarSession, run_id: str = "") -> ResolvedRun:
    """Resolve a run handle, or raise LookupError for the error envelope.

    Empty ``run_id`` returns the session's live state (``LookupError("")`` when
    nothing has been fitted); otherwise the process-wide cache entry, else
    ``LookupError(run_id)``.
    """
    if not run_id:
        if session.model is None:
            raise LookupError("")
        return ResolvedRun(
            run_id=session.latest_run_id or "",
            model=session.model,
            data=session.data,
            dataset_id=session.dataset_id,
            embeddings=session.embeddings,
            projection_fingerprint=session.projection_fingerprint,
            pca_fingerprint=session.pca_fingerprint,
        )
    entry = model_cache.get(run_id)
    if entry is None:
        raise LookupError(run_id)
    return entry


def is_latest(session: _PulsarSession, resolved: ResolvedRun) -> bool:
    """True when ``resolved`` is the session's latest published run."""
    return (
        session.latest_run_id is not None and resolved.run_id == session.latest_run_id
    )


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

    sweep_revision_before = session.sweep_revision
    df = await run_session_worker(
        session,
        _read_dataset_file,
        normalized_path,
        heavy=False,
        mutates_session=False,
    )
    # Bind on the loop thread (atomic w.r.t. other coroutines; no lock needed and
    # the loop never waits on a worker). If a sweep published while we were
    # reading, its data binding is newer than this read, so do not clobber it:
    # return the frame to the caller without rebinding the session.
    if session.sweep_revision == sweep_revision_before:
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
