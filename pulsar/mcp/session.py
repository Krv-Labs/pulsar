from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastmcp import Context
from fastmcp.exceptions import ToolError

from pulsar.pipeline import ThemaRS
from pulsar.mcp.registry import registry
from pulsar.mcp.interpreter import FeatureEvidenceIndex

logger = logging.getLogger(__name__)


@dataclass
class SweepRecord:
    timestamp: float
    config_yaml: str
    metrics: dict
    dataset_id: str | None = None


@dataclass
class _PulsarSession:
    """Session state for a single MCP client."""

    model: ThemaRS | None = None
    data: pd.DataFrame | None = None
    clusters: pd.Series | None = None
    embeddings: list | None = None  # cached PCA output from last fit
    pca_fingerprint: str | None = None  # SHA256 of (data_path, dims, seeds, n_rows)
    sweep_history: list[SweepRecord] = field(default_factory=list)
    dataset_id: str | None = None
    latest_run_id: str | None = None
    report_exclude_columns: list[str] | None = None
    data_dataset_id: str | None = None
    data_path: str | None = None
    feature_evidence_index: FeatureEvidenceIndex | None = None
    feature_evidence_fingerprint: str | None = None
    feature_evidence_cluster_meta: dict[str, Any] | None = None
    active_config_yaml: str | None = None
    active_config_dataset_id: str | None = None

    def calculate_memory_mb(self) -> float:
        """Estimate current session memory footprint in MB."""
        bytes_total = 0
        if self.data is not None:
            # Deep memory usage for Pandas (captures string objects etc)
            bytes_total += self.data.memory_usage(deep=True).sum()

        if self.model is not None and self.model._weighted_adjacency is not None:
            bytes_total += self.model._weighted_adjacency.nbytes

        if self.embeddings is not None:
            for emb in self.embeddings:
                if hasattr(emb, "nbytes"):
                    bytes_total += emb.nbytes

        return round(float(bytes_total) / (1024 * 1024), 2)


_sessions: OrderedDict[str, _PulsarSession] = OrderedDict()
_MAX_SESSIONS = int(os.environ.get("PULSAR_MAX_SESSIONS", "3"))


def _session_key(ctx: Context | None) -> str:
    """Get the session key from context (session_id or 'default' for STDIO)."""
    if ctx is None:
        return "default"
    return ctx.session_id or "default"


def _get_session(ctx: Context) -> _PulsarSession:
    """Get or create session state for the current client with LRU eviction."""
    key = _session_key(ctx)

    if key in _sessions:
        # Move to end (mark as recently used)
        _sessions.move_to_end(key)
        return _sessions[key]

    # Create new session
    if len(_sessions) >= _MAX_SESSIONS:
        # Evict oldest session
        evicted_key, evicted_session = _sessions.popitem(last=False)
        logger.info(
            "Evicting oldest session '%s' to free memory (Max sessions: %d)",
            evicted_key,
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
    # Rebinding the underlying data invalidates any cached feature evidence,
    # which was computed against the previous DataFrame. (The sweep path also
    # invalidates explicitly; the redundant call here is idempotent.)
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


async def _load_session_dataframe(
    session: _PulsarSession,
    *,
    dataset_id: str = "",
    data_path: str = "",
) -> tuple[pd.DataFrame, str]:
    if dataset_id:
        resolved_path = _resolve_dataset_path(dataset_id)
        normalized_path = _normalize_data_path(resolved_path)
        if (
            session.data is not None
            and session.data_dataset_id == dataset_id
            and session.data_path == normalized_path
        ):
            session.dataset_id = dataset_id
            return session.data, normalized_path

        df = await asyncio.to_thread(_read_dataset_file, normalized_path)
        _bind_session_data(
            session,
            df,
            dataset_id=dataset_id,
            data_path=normalized_path,
        )
        return df, normalized_path

    if not data_path:
        raise ToolError("Provide either data_path or dataset_id.")

    normalized_path = _normalize_data_path(data_path)
    if (
        session.data is not None
        and session.data_dataset_id is None
        and session.data_path == normalized_path
    ):
        return session.data, normalized_path

    df = await asyncio.to_thread(_read_dataset_file, normalized_path)
    _bind_session_data(session, df, data_path=normalized_path)
    return df, normalized_path


def _invalidate_feature_evidence_cache(session: _PulsarSession) -> None:
    session.feature_evidence_index = None
    session.feature_evidence_fingerprint = None
    session.feature_evidence_cluster_meta = None
    session.clusters = None


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
    giant_fraction = float(metrics.get("giant_fraction", 0.0))
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
    if giant_fraction > 0.95:
        return (
            "connected",
            is_connected,
            "Proceed to clustering; graph structure is suitable for interpretive analysis.",
        )
    return (
        "connected",
        is_connected,
        "Proceed to clustering; graph structure is suitable for interpretive analysis.",
    )


def _pca_cache_status(
    session: _PulsarSession,
    cfg: Any,
) -> tuple[list | None, dict[str, Any]]:
    """Return reusable PCA embeddings and an operational status payload."""
    from pulsar.runtime.fingerprint import pca_fingerprint

    status: dict[str, Any] = {
        "scope": "session",
        "status": "miss",
        "reason": "no_cached_embeddings",
    }
    if session.embeddings is None:
        return None, status
    if session.data is None:
        status["reason"] = "no_session_data"
        return None, status
    if session.pca_fingerprint is None:
        status["reason"] = "no_cached_fingerprint"
        return None, status

    fingerprint = pca_fingerprint(cfg, len(session.data), session.data)
    if fingerprint != session.pca_fingerprint:
        status["reason"] = "fingerprint_mismatch"
        return None, status

    return (
        session.embeddings,
        {
            "scope": "session",
            "status": "hit",
            "reason": "fingerprint_match",
        },
    )
