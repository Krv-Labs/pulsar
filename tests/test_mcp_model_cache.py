"""Tests for the process-wide MCP ModelCache and resolve_run."""

from __future__ import annotations

import pandas as pd
import pytest

from pulsar.mcp import session as sess
from pulsar.mcp.session import (
    ModelCache,
    ResolvedRun,
    SweepRecord,
    _PulsarSession,
    is_latest,
    resolve_run,
)


class FakeModel:
    pass


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    cache = ModelCache(capacity=3)
    monkeypatch.setattr(sess, "model_cache", cache)
    return cache


def _run(run_id: str) -> ResolvedRun:
    return ResolvedRun(run_id, FakeModel(), None, None, None, None)


def _publish(session: _PulsarSession, run_id: str, model=None):
    model = model or FakeModel()
    session.publish_sweep(
        model=model,
        data=pd.DataFrame({"a": [1.0, 2.0]}),
        dataset_id="ds1",
        embeddings=[],
        projection_fingerprint="fp",
        record=SweepRecord(timestamp=0.0, config_yaml="", metrics={}),
        run_id=run_id,
    )
    return model


def test_lru_eviction_order(fresh_cache):
    for rid in ("a", "b", "c"):
        fresh_cache.put(_run(rid))
    assert fresh_cache.get("a") is not None  # touch a; b is now oldest
    fresh_cache.put(_run("d"))
    assert fresh_cache.get("b") is None
    assert fresh_cache.stats()["run_ids"] == ["c", "a", "d"]


def test_capacity_from_env(monkeypatch):
    monkeypatch.setenv("PULSAR_MCP_MODEL_CACHE_SIZE", "5")
    assert ModelCache().capacity == 5
    monkeypatch.setenv("PULSAR_MCP_MODEL_CACHE_SIZE", "0")
    assert ModelCache().capacity == 1
    monkeypatch.delenv("PULSAR_MCP_MODEL_CACHE_SIZE")
    assert ModelCache().capacity == 3


def test_capacity_one_keeps_newest():
    cache = ModelCache(capacity=1)
    cache.put(_run("a"))
    cache.put(_run("b"))
    assert len(cache) == 1
    assert cache.get("a") is None and cache.get("b") is not None


def test_evict_and_clear(fresh_cache):
    fresh_cache.put(_run("a"))
    fresh_cache.put(_run("b"))
    fresh_cache.evict("a")
    fresh_cache.evict("missing")
    assert len(fresh_cache) == 1
    fresh_cache.clear()
    assert len(fresh_cache) == 0


def test_stats_shape(fresh_cache):
    fresh_cache.put(_run("a"))
    stats = fresh_cache.stats()
    assert stats == {
        "size": 1,
        "capacity": 3,
        "run_ids": ["a"],
        "memory_bytes": 0,
        "entries": [{"run_id": "a", "memory_bytes": 0}],
        "model_bytes_estimated": False,
    }


def test_stats_memory_counts_frame_and_embeddings(fresh_cache):
    import numpy as np

    df = pd.DataFrame({"a": np.arange(100, dtype=float)})
    emb = np.zeros((10, 4))
    fresh_cache.put(ResolvedRun("m", FakeModel(), df, None, [emb], None))
    stats = fresh_cache.stats()
    expected = int(df.memory_usage(deep=True).sum()) + emb.nbytes
    assert stats["memory_bytes"] == expected
    assert stats["entries"] == [{"run_id": "m", "memory_bytes": expected}]
    assert stats["model_bytes_estimated"] is False


def test_publish_sweep_populates_cache_same_object(fresh_cache):
    session = _PulsarSession()
    model = _publish(session, "r1")
    cached = fresh_cache.get("r1")
    assert cached is not None
    assert cached.model is model and cached.model is session.model
    assert cached.data is session.data
    assert cached.dataset_id == "ds1"
    assert cached.projection_fingerprint == "fp"
    assert cached.pca_fingerprint == "fp"


def test_resolve_run_empty_known_unknown():
    session = _PulsarSession()
    with pytest.raises(LookupError) as exc:
        resolve_run(session)
    assert exc.value.args == ("",)

    m1 = _publish(session, "r1")
    m2 = _publish(session, "r2")
    live = resolve_run(session)
    assert live.model is m2 and live.run_id == "r2"
    old = resolve_run(session, "r1")
    assert old.model is m1
    assert is_latest(session, live) and not is_latest(session, old)
    with pytest.raises(LookupError) as exc:
        resolve_run(session, "nope")
    assert exc.value.args == ("nope",)


def test_session_eviction_does_not_evict_cache(fresh_cache):
    session = _PulsarSession()
    _publish(session, "r1")
    del session
    assert resolve_run(_PulsarSession(), "r1").run_id == "r1"
