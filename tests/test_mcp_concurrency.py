from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import statistics
import threading
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from pulsar.exports import export_dataset_bundle
from pulsar.mcp import session as session_mod
from pulsar.mcp.tools.clustering import generate_cluster_dossier
from pulsar.mcp.tools.config import create_config
from pulsar.mcp.tools.ingestion import ingest_dataset
from pulsar.mcp.tools.meta import get_runtime_context
from pulsar.mcp.tools.reporting import export_labeled_data
from pulsar.mcp.tools.sweeping import get_sweep_history, run_topological_sweep
from pulsar.pipeline import ThemaRS
from pulsar.runtime.utils import rayon_thread_override


async def _noop_progress(*_args, **_kwargs) -> None:
    return None


def _make_ctx(session_id: str) -> SimpleNamespace:
    return SimpleNamespace(session_id=session_id, report_progress=_noop_progress)


def _write_synthetic_csv(path: Path, n_rows: int = 120) -> Path:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "f1": np.concatenate(
                [
                    rng.normal(-2.0, 0.4, n_rows // 2),
                    rng.normal(2.0, 0.4, n_rows // 2),
                ]
            ),
            "f2": np.concatenate(
                [
                    rng.normal(1.5, 0.4, n_rows // 2),
                    rng.normal(-1.5, 0.4, n_rows // 2),
                ]
            ),
            "f3": rng.normal(0.0, 1.0, n_rows),
            "f4": rng.normal(0.5, 0.8, n_rows),
        }
    )
    df.to_csv(path, index=False)
    return path


async def _ingest_and_configure(
    csv_path: Path, ctx: SimpleNamespace
) -> tuple[str, str]:
    ingest_res = json.loads(await ingest_dataset(str(csv_path), ctx=ctx))
    dataset_id = ingest_res["dataset_id"]
    cfg_res = json.loads(await create_config(dataset_id=dataset_id, ctx=ctx))
    return dataset_id, cfg_res["config_yaml"]


async def _wait_for_event(event: threading.Event) -> None:
    while not event.is_set():
        await asyncio.sleep(0.002)


async def _wait_for_worker_drain(session: session_mod._PulsarSession) -> None:
    for _ in range(50):
        if not session._worker_mutation_lock.locked() and session.active_ops == 0:
            return
        await asyncio.sleep(0.005)


def _make_cancellable_fit(
    worker_entered: threading.Event,
    allow_worker_check: threading.Event,
):
    def _cancellable_fit(self: ThemaRS, *_args, **kwargs):
        progress_callback = kwargs.get("progress_callback")
        if progress_callback is not None:
            progress_callback("load", 0.05)
        worker_entered.set()
        allow_worker_check.wait(timeout=2.0)
        if progress_callback is not None:
            progress_callback("pca", 0.35)
        return self

    return _cancellable_fit


@pytest.fixture(autouse=True)
def _clean_sessions():
    session_mod._sessions.clear()
    yield
    session_mod._sessions.clear()


def test_concurrent_ping_during_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _run() -> None:
        csv_path = _write_synthetic_csv(tmp_path / "synthetic.csv", n_rows=160)
        ctx = _make_ctx("concurrency-ping")
        dataset_id, config_yaml = await _ingest_and_configure(csv_path, ctx)

        started_in_worker = threading.Event()
        orig_fit = ThemaRS.fit

        def _paced_fit(self: ThemaRS, *args, **kwargs):
            started_in_worker.set()
            time.sleep(0.08)
            return orig_fit(self, *args, **kwargs)

        monkeypatch.setattr(ThemaRS, "fit", _paced_fit)
        sweep_task = asyncio.create_task(
            run_topological_sweep(
                config_yaml=config_yaml,
                dataset_id=dataset_id,
                response_format="json",
                ctx=ctx,
            )
        )
        await _wait_for_event(started_in_worker)

        latencies_ms: list[float] = []
        for _ in range(8):
            t0 = time.perf_counter()
            ctx_out, hist_out = await asyncio.gather(
                get_runtime_context(ctx=ctx),
                get_sweep_history(response_format="json", ctx=ctx),
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            assert json.loads(ctx_out)["session_id"] == "concurrency-ping"
            assert json.loads(hist_out)["status"] == "ok"
            await asyncio.sleep(0.005)

        sweep_res = json.loads(await sweep_task)
        assert sweep_res["status"] == "ok"
        assert statistics.median(latencies_ms) < 50.0

    asyncio.run(_run())


def test_cancelled_sweep_preserves_session_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _run() -> None:
        csv_path = _write_synthetic_csv(tmp_path / "cancel_sweep.csv", n_rows=100)
        ctx = _make_ctx("cancel-revision")
        dataset_id, config_yaml = await _ingest_and_configure(csv_path, ctx)

        first_res = json.loads(
            await run_topological_sweep(
                config_yaml=config_yaml,
                dataset_id=dataset_id,
                response_format="json",
                ctx=ctx,
            )
        )
        run_a = first_res["run_id"]
        session = session_mod._get_session(ctx)
        assert (session.revision, session.latest_run_id, session.active_ops) == (
            1,
            run_a,
            0,
        )

        worker_entered = threading.Event()
        allow_worker_check = threading.Event()
        monkeypatch.setattr(
            ThemaRS,
            "fit",
            _make_cancellable_fit(worker_entered, allow_worker_check),
        )

        second_task = asyncio.create_task(
            run_topological_sweep(
                config_yaml=config_yaml,
                dataset_id=dataset_id,
                response_format="json",
                ctx=ctx,
            )
        )
        await _wait_for_event(worker_entered)
        second_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second_task

        allow_worker_check.set()
        await _wait_for_worker_drain(session)
        assert (
            session.revision,
            session.latest_run_id,
            len(session.sweep_history),
            session.active_ops,
        ) == (1, run_a, 1, 0)

    asyncio.run(_run())


def test_lru_eviction_skips_active_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_mod, "_MAX_SESSIONS", 3)

    s1 = session_mod._get_session(_make_ctx("s1"))
    s2 = session_mod._get_session(_make_ctx("s2"))
    s3 = session_mod._get_session(_make_ctx("s3"))

    s1.active_ops = 1
    assert session_mod._sessions["s2"] is s2
    assert list(session_mod._sessions.keys()) == ["s1", "s2", "s3"]

    s4 = session_mod._get_session(_make_ctx("s4"))
    assert "s1" in session_mod._sessions
    assert "s2" not in session_mod._sessions
    assert session_mod._sessions["s1"] is s1
    assert session_mod._sessions["s3"] is s3
    assert session_mod._sessions["s4"] is s4


def _make_failing_parquet_writer(orig_write_table, fail_on_call: int = 2):
    calls = {"count": 0}

    def _writer(table, where, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] >= fail_on_call:
            raise OSError("Simulated mid-bundle disk failure")
        return orig_write_table(table, where, *args, **kwargs)

    return _writer


def test_atomic_export_bundle_fault_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _run() -> None:
        csv_path = _write_synthetic_csv(tmp_path / "bundle_fault.csv", n_rows=80)
        ctx = _make_ctx("bundle-fault")
        dataset_id, config_yaml = await _ingest_and_configure(csv_path, ctx)
        await run_topological_sweep(
            config_yaml=config_yaml,
            dataset_id=dataset_id,
            response_format="json",
            ctx=ctx,
        )
        session = session_mod._get_session(ctx)
        assert session.model is not None
        labels = np.zeros(len(session.model.data), dtype=int)
        output_dir = tmp_path / "exports"
        orig_write_table = pq.write_table

        monkeypatch.setattr(
            pq, "write_table", _make_failing_parquet_writer(orig_write_table)
        )
        with pytest.raises(OSError, match="Simulated mid-bundle disk failure"):
            export_dataset_bundle(
                session.model, output_dir, "fresh_bundle", cluster_labels=labels
            )
        assert not (output_dir / "fresh_bundle").exists()
        assert list(output_dir.iterdir()) == []

        monkeypatch.setattr(pq, "write_table", orig_write_table)
        manifest = export_dataset_bundle(
            session.model, output_dir, "existing_bundle", cluster_labels=labels
        )
        manifest_path = output_dir / "existing_bundle" / "export_manifest.json"
        original_manifest_bytes = manifest_path.read_bytes()
        assert manifest["counts"]["nodes"] == 80

        monkeypatch.setattr(
            pq, "write_table", _make_failing_parquet_writer(orig_write_table)
        )
        with pytest.raises(OSError, match="Simulated mid-bundle disk failure"):
            export_dataset_bundle(
                session.model,
                output_dir,
                "existing_bundle",
                cluster_labels=np.ones(len(session.model.data), dtype=int),
            )
        assert manifest_path.read_bytes() == original_manifest_bytes
        assert [p.name for p in output_dir.iterdir()] == ["existing_bundle"]

    asyncio.run(_run())


def test_atomic_labeled_csv_fault_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _run() -> None:
        csv_path = _write_synthetic_csv(tmp_path / "labeled_fault.csv", n_rows=80)
        ctx = _make_ctx("labeled-fault")
        dataset_id, config_yaml = await _ingest_and_configure(csv_path, ctx)

        await run_topological_sweep(
            config_yaml=config_yaml,
            dataset_id=dataset_id,
            response_format="json",
            ctx=ctx,
        )
        dossier_res = json.loads(
            await generate_cluster_dossier(
                method="components",
                response_format="json",
                ctx=ctx,
            )
        )
        assignment_id = dossier_res["cluster_assignment_id"]
        session = session_mod._get_session(ctx)
        assert session.clusters is not None
        unique_ids = sorted(int(c) for c in session.clusters.unique())
        cluster_names = {str(cid): f"Cluster {cid}" for cid in unique_ids}

        dest_csv = tmp_path / "out" / "labeled.csv"

        def _failing_to_csv(self: pd.DataFrame, path_or_buf=None, *_args, **_kwargs):
            if path_or_buf is not None:
                Path(path_or_buf).write_text("partial,corrupt\n1,", encoding="utf-8")
            raise OSError("Simulated CSV write failure")

        monkeypatch.setattr(pd.DataFrame, "to_csv", _failing_to_csv)

        res = json.loads(
            await export_labeled_data(
                cluster_assignment_id=assignment_id,
                cluster_names=cluster_names,
                output_path=str(dest_csv),
                ctx=ctx,
            )
        )
        assert res["status"] == "error"
        assert not dest_csv.exists()
        assert list(dest_csv.parent.iterdir()) == []

    asyncio.run(_run())


def test_rayon_thread_override_does_not_mutate_environ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
    before = dict(os.environ)

    with rayon_thread_override(4):
        assert "RAYON_NUM_THREADS" not in os.environ
        assert dict(os.environ) == before

    assert dict(os.environ) == before
