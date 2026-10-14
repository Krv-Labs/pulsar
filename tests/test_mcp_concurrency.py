from __future__ import annotations

import asyncio
import concurrent.futures
import os
import statistics
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from pulsar.exports import export_dataset_bundle
from pulsar.mcp import registry as registry_mod
from pulsar.mcp import session as session_mod
from pulsar.mcp.tools import clustering as clustering_mod
from pulsar.mcp.tools.clustering import generate_cluster_dossier
from pulsar.mcp.tools.config import create_config
from pulsar.mcp.tools.ingestion import ingest_dataset
from pulsar.mcp.tools.meta import get_runtime_context
from pulsar.mcp.tools.preprocessing import recommend_preprocessing
from pulsar.mcp.tools.reporting import export_labeled_data
from pulsar.mcp.tools.sweeping import get_sweep_history, run_topological_sweep
from pulsar.pipeline import ThemaRS
from pulsar.runtime.utils import rayon_thread_override
from tests.mcp_helpers import payload_of


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
    ingest_res = payload_of(await ingest_dataset(str(csv_path), ctx=ctx))
    dataset_id = ingest_res["dataset_id"]
    cfg_res = payload_of(await create_config(dataset_id=dataset_id, ctx=ctx))
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
def _clean_sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cache_dir = tmp_path / "pulsar_mcp_cache"
    monkeypatch.setattr(registry_mod, "_CACHE_DIR", cache_dir)
    monkeypatch.setattr(registry_mod, "_DATASETS_PATH", cache_dir / "datasets.json")
    monkeypatch.setattr(registry_mod, "_DATASET_FILES_DIR", cache_dir / "datasets")
    monkeypatch.setattr(registry_mod, "_UPLOADS_DIR", cache_dir / "uploads")
    monkeypatch.setattr(registry_mod, "_RUNS_DIR", cache_dir / "runs")
    monkeypatch.setattr(
        registry_mod, "_CLUSTER_ASSIGNMENTS_DIR", cache_dir / "cluster_assignments"
    )
    monkeypatch.setattr(registry_mod, "_LOCK_PATH", cache_dir / ".registry.lock")
    registry_mod.MCPRegistry()
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
            assert payload_of(ctx_out)["session_id"] == "concurrency-ping"
            assert payload_of(hist_out)["status"] == "ok"
            await asyncio.sleep(0.005)

        sweep_res = payload_of(await sweep_task)
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

        first_res = payload_of(
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
        dossier_res = payload_of(
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

        res = payload_of(
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


def test_loop_stays_responsive_when_binding_dataset_during_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F1: binding a new dataset must not wait on the sweep's mutation lock."""

    async def _run() -> None:
        ctx = _make_ctx("loop-block")
        path_a = _write_synthetic_csv(tmp_path / "a.csv", n_rows=100)
        path_b = _write_synthetic_csv(tmp_path / "b.csv", n_rows=90)
        dataset_a, config_yaml = await _ingest_and_configure(path_a, ctx)
        dataset_b = payload_of(await ingest_dataset(str(path_b), ctx=ctx))["dataset_id"]

        entered = threading.Event()
        orig_fit = ThemaRS.fit

        def _slow_fit(self: ThemaRS, *args, **kwargs):
            entered.set()
            time.sleep(1.0)
            return orig_fit(self, *args, **kwargs)

        monkeypatch.setattr(ThemaRS, "fit", _slow_fit)
        sweep_task = asyncio.create_task(
            run_topological_sweep(
                config_yaml=config_yaml,
                dataset_id=dataset_a,
                response_format="json",
                ctx=ctx,
            )
        )
        await _wait_for_event(entered)
        session = session_mod._get_session(ctx)
        assert session._worker_mutation_lock.locked()

        probe_task = asyncio.create_task(
            recommend_preprocessing(
                dataset_id=dataset_b, response_format="json", ctx=ctx
            )
        )
        t0 = time.perf_counter()
        await asyncio.sleep(0.05)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 300.0, f"event loop stalled for {elapsed_ms:.0f} ms"
        t0 = time.perf_counter()
        await get_runtime_context(ctx=ctx)
        assert (time.perf_counter() - t0) * 1000.0 < 300.0

        # The light tool finished binding dataset B while the sweep was running.
        await probe_task
        assert session.data_dataset_id == dataset_b
        assert not sweep_task.done()

        assert payload_of(await sweep_task)["status"] == "ok"
        assert session.data_dataset_id == dataset_a

    asyncio.run(_run())


def test_cancel_before_start_does_not_leak_active_ops() -> None:
    """F4: a job cancelled while queued in the executor releases active_ops."""

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1))
        session = session_mod._PulsarSession()
        ran: list[bool] = []

        blocker = asyncio.create_task(asyncio.to_thread(time.sleep, 0.3))
        await asyncio.sleep(0.02)
        task = asyncio.create_task(
            session_mod.run_session_worker(
                session,
                lambda: ran.append(True),
                heavy=False,
                mutates_session=False,
            )
        )
        await asyncio.sleep(0.02)
        assert session.active_ops == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await blocker
        await asyncio.sleep(0.05)
        assert session.active_ops == 0
        assert ran == [], "cancelled job must not execute after being dequeued"

    asyncio.run(_run())


def test_heavy_admission_queues_on_loop_not_executor_threads() -> None:
    """F4: heavy requests waiting on admission do not pin executor threads."""

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=2))
        session = session_mod._PulsarSession()
        gate = threading.Event()
        started = threading.Event()

        def _hold() -> str:
            started.set()
            gate.wait(timeout=3.0)
            return "held"

        heavy = [
            asyncio.create_task(
                session_mod.run_session_worker(session, _hold, heavy=True)
            )
            for _ in range(4)
        ]
        await _wait_for_event(started)
        await asyncio.sleep(0.05)
        # Only one heavy job is admitted (default PULSAR_MCP_MAX_CONCURRENT_HEAVY=1);
        # the free executor thread still serves a light job immediately.
        assert session_mod._MAX_CONCURRENT_HEAVY == 1
        light = await asyncio.wait_for(
            session_mod.run_session_worker(session, lambda: "light", heavy=False),
            timeout=1.0,
        )
        assert light == "light"
        assert session.active_ops == 4

        gate.set()
        assert await asyncio.gather(*heavy) == ["held"] * 4
        assert session.active_ops == 0

    asyncio.run(_run())


def test_concurrent_sweeps_serialize_last_writer_wins(tmp_path: Path) -> None:
    """Two sweeps on one session both succeed; the later publish is the live run."""

    async def _run() -> None:
        ctx = _make_ctx("sweep-serial")
        csv_path = _write_synthetic_csv(tmp_path / "serial.csv", n_rows=80)
        dataset_id, config_yaml = await _ingest_and_configure(csv_path, ctx)
        session = session_mod._get_session(ctx)

        raw = await asyncio.gather(
            *(
                run_topological_sweep(
                    config_yaml=config_yaml,
                    dataset_id=dataset_id,
                    response_format="json",
                    ctx=ctx,
                )
                for _ in range(2)
            )
        )
        first, second = (payload_of(r) for r in raw)

        assert first["status"] == "ok"
        assert second["status"] == "ok"
        assert first["run_id"] != second["run_id"]
        # The mutation lock is FIFO, so the second call publishes last.
        assert session.latest_run_id == second["run_id"]
        assert len(session.sweep_history) == 2
        assert not session._worker_mutation_lock.locked()
        assert session.active_ops == 0
        run_files = {p.stem for p in registry_mod._RUNS_DIR.glob("*.json")}
        assert run_files == {first["run_id"], second["run_id"]}
        assert list(registry_mod._RUNS_DIR.glob(".*")) == []

    asyncio.run(_run())


def test_dossier_refused_when_sweep_replaced_model_persists_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dossier computed against a replaced run is refused; no file is left."""

    async def _run() -> None:
        ctx = _make_ctx("dossier-conflict")
        csv_path = _write_synthetic_csv(tmp_path / "dossier.csv", n_rows=80)
        dataset_id, config_yaml = await _ingest_and_configure(csv_path, ctx)
        sweep = payload_of(
            await run_topological_sweep(
                config_yaml=config_yaml,
                dataset_id=dataset_id,
                response_format="json",
                ctx=ctx,
            )
        )
        session = session_mod._get_session(ctx)
        assert session.latest_run_id == sweep["run_id"]

        entered = threading.Event()
        release = threading.Event()
        orig_resolve = clustering_mod.resolve_clusters

        def _gated_resolve(*args, **kwargs):
            entered.set()
            release.wait(timeout=5.0)
            return orig_resolve(*args, **kwargs)

        monkeypatch.setattr(clustering_mod, "resolve_clusters", _gated_resolve)
        task = asyncio.create_task(
            generate_cluster_dossier(
                method="components", response_format="json", ctx=ctx
            )
        )
        await _wait_for_event(entered)
        # A newer sweep replaced the live run while the dossier computed.
        session.latest_run_id = "run_replaced_by_newer_sweep"
        release.set()
        res = payload_of(await task)

        assert res["status"] == "error"
        assert res["tool"] == "generate_cluster_dossier"
        assert res["error_code"] == "SESSION_REVISION_CONFLICT"
        assert "Active run changed" in res["reason"]
        assert res["agent_action"]
        assert session.clusters is None
        assert session.cluster_assignment_id is None
        assert session.active_ops == 0
        assert not session._worker_mutation_lock.locked()
        # No committed assignment and no staged temp file left behind.
        ca_dir = registry_mod._CLUSTER_ASSIGNMENTS_DIR
        assert [p.name for p in ca_dir.iterdir()] == []

    asyncio.run(_run())


def test_staged_run_and_assignment_visible_only_after_commit() -> None:
    reg = registry_mod.registry
    staged_ca = reg.stage_cluster_assignment(
        run_id="run_x",
        dataset_id=None,
        method="components",
        interpretation_edge_weight_threshold=0.0,
        threshold_source="explicit",
        construction_threshold=0.0,
        labels=[0, 1, 1, 2],
    )
    staged_run = reg.stage_run(
        dataset_id=None,
        config_yaml="run: {}",
        metrics={},
        resolved_construction_threshold=0.0,
        graph_summary={},
    )
    ca_id = staged_ca.record.cluster_assignment_id
    run_id = staged_run.record.run_id

    assert reg.get_cluster_assignment(ca_id) is None
    assert reg.get_run(run_id) is None
    assert list(registry_mod._CLUSTER_ASSIGNMENTS_DIR.glob("*.json")) == []
    assert list(registry_mod._RUNS_DIR.glob("*.json")) == []
    assert staged_ca.temp_path.exists() and staged_run.temp_path.exists()

    reg.commit_staged(staged_ca)
    reg.commit_staged(staged_run)
    loaded = reg.get_cluster_assignment(ca_id)
    assert loaded is not None and loaded.labels == [0, 1, 1, 2]
    assert loaded.cluster_ids == [0, 1, 2]
    assert reg.get_run(run_id) is not None
    assert not staged_ca.temp_path.exists() and not staged_run.temp_path.exists()

    discarded = reg.stage_cluster_assignment(
        run_id=None,
        dataset_id=None,
        method="components",
        interpretation_edge_weight_threshold=0.0,
        threshold_source="explicit",
        construction_threshold=0.0,
        labels=[0],
    )
    reg.discard_staged(discarded)
    reg.discard_staged(discarded)  # idempotent
    assert not discarded.temp_path.exists()
    assert reg.get_cluster_assignment(discarded.record.cluster_assignment_id) is None


def test_staging_does_not_take_registry_process_lock() -> None:
    """Staging must not wait on the registry lock a long upload/hash may hold."""
    reg = registry_mod.registry
    held = threading.Event()
    done = threading.Event()

    def _hold_lock() -> None:
        with registry_mod._PROCESS_LOCK:
            held.set()
            done.wait(timeout=5.0)

    holder = threading.Thread(target=_hold_lock)
    holder.start()
    try:
        assert held.wait(timeout=2.0)
        result: list[object] = []
        stager = threading.Thread(
            target=lambda: result.append(
                reg.stage_cluster_assignment(
                    run_id=None,
                    dataset_id=None,
                    method="components",
                    interpretation_edge_weight_threshold=0.0,
                    threshold_source="explicit",
                    construction_threshold=0.0,
                    labels=[0, 1],
                )
            )
        )
        stager.start()
        stager.join(timeout=2.0)
        assert not stager.is_alive(), "stage blocked on _PROCESS_LOCK"
        reg.discard_staged(result[0])
    finally:
        done.set()
        holder.join()


def test_runtime_context_reports_transport_and_model_continuity() -> None:
    async def _run() -> None:
        ctx = _make_ctx("transport-ctx")
        stdio = payload_of(await get_runtime_context(ctx=ctx))
        assert stdio["transport_assumption"] == "stdio-single-client"
        assert "stateless HTTP" in stdio["model_continuity"].replace(
            "Stateless", "stateless"
        )
        ctx.transport = "streamable-http"
        http = payload_of(await get_runtime_context(ctx=ctx))
        assert http["transport_assumption"] == "streamable-http"

    asyncio.run(_run())


def test_dossier_cancelled_after_staging_leaves_no_staged_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancel that lands while the worker is staging must not orphan the temp."""

    async def _run() -> None:
        ctx = _make_ctx("dossier-cancel-after-stage")
        csv_path = _write_synthetic_csv(tmp_path / "dossier.csv", n_rows=80)
        dataset_id, config_yaml = await _ingest_and_configure(csv_path, ctx)
        await run_topological_sweep(
            config_yaml=config_yaml,
            dataset_id=dataset_id,
            response_format="json",
            ctx=ctx,
        )
        session = session_mod._get_session(ctx)

        reg = registry_mod.registry
        staged_done = threading.Event()
        release = threading.Event()
        orig_stage = reg.stage_cluster_assignment

        def _gated_stage(*args, **kwargs):
            staged = orig_stage(*args, **kwargs)
            staged_done.set()
            release.wait(timeout=5.0)
            return staged

        monkeypatch.setattr(reg, "stage_cluster_assignment", _gated_stage)
        task = asyncio.create_task(
            generate_cluster_dossier(
                method="components", response_format="json", ctx=ctx
            )
        )
        await _wait_for_event(staged_done)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()

        deadline = time.monotonic() + 5.0
        while session.active_ops != 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert session.active_ops == 0
        assert not session._worker_mutation_lock.locked()
        assert session.clusters is None
        ca_dir = registry_mod._CLUSTER_ASSIGNMENTS_DIR
        assert [p.name for p in ca_dir.iterdir()] == []

    asyncio.run(_run())


def test_registry_init_sweeps_only_old_staged_files() -> None:
    runs_dir = registry_mod._RUNS_DIR
    ca_dir = registry_mod._CLUSTER_ASSIGNMENTS_DIR
    old_run = runs_dir / ".run_old.json.deadbeef.staged"
    fresh_run = runs_dir / ".run_new.json.cafebabe.staged"
    old_ca = ca_dir / ".ca_old.json.deadbeef.staged"
    committed = runs_dir / "run_real.json"
    for p in (old_run, fresh_run, old_ca, committed):
        p.write_text("{}", encoding="utf-8")
    stale = time.time() - 2 * registry_mod._STAGED_ORPHAN_MAX_AGE_S
    os.utime(old_run, (stale, stale))
    os.utime(old_ca, (stale, stale))

    # Init sweeps on construction; a second explicit call finds nothing left.
    reg = registry_mod.MCPRegistry()
    assert not old_run.exists()
    assert not old_ca.exists()
    assert fresh_run.exists()
    assert committed.exists()
    assert reg.sweep_orphaned_staged() == 0

    # The explicit call reports what it removed and leaves fresh files alone.
    os.utime(fresh_run, (stale, stale))
    assert reg.sweep_orphaned_staged() == 1
    assert not fresh_run.exists()
    assert committed.exists()


def test_dossier_refused_when_data_rebound_during_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dataset rebind mid-dossier must not publish labels for the old rows."""

    async def _run() -> None:
        ctx = _make_ctx("dossier-rebind")
        csv_path = _write_synthetic_csv(tmp_path / "rebind.csv", n_rows=80)
        dataset_id, config_yaml = await _ingest_and_configure(csv_path, ctx)
        await run_topological_sweep(
            config_yaml=config_yaml,
            dataset_id=dataset_id,
            response_format="json",
            ctx=ctx,
        )
        session = session_mod._get_session(ctx)
        run_id_before = session.latest_run_id

        entered = threading.Event()
        release = threading.Event()
        orig_resolve = clustering_mod.resolve_clusters

        def _gated_resolve(*args, **kwargs):
            entered.set()
            release.wait(timeout=5.0)
            return orig_resolve(*args, **kwargs)

        monkeypatch.setattr(clustering_mod, "resolve_clusters", _gated_resolve)
        task = asyncio.create_task(
            generate_cluster_dossier(
                method="components", response_format="json", ctx=ctx
            )
        )
        await _wait_for_event(entered)
        # Another tool binds a different (smaller) dataset; the run id is unchanged.
        session_mod._bind_session_data(session, session.data.iloc[:50].copy())
        release.set()
        res = json.loads(await task)

        assert session.latest_run_id == run_id_before
        assert res["status"] == "error"
        assert res["error_code"] == "SESSION_REVISION_CONFLICT"
        assert "rebound" in res["reason"]
        assert session.clusters is None
        assert len(session.data) == 50
        assert session.active_ops == 0
        ca_dir = registry_mod._CLUSTER_ASSIGNMENTS_DIR
        assert [p.name for p in ca_dir.iterdir()] == []

    asyncio.run(_run())


def test_bundle_reexport_keeps_unrelated_files_and_supports_nested_slug(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        csv_path = _write_synthetic_csv(tmp_path / "bundle_keep.csv", n_rows=80)
        ctx = _make_ctx("bundle-keep")
        dataset_id, config_yaml = await _ingest_and_configure(csv_path, ctx)
        await run_topological_sweep(
            config_yaml=config_yaml,
            dataset_id=dataset_id,
            response_format="json",
            ctx=ctx,
        )
        model = session_mod._get_session(ctx).model
        labels = np.zeros(len(model.data), dtype=int)
        output_dir = tmp_path / "exports"

        export_dataset_bundle(
            model, output_dir, "team/run1", cluster_labels=labels, include_clean=True
        )
        bundle = output_dir / "team" / "run1"
        notes = bundle / "NOTES.md"
        notes.write_text("keep me", encoding="utf-8")
        (bundle / "graph" / "figure.png").write_bytes(b"png")

        export_dataset_bundle(
            model,
            output_dir,
            "team/run1",
            cluster_labels=np.ones(len(model.data), dtype=int),
            include_clean=False,
        )

        assert notes.read_text(encoding="utf-8") == "keep me"
        assert (bundle / "graph" / "figure.png").exists()
        assert not (bundle / "tabular" / "clean.parquet").exists()
        nodes = pq.read_table(bundle / "graph" / "nodes.parquet").to_pandas()
        assert len(nodes) == 80
        # No staging directories are left beside the bundle.
        assert [p.name for p in (output_dir / "team").iterdir()] == ["run1"]

    asyncio.run(_run())
