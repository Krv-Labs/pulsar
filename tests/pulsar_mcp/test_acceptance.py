"""Stage 1 acceptance tests — Pulsar MCP service (build-spec §4).

Covers the sharp, locally-verifiable gates:
  - artifact round-trips DERIVED artifacts; interpret funcs run off the cold view (no PyO3)
  - phase-gated sweep preparation; async sweep ENQUEUES only with a validated config
  - over-row-cap ingest/sweep => structured error (not crash); noise => no_reliable_structure
  - curated tools interpret off the persisted artifact and carry a viz_payload (H0 vocab)
  - MCP initialize over Streamable HTTP: bearer => 200, missing/wrong bearer => 401

Run: `uv run pytest tests/pulsar_mcp/test_acceptance.py -q`
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PENGUINS = os.path.join(REPO, "demos", "penguins", "penguins.csv")
PENGUINS_CFG_PATH = os.path.join(REPO, "demos", "penguins", "initial_sweep_params.yaml")


def _penguins_cfg() -> dict:
    with open(PENGUINS_CFG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def store_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OBJECT_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("JOB_QUEUE_DIR", str(tmp_path / "queue"))
    yield


# --------------------------------------------------------------------------- #
# artifact round-trip + cold interpret (no live model / PyO3)
# --------------------------------------------------------------------------- #
def test_artifact_roundtrip_and_cold_interpret(store_env):
    from pulsar.artifacts import dump_artifact, load_artifact
    from pulsar.config import load_config
    from pulsar.mcp.diagnostics import diagnose_model
    from pulsar.mcp.interpreter import build_feature_evidence_index, resolve_clusters
    from pulsar.mcp.store import artifact_prefix, get_object_store
    from pulsar.pipeline import ThemaRS

    df = pd.read_csv(PENGUINS)
    model = ThemaRS(load_config(_penguins_cfg())).fit(df)
    w_ref = np.array(model._weighted_adjacency, copy=True)
    cr_live = resolve_clusters(model, method="auto")

    store = get_object_store()
    prefix = artifact_prefix("u", "ds", "cfg")
    art = json.loads(
        json.dumps(dump_artifact(model, dataset_id="ds", config_hash="cfg", prefix=prefix, store=store))
    )
    assert art["pulsarVersion"] == "0.2.3"  # D17 version pin recorded
    assert art["weightedAdjacency"]["format"] == "csr"

    del model  # no live model may leak into interpretation below
    view = load_artifact(art, store)

    assert np.allclose(view.weighted_adjacency, w_ref)
    assert view.cosmic_graph.number_of_nodes() == len(df)
    gm = diagnose_model(view)
    assert gm.n_nodes == len(df)
    cr = resolve_clusters(view, method="auto")
    assert list(cr.labels) == list(cr_live.labels)  # identical clustering off the cold view
    fei = build_feature_evidence_index(view, view.data, cr.labels)
    assert len(fei.cluster_bundles) >= 1


# --------------------------------------------------------------------------- #
# async sweep: enqueue -> worker -> status=done + artifact persisted
# --------------------------------------------------------------------------- #
def test_sweep_enqueue_worker_status(store_env):
    from pulsar.mcp.jobs import config_hash, get_job_queue
    from pulsar.mcp.store import get_object_store
    from pulsar.mcp.worker import run_job

    q = get_job_queue()
    store = get_object_store()
    cfg = _penguins_cfg()
    ch = config_hash(cfg)
    jid = q.enqueue(
        {"user_id": "u", "dataset_id": "ds", "config_hash": ch, "data_path": PENGUINS, "config": cfg}
    )
    assert q.status(jid)["status"] == "queued"  # enqueue does not block
    run_job(q.claim(), queue=q, store=store)
    st = q.status(jid)
    assert st["status"] == "done"
    assert st["structure_status"] == "ok"
    assert st["artifact_ref"]["datasetId"] == "ds"
    assert st["peak_rss_mb"] is not None and st["vcpu_ms"] is not None
    assert store.exists(f"u/ds/{ch}/artifact.json")


# --------------------------------------------------------------------------- #
# admission: over row cap => structured error, not crash
# --------------------------------------------------------------------------- #
def test_admission_over_row_cap(store_env):
    from pulsar.mcp.jobs import config_hash, get_job_queue
    from pulsar.mcp.store import get_object_store
    from pulsar.mcp.worker import run_job

    q = get_job_queue()
    store = get_object_store()
    cfg = _penguins_cfg()
    cfg.setdefault("output", {})["max_rows"] = 100  # penguins n=344 > 100
    jid = q.enqueue(
        {"user_id": "u", "dataset_id": "dsc", "config_hash": config_hash(cfg), "data_path": PENGUINS, "config": cfg}
    )
    run_job(q.claim(), queue=q, store=store)
    st = q.status(jid)
    assert st["status"] == "error"
    assert "row cap exceeded" in st["error"]


# --------------------------------------------------------------------------- #
# validity: noise-shaped data => no_reliable_structure (no crash)
# --------------------------------------------------------------------------- #
def test_noise_no_reliable_structure(store_env, tmp_path):
    from pulsar.mcp.jobs import config_hash, get_job_queue
    from pulsar.mcp.store import get_object_store
    from pulsar.mcp.worker import run_job

    q = get_job_queue()
    store = get_object_store()
    rng = np.random.default_rng(0)
    noise = pd.DataFrame(rng.normal(size=(200, 12)), columns=[f"f{i}" for i in range(12)])
    p = str(tmp_path / "noise.csv")
    noise.to_csv(p, index=False)
    cfg = {
        "sweep": {"pca": {"dimensions": [3], "seed": [42]}, "ball_mapper": {"epsilon": [0.5]}},
        "cosmic_graph": {"construction_threshold": "auto"},
        "output": {"n_reps": 2},
    }
    jid = q.enqueue(
        {"user_id": "u", "dataset_id": "dsn", "config_hash": config_hash(cfg), "data_path": p, "config": cfg}
    )
    run_job(q.claim(), queue=q, store=store)
    st = q.status(jid)
    assert st["status"] == "done"
    assert st["structure_status"] == "no_reliable_structure"


# --------------------------------------------------------------------------- #
# curated tools interpret off the persisted artifact + carry a viz_payload
# --------------------------------------------------------------------------- #
def test_curated_interpret_with_viz(store_env):
    import pulsar.mcp.tools.curated as C
    from pulsar.mcp.jobs import get_job_queue
    from pulsar.mcp.store import get_object_store
    from pulsar.mcp.worker import run_job

    store = get_object_store()
    q = get_job_queue()

    async def run():
        df = pd.read_csv(PENGUINS)
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        store.put("up/p.parquet", buf.getvalue())
        ds = json.loads(await C.ingest_dataset("up/p.parquet"))["structured"]["datasetId"]
        prep = json.loads(await C.prepare_sweep(ds, intent="penguins_demo"))
        assert prep["structured"]["phaseState"] == "validated"
        rs = json.loads(await C.run_topological_sweep(ds, config=prep["structured"]["config"]))
        ch = prep["structured"]["configHash"]
        assert rs["structured"]["status"] == "queued"
        assert rs["structured"]["artifactRef"]["configHash"] == ch
        run_job(q.claim(), queue=q, store=store)
        st = json.loads(await C.get_sweep_status(rs["structured"]["jobId"]))
        assert st["structured"]["status"] == "done"

        rd = json.loads(await C.diagnose_cosmic_graph(ds, ch))
        assert "n_nodes" in rd["structured"]
        assert rd["vizPayload"]["kind"] == "cosmic_graph"

        ts = json.loads(await C.get_threshold_stability_curve(ds, ch))
        assert ts["vizPayload"]["kind"] == "threshold_stability"
        assert ts["structured"]["detail"] == "summary"
        assert "thresholds" not in ts["structured"]
        assert ts["structured"]["curve_sample"]
        ts_full = json.loads(await C.get_threshold_stability_curve(ds, ch, detail="full"))
        assert ts_full["structured"]["thresholds"]
        assert ts_full["structured"]["component_counts"]

        sk = json.loads(await C.get_topological_skeleton(ds, ch, detail="nodes", max_nodes=5))
        assert sk["vizPayload"]["kind"] == "cosmic_graph"
        assert sk["structured"]["graph"]["detail"] == "nodes"
        assert "topological_summary" in sk["structured"]["graph"]
        sk_edges = json.loads(await C.get_topological_skeleton(ds, ch, detail="edges", max_edges=5))
        assert sk_edges["structured"]["graph"]["edges_returned"] <= 5

        rdo = json.loads(await C.generate_cluster_dossier(ds, ch))
        assert rdo["vizPayload"]["kind"] == "manifold3d"
        assert len(rdo["vizPayload"]["points"][0]) == 3  # real 3-D projection

        matrix = json.loads(await C.get_cluster_signal_matrix(ds, ch, max_clusters=3))
        assert "Topological Signal Matrix" in matrix["markdown"]
        assert matrix["structured"]["signal_matrix"]["clusters_returned"] <= 3
        matrix_json = json.loads(
            await C.get_cluster_signal_matrix(ds, ch, max_clusters=3, return_markdown=False)
        )
        assert "numeric_rows" in matrix_json["structured"]["signal_matrix"]

    asyncio.run(run())


def test_curated_artifact_analysis_tools_are_tenant_scoped(store_env):
    import pulsar.mcp.tools.curated as C
    from pulsar.mcp.jobs import get_job_queue
    from pulsar.mcp.store import get_object_store
    from pulsar.mcp.worker import run_job

    store = get_object_store()
    q = get_job_queue()

    async def run():
        df = pd.read_csv(PENGUINS)
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        store.put("up/p.parquet", buf.getvalue())
        ds = json.loads(await C.ingest_dataset("up/p.parquet", user_id="alice"))["structured"]["datasetId"]
        prep = json.loads(await C.prepare_sweep(ds, intent="penguins_demo", user_id="alice"))
        rs = json.loads(await C.run_topological_sweep(ds, config=prep["structured"]["config"], user_id="alice"))
        ch = prep["structured"]["configHash"]
        run_job(q.claim(), queue=q, store=store)
        st = json.loads(await C.get_sweep_status(rs["structured"]["jobId"]))
        assert st["structured"]["status"] == "done"

        alice_threshold = json.loads(await C.get_threshold_stability_curve(ds, ch, user_id="alice"))
        assert alice_threshold["structured"]["status"] == "ok"
        with pytest.raises(Exception, match="No artifact"):
            await C.get_threshold_stability_curve(ds, ch, user_id="bob")
        with pytest.raises(Exception, match="No artifact"):
            await C.get_topological_skeleton(ds, ch, user_id="bob")
        with pytest.raises(Exception, match="No artifact"):
            await C.get_cluster_signal_matrix(ds, ch, user_id="bob")

    asyncio.run(run())


def test_curated_sweep_rejects_missing_prepared_config(store_env):
    import pulsar.mcp.tools.curated as C
    from pulsar.mcp.store import get_object_store

    async def run():
        df = pd.read_csv(PENGUINS)
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        store = get_object_store()
        store.put("up/p.parquet", buf.getvalue())
        ds = json.loads(await C.ingest_dataset("up/p.parquet"))["structured"]["datasetId"]
        with pytest.raises(Exception, match="Validated config required"):
            await C.run_topological_sweep(ds)

    asyncio.run(run())


def test_sync_to_pulsar_one_new_snapshot(store_env):
    import pulsar.mcp.tools.curated as C
    from pulsar.mcp.jobs import get_job_queue
    from pulsar.mcp.store import get_object_store
    from pulsar.mcp.worker import run_job

    store = get_object_store()
    q = get_job_queue()

    async def run():
        df = pd.read_csv(PENGUINS)
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        store.put("up/p.parquet", buf.getvalue())
        ds = json.loads(await C.ingest_dataset("up/p.parquet"))["structured"]["datasetId"]
        df2 = df.copy()
        df2["synthetic_flag"] = 1.0
        b2 = io.BytesIO()
        df2.to_parquet(b2, index=False)
        store.put("up/mod.parquet", b2.getvalue())
        rsy = json.loads(await C.sync_to_pulsar(ds, "up/mod.parquet"))
        assert rsy["structured"]["newDatasetId"] != ds  # ONE new-fingerprint snapshot
        assert rsy["structured"]["recommendedNextTool"] == "prepare_sweep"
        prep = json.loads(await C.prepare_sweep(rsy["structured"]["newDatasetId"], intent="synced_penguins"))
        rsy = json.loads(
            await C.run_topological_sweep(
                rsy["structured"]["newDatasetId"], config=prep["structured"]["config"]
            )
        )
        run_job(q.claim(), queue=q, store=store)
        st = json.loads(await C.get_sweep_status(rsy["structured"]["jobId"]))
        assert st["structured"]["status"] == "done"

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# curated config/preprocessing helpers run over the tenant-scoped object store
# --------------------------------------------------------------------------- #
def test_curated_preprocessing_helpers(store_env):
    import pulsar.mcp.tools.curated as C
    from pulsar.mcp.store import get_object_store

    store = get_object_store()

    async def run():
        df = pd.read_csv(PENGUINS)
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        store.put("up/p.parquet", buf.getvalue())
        ds = json.loads(await C.ingest_dataset("up/p.parquet"))["structured"]["datasetId"]

        # workflow guide is stateless + curated-accurate (names only curated tools)
        wf = json.loads(await C.get_workflow_guide())
        assert "prepare_sweep" in wf["markdown"]

        # recommend / probe load via the object store (penguins has missing values)
        rec = json.loads(await C.recommend_preprocessing(ds))
        assert rec["markdown"] and rec["structured"]
        pc = json.loads(await C.probe_columns(ds, ["species", "bill_length_mm"]))
        assert pc["structured"]

        # a known-good config (from prepare_sweep) validates PASS
        prep = json.loads(await C.prepare_sweep(ds, intent="penguins_demo"))
        vok = json.loads(await C.validate_preprocessing_config(ds, prep["structured"]["configYaml"]))
        assert vok["structured"]["valid"] is True

    asyncio.run(run())


def test_curated_preprocessing_is_tenant_scoped(store_env):
    """The ported tools take user_id and must not resolve another tenant's dataset."""
    import pulsar.mcp.tools.curated as C
    from pulsar.mcp.store import get_object_store

    store = get_object_store()

    async def run():
        df = pd.read_csv(PENGUINS)
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        store.put("up/p.parquet", buf.getvalue())
        ds = json.loads(await C.ingest_dataset("up/p.parquet", user_id="alice"))["structured"]["datasetId"]

        assert json.loads(await C.recommend_preprocessing(ds, user_id="alice"))["structured"]
        with pytest.raises(Exception, match="not found"):
            await C.recommend_preprocessing(ds, user_id="bob")

    asyncio.run(run())


def test_curated_exclude_column_via_create_refine_prepare(store_env):
    """The discoverable exclude-a-column path: create_config -> refine_config(drop_columns) ->
    prepare_sweep. Proves the excluded column is actually dropped from the validated config (the
    session that 'assumed' species was excluded was silently wrong — this is the real fix)."""
    import yaml as _yaml

    import pulsar.mcp.tools.curated as C
    from pulsar.mcp.store import get_object_store

    store = get_object_store()

    async def run():
        df = pd.read_csv(PENGUINS)
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        store.put("up/p.parquet", buf.getvalue())
        ds = json.loads(await C.ingest_dataset("up/p.parquet"))["structured"]["datasetId"]

        cc = json.loads(await C.create_config(ds))
        cfg_yaml = cc["structured"]["config_yaml"]
        assert "drop_columns" in cfg_yaml  # the exclusion key is discoverable in the built config

        rc = json.loads(await C.refine_config(cfg_yaml, {"preprocessing.drop_columns": ["species"]}))
        assert rc["structured"]["status"] == "ok"
        refined = _yaml.safe_load(rc["structured"]["config_yaml"])

        prep = json.loads(await C.prepare_sweep(ds, config=refined))
        assert "species" in prep["structured"]["config"]["preprocessing"]["drop_columns"]

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# MCP initialize over Streamable HTTP: bearer=200, missing/wrong=401
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def http_server(tmp_path):
    import httpx

    port = _free_port()
    token = "testtoken"
    env = dict(
        os.environ,
        PULSAR_TRANSPORT="http",
        INTERNAL_MCP_TOKEN=token,
        PULSAR_MCP_HOST="127.0.0.1",
        PULSAR_MCP_PORT=str(port),
        PULSAR_MCP_PATH="/mcp",
        PULSAR_TOOLSET="curated",
        OBJECT_STORE_DIR=str(tmp_path / "s"),
        JOB_QUEUE_DIR=str(tmp_path / "q"),
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "pulsar.mcp.server"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(80):
            try:
                if httpx.get(base + "/healthz", timeout=1).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            pytest.fail("pulsar MCP HTTP server did not become healthy")
        yield base, token
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


def test_streamable_http_bearer_auth(http_server):
    import httpx

    base, token = http_server
    init = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "acceptance", "version": "0"},
        },
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

    assert httpx.get(base + "/healthz", timeout=5).status_code == 200  # unauthenticated liveness
    assert httpx.post(base + "/mcp", json=init, headers=headers, timeout=10).status_code == 401
    bad = httpx.post(base + "/mcp", json=init, headers={**headers, "Authorization": "Bearer wrong"}, timeout=10)
    assert bad.status_code == 401
    good = httpx.post(
        base + "/mcp", json=init, headers={**headers, "Authorization": f"Bearer {token}"}, timeout=10
    )
    assert good.status_code == 200  # MCP initialize succeeds with the bearer
