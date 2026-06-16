"""Regression tests for agent-safe config references."""
from __future__ import annotations

import asyncio
import io
import json
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PENGUINS = os.path.join(REPO, "demos", "penguins", "penguins.csv")


def test_curated_create_refine_prepare_uses_config_ref(tmp_path, monkeypatch):
    import pulsar.mcp.tools.curated as C
    from pulsar.mcp.store import get_object_store

    monkeypatch.setenv("OBJECT_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("JOB_QUEUE_DIR", str(tmp_path / "queue"))
    store = get_object_store()

    async def run():
        df = pd.read_csv(PENGUINS)
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        store.put("up/p.parquet", buf.getvalue())
        ds = json.loads(await C.ingest_dataset("up/p.parquet"))["structured"]["datasetId"]

        created = json.loads(await C.create_config(ds))["structured"]
        assert created["config_ref"].startswith("cfg_")

        refined = json.loads(
            await C.refine_config(
                dataset_id=ds,
                config_ref=created["config_ref"],
                overrides={"preprocessing.drop_columns": ["species"]},
            )
        )["structured"]
        assert refined["config_ref"].startswith("cfg_")
        assert refined["config_ref"] != created["config_ref"]

        prepared = json.loads(await C.prepare_sweep(ds, config_ref=refined["config_ref"]))["structured"]
        assert prepared["source"] == "config_ref"
        assert prepared["config"]["preprocessing"]["drop_columns"] == ["species"]

    asyncio.run(run())


def test_refine_config_accepts_legacy_escaped_newline_yaml(tmp_path, monkeypatch):
    import pulsar.mcp.tools.curated as C
    from pulsar.mcp.store import get_object_store

    monkeypatch.setenv("OBJECT_STORE_DIR", str(tmp_path / "store"))
    store = get_object_store()

    async def run():
        df = pd.read_csv(PENGUINS)
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        store.put("up/p.parquet", buf.getvalue())
        ds = json.loads(await C.ingest_dataset("up/p.parquet"))["structured"]["datasetId"]
        created = json.loads(await C.create_config(ds))["structured"]
        escaped_yaml = created["config_yaml"].replace("\n", "\\n")

        refined = json.loads(
            await C.refine_config(
                escaped_yaml,
                overrides={"preprocessing.drop_columns": ["species"]},
                dataset_id=ds,
            )
        )["structured"]

        assert refined["status"] == "ok"
        assert refined["config"]["preprocessing"]["drop_columns"] == ["species"]

    asyncio.run(run())
