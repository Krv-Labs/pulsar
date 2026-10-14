"""Wire-level contract tests for the reporting, preprocessing and ingestion tools.

FastMCP validates ``structured_content`` against ``output_schema`` only when a
tool is called through a server, so every tool here is exercised through an
in-memory ``fastmcp.Client``.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastmcp import Client, FastMCP

from pulsar.mcp import registry as registry_mod
from pulsar.mcp import session as session_mod
from pulsar.mcp.registry import registry
from pulsar.mcp.server import register_server_tools
from pulsar.mcp.toolmeta import tool_meta_of
from pulsar.mcp.tools import ingestion as ingestion_mod
from pulsar.mcp.tools import preprocessing as preprocessing_mod
from pulsar.mcp.tools import reporting as reporting_mod

# tool name -> (read_only, idempotent)
EXPECTED_ANNOTATIONS: dict[str, tuple[bool, bool | None]] = {
    "probe_columns": (True, True),
    "recommend_preprocessing": (True, True),
    "repair_preprocessing_config": (True, True),
    "validate_preprocessing_config": (True, True),
    "ingest_dataset": (False, None),
    "sample_longitudinal_entities": (False, None),
    "begin_dataset_upload": (False, None),
    "append_dataset_chunk": (False, None),
    "finalize_dataset_upload": (False, None),
    "export_labeled_data": (False, True),
    "export_html_report": (False, True),
    "export_dataset_bundle": (False, None),
}
UPLOAD_TOOLS = (
    ingestion_mod.begin_dataset_upload,
    ingestion_mod.append_dataset_chunk,
    ingestion_mod.finalize_dataset_upload,
)
TIMEOUT_TOOLS = {"export_html_report", "export_dataset_bundle"}


@pytest.fixture(autouse=True)
def _isolate_mcp_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


def _write_csv(path: Path, n_rows: int = 24) -> Path:
    lines = ["f1,f2,cat,pid,t"] + [
        f"{i * 0.25},{(i % 6) * 0.5},{'a' if i % 2 == 0 else 'b'},{i // 4},{i % 4}"
        for i in range(n_rows)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _server() -> FastMCP:
    server = FastMCP("t")
    register_server_tools(server)
    # The upload trio is env-gated out of ALL_TOOLS_LIST; register it here with
    # the same metadata so its schemas are validated through the server too.
    from mcp.types import ToolAnnotations

    from pulsar.mcp.tools import ALL_TOOLS_LIST

    registered = {fn.__name__ for fn in ALL_TOOLS_LIST}
    for fn in UPLOAD_TOOLS:
        if fn.__name__ in registered:
            continue
        meta = tool_meta_of(fn)
        server.tool(
            annotations=ToolAnnotations(
                read_only_hint=meta.read_only,
                destructive_hint=meta.destructive,
                idempotent_hint=meta.idempotent,
                open_world_hint=False,
            ),
            output_schema=meta.output_schema,
        )(fn)
    return server


async def _call(client: Client, name: str, args: dict[str, Any], *, error=None):
    res = await client.call_tool(name, args, raise_on_error=False)
    if error is None:
        assert res.is_error is False, (name, args, res.content)
        assert isinstance(res.structured_content, dict), name
    else:
        assert res.is_error is True, (name, args)
        assert res.structured_content["error_code"] == error, res.structured_content
        assert res.structured_content["agent_action"]
        assert res.structured_content["tool"] == name
    return res


def test_every_owned_tool_declares_metadata_and_described_params():
    async def _run():
        server = _server()
        async with Client(server) as client:
            tools = {t.name: t for t in await client.list_tools()}
        for name, (read_only, idempotent) in EXPECTED_ANNOTATIONS.items():
            tool = tools[name]
            assert tool.annotations.read_only_hint is read_only, name
            assert tool.annotations.idempotent_hint is idempotent, name
            assert tool.annotations.open_world_hint is False, name
            assert tool.annotations.destructive_hint is False, name
            assert tool.output_schema and tool.output_schema.get("properties"), name
            assert len(tool.description) < 700, name
            for prop, schema in tool.input_schema.get("properties", {}).items():
                assert schema.get("description"), (name, prop)
        for name in TIMEOUT_TOOLS:
            got = await server.get_tool(name)
            assert got.timeout == reporting_mod._TOOL_TIMEOUT_S
        for name in set(EXPECTED_ANNOTATIONS) - TIMEOUT_TOOLS:
            assert (await server.get_tool(name)).timeout is None, name
        for name in (
            "export_labeled_data",
            "export_html_report",
            "export_dataset_bundle",
        ):
            assert "run_id" in tools[name].input_schema["properties"]
        for name in (
            "probe_columns",
            "recommend_preprocessing",
            "repair_preprocessing_config",
            "validate_preprocessing_config",
        ):
            assert "run_id" not in tools[name].input_schema["properties"]

    asyncio.run(_run())


def test_ingestion_tools_through_client(tmp_path):
    async def _run():
        csv = _write_csv(tmp_path / "panel.csv")
        async with Client(_server()) as client:
            res = await _call(client, "ingest_dataset", {"path": str(csv)})
            dataset_id = res.structured_content["dataset_id"]
            res = await _call(
                client,
                "sample_longitudinal_entities",
                {
                    "dataset_id": dataset_id,
                    "entity_column": "pid",
                    "max_entities": 3,
                    "min_observations": 2,
                    "time_column": "t",
                    "min_time_points": 2,
                },
            )
            assert res.structured_content["sampled_entities"] == 3
            assert res.structured_content["source_dataset_id"] == dataset_id
            # optional-field variant: no time column
            res = await _call(
                client,
                "sample_longitudinal_entities",
                {"dataset_id": dataset_id, "entity_column": "pid"},
            )
            assert res.structured_content["time_column"] is None

            # Upload trio: base64 and utf-8 variants.
            res = await _call(client, "begin_dataset_upload", {"filename": "u.csv"})
            upload_id = res.structured_content["upload_id"]
            body = csv.read_bytes()
            res = await _call(
                client,
                "append_dataset_chunk",
                {
                    "upload_id": upload_id,
                    "chunk": base64.b64encode(body[:40]).decode(),
                },
            )
            assert res.structured_content["bytes_received"] == 40
            await _call(
                client,
                "append_dataset_chunk",
                {
                    "upload_id": upload_id,
                    "chunk": body[40:].decode("utf-8"),
                    "encoding": "utf-8",
                },
            )
            res = await _call(
                client, "finalize_dataset_upload", {"upload_id": upload_id}
            )
            assert (
                res.structured_content["dataset_id"].startswith("ds_")
                or res.structured_content["dataset_id"]
            )

    asyncio.run(_run())


def test_error_paths_for_ingestion_and_preprocessing(tmp_path):
    async def _run():
        csv = _write_csv(tmp_path / "e.csv")
        async with Client(_server()) as client:
            # validate before any data is loaded into the session
            await _call(
                client,
                "validate_preprocessing_config",
                {"config_yaml": "run: {name: x}"},
                error="SESSION_DATA_MISSING",
            )
            await _call(
                client,
                "ingest_dataset",
                {"path": str(tmp_path / "missing.csv")},
                error="FILE_NOT_FOUND",
            )
            dataset_id = (
                await _call(client, "ingest_dataset", {"path": str(csv)})
            ).structured_content["dataset_id"]
            await _call(
                client,
                "sample_longitudinal_entities",
                {"dataset_id": "ds_nope", "entity_column": "pid"},
                error="DATASET_ID_UNKNOWN",
            )
            await _call(
                client,
                "sample_longitudinal_entities",
                {"dataset_id": dataset_id, "entity_column": "pid", "max_entities": 0},
                error="INVALID_SAMPLE_SIZE",
            )
            await _call(
                client,
                "append_dataset_chunk",
                {"upload_id": "upload_nope", "chunk": "YQ=="},
                error="UPLOAD_ID_UNKNOWN",
            )
            await _call(
                client,
                "finalize_dataset_upload",
                {"upload_id": "upload_nope"},
                error="UPLOAD_ID_UNKNOWN",
            )
            await _call(
                client,
                "probe_columns",
                {"dataset_id": dataset_id, "columns": [f"c{i}" for i in range(21)]},
                error="TOO_MANY_COLUMNS",
            )
            await _call(
                client,
                "probe_columns",
                {"dataset_id": "ds_nope", "columns": ["f1"]},
                error="DATASET_ID_UNKNOWN",
            )
            await _call(
                client,
                "recommend_preprocessing",
                {},
                error="MISSING_INPUT",
            )
            await _call(
                client,
                "recommend_preprocessing",
                {"dataset_id": "ds_nope"},
                error="DATASET_ID_UNKNOWN",
            )
            await _call(
                client,
                "recommend_preprocessing",
                {"dataset_id": dataset_id, "rationale_limit": 0},
                error="INVALID_RATIONALE_LIMIT",
            )
            await _call(
                client,
                "repair_preprocessing_config",
                {
                    "error_message": "x",
                    "config_yaml": "run: {name: x}",
                    "dataset_id": "ds_nope",
                },
                error="DATASET_ID_UNKNOWN",
            )
            await _call(
                client,
                "repair_preprocessing_config",
                {
                    "error_message": "something unrelated",
                    "config_yaml": "run: {name: x}",
                    "dataset_id": dataset_id,
                },
                error="ERROR_PATTERN_UNRECOGNIZED",
            )
            await _call(
                client,
                "repair_preprocessing_config",
                {
                    "error_message": "x",
                    "config_yaml": "- just\n- a list",
                    "dataset_id": dataset_id,
                },
                error="CONFIG_YAML_INVALID",
            )

    asyncio.run(_run())


def test_export_run_id_and_precondition_errors(tmp_path):
    async def _run():
        out = tmp_path / "out.csv"
        csv = _write_csv(tmp_path / "r.csv")
        dataset = registry.register_dataset(str(csv))
        assignment = registry.save_cluster_assignment(
            run_id="run_saved",
            dataset_id=dataset.dataset_id,
            method="components",
            interpretation_edge_weight_threshold=0.25,
            threshold_source="explicit",
            construction_threshold=0.25,
            labels=[0, 1] * 12,
        )
        aid = assignment.cluster_assignment_id
        async with Client(_server()) as client:
            names = {"0": "A", "1": "B"}
            await _call(
                client,
                "export_labeled_data",
                {
                    "cluster_assignment_id": "ca_nope",
                    "cluster_names": names,
                    "output_path": str(out),
                },
                error="CLUSTER_ASSIGNMENT_ID_UNKNOWN",
            )
            await _call(
                client,
                "export_labeled_data",
                {
                    "cluster_assignment_id": aid,
                    "cluster_names": names,
                    "output_path": str(out),
                    "run_id": "run_gone",
                },
                error="RUN_ID_UNKNOWN",
            )
            await _call(
                client,
                "export_labeled_data",
                {
                    "cluster_assignment_id": aid,
                    "cluster_names": {"x": "A"},
                    "output_path": str(out),
                },
                error="CLUSTER_NAMES_INVALID",
            )
            ok = await _call(
                client,
                "export_labeled_data",
                {
                    "cluster_assignment_id": aid,
                    "cluster_names": names,
                    "output_path": str(out),
                },
            )
            assert ok.structured_content["row_count"] == 24
            for tool, args in (
                ("export_html_report", {}),
                (
                    "export_dataset_bundle",
                    {"output_dir": str(tmp_path / "b"), "slug": "s"},
                ),
            ):
                await _call(
                    client, tool, {**args, "run_id": "run_gone"}, error="RUN_ID_UNKNOWN"
                )
                await _call(client, tool, args, error="MODEL_NOT_LOADED")

    asyncio.run(_run())


def test_full_loop_variants_and_exports(tmp_path):
    async def _run():
        csv = _write_csv(tmp_path / "loop.csv")
        async with Client(_server()) as client:
            dataset_id = (
                await _call(client, "ingest_dataset", {"path": str(csv)})
            ).structured_content["dataset_id"]

            for detail in ("summary", "full"):
                for fmt in ("markdown", "json"):
                    res = await _call(
                        client,
                        "recommend_preprocessing",
                        {
                            "dataset_id": dataset_id,
                            "detail": detail,
                            "response_format": fmt,
                        },
                    )
                    sc = res.structured_content
                    assert sc["detail"] == detail
                    assert ("rationale" in sc) is (detail == "full")
                    text = res.content[0].text
                    assert text.startswith("#") is (fmt == "markdown")
            summary = (
                await _call(
                    client, "recommend_preprocessing", {"dataset_id": dataset_id}
                )
            ).structured_content
            assert summary["preprocessing_yaml"]

            for fmt in ("markdown", "json"):
                res = await _call(
                    client,
                    "probe_columns",
                    {
                        "dataset_id": dataset_id,
                        "columns": ["f1", "cat", "nope"],
                        "response_format": fmt,
                    },
                )
                assert res.structured_content["missing_columns"] == ["nope"]
                assert res.structured_content["columns_returned"] == 2
                assert res.content[0].text.startswith("#") is (fmt == "markdown")

            cfg = await _call(client, "create_config", {"dataset_id": dataset_id})
            config_yaml = cfg.structured_content["config_yaml"]
            await _call(
                client,
                "validate_preprocessing_config",
                {"config_yaml": "- not\n- a mapping"},
                error="CONFIG_YAML_INVALID",
            )
            res = await _call(
                client, "validate_preprocessing_config", {"config_yaml": config_yaml}
            )
            assert res.structured_content["valid"] is True
            assert res.structured_content["input_rows"] == 24

            res = await _call(
                client,
                "repair_preprocessing_config",
                {
                    "error_message": "NaN values remain after imputation: 'f1' (3 rows)",
                    "config_yaml": config_yaml,
                    "dataset_id": dataset_id,
                },
            )
            sc = res.structured_content
            assert sc["change_count"] == 1 and sc["changes"][0]["column"] == "f1"
            assert "impute" in sc["config_yaml"]
            assert "## Patched Config" in res.content[0].text

            sweep = await _call(
                client,
                "run_topological_sweep",
                {"dataset_id": dataset_id, "response_format": "json"},
            )
            run1 = sweep.structured_content["run_id"]
            doss = await _call(
                client, "generate_cluster_dossier", {"detail": "summary"}
            )
            aid = doss.structured_content["cluster_assignment_id"]
            cluster_ids = registry.get_cluster_assignment(aid).cluster_ids
            names = {str(c): f"cohort {c}" for c in cluster_ids}

            labeled = tmp_path / "labeled.csv"
            res = await _call(
                client,
                "export_labeled_data",
                {
                    "cluster_assignment_id": aid,
                    "cluster_names": names,
                    "output_path": str(labeled),
                    "run_id": run1,
                },
            )
            assert res.structured_content["row_count"] == 24 and labeled.exists()

            res = await _call(client, "export_html_report", {})
            assert "warning" in res.structured_content
            res = await _call(
                client,
                "export_html_report",
                {"cluster_names": names, "run_id": run1},
            )
            assert "warning" not in res.structured_content
            assert Path(res.structured_content["report_path"]).exists()
            await _call(
                client,
                "export_html_report",
                {"cluster_names": {"zz": "bad"}},
                error="CLUSTER_NAMES_INVALID",
            )

            res = await _call(
                client,
                "export_dataset_bundle",
                {
                    "output_dir": str(tmp_path / "bundle"),
                    "slug": "s1",
                    "cluster_names": names,
                    "include_clean": True,
                    "run_id": run1,
                },
            )
            assert Path(res.structured_content["bundle_path"]).is_dir()
            res = await _call(
                client,
                "export_dataset_bundle",
                {
                    "output_dir": str(tmp_path / "bundle"),
                    "slug": "s2",
                    "cluster_assignment_id": aid,
                    "layout": "zeros",
                    "edges_threshold": 0.1,
                    "write_manifest": False,
                },
            )
            await _call(
                client,
                "export_dataset_bundle",
                {
                    "output_dir": str(tmp_path / "bundle"),
                    "slug": "s3",
                    "edges_threshold": 7.0,
                },
                error="INVALID_EDGES_THRESHOLD",
            )

            # A second sweep makes the session clusters belong to an older run.
            sweep2 = await _call(
                client,
                "run_topological_sweep",
                {"dataset_id": dataset_id, "response_format": "json"},
            )
            run2 = sweep2.structured_content["run_id"]
            assert run2 != run1
            # A new sweep drops the session's clusters entirely.
            await _call(
                client,
                "export_html_report",
                {"run_id": run2},
                error="CLUSTERS_MISSING",
            )
            # Clusters that still carry the older run's stamp are reported stale
            # for the newer run.
            (session,) = session_mod._sessions.values()
            session.clusters = pd.Series([0, 1] * 12)
            session.clusters_run_id = run1
            await _call(
                client,
                "export_html_report",
                {"run_id": run2},
                error="CLUSTERS_STALE",
            )
            await _call(
                client,
                "export_dataset_bundle",
                {"output_dir": str(tmp_path / "bundle"), "slug": "s4", "run_id": run2},
                error="CLUSTERS_STALE",
            )
            # The older run is still addressable and matches the clusters.
            await _call(client, "export_html_report", {"run_id": run1})

    asyncio.run(_run())


def test_schemas_are_declared_on_every_owned_function():
    for mod in (ingestion_mod, preprocessing_mod, reporting_mod):
        for name in dir(mod):
            fn = getattr(mod, name)
            if name in EXPECTED_ANNOTATIONS and callable(fn):
                meta = tool_meta_of(fn)
                assert meta is not None, name
                assert meta.output_schema["type"] == "object", name
