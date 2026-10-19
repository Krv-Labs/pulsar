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
import yaml

from pulsar.mcp.toolmeta import default_tool_timeout
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
    "draft_preprocessing_plan": (False, False),
    "edit_preprocessing_plan": (False, False),
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
TIMEOUT_TOOLS = {
    "export_html_report",
    "export_dataset_bundle",
    "draft_preprocessing_plan",
}


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
            assert got.timeout == default_tool_timeout()
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
            "draft_preprocessing_plan",
            "edit_preprocessing_plan",
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
            res = await _call(
                client,
                "validate_preprocessing_config",
                {"config_yaml": "run: {name: x}"},
                error="SESSION_DATA_MISSING",
            )
            assert "dataset_id" in res.structured_content["agent_action"]
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
                "draft_preprocessing_plan",
                {"dataset_id": ""},
                error="MISSING_INPUT",
            )
            await _call(
                client,
                "draft_preprocessing_plan",
                {"dataset_id": "ds_nope"},
                error="DATASET_ID_UNKNOWN",
            )
            await _call(
                client,
                "draft_preprocessing_plan",
                {"dataset_id": dataset_id, "target_columns": ["no_such_col"]},
                error="INVALID_ARGUMENT",
            )
            await _call(
                client,
                "edit_preprocessing_plan",
                {"plan_id": "pp_nope"},
                error="PLAN_ID_UNKNOWN",
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


def test_imputation_ensemble_plan_config_sweep_and_cache_through_client(tmp_path):
    async def _run():
        csv = tmp_path / "ensemble.csv"
        pd.DataFrame(
            {
                "visits": [1, 2, 4, None, 3, 2] * 10,
                "score": [None if i % 5 == 0 else (i % 7) + 0.25 for i in range(60)],
                "other": [(i % 11) * 0.1 for i in range(60)],
            }
        ).to_csv(csv, index=False)
        async with Client(_server()) as client:
            dataset_id = (
                await _call(client, "ingest_dataset", {"path": str(csv)})
            ).structured_content["dataset_id"]
            plan = (
                await _call(
                    client,
                    "draft_preprocessing_plan",
                    {
                        "dataset_id": dataset_id,
                        "stakes": False,
                        "detail": "full",
                    },
                )
            ).structured_content
            question = next(
                q
                for q in plan["questions"]
                if q["topic"] == "blank_meaning"
                and q["applies_to_preview"] == ["visits"]
            )
            edited = (
                await _call(
                    client,
                    "edit_preprocessing_plan",
                    {
                        "plan_id": plan["plan_id"],
                        "base_revision": plan["revision"],
                        "answers": [
                            {
                                "question_id": question["question_id"],
                                "option_ids": ["none", "unknown"],
                            }
                        ],
                        "edits": [
                            {
                                "column": "score",
                                "impute_method": ["fill_mean", "sample_normal"],
                                "indicator": "none",
                            }
                        ],
                    },
                )
            ).structured_content
            assert edited["validation"]["valid"] is True
            created = (
                await _call(
                    client,
                    "create_config",
                    {
                        "dataset_id": dataset_id,
                        "plan_id": plan["plan_id"],
                    },
                )
            ).structured_content
            cfg = yaml.safe_load(created["config_yaml"])
            candidates = cfg["preprocessing"]["impute"]["visits"]
            assert {s["method"]: s.get("indicator", True) for s in candidates} == {
                "fill_value": False,
                "fill_median": True,
            }
            strategy = created["sweep_strategy"]
            assert strategy["imputation_variants"] == 2
            first = (
                await _call(
                    client,
                    "run_topological_sweep",
                    {
                        "dataset_id": dataset_id,
                        "detail": "full",
                    },
                )
            ).structured_content
            assert not first["projection_cached"]
            assert first["metrics"]["n_ball_maps"] == strategy["estimated_ball_maps"]

            # An epsilon-only retry reuses every realization's cached embeddings.
            cfg["sweep"]["ball_mapper"]["epsilon"] = {"values": [0.5]}
            cached = (
                await _call(
                    client,
                    "run_topological_sweep",
                    {
                        "dataset_id": dataset_id,
                        "config_yaml": yaml.safe_dump(cfg),
                        "detail": "full",
                    },
                )
            ).structured_content
            assert cached["projection_cached"]
            projection_count = len(strategy["projection_dimensions"]) * len(
                strategy["projection_seeds"]
            )
            assert cached["metrics"]["n_ball_maps"] == 2 * projection_count

            # Extra stochastic draws invalidate the cache and contribute new maps.
            refined = (
                await _call(
                    client, "refine_config", {"overrides": {"n_imputations": 3}}
                )
            ).structured_content
            assert (
                yaml.safe_load(refined["config_yaml"])["preprocessing"]["n_imputations"]
                == 3
            )
            expanded = (
                await _call(
                    client,
                    "run_topological_sweep",
                    {
                        "dataset_id": dataset_id,
                        "detail": "full",
                    },
                )
            ).structured_content
            assert not expanded["projection_cached"]
            assert expanded["metrics"]["n_ball_maps"] == 3 * projection_count

    asyncio.run(_run())


def test_create_config_cost_deduplicates_deterministic_candidates_through_client(
    tmp_path,
):
    async def _run():
        csv = tmp_path / "deterministic.csv"
        pd.DataFrame(
            {"x": [1.5, None, 2.5, 3.5, 4.5] * 8, "y": list(range(40))}
        ).to_csv(csv, index=False)
        async with Client(_server()) as client:
            dataset_id = (
                await _call(client, "ingest_dataset", {"path": str(csv)})
            ).structured_content["dataset_id"]
            plan = (
                await _call(
                    client,
                    "draft_preprocessing_plan",
                    {"dataset_id": dataset_id, "stakes": False},
                )
            ).structured_content
            await _call(
                client,
                "edit_preprocessing_plan",
                {
                    "plan_id": plan["plan_id"],
                    "edits": [
                        {"column": "x", "impute_method": ["fill_mean", "fill_mean"]}
                    ],
                },
            )
            created = (
                await _call(
                    client,
                    "create_config",
                    {"dataset_id": dataset_id, "plan_id": plan["plan_id"]},
                )
            ).structured_content
            strategy = created["sweep_strategy"]
            assert strategy["imputation_variants"] == 1
            sweep = (
                await _call(
                    client,
                    "run_topological_sweep",
                    {"dataset_id": dataset_id, "detail": "full"},
                )
            ).structured_content
            assert sweep["metrics"]["n_ball_maps"] == strategy["estimated_ball_maps"]

    asyncio.run(_run())


def test_full_loop_variants_and_exports(tmp_path):
    async def _run():
        csv = _write_csv(tmp_path / "loop.csv")
        async with Client(_server()) as client:
            dataset_id = (
                await _call(client, "ingest_dataset", {"path": str(csv)})
            ).structured_content["dataset_id"]

            plan_ids: list[str] = []
            for detail in ("summary", "full"):
                for fmt in ("markdown", "json"):
                    res = await _call(
                        client,
                        "draft_preprocessing_plan",
                        {
                            "dataset_id": dataset_id,
                            "detail": detail,
                            "response_format": fmt,
                        },
                    )
                    sc = res.structured_content
                    assert sc["detail"] == detail
                    assert ("entries" in sc) is (detail == "full")
                    assert ("preprocessing_yaml" in sc) is (detail == "full")
                    assert sc["preprocessing_yaml_chars"] > 0
                    text = res.content[0].text
                    assert text.startswith("#") is (fmt == "markdown")
                    plan_ids.append(sc["plan_id"])

            assert len(plan_ids) == 4
            last_plan_id = plan_ids[-1]
            # Edit plan (markdown + json) on distinct fresh drafts so q1 is open in each
            for pid_for_fmt, fmt in zip(plan_ids[-2:], ("markdown", "json")):
                edit_res = await _call(
                    client,
                    "edit_preprocessing_plan",
                    {
                        "plan_id": pid_for_fmt,
                        "answers": [
                            {"question_id": "q1", "option_id": "all_amounts"},
                            {"question_id": "q2", "option_id": "none_of_them"},
                        ],
                        "response_format": fmt,
                    },
                )
                esc = edit_res.structured_content
                assert esc["plan_id"] == pid_for_fmt
                assert esc["revision"] >= 2
                assert esc["validation"]["valid"] is True
                assert edit_res.content[0].text.startswith("#") is (fmt == "markdown")

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

            # create_config with explicit plan_id
            cfg = await _call(
                client,
                "create_config",
                {"dataset_id": dataset_id, "plan_id": last_plan_id},
            )
            config_yaml = cfg.structured_content["config_yaml"]
            assert (
                cfg.structured_content["preprocessing_plan"]["source"]
                == "explicit_plan"
            )
            assert (
                cfg.structured_content["preprocessing_plan"]["plan_id"] == last_plan_id
            )
            fp = cfg.structured_content["preprocessing_plan"]["plan_fingerprint"]
            assert f"# preprocessing plan {last_plan_id} revision " in config_yaml
            assert f"({fp})" in config_yaml

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


def test_draft_preprocessing_plan_dataset_id_is_required_in_schema():
    async def _run():
        async with Client(_server()) as client:
            tools = {t.name: t for t in await client.list_tools()}
        schema = tools["draft_preprocessing_plan"].input_schema
        assert "dataset_id" in schema["required"]
        assert "default" not in schema["properties"]["dataset_id"]
        assert "Required" in schema["properties"]["dataset_id"]["description"]
        edit_schema = tools["edit_preprocessing_plan"].input_schema
        assert "plan_id" in edit_schema["required"]
        validate = tools["validate_preprocessing_config"].input_schema
        assert "dataset_id" in validate["properties"]
        assert "dataset_id" not in validate.get("required", [])

    asyncio.run(_run())


def test_validate_preprocessing_config_by_dataset_id_is_stateless_and_read_only(
    tmp_path,
):
    async def _run():
        csv = _write_csv(tmp_path / "v.csv")
        async with Client(_server()) as client:
            dataset_id = (
                await _call(client, "ingest_dataset", {"path": str(csv)})
            ).structured_content["dataset_id"]
            cfg = (
                await _call(client, "create_config", {"dataset_id": dataset_id})
            ).structured_content["config_yaml"]
            # Simulate a stateless transport / another frame bound last.
            session_mod._sessions.clear()
            res = await _call(
                client,
                "validate_preprocessing_config",
                {"config_yaml": cfg, "dataset_id": dataset_id},
            )
            assert res.structured_content["valid"] is True
            assert res.structured_content["input_rows"] == 24
            # Read-only: the session was not rebound to this dataset.
            for session in session_mod._sessions.values():
                assert session.data is None
            await _call(
                client,
                "validate_preprocessing_config",
                {"config_yaml": cfg, "dataset_id": "ds_nope"},
                error="DATASET_ID_UNKNOWN",
            )

    asyncio.run(_run())


def test_validate_preprocessing_config_does_not_rebind_other_session_data(tmp_path):
    async def _run():
        a = _write_csv(tmp_path / "a.csv", n_rows=24)
        b = _write_csv(tmp_path / "b.csv", n_rows=12)
        async with Client(_server()) as client:
            id_a = (
                await _call(client, "ingest_dataset", {"path": str(a)})
            ).structured_content["dataset_id"]
            id_b = (
                await _call(client, "ingest_dataset", {"path": str(b)})
            ).structured_content["dataset_id"]
            cfg = (
                await _call(client, "create_config", {"dataset_id": id_a})
            ).structured_content["config_yaml"]
            await _call(
                client,
                "draft_preprocessing_plan",
                {"dataset_id": id_a, "stakes": False},
            )
            bound = {k: v.data_dataset_id for k, v in session_mod._sessions.items()}
            res = await _call(
                client,
                "validate_preprocessing_config",
                {"config_yaml": cfg, "dataset_id": id_b},
            )
            assert res.structured_content["input_rows"] == 12
            after = {k: v.data_dataset_id for k, v in session_mod._sessions.items()}
            assert after == bound

    asyncio.run(_run())


def test_plan_edit_conflicts_and_create_config_provenance(tmp_path: Path):
    from tests.fixtures.preprocessing_traps import make_preprocessing_traps

    async def _run():
        traps_csv = tmp_path / "traps.csv"
        make_preprocessing_traps(n=120, seed=0).to_csv(traps_csv, index=False)
        other_csv = _write_csv(tmp_path / "other.csv", n_rows=20)
        async with Client(_server()) as client:
            ds_traps = (
                await _call(client, "ingest_dataset", {"path": str(traps_csv)})
            ).structured_content["dataset_id"]
            ds_other = (
                await _call(client, "ingest_dataset", {"path": str(other_csv)})
            ).structured_content["dataset_id"]

            # 1. create_config WITHOUT plan_id drafts baseline defaults and warns PREPROCESSING_QUESTIONS_OPEN
            cfg_no_plan = (
                await _call(client, "create_config", {"dataset_id": ds_traps})
            ).structured_content
            assert cfg_no_plan["preprocessing_plan"]["source"] == "baseline_defaults"
            assert cfg_no_plan["preprocessing_plan"]["plan_id"].startswith("pp_")
            assert cfg_no_plan["preprocessing_plan"]["open_questions"] > 0
            preview = cfg_no_plan["preprocessing_plan"]["open_questions_preview"]
            assert 1 <= len(preview) <= 3
            assert all(
                {"question_id", "topic", "rows_affected", "stakes"} <= set(item)
                and item["stakes"] is not None
                for item in preview
            )
            open_warn = next(
                w
                for w in cfg_no_plan["warnings"]
                if w["code"] == "PREPROCESSING_QUESTIONS_OPEN"
            )
            assert cfg_no_plan["preprocessing_plan"]["plan_id"] in open_warn["message"]
            assert "open_questions_preview=" in open_warn["message"]
            assert (
                "call draft_preprocessing_plan(dataset_id) for the full cards, "
                "or edit_preprocessing_plan(plan_id, ...)"
            ) in open_warn["message"]

            # 2. draft_preprocessing_plan (summary + full × json + markdown)
            for fmt in ("json", "markdown"):
                full_res = await _call(
                    client,
                    "draft_preprocessing_plan",
                    {
                        "dataset_id": ds_traps,
                        "stakes": False,
                        "detail": "full",
                        "response_format": fmt,
                    },
                )
                assert full_res.structured_content["detail"] == "full"
                assert "entries" in full_res.structured_content
                if fmt == "markdown":
                    assert "- Most-missing columns:" in full_res.content[0].text

            draft_sc = (
                await _call(
                    client,
                    "draft_preprocessing_plan",
                    {
                        "dataset_id": ds_traps,
                        "stakes": True,
                        "evidence_limit": 3,
                        "response_format": "json",
                    },
                )
            ).structured_content
            plan_id = draft_sc["plan_id"]
            assert draft_sc["revision"] == 1
            assert draft_sc["counts"]["open_questions"] >= 4
            mm_cols = draft_sc["evidence"]["most_missing_columns"]
            assert 1 <= len(mm_cols) <= 3
            assert draft_sc["evidence"]["most_missing_columns_omitted"] >= 1
            assert [
                (round(-c["missing_fraction"], 6), c["column"]) for c in mm_cols
            ] == sorted(
                (round(-c["missing_fraction"], 6), c["column"]) for c in mm_cols
            )

            # Unknown plan_id on edit -> PLAN_ID_UNKNOWN
            await _call(
                client,
                "edit_preprocessing_plan",
                {
                    "plan_id": "pp_unknown",
                    "edits": [{"column": "bmi", "role": "drop"}],
                },
                error="PLAN_ID_UNKNOWN",
            )

            # Empty answers + empty edits -> INVALID_ARGUMENT
            await _call(
                client,
                "edit_preprocessing_plan",
                {"plan_id": plan_id},
                error="INVALID_ARGUMENT",
            )

            # Answer a question at base_revision=1 -> revision 2
            q_id = draft_sc["questions"][0]["question_id"]
            opt_id = draft_sc["questions"][0]["options"][0]["option_id"]
            edit_sc = (
                await _call(
                    client,
                    "edit_preprocessing_plan",
                    {
                        "plan_id": plan_id,
                        "base_revision": 1,
                        "answers": [{"question_id": q_id, "option_id": opt_id}],
                        "source": "user",
                        "response_format": "json",
                    },
                )
            ).structured_content
            assert edit_sc["revision"] == 2
            assert "confirmed_without_change" in edit_sc["counts"]

            # Stale base_revision=1 -> SESSION_REVISION_CONFLICT
            await _call(
                client,
                "edit_preprocessing_plan",
                {
                    "plan_id": plan_id,
                    "base_revision": 1,
                    "edits": [{"column": "bmi", "role": "drop"}],
                },
                error="SESSION_REVISION_CONFLICT",
            )

            # Invalid answer (unknown question_id and bad option_id) -> INVALID_ARGUMENT
            await _call(
                client,
                "edit_preprocessing_plan",
                {
                    "plan_id": plan_id,
                    "answers": [{"question_id": "q_nope", "option_id": "none"}],
                },
                error="INVALID_ARGUMENT",
            )
            await _call(
                client,
                "edit_preprocessing_plan",
                {
                    "plan_id": plan_id,
                    "answers": [{"question_id": q_id, "option_id": "bad_option_id"}],
                },
                error="INVALID_ARGUMENT",
            )
            # D5: answering outcome_columns with invalid option_id (e.g. 'none') gives targeted agent_action
            outcome_q_id = f"q{draft_sc['counts']['open_questions']}"
            outcome_err = await _call(
                client,
                "edit_preprocessing_plan",
                {
                    "plan_id": plan_id,
                    "answers": [{"question_id": outcome_q_id, "option_id": "none"}],
                },
                error="INVALID_ARGUMENT",
            )
            assert (
                outcome_err.structured_content["details"]["topic"] == "outcome_columns"
            )
            assert (
                outcome_err.structured_content["agent_action"]
                == "To mark outcome columns, call edit_preprocessing_plan with edits=[{column, role: 'target'}]."
            )

            # Invalid edit -> INVALID_ARGUMENT
            await _call(
                client,
                "edit_preprocessing_plan",
                {
                    "plan_id": plan_id,
                    "edits": [{"column": "unknown_col", "role": "drop"}],
                },
                error="INVALID_ARGUMENT",
            )
            no_blanks_err = await _call(
                client,
                "edit_preprocessing_plan",
                {
                    "plan_id": plan_id,
                    "edits": [
                        {
                            "column": "age",
                            "impute": {"method": "fill_median"},
                        }
                    ],
                },
                error="INVALID_ARGUMENT",
            )
            assert (
                no_blanks_err.structured_content["agent_action"]
                == "Remove the impute/indicator edit for this column; it has no blanks."
            )

            # Edit that causes dry-run failure still returns tool success with validation.valid == False
            bad_edit = (
                await _call(
                    client,
                    "edit_preprocessing_plan",
                    {
                        "plan_id": plan_id,
                        "edits": [
                            {
                                "column": "patient_id",
                                "role": "feature",
                                "encode": None,
                            }
                        ],
                        "response_format": "json",
                    },
                )
            ).structured_content
            assert bad_edit["validation"]["valid"] is False
            assert bad_edit["validation"]["error"]
            assert bad_edit["processed_space"] is None

            # Restore patient_id role="id"
            fixed_edit = (
                await _call(
                    client,
                    "edit_preprocessing_plan",
                    {
                        "plan_id": plan_id,
                        "edits": [{"column": "patient_id", "role": "id"}],
                        "response_format": "json",
                    },
                )
            ).structured_content
            assert fixed_edit["validation"]["valid"] is True

            # 3. R1: Re-drafting gets a NEW plan_id and keeps the edited plan intact
            redraft_sc = (
                await _call(
                    client,
                    "draft_preprocessing_plan",
                    {
                        "dataset_id": ds_traps,
                        "stakes": False,
                        "response_format": "json",
                    },
                )
            ).structured_content
            assert redraft_sc["plan_id"] != plan_id
            assert redraft_sc["revision"] == 1
            # Editing the first plan with a stale revision after re-draft still conflicts
            await _call(
                client,
                "edit_preprocessing_plan",
                {
                    "plan_id": plan_id,
                    "base_revision": 1,
                    "edits": [{"column": "bmi", "role": "drop"}],
                },
                error="SESSION_REVISION_CONFLICT",
            )
            # create_config without plan_id after edits warns PREPROCESSING_PLAN_NOT_USED with edited plan_id
            cfg_after_edit = (
                await _call(client, "create_config", {"dataset_id": ds_traps})
            ).structured_content
            assert cfg_after_edit["preprocessing_plan"]["plan_id"] not in {
                plan_id,
                redraft_sc["plan_id"],
            }
            not_used_warn = next(
                w
                for w in cfg_after_edit["warnings"]
                if w["code"] == "PREPROCESSING_PLAN_NOT_USED"
            )
            assert plan_id in not_used_warn["message"]
            assert f"revision {fixed_edit['revision']}" in not_used_warn["message"]
            assert f"Pass plan_id={plan_id} to use it." in not_used_warn["message"]

            # 4. create_config with wrong dataset's plan_id -> INVALID_ARGUMENT
            await _call(
                client,
                "create_config",
                {"dataset_id": ds_other, "plan_id": plan_id},
                error="INVALID_ARGUMENT",
            )
            # Unknown plan_id -> INVALID_ARGUMENT
            await _call(
                client,
                "create_config",
                {"dataset_id": ds_traps, "plan_id": "pp_missing"},
                error="INVALID_ARGUMENT",
            )

    asyncio.run(_run())


def test_validate_preprocessing_config_malformed_yaml_is_config_yaml_invalid(
    tmp_path: Path,
) -> None:
    """Malformed YAML gets the shared CONFIG_YAML_INVALID code, not a generic failure."""

    async def _run() -> None:
        csv = _write_csv(tmp_path / "v.csv")
        async with Client(_server()) as client:
            dataset_id = (
                await _call(client, "ingest_dataset", {"path": str(csv)})
            ).structured_content["dataset_id"]
            res = await _call(
                client,
                "validate_preprocessing_config",
                {"config_yaml": "nope: [", "dataset_id": dataset_id},
                error="CONFIG_YAML_INVALID",
            )
            assert res.content[0].text.strip().startswith("{")
            assert res.structured_content["details"] == {"valid": False}

    asyncio.run(_run())


def test_edit_preprocessing_plan_concurrent_lost_update_guard(tmp_path: Path) -> None:
    """Concurrent edits on the same base_revision allow one update and reject the racing one."""
    from tests.fixtures.preprocessing_traps import make_preprocessing_traps

    async def _run() -> None:
        traps_csv = tmp_path / "traps.csv"
        make_preprocessing_traps(n=80, seed=0).to_csv(traps_csv, index=False)
        async with Client(_server()) as client:
            ds_id = (
                await _call(client, "ingest_dataset", {"path": str(traps_csv)})
            ).structured_content["dataset_id"]
            draft = (
                await _call(
                    client,
                    "draft_preprocessing_plan",
                    {"dataset_id": ds_id, "stakes": False, "response_format": "json"},
                )
            ).structured_content
            plan_id = draft["plan_id"]
            res_a, res_b = await asyncio.gather(
                client.call_tool(
                    "edit_preprocessing_plan",
                    {
                        "plan_id": plan_id,
                        "base_revision": 1,
                        "edits": [{"column": "bmi", "role": "drop"}],
                        "response_format": "json",
                    },
                    raise_on_error=False,
                ),
                client.call_tool(
                    "edit_preprocessing_plan",
                    {
                        "plan_id": plan_id,
                        "base_revision": 1,
                        "edits": [{"column": "age", "role": "drop"}],
                        "response_format": "json",
                    },
                    raise_on_error=False,
                ),
            )
            statuses = sorted(
                [
                    res_a.structured_content["status"],
                    res_b.structured_content["status"],
                ]
            )
            assert statuses == ["error", "ok"]
            ok_sc = (
                res_a.structured_content
                if res_a.structured_content["status"] == "ok"
                else res_b.structured_content
            )
            err_sc = (
                res_a.structured_content
                if res_a.structured_content["status"] == "error"
                else res_b.structured_content
            )
            assert ok_sc["revision"] == 2
            assert err_sc["error_code"] == "SESSION_REVISION_CONFLICT"

    asyncio.run(_run())


def test_plan_workers_internal_keyerror_is_not_masked_as_unknown_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Internal KeyError inside plan workers maps to DRAFT_PLAN_FAILED / EDIT_PLAN_FAILED."""

    async def _run() -> None:
        csv = _write_csv(tmp_path / "v.csv")
        async with Client(_server()) as client:
            dataset_id = (
                await _call(client, "ingest_dataset", {"path": str(csv)})
            ).structured_content["dataset_id"]
            draft = (
                await _call(
                    client,
                    "draft_preprocessing_plan",
                    {"dataset_id": dataset_id, "stakes": False},
                )
            ).structured_content
            plan_id = draft["plan_id"]

            def _boom(*_args: object, **_kwargs: object) -> object:
                raise KeyError("boom")

            monkeypatch.setattr(preprocessing_mod, "_draft_plan_worker", _boom)
            monkeypatch.setattr(preprocessing_mod, "_edit_plan_worker", _boom)

            await _call(
                client,
                "draft_preprocessing_plan",
                {"dataset_id": dataset_id, "stakes": False},
                error="DRAFT_PLAN_FAILED",
            )
            await _call(
                client,
                "edit_preprocessing_plan",
                {
                    "plan_id": plan_id,
                    "edits": [{"column": "a", "role": "drop"}],
                },
                error="EDIT_PLAN_FAILED",
            )

    asyncio.run(_run())


def test_validate_preprocessing_config_dummy_count_excludes_missingness_markers_and_measures_nans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """validate_preprocessing_config separates dummy_expansion_count from missingness_flag_count and measures nan_remaining."""
    from pulsar.preprocessing import PreprocessLayout

    async def _run() -> None:
        nan = float("nan")
        df = pd.DataFrame(
            {
                "num_miss": [1.0, nan, 3.0, 4.0, nan, 6.0],
                "cat": ["red", "blue", "green", "red", "blue", "green"],
            }
        )
        csv = tmp_path / "dummy_markers.csv"
        df.to_csv(csv, index=False)
        cfg_yaml = """
preprocessing:
  impute:
    num_miss:
      method: fill_median
      indicator: true
  encode:
    cat:
      method: one_hot
      drop_first: false
"""
        async with Client(_server()) as client:
            dataset_id = (
                await _call(client, "ingest_dataset", {"path": str(csv)})
            ).structured_content["dataset_id"]
            res = (
                await _call(
                    client,
                    "validate_preprocessing_config",
                    {"config_yaml": cfg_yaml, "dataset_id": dataset_id},
                )
            ).structured_content
            assert res["missingness_flag_count"] == 1
            assert res["dummy_expansion_count"] == 3
            assert res["nan_remaining"] == 0

            # Also verify nan_remaining is computed from df_proc rather than hard-coded.
            def _fake_preprocess(
                data: pd.DataFrame, _cfg: object
            ) -> tuple[pd.DataFrame, PreprocessLayout]:
                out = pd.DataFrame(
                    {
                        "num_miss": [1.0, nan, 3.0, 4.0, nan, 6.0],
                        "num_miss_was_missing": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                    }
                )
                layout = PreprocessLayout(
                    feature_names=tuple(out.columns),
                    vocab={},
                    n_rows=len(out),
                )
                return out, layout

            monkeypatch.setattr(
                preprocessing_mod, "preprocess_dataframe", _fake_preprocess
            )
            res_nan = (
                await _call(
                    client,
                    "validate_preprocessing_config",
                    {"config_yaml": cfg_yaml, "dataset_id": dataset_id},
                )
            ).structured_content
            assert res_nan["nan_remaining"] == 2

    asyncio.run(_run())


def test_calibrate_processed_space_catches_unexpected_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_calibrate_processed_space logs and returns None when preprocess_dataframe raises RuntimeError."""
    import pulsar.mcp.preprocessing as mcp_prep_mod
    from pulsar.mcp.preprocessing_plan import draft_plan

    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
    plan = draft_plan(df, dataset_id="ds_test", stakes=False)

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(mcp_prep_mod, "preprocess_dataframe", _boom)
    assert mcp_prep_mod._calibrate_processed_space(df, plan) is None
