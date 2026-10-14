"""Wire-level contract tests for the sweep / config / meta MCP tools.

FastMCP validates ``structured_content`` against ``output_schema`` only when a
call goes through the server, so every tool here is exercised via an in-memory
``fastmcp.Client`` for every ``detail`` x ``response_format`` variant.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastmcp import Client, FastMCP

from pulsar.mcp import registry as registry_mod
from pulsar.mcp import session as session_mod
from pulsar.mcp.server import register_server_tools
from tests.mcp_helpers import payload_of, text_of

READ_ONLY = {
    "get_sweep_history": True,
    "compare_sweeps": True,
    "validate_config": True,
    "get_workflow_guide": True,
    "get_runtime_context": True,
    "characterize_dataset": True,
    "run_topological_sweep": False,
    "create_config": False,
    "refine_config": False,
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
    session_mod.model_cache.clear()
    yield
    session_mod._sessions.clear()
    session_mod.model_cache.clear()


def _write_sample_csv(csv_path: Path, n_rows: int = 24) -> Path:
    lines = ["f1,f2,cat"] + [
        f"{i * 0.25},{(i % 6) * 0.5},{'a' if i % 2 == 0 else 'b'}"
        for i in range(n_rows)
    ]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


def _server() -> FastMCP:
    server = FastMCP("t")
    register_server_tools(server)
    return server


def _run(coro):
    return asyncio.run(coro)


async def _ok(client: Client, name: str, args: dict | None = None):
    res = await client.call_tool(name, args or {})
    assert res.is_error is False, (name, args, text_of(res))
    assert isinstance(res.structured_content, dict)
    return res


async def _err(client: Client, name: str, args: dict, code: str):
    res = await client.call_tool(name, args, raise_on_error=False)
    assert res.is_error is True, (name, args)
    assert res.structured_content["error_code"] == code
    assert res.structured_content["agent_action"]
    assert res.structured_content["tool"] == name
    return res


def test_wire_metadata_for_owned_tools():
    async def go():
        async with Client(_server()) as client:
            tools = {t.name: t for t in await client.list_tools()}
            for name, read_only in READ_ONLY.items():
                t = tools[name]
                assert t.annotations.read_only_hint is read_only, name
                assert t.annotations.open_world_hint is False, name
                assert t.output_schema, name
                assert t.description and t.description.strip(), name
                assert len(t.description) < 900, name
                for pname, prop in (t.input_schema.get("properties") or {}).items():
                    assert prop.get("description", "").strip(), (name, pname)
            for name in (
                "get_sweep_history",
                "compare_sweeps",
                "validate_config",
                "get_workflow_guide",
                "get_runtime_context",
                "characterize_dataset",
            ):
                assert tools[name].annotations.idempotent_hint is True, name

    _run(go())


def test_all_variants_validate_through_client(tmp_path: Path):
    csv = _write_sample_csv(tmp_path / "d.csv")

    async def go():
        async with Client(_server()) as client:
            # runtime context / guide (empty session first, then later again)
            ctx = await _ok(client, "get_runtime_context")
            p = payload_of(ctx)
            assert p["model_cache"]["size"] == 0
            assert p["transport"]
            guide = await _ok(client, "get_workflow_guide")
            assert payload_of(guide)["guide"] == text_of(guide)

            # empty history, every variant
            for detail in ("table", "summary", "full"):
                for fmt in ("markdown", "json"):
                    await _ok(
                        client,
                        "get_sweep_history",
                        {"detail": detail, "response_format": fmt},
                    )

            ing = await _ok(client, "ingest_dataset", {"path": str(csv)})
            dataset_id = payload_of(ing)["dataset_id"]

            # characterize: both formats share structured content
            md = await _ok(
                client,
                "characterize_dataset",
                {"dataset_id": dataset_id, "response_format": "markdown"},
            )
            js = await _ok(
                client,
                "characterize_dataset",
                {"dataset_id": dataset_id, "response_format": "json"},
            )
            assert md.structured_content == js.structured_content
            assert text_of(md) != text_of(js)
            await _ok(client, "characterize_dataset", {"csv_path": str(csv)})

            cfg = await _ok(client, "create_config", {"dataset_id": dataset_id})
            config_yaml = payload_of(cfg)["config_yaml"]

            val = await _ok(
                client,
                "validate_config",
                {"config_yaml": config_yaml, "dataset_id": dataset_id},
            )
            assert payload_of(val)["normalized_config_yaml"]

            # refine: three formats, same structured content
            overrides = {
                "overrides": {"cosmic_graph": {"construction_threshold": "auto"}}
            }
            refined = [
                await _ok(
                    client, "refine_config", {**overrides, "response_format": fmt}
                )
                for fmt in ("json", "markdown", "yaml")
            ]
            assert (
                refined[0].structured_content
                == refined[1].structured_content
                == refined[2].structured_content
            )
            assert (
                text_of(refined[2]).strip()
                == refined[0].structured_content["config_yaml"].strip()
            )
            await _ok(client, "refine_config", {"config_yaml": config_yaml})

            # sweep: detail x response_format, same structured content per detail
            sweeps = {}
            for detail in ("summary", "full"):
                for fmt in ("markdown", "json"):
                    res = await _ok(
                        client,
                        "run_topological_sweep",
                        {
                            "dataset_id": dataset_id,
                            "detail": detail,
                            "response_format": fmt,
                            "include_config_yaml": detail == "summary",
                            "component_limit": 3,
                        },
                    )
                    sweeps[(detail, fmt)] = res
                    assert res.structured_content["detail"] == detail
            run_ids = [r.structured_content["run_id"] for r in sweeps.values()]
            assert len(set(run_ids)) == 4
            assert "metrics" in sweeps[("full", "json")].structured_content
            assert "key_metrics" in sweeps[("summary", "json")].structured_content
            assert text_of(sweeps[("summary", "markdown")]).startswith("# Sweep Run")

            # history with real runs, every variant
            for detail in ("table", "summary", "full"):
                for fmt in ("markdown", "json"):
                    res = await _ok(
                        client,
                        "get_sweep_history",
                        {"detail": detail, "response_format": fmt},
                    )
                    assert res.structured_content["n_runs"] == 4
            await _ok(
                client, "compare_sweeps", {"run_a": run_ids[0], "run_b": run_ids[1]}
            )

            ctx = await _ok(client, "get_runtime_context")
            mc = payload_of(ctx)["model_cache"]
            assert mc["size"] >= 1 and run_ids[-1] in mc["run_ids"]

    _run(go())


def test_error_paths_reach_client_with_error_code(tmp_path: Path):
    async def go():
        async with Client(_server()) as client:
            await _err(
                client,
                "characterize_dataset",
                {"dataset_id": "nope"},
                "DATASET_ID_UNKNOWN",
            )
            await _err(
                client,
                "characterize_dataset",
                {},
                "DATASET_NOT_SPECIFIED",
            )
            await _err(
                client, "create_config", {"dataset_id": "nope"}, "DATASET_ID_UNKNOWN"
            )
            await _err(client, "refine_config", {}, "ACTIVE_CONFIG_MISSING")
            await _err(
                client,
                "validate_config",
                {"config_yaml": "dataset: {path: x}\nbogus: 1\n"},
                "YAML_SCHEMA_MISMATCH",
            )
            await _err(
                client,
                "run_topological_sweep",
                {},
                "CONFIG_NOT_PROVIDED",
            )
            await _err(
                client,
                "run_topological_sweep",
                {"component_limit": 0},
                "INVALID_COMPONENT_LIMIT",
            )
            await _err(
                client,
                "compare_sweeps",
                {"run_a": "missing", "run_b": "missing"},
                "RUN_ID_UNKNOWN",
            )
            # get_sweep_history / get_workflow_guide / get_runtime_context have no
            # reachable domain error through the wire: an invalid enum is rejected
            # by FastMCP input validation before the tool runs.
            res = await client.call_tool(
                "get_sweep_history", {"detail": "bogus"}, raise_on_error=False
            )
            assert res.is_error is True

    _run(go())


def test_sweep_history_defensive_errors_have_codes():
    from pulsar.mcp.tools.sweeping import get_sweep_history

    async def go():
        bad = await get_sweep_history(detail="bogus")  # type: ignore[arg-type]
        assert bad.is_error is True
        assert bad.structured_content["error_code"] == "INVALID_DETAIL"
        bad = await get_sweep_history(response_format="x")  # type: ignore[arg-type]
        assert bad.structured_content["error_code"] == "UNKNOWN_RESPONSE_FORMAT"
        assert json.loads(bad.content[0].text) == bad.structured_content

    _run(go())
