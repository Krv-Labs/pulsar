from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastmcp import Client, Context
from fastmcp.server.dependencies import _background_task_headers
from fastmcp.utilities.tasks import TaskConfig
import pytest

from pulsar.mcp import registry as registry_mod
from pulsar.mcp import session as session_mod
from pulsar.mcp.server import AgnosticFastMCP, register_server_tools
from pulsar.mcp.tools import _sweep_task_config, _tasks_enabled


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


def _write_sample_csv(csv_path: Path, n_rows: int = 24) -> Path:
    lines = ["f1,f2,cat"] + [
        f"{i * 0.25},{(i % 6) * 0.5},{'a' if i % 2 == 0 else 'b'}"
        for i in range(n_rows)
    ]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


def _extract_tool_json(result: Any) -> dict[str, Any]:
    blocks = result.content if hasattr(result, "content") else result
    return json.loads(blocks[0].text)


async def _get_task_enabled_tools() -> tuple[set[str], str, str]:
    server = AgnosticFastMCP("test-sweep-task")
    register_server_tools(server)
    tools = await server.list_tools()
    enabled_names = {t.name for t in tools if t.task_config.mode != "forbidden"}
    sweep_tool = await server.get_tool("run_topological_sweep")
    diag_tool = await server.get_tool("diagnose_cosmic_graph")
    assert sweep_tool is not None
    assert diag_tool is not None
    return enabled_names, sweep_tool.task_config.mode, diag_tool.task_config.mode


async def _run_sweep_and_diagnose_workflow(
    csv_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    server = AgnosticFastMCP("test-continuity")
    register_server_tools(server)
    async with Client(server) as client:
        ingest_res = await client.call_tool(
            "ingest_dataset",
            {"path": str(csv_path)},
        )
        dataset_id = _extract_tool_json(ingest_res)["dataset_id"]
        await client.call_tool("create_config", {"dataset_id": dataset_id})
        sweep_res = await client.call_tool(
            "run_topological_sweep",
            {"dataset_id": dataset_id, "response_format": "json"},
        )
        diag_res = await client.call_tool(
            "diagnose_cosmic_graph",
            {"response_format": "json"},
        )
    return _extract_tool_json(sweep_res), _extract_tool_json(diag_res)


def test_sweep_task_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PULSAR_MCP_ENABLE_TASKS", raising=False)
    assert _tasks_enabled() is False
    assert _sweep_task_config() is False
    enabled_names, sweep_mode, diag_mode = asyncio.run(_get_task_enabled_tools())
    assert enabled_names == set()
    assert sweep_mode == "forbidden"
    assert diag_mode == "forbidden"


def test_sweep_task_enabled_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PULSAR_MCP_ENABLE_TASKS", "1")
    assert _tasks_enabled() is True
    cfg = _sweep_task_config()
    assert isinstance(cfg, TaskConfig)
    assert cfg.mode == "optional"
    enabled_names, sweep_mode, diag_mode = asyncio.run(_get_task_enabled_tools())
    assert enabled_names == {"run_topological_sweep"}
    assert sweep_mode == "optional"
    assert diag_mode == "forbidden"


def test_session_key_background_task_context() -> None:
    server = AgnosticFastMCP("test-bg-ctx")
    ctx = Context(fastmcp=server)
    assert session_mod._session_key(ctx) == "default"
    token = _background_task_headers.set({"mcp-session-id": "http-sess-42"})
    try:
        assert session_mod._session_key(ctx) == "http-sess-42"
    finally:
        _background_task_headers.reset(token)


def test_client_session_continuity_across_sweep_and_diagnose(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PULSAR_MCP_ENABLE_TASKS", "1")
    csv_path = _write_sample_csv(tmp_path / "sample.csv", n_rows=24)
    sweep_payload, diag_payload = asyncio.run(
        _run_sweep_and_diagnose_workflow(csv_path),
    )
    assert sweep_payload.get("status") == "ok"
    assert sweep_payload["key_metrics"]["n_nodes"] == 24
    assert diag_payload.get("status") == "ok"
    assert diag_payload["scale"]["n_nodes"] == 24
    assert diag_payload["run_id"] == sweep_payload["run_id"]

