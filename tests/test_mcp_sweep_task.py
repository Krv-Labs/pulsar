from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import fastmcp_tasks
import pytest
from fastmcp import Client, Context, FastMCP
from fastmcp.server.dependencies import _background_task_headers
from fastmcp.utilities.tasks import TaskConfig

from tests.mcp_helpers import payload_of
from pulsar.mcp import registry as registry_mod
from pulsar.mcp import session as session_mod
from pulsar.mcp.server import register_server_tools
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
    if hasattr(result, "content") and hasattr(result, "structured_content"):
        return payload_of(result)
    blocks = result.content if hasattr(result, "content") else result
    return json.loads(blocks[0].text)


async def _get_task_enabled_tools() -> tuple[set[str], str, str]:
    server = FastMCP("test-sweep-task")
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
    server = FastMCP("test-continuity")
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
    server = FastMCP("test-bg-ctx")
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


# ---------------------------------------------------------------------------
# F2: tasks must be real (extension serves tasks/* methods) or not advertised
# ---------------------------------------------------------------------------
def test_tasks_extension_registered_with_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastmcp.utilities.tasks import TASKS_EXTENSION_ID

    monkeypatch.setenv("PULSAR_MCP_ENABLE_TASKS", "1")
    server = FastMCP("test-ext")
    register_server_tools(server)
    ext = server._extensions[TASKS_EXTENSION_ID]
    assert isinstance(ext, fastmcp_tasks.TasksExtension)
    assert len(ext.methods()) > 0


def test_tasks_not_advertised_when_fastmcp_tasks_missing(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import builtins

    from fastmcp.utilities.tasks import TASKS_EXTENSION_ID

    real_import = builtins.__import__

    def _no_tasks(name, *args, **kwargs):
        if name == "fastmcp_tasks" or name.startswith("fastmcp_tasks."):
            raise ImportError("simulated missing fastmcp-tasks")
        return real_import(name, *args, **kwargs)

    monkeypatch.setenv("PULSAR_MCP_ENABLE_TASKS", "1")
    monkeypatch.setattr(builtins, "__import__", _no_tasks)
    server = FastMCP("test-missing")
    with caplog.at_level("ERROR", logger="pulsar.mcp.server"):
        register_server_tools(server)
    assert TASKS_EXTENSION_ID not in server._extensions
    assert "fastmcp[tasks]" in caplog.text
    tools = asyncio.run(server.list_tools())
    assert all(t.task_config.mode == "forbidden" for t in tools)


async def _task_roundtrip(csv_path: Path) -> tuple[dict, dict, dict]:
    server = FastMCP("test-task-e2e")
    register_server_tools(server)
    async with Client(server) as client:
        ingest = _extract_tool_json(
            await client.call_tool("ingest_dataset", {"path": str(csv_path)})
        )
        dataset_id = ingest["dataset_id"]
        await client.call_tool("create_config", {"dataset_id": dataset_id})
        handle = await fastmcp_tasks.call_tool_task(
            client,
            "run_topological_sweep",
            {"dataset_id": dataset_id, "response_format": "json"},
        )
        task_result = await asyncio.wait_for(handle.result(), timeout=120)
        ctx_res = await client.call_tool("get_runtime_context", {})
    return (
        _extract_tool_json(task_result),
        _extract_tool_json(ctx_res),
        {"dataset_id": dataset_id},
    )


def test_sweep_runs_as_background_task_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PULSAR_MCP_ENABLE_TASKS", "1")
    csv_path = _write_sample_csv(tmp_path / "sample.csv", n_rows=24)
    sweep, runtime, ids = asyncio.run(_task_roundtrip(csv_path))
    assert sweep.get("status") == "ok"
    assert sweep["run_id"]
    assert sweep["key_metrics"]["n_nodes"] == 24
    assert runtime.get("latest_run_id") == sweep["run_id"]


# ---------------------------------------------------------------------------
# F3: "default" session only for stdio / in-memory transports
# ---------------------------------------------------------------------------
def _fake_ctx(
    transport: str | None,
    *,
    header: str | None = None,
    session_id: str = "per-request-uuid",
) -> SimpleNamespace:
    headers = {"mcp-session-id": header} if header else {}
    return SimpleNamespace(
        request_context=SimpleNamespace(
            session=None, request=SimpleNamespace(headers=headers)
        ),
        _session=None,
        transport=transport,
        session_id=session_id,
        is_background_task=False,
    )


def test_session_key_stdio_is_default() -> None:
    assert session_mod._extract_context_session_id(_fake_ctx("stdio")) == "default"


def test_session_key_in_memory_is_default() -> None:
    assert session_mod._extract_context_session_id(_fake_ctx(None)) == "default"


def test_session_key_http_with_header_uses_header() -> None:
    ctx = _fake_ctx("streamable-http", header="hdr-sess-1")
    assert session_mod._extract_context_session_id(ctx) == "hdr-sess-1"


def test_session_key_http_without_header_is_isolated_not_default() -> None:
    a = session_mod._extract_context_session_id(
        _fake_ctx("streamable-http", session_id="req-a")
    )
    b = session_mod._extract_context_session_id(
        _fake_ctx("streamable-http", session_id="req-b")
    )
    assert (a, b) == ("req-a", "req-b")
    assert "default" not in (a, b)


def test_session_key_in_memory_client_is_default() -> None:
    server = FastMCP("test-inmem-key")
    seen: list[str] = []

    @server.tool
    async def whoami(ctx: Context) -> str:
        seen.append(session_mod._session_key(ctx))
        return "ok"

    async def _run() -> None:
        async with Client(server) as client:
            await client.call_tool("whoami", {})

    asyncio.run(_run())
    assert seen == ["default"]


def test_sweep_progress_reaches_real_client_from_worker(tmp_path: Path) -> None:
    """In-fit progress from the worker thread must reach the client.

    The worker runs via run_in_executor, which does not copy contextvars; if
    the request context is lost, ctx.report_progress silently no-ops and only
    the loop-side events ("format response", "complete") arrive.
    """
    csv_path = _write_sample_csv(tmp_path / "progress.csv")
    messages: list[str | None] = []

    async def _on_progress(
        progress: float, total: float | None, message: str | None
    ) -> None:
        messages.append(message)

    async def _run() -> None:
        server = FastMCP("test-progress")
        register_server_tools(server)
        async with Client(server, progress_handler=_on_progress) as client:
            ingest_res = await client.call_tool(
                "ingest_dataset", {"path": str(csv_path)}
            )
            dataset_id = _extract_tool_json(ingest_res)["dataset_id"]
            await client.call_tool("create_config", {"dataset_id": dataset_id})
            messages.clear()
            await client.call_tool(
                "run_topological_sweep",
                {"dataset_id": dataset_id, "response_format": "json"},
            )

    asyncio.run(_run())
    in_fit = [m for m in messages if m not in ("format response", "complete")]
    assert in_fit, f"no in-fit progress reached the client: {messages}"
