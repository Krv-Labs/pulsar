#!/usr/bin/env python3
"""Smoke-test Pulsar MCP: import path, live handshake, and stdio launch.

Used by release CI on every published wheel platform (Linux x86_64/ARM, Windows,
macOS Intel/ARM) and by the dedicated macOS Intel CI job. Exercises the same
launch surface Gemini/Claude hit via ``uvx ... pulsar-mcp`` (stdio MCP).
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import sysconfig
from pathlib import Path


_REQUIRED_TOOLS = frozenset(
    {
        "ingest_dataset",
        "sample_longitudinal_entities",
        "create_config",
        "run_topological_sweep",
        "diagnose_cosmic_graph",
        "generate_cluster_dossier",
        "get_workflow_guide",
        "get_runtime_context",
    }
)


def _check_imports() -> None:
    importlib.import_module("pulsar._pulsar")

    server = importlib.import_module("pulsar.mcp.server")
    tools = importlib.import_module("pulsar.mcp.tools")
    names = {fn.__name__ for fn in tools.ALL_TOOLS_LIST}
    missing = _REQUIRED_TOOLS - names
    if missing:
        raise SystemExit(f"missing tools: {sorted(missing)}")
    if getattr(server, "mcp", None) is None:
        raise SystemExit("pulsar.mcp.server.mcp missing")

    from importlib.metadata import entry_points

    eps = entry_points()
    if hasattr(eps, "select"):
        scripts = {ep.name for ep in eps.select(group="console_scripts")}
    else:  # pragma: no cover - Python 3.10
        scripts = {ep.name for ep in eps.get("console_scripts", [])}
    if "pulsar-mcp" not in scripts:
        raise SystemExit(f"pulsar-mcp entry point missing; have={sorted(scripts)[:20]}")

    print(f"ok: imports (_pulsar + {len(names)} tools + entry point)")


async def _assert_healthy(client, label: str) -> None:
    if not client.is_connected():
        raise SystemExit(f"{label}: client not connected")

    tools = await client.list_tools()
    names = {t.name for t in tools}
    missing = _REQUIRED_TOOLS - names
    if missing:
        raise SystemExit(f"{label}: missing tools from list_tools: {sorted(missing)}")

    result = await client.call_tool("get_workflow_guide", {"wait_for_previous": True})
    text = str(result)
    if "Pulsar Topological Analysis Workflow" not in text:
        raise SystemExit(f"{label}: get_workflow_guide returned unexpected payload")

    runtime = await client.call_tool("get_runtime_context", {})
    context = json.loads(runtime.content[0].text)
    if not context.get("session_id") or "session_model_loaded" not in context:
        raise SystemExit(f"{label}: get_runtime_context returned incomplete context")

    print(
        f"ok: {label} ({len(names)} tools + get_workflow_guide + get_runtime_context)"
    )


async def _check_inprocess() -> None:
    from fastmcp import Client
    from pulsar.mcp.server import mcp

    async with Client(mcp) as client:
        await _assert_healthy(client, "in-process")


def _pulsar_mcp_command() -> tuple[str, list[str]]:
    """Resolve the entry point from this Python environment, independent of cwd/PATH."""
    scripts = Path(sysconfig.get_path("scripts"))
    for name in ("pulsar-mcp", "pulsar-mcp.exe"):
        command = scripts / name
        if command.is_file():
            return str(command), []
    return sys.executable, ["-I", "-m", "pulsar.mcp.server"]


async def _check_stdio_launch() -> None:
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    command, args = _pulsar_mcp_command()

    # Quiet-ish child process; banner may still print depending on FastMCP version.
    env = {**os.environ, "FASTMCP_SHOW_SERVER_BANNER": "false"}
    transport = StdioTransport(command=command, args=args, env=env)
    async with Client(transport) as client:
        await _assert_healthy(client, "stdio launch")


def main() -> int:
    _check_imports()
    asyncio.run(_check_inprocess())
    asyncio.run(_check_stdio_launch())
    print("ok: MCP healthy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
