#!/usr/bin/env python3
"""Smoke-test the Pulsar MCP import path (native extension + server + entry point).

Used by macOS Intel CI to catch arch/ABI and packaging failures that unit tests
on Linux do not exercise.
"""

from __future__ import annotations

import importlib
import sys


_REQUIRED_TOOLS = frozenset(
    {
        "ingest_dataset",
        "create_config",
        "run_topological_sweep",
        "diagnose_cosmic_graph",
        "generate_cluster_dossier",
        "get_workflow_guide",
    }
)


def main() -> int:
    # Native extension — fails loud on arch/ABI mismatch.
    importlib.import_module("pulsar._pulsar")

    server = importlib.import_module("pulsar.mcp.server")
    tools = importlib.import_module("pulsar.mcp.tools")
    names = {fn.__name__ for fn in tools.ALL_TOOLS_LIST}
    missing = _REQUIRED_TOOLS - names
    if missing:
        print(f"missing tools: {sorted(missing)}", file=sys.stderr)
        return 1
    if getattr(server, "mcp", None) is None:
        print("pulsar.mcp.server.mcp missing", file=sys.stderr)
        return 1

    from importlib.metadata import entry_points

    eps = entry_points()
    if hasattr(eps, "select"):
        scripts = {ep.name for ep in eps.select(group="console_scripts")}
    else:  # pragma: no cover - Python 3.10
        scripts = {ep.name for ep in eps.get("console_scripts", [])}
    if "pulsar-mcp" not in scripts:
        print(
            f"pulsar-mcp entry point missing; have={sorted(scripts)[:20]}",
            file=sys.stderr,
        )
        return 1

    print(f"ok: _pulsar + mcp ({len(names)} tools) + pulsar-mcp entry point")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
