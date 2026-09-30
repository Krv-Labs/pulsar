from __future__ import annotations

import importlib.util
import sys
import sysconfig
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "smoke_mcp", Path(__file__).parents[1] / "scripts/smoke_mcp.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_SMOKE_MCP = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SMOKE_MCP)


def test_smoke_launch_uses_entry_point_from_active_environment(tmp_path, monkeypatch):
    scripts = tmp_path / "bin"
    scripts.mkdir()
    entry_point = scripts / "pulsar-mcp"
    entry_point.touch()
    monkeypatch.setattr(sysconfig, "get_path", lambda _: str(scripts))
    monkeypatch.setenv("PATH", "/foreign/environment/bin")

    assert _SMOKE_MCP._pulsar_mcp_command() == (str(entry_point), [])


def test_smoke_module_fallback_isolates_checkout(tmp_path, monkeypatch):
    scripts = tmp_path / "bin"
    scripts.mkdir()
    monkeypatch.setattr(sysconfig, "get_path", lambda _: str(scripts))
    monkeypatch.setattr(sys, "executable", "/active/environment/bin/python")

    assert _SMOKE_MCP._pulsar_mcp_command() == (
        "/active/environment/bin/python",
        ["-I", "-m", "pulsar.mcp.server"],
    )
