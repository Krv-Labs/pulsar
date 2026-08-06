"""Resolve the MCP launch command and args to record in harness configs."""

from __future__ import annotations

import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path

from pulsar._version import get_version

PACKAGE_SPEC = "thema-pulsar[mcp]"
SERVER_SCRIPT = "pulsar-mcp"


@dataclass(frozen=True)
class LaunchSpec:
    """Absolute command path and args written into agent MCP configs."""

    command: Path
    args: tuple[str, ...]
    mode: str  # "uvx" or "pipx"

    @property
    def command_str(self) -> str:
        return str(self.command)


def uvx_args(*, pin_version: bool = False) -> tuple[str, ...]:
    if pin_version:
        version = get_version()
        if version and not version.endswith("+unknown"):
            spec = f"thema-pulsar=={version}[mcp]"
            return ("--from", spec, SERVER_SCRIPT)
    return ("--from", PACKAGE_SPEC, SERVER_SCRIPT)


def resolve_launch_spec(
    *,
    mode: str = "uvx",
    pin_version: bool = False,
) -> LaunchSpec:
    if mode == "pipx":
        return _resolve_pipx()
    if mode != "uvx":
        raise RuntimeError(f"unknown launch mode: {mode!r} (supported: uvx, pipx)")
    return _resolve_uvx(pin_version=pin_version)


def _resolve_uvx(*, pin_version: bool) -> LaunchSpec:
    found = shutil.which("uvx")
    if not found:
        raise RuntimeError(
            "uvx is not on PATH — install uv from https://docs.astral.sh/uv/ "
            "and ensure uvx is available"
        )
    command = Path(found)
    if not command.is_absolute():
        command = command.resolve()
    return LaunchSpec(command=command, args=uvx_args(pin_version=pin_version), mode="uvx")


def _resolve_pipx() -> LaunchSpec:
    found = shutil.which(SERVER_SCRIPT)
    if not found:
        raise RuntimeError(
            f"{SERVER_SCRIPT} is not on PATH — run "
            f'pipx install "{PACKAGE_SPEC}" first'
        )
    command = Path(found)
    if not command.is_absolute():
        command = command.resolve()
    return LaunchSpec(command=command, args=(), mode="pipx")


def names_pulsar(command: str) -> bool:
    name = Path(command).name
    return name in {"uvx", "uvx.exe", SERVER_SCRIPT, f"{SERVER_SCRIPT}.exe"}


def args_match(spec: LaunchSpec, args: list[str] | None) -> bool:
    if spec.mode == "pipx":
        return args in (None, [])
    expected = list(spec.args)
    if args is None:
        return False
    return list(args) == expected


def owns_entry(spec: LaunchSpec, command: str, args: list[str] | None) -> bool:
    return names_pulsar(command) and args_match(spec, args)


def drift(recorded: str, expected: LaunchSpec) -> str | None:
    path = Path(recorded)
    if not path.is_absolute():
        return f"`{recorded}` is not an absolute path"
    if not path.exists():
        return f"`{recorded}` no longer exists"
    if not path.is_file():
        return f"`{recorded}` is not a file"
    if os.name != "nt" and not os.access(path, os.X_OK):
        mode = path.stat().st_mode
        if not (mode & stat.S_IXUSR):
            return f"`{recorded}` is not executable"
    if not same_file(path, expected.command):
        return (
            f"`{recorded}` is a different binary than the expected "
            f"{expected.command}"
        )
    return None


def same_file(left: Path, right: Path) -> bool:
    try:
        return left.resolve().samefile(right.resolve())
    except OSError:
        return False
