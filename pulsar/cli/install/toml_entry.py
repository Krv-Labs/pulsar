"""Read, classify, write, and remove Pulsar MCP entries in Codex TOML configs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from pulsar.cli.install.artifact import Inspection, SERVER_KEY, State
from pulsar.cli.install.command import LaunchSpec, drift, names_pulsar, owns_entry
from pulsar.cli.install.fsops import WriteOutcome, atomic_write

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w


def inspect(path: Path, launch: LaunchSpec) -> Inspection:
    if not path.is_file():
        return Inspection.plain(State.ABSENT)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return Inspection.conflict(f"parsing {path}: {exc}")

    entry = _entry_of(data)
    if entry is None:
        return Inspection.plain(State.ABSENT)

    command = entry.get("command", "")
    if not isinstance(command, str):
        command = ""
    args = _args_list(entry.get("args"))

    if not names_pulsar(command) or not owns_entry(launch, command, args):
        return Inspection.conflict(
            f"[mcp_servers.{SERVER_KEY}] in {path} is an entry pulsar did not "
            "write — inspect it by hand"
        )

    if reason := drift(command, launch):
        return Inspection.incomplete(reason)

    return Inspection.plain(State.ACTIVE)


def inspect_loose(path: Path) -> Inspection:
    if not path.is_file():
        return Inspection.plain(State.ABSENT)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return Inspection.conflict(f"parsing {path}: {exc}")

    entry = _entry_of(data)
    if entry is None:
        return Inspection.plain(State.ABSENT)

    command = entry.get("command", "")
    if not isinstance(command, str):
        command = ""
    if not _is_removable(command, _args_list(entry.get("args"))):
        return Inspection.conflict(
            f"[mcp_servers.{SERVER_KEY}] in {path} is an entry pulsar did not "
            "write — inspect it by hand"
        )
    return Inspection.plain(State.ACTIVE)


def write(path: Path, launch: LaunchSpec, backup: bool) -> WriteOutcome:
    if path.is_file():
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise RuntimeError(f"parsing {path}: {exc}") from exc
    else:
        data = {}

    servers = data.setdefault("mcp_servers", {})
    if not isinstance(servers, dict):
        raise RuntimeError(f"{path} `mcp_servers` must be a table")
    servers[SERVER_KEY] = {
        "command": launch.command_str,
        "args": list(launch.args),
    }
    contents = tomli_w.dumps(data)
    if not contents.endswith("\n"):
        contents += "\n"
    return atomic_write(path, contents, backup)


def remove(path: Path, dry_run: bool) -> bool:
    if not path.is_file():
        return False
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        return False

    servers = data.get("mcp_servers")
    if not isinstance(servers, dict):
        return False
    entry = servers.get(SERVER_KEY)
    if not isinstance(entry, dict):
        return False

    command = entry.get("command", "")
    if not isinstance(command, str):
        return False
    if not _is_removable(command, _args_list(entry.get("args"))):
        return False

    servers.pop(SERVER_KEY, None)
    if not servers:
        data.pop("mcp_servers", None)
    if dry_run:
        return True
    contents = tomli_w.dumps(data)
    if not contents.endswith("\n"):
        contents += "\n"
    atomic_write(path, contents, backup=False)
    return True


def _entry_of(data: dict[str, Any]) -> dict[str, Any] | None:
    servers = data.get("mcp_servers")
    if not isinstance(servers, dict):
        return None
    entry = servers.get(SERVER_KEY)
    if isinstance(entry, dict):
        return entry
    return None


def _is_removable(command: str, args: list[str] | None) -> bool:
    if not names_pulsar(command):
        return False
    if Path(command).name in {"pulsar-mcp", "pulsar-mcp.exe"}:
        return args in (None, [])
    if args is None:
        return False
    return len(args) >= 3 and args[0] == "--from" and args[-1] == "pulsar-mcp"


def _args_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        out.append(item)
    return out
