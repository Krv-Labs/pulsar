"""Read, classify, write, and remove Pulsar MCP entries in Codex TOML configs."""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import tomlkit
from tomlkit.exceptions import ParseError

from pulsar.cli.install.artifact import Artifact, Inspection, SERVER_KEY, State
from pulsar.cli.install.command import (
    LaunchSpec,
    drift,
    entry_args,
    owns_entry,
    recorded_drift,
)
from pulsar.cli.install.fsops import WriteOutcome, atomic_write, is_dangling_symlink


def inspect(artifact: Artifact, path: Path, launch: LaunchSpec) -> Inspection:
    if is_dangling_symlink(path):
        return Inspection.conflict(f"{path} is a dangling symlink")
    if not path.is_file():
        return Inspection.plain(State.ABSENT)
    try:
        data = tomlkit.parse(path.read_text(encoding="utf-8"))
    except ParseError as exc:
        return Inspection.conflict(f"parsing {path}: {exc}")

    entry = _entry_of(artifact, data)
    if entry is None:
        return Inspection.plain(State.ABSENT)

    command = entry.get("command", "")
    if not isinstance(command, str):
        command = ""
    args = entry_args(entry.get("args"))

    if not owns_entry(command, args):
        return Inspection.conflict(
            f"[mcp_servers.{SERVER_KEY}] in {path} is an entry pulsar did not "
            "write — inspect it by hand"
        )

    if reason := drift(command, args, launch):
        return Inspection.incomplete(reason)

    return Inspection.plain(State.ACTIVE)


def inspect_loose(artifact: Artifact, path: Path) -> Inspection:
    if is_dangling_symlink(path):
        return Inspection.conflict(f"{path} is a dangling symlink")
    if not path.is_file():
        return Inspection.plain(State.ABSENT)
    try:
        data = tomlkit.parse(path.read_text(encoding="utf-8"))
    except ParseError as exc:
        return Inspection.conflict(f"parsing {path}: {exc}")

    entry = _entry_of(artifact, data)
    if entry is None:
        return Inspection.plain(State.ABSENT)

    command = entry.get("command", "")
    if not isinstance(command, str):
        command = ""
    if not owns_entry(command, entry_args(entry.get("args"))):
        return Inspection.conflict(
            f"[mcp_servers.{SERVER_KEY}] in {path} is an entry pulsar did not "
            "write — inspect it by hand"
        )
    if reason := recorded_drift(command):
        return Inspection.incomplete(reason)
    return Inspection.plain(State.ACTIVE)


def write(
    artifact: Artifact, path: Path, launch: LaunchSpec, backup: bool
) -> WriteOutcome:
    if path.is_file():
        try:
            data = tomlkit.parse(path.read_text(encoding="utf-8"))
        except ParseError as exc:
            raise RuntimeError(f"parsing {path}: {exc}") from exc
    else:
        data = tomlkit.document()

    container = artifact.container_key()
    servers = data.get(container)
    if servers is None:
        servers = tomlkit.table()
        data.add(container, servers)
    if not isinstance(servers, MutableMapping):
        raise RuntimeError(f"{path} `{container}` must be a table")
    entry = tomlkit.table()
    entry.add("command", launch.command_str)
    entry.add("args", list(launch.args))
    servers[SERVER_KEY] = entry
    contents = tomlkit.dumps(data)
    if not contents.endswith("\n"):
        contents += "\n"
    return atomic_write(path, contents, backup)


def remove(artifact: Artifact, path: Path, dry_run: bool) -> bool:
    if not path.is_file():
        return False
    try:
        data = tomlkit.parse(path.read_text(encoding="utf-8"))
    except ParseError:
        return False

    servers = data.get(artifact.container_key())
    if not isinstance(servers, MutableMapping):
        return False
    entry = servers.get(SERVER_KEY)
    if not isinstance(entry, MutableMapping):
        return False

    command = entry.get("command", "")
    if not isinstance(command, str):
        return False
    if not owns_entry(command, entry_args(entry.get("args"))):
        return False

    servers.pop(SERVER_KEY, None)
    if not servers:
        data.pop(artifact.container_key(), None)
    if dry_run:
        return True
    contents = tomlkit.dumps(data)
    if not contents.endswith("\n"):
        contents += "\n"
    atomic_write(path, contents, backup=False)
    return True


def _entry_of(artifact: Artifact, data: Any) -> MutableMapping[str, Any] | None:
    servers = data.get(artifact.container_key())
    if not isinstance(servers, MutableMapping):
        return None
    entry = servers.get(SERVER_KEY)
    if isinstance(entry, MutableMapping):
        return entry
    return None
