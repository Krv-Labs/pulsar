"""Read, classify, write, and remove Pulsar MCP entries in JSON configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pulsar.cli.install.artifact import (
    Artifact,
    Inspection,
    SERVER_KEY,
    State,
)
from pulsar.cli.install.command import LaunchSpec, drift, names_pulsar, owns_entry
from pulsar.cli.install.fsops import (
    WriteOutcome,
    read_json_object,
    read_jsonc_object,
    write_json_object,
)


def inspect(artifact: Artifact, path: Path, launch: LaunchSpec) -> Inspection:
    try:
        if artifact.is_jsonc():
            data, comments = read_jsonc_object(path)
        else:
            data = read_json_object(path)
            comments = False
    except (ValueError, json.JSONDecodeError) as exc:
        return Inspection.conflict(str(exc))

    found = _classify(artifact, data, path, launch)
    if found.state in (State.ACTIVE, State.CONFLICT):
        return found
    if comments:
        entry = json.dumps(_fresh_entry(artifact, launch))
        return Inspection.conflict(
            f"{path} contains comments, which pulsar will not rewrite — "
            f'add "{SERVER_KEY}": {entry} to its '
            f'"{artifact.container_key()}" object by hand'
        )
    return found


def inspect_loose(artifact: Artifact, path: Path) -> Inspection:
    try:
        if artifact.is_jsonc():
            data, comments = read_jsonc_object(path)
        else:
            data = read_json_object(path)
            comments = False
    except (ValueError, json.JSONDecodeError) as exc:
        return Inspection.conflict(str(exc))

    entry = _entry_of(artifact, data)
    if entry is None:
        return Inspection.plain(State.ABSENT)
    if comments:
        return Inspection.conflict(
            f"{path} contains comments — remove its `{SERVER_KEY}` entry by hand"
        )

    command = entry.get("command")
    if not isinstance(command, str):
        command = ""
    if not _is_removable(command, _args_list(entry.get("args"))):
        return Inspection.conflict(
            f"`{SERVER_KEY}` in {path} is an MCP entry pulsar did not write "
            "— inspect it by hand"
        )
    if artifact.wants_stdio_type() and entry.get("type") != "stdio":
        return Inspection.incomplete(
            f'`{SERVER_KEY}` in {path} is missing "type": "stdio"'
        )
    return Inspection.plain(State.ACTIVE)


def write(
    artifact: Artifact, path: Path, launch: LaunchSpec, backup: bool
) -> WriteOutcome:
    if artifact.is_jsonc():
        data, _ = read_jsonc_object(path)
    else:
        data = read_json_object(path)

    container = data.setdefault(artifact.container_key(), {})
    if not isinstance(container, dict):
        raise RuntimeError(
            f"{path} `{artifact.container_key()}` must be an object"
        )
    entry = container.setdefault(SERVER_KEY, {})
    if not isinstance(entry, dict):
        raise RuntimeError(f"{path} `{SERVER_KEY}` must be an object")
    _set_owned_fields(entry, artifact, launch)
    return write_json_object(path, data, backup)


def remove(artifact: Artifact, path: Path, dry_run: bool) -> bool:
    if not path.is_file():
        return False
    try:
        if artifact.is_jsonc():
            data, comments = read_jsonc_object(path)
        else:
            data = read_json_object(path)
            comments = False
    except (ValueError, json.JSONDecodeError):
        return False

    if not _is_removable_entry(artifact, data):
        return False
    if comments:
        raise RuntimeError(
            f"{path} contains comments — remove its `{SERVER_KEY}` entry by hand"
        )

    container = data.get(artifact.container_key())
    if not isinstance(container, dict):
        return False
    container.pop(SERVER_KEY, None)
    if not container:
        data.pop(artifact.container_key(), None)
    if not dry_run:
        write_json_object(path, data, backup=False)
    return True


def _classify(
    artifact: Artifact, data: dict[str, Any], path: Path, launch: LaunchSpec
) -> Inspection:
    entry = _entry_of(artifact, data)
    if entry is None:
        return Inspection.plain(State.ABSENT)

    command = entry.get("command")
    if not isinstance(command, str):
        command = ""
    args = _args_list(entry.get("args"))

    if not names_pulsar(command) or not owns_entry(launch, command, args):
        return Inspection.conflict(
            f"`{SERVER_KEY}` in {path} is an MCP entry pulsar did not write "
            "— inspect it by hand"
        )

    if reason := drift(command, launch):
        return Inspection.incomplete(reason)

    if artifact.wants_stdio_type() and entry.get("type") != "stdio":
        return Inspection.incomplete(
            f'`{SERVER_KEY}` in {path} is missing "type": "stdio"'
        )

    return Inspection.plain(State.ACTIVE)


def _entry_of(artifact: Artifact, data: dict[str, Any]) -> dict[str, Any] | None:
    container = data.get(artifact.container_key())
    if not isinstance(container, dict):
        return None
    entry = container.get(SERVER_KEY)
    if isinstance(entry, dict):
        return entry
    return None


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


def _is_removable_entry(artifact: Artifact, data: dict[str, Any]) -> bool:
    entry = _entry_of(artifact, data)
    if entry is None:
        return False
    command = entry.get("command")
    if not isinstance(command, str):
        return False
    return _is_removable(command, _args_list(entry.get("args")))


def _is_removable(command: str, args: list[str] | None) -> bool:
    if not names_pulsar(command):
        return False
    if Path(command).name in {"pulsar-mcp", "pulsar-mcp.exe"}:
        return args in (None, [])
    if args is None:
        return False
    return len(args) >= 3 and args[0] == "--from" and args[-1] == "pulsar-mcp"


def _set_owned_fields(entry: dict[str, Any], artifact: Artifact, launch: LaunchSpec) -> None:
    if artifact.wants_stdio_type():
        entry["type"] = "stdio"
    entry["command"] = launch.command_str
    entry["args"] = list(launch.args)


def _fresh_entry(artifact: Artifact, launch: LaunchSpec) -> dict[str, Any]:
    entry: dict[str, Any] = {}
    _set_owned_fields(entry, artifact, launch)
    return entry
