"""MCP registration artifact types and ownership markers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pulsar.cli.install.command import LaunchSpec
    from pulsar.cli.install.fsops import WriteOutcome

SERVER_KEY = "pulsar"


class ArtifactKind(Enum):
    MCP_JSON = "mcp_json"
    MCP_TOML = "mcp_toml"
    VSCODE_JSONC = "vscode_jsonc"


class State(Enum):
    ACTIVE = "active"
    INCOMPLETE = "incomplete"
    CONFLICT = "conflict"
    ABSENT = "absent"


@dataclass
class Inspection:
    state: State
    detail: str | None = None

    @classmethod
    def plain(cls, state: State) -> Inspection:
        return cls(state=state)

    @classmethod
    def incomplete(cls, detail: str) -> Inspection:
        return cls(state=State.INCOMPLETE, detail=detail)

    @classmethod
    def conflict(cls, detail: str) -> Inspection:
        return cls(state=State.CONFLICT, detail=detail)


@dataclass(frozen=True)
class Artifact:
    kind: ArtifactKind

    def container_key(self) -> str:
        if self.kind == ArtifactKind.VSCODE_JSONC:
            return "servers"
        if self.kind == ArtifactKind.MCP_TOML:
            return "mcp_servers"
        return "mcpServers"

    def wants_stdio_type(self) -> bool:
        return self.kind == ArtifactKind.VSCODE_JSONC

    def is_jsonc(self) -> bool:
        return self.kind == ArtifactKind.VSCODE_JSONC

    def inspect(self, path: Path, spec: LaunchSpec) -> Inspection:
        if self.kind == ArtifactKind.MCP_TOML:
            from pulsar.cli.install import toml_entry

            return toml_entry.inspect(path, spec)
        from pulsar.cli.install import json_entry

        return json_entry.inspect(self, path, spec)

    def apply(self, path: Path, launch: LaunchSpec) -> WriteOutcome | None:
        inspection = self.inspect(path, launch)
        if inspection.state == State.ACTIVE:
            return None
        if inspection.state == State.CONFLICT:
            detail = inspection.detail or f"{path} cannot be updated safely"
            raise RuntimeError(detail)
        # Back up on repair too: a drifted entry may have been hand-written or
        # left by another launch mode. atomic_write skips absent files.
        if self.kind == ArtifactKind.MCP_TOML:
            from pulsar.cli.install import toml_entry

            return toml_entry.write(path, launch, backup=True)
        from pulsar.cli.install import json_entry

        return json_entry.write(self, path, launch, backup=True)

    def inspect_loose(self, path: Path) -> Inspection:
        if self.kind == ArtifactKind.MCP_TOML:
            from pulsar.cli.install import toml_entry

            return toml_entry.inspect_loose(path)
        from pulsar.cli.install import json_entry

        return json_entry.inspect_loose(self, path)

    def remove(self, path: Path, dry_run: bool) -> bool:
        if self.kind == ArtifactKind.MCP_TOML:
            from pulsar.cli.install import toml_entry

            return toml_entry.remove(path, dry_run)
        from pulsar.cli.install import json_entry

        return json_entry.remove(self, path, dry_run)
