"""Supported agent harnesses and their MCP registration targets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pulsar.cli.install.artifact import Artifact, ArtifactKind
from pulsar.cli.install import paths


@dataclass(frozen=True)
class HarnessSpec:
    id: str
    name: str
    artifact: Artifact
    config_path: Callable[[Path], Path]
    active_msg: str
    absent_msg: str
    detect: Callable[[Path], bool]
    note: Callable[[Path], str | None]


def _no_note(_home: Path) -> str | None:
    return None


def _detect_claude(home: Path) -> bool:
    return (home / ".claude").is_dir()


def _detect_codex(home: Path) -> bool:
    return (home / ".codex").is_dir()


def _detect_gemini(home: Path) -> bool:
    return (home / ".gemini").is_dir()


def _detect_copilot(home: Path) -> bool:
    return (home / ".copilot").is_dir()


def _detect_cursor(home: Path) -> bool:
    return (home / ".cursor").is_dir()


def _parent_is_dir(path: Path) -> bool:
    parent = path.parent
    return parent.is_dir()


def _migration_marker(home: Path) -> Path:
    return home / ".gemini" / "config" / ".migrated"


def _antigravity_data_dirs(home: Path) -> list[Path]:
    return [
        home / ".gemini" / name
        for name in ("antigravity", "antigravity-cli", "antigravity-ide")
    ]


def _detect_antigravity(home: Path) -> bool:
    if _migration_marker(home).exists():
        return True
    return any(path.is_dir() for path in _antigravity_data_dirs(home))


def _antigravity_note(home: Path) -> str | None:
    if _migration_marker(home).exists():
        return None
    if not any(path.is_dir() for path in _antigravity_data_dirs(home)):
        return None
    return (
        "Antigravity has not migrated MCP config to ~/.gemini/config/ yet — "
        "the entry may be overwritten on next Antigravity launch"
    )


HARNESSES: tuple[HarnessSpec, ...] = (
    HarnessSpec(
        id="claude",
        name="Claude Code",
        artifact=Artifact(ArtifactKind.MCP_JSON),
        config_path=paths.claude_config,
        active_msg="MCP server registered in ~/.claude.json",
        absent_msg="no MCP server entry in ~/.claude.json",
        detect=_detect_claude,
        note=_no_note,
    ),
    HarnessSpec(
        id="claude-desktop",
        name="Claude Desktop",
        artifact=Artifact(ArtifactKind.MCP_JSON),
        config_path=paths.claude_desktop_config,
        active_msg="MCP server registered in the Claude Desktop config",
        absent_msg="no MCP server entry in the Claude Desktop config",
        detect=lambda home: _parent_is_dir(paths.claude_desktop_config(home)),
        note=_no_note,
    ),
    HarnessSpec(
        id="codex",
        name="Codex CLI",
        artifact=Artifact(ArtifactKind.MCP_TOML),
        config_path=paths.codex_config,
        active_msg="[mcp_servers.pulsar] present in ~/.codex/config.toml",
        absent_msg="no [mcp_servers.pulsar] in ~/.codex/config.toml",
        detect=_detect_codex,
        note=_no_note,
    ),
    HarnessSpec(
        id="gemini",
        name="Gemini CLI",
        artifact=Artifact(ArtifactKind.MCP_JSON),
        config_path=paths.gemini_config,
        active_msg="MCP server registered in ~/.gemini/settings.json",
        absent_msg="no MCP server entry in ~/.gemini/settings.json",
        detect=_detect_gemini,
        note=_no_note,
    ),
    HarnessSpec(
        id="copilot",
        name="GitHub Copilot CLI",
        artifact=Artifact(ArtifactKind.MCP_JSON),
        config_path=paths.copilot_config,
        active_msg="MCP server registered in ~/.copilot/mcp-config.json",
        absent_msg="no MCP server entry in ~/.copilot/mcp-config.json",
        detect=_detect_copilot,
        note=_no_note,
    ),
    HarnessSpec(
        id="cursor",
        name="Cursor",
        artifact=Artifact(ArtifactKind.MCP_JSON),
        config_path=paths.cursor_config,
        active_msg="MCP server registered in ~/.cursor/mcp.json",
        absent_msg="no MCP server entry in ~/.cursor/mcp.json",
        detect=_detect_cursor,
        note=_no_note,
    ),
    HarnessSpec(
        id="vscode",
        name="VS Code",
        artifact=Artifact(ArtifactKind.VSCODE_JSONC),
        config_path=paths.vscode_config,
        active_msg="servers.pulsar present in the VS Code user mcp.json",
        absent_msg="no servers.pulsar in the VS Code user mcp.json",
        detect=lambda home: _parent_is_dir(paths.vscode_config(home)),
        note=_no_note,
    ),
    HarnessSpec(
        id="antigravity",
        name="Google Antigravity",
        artifact=Artifact(ArtifactKind.MCP_JSON),
        config_path=paths.antigravity_config,
        active_msg="MCP server registered in ~/.gemini/config/mcp_config.json",
        absent_msg="no MCP server entry in ~/.gemini/config/mcp_config.json",
        detect=_detect_antigravity,
        note=_antigravity_note,
    ),
)


def harness_ids() -> list[str]:
    return [spec.id for spec in HARNESSES]


def get_harness(harness_id: str) -> HarnessSpec | None:
    lowered = harness_id.lower()
    for spec in HARNESSES:
        if spec.id == lowered:
            return spec
    return None
