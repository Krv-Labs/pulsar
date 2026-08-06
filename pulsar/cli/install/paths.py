"""Per-OS config locations for every supported harness."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def home_dir() -> Path:
    for key in ("HOME", "USERPROFILE"):
        value = os.environ.get(key)
        if value:
            return Path(value)
    raise RuntimeError(
        "cannot resolve home directory (HOME and USERPROFILE are unset)"
    )


def app_data(home: Path) -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata)
    return home / "AppData" / "Roaming"


def claude_config(home: Path) -> Path:
    return home / ".claude.json"


def codex_config(home: Path) -> Path:
    return home / ".codex" / "config.toml"


def gemini_config(home: Path) -> Path:
    return home / ".gemini" / "settings.json"


def copilot_config(home: Path) -> Path:
    return home / ".copilot" / "mcp-config.json"


def cursor_config(home: Path) -> Path:
    return home / ".cursor" / "mcp.json"


def antigravity_config(home: Path) -> Path:
    return home / ".gemini" / "config" / "mcp_config.json"


def claude_desktop_config(home: Path) -> Path:
    if sys.platform == "win32":
        return app_data(home) / "Claude" / "claude_desktop_config.json"
    if sys.platform == "darwin":
        return (
            home
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
    return home / ".config" / "Claude" / "claude_desktop_config.json"


def vscode_config(home: Path) -> Path:
    if sys.platform == "win32":
        return app_data(home) / "Code" / "User" / "mcp.json"
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "Code" / "User" / "mcp.json"
    return home / ".config" / "Code" / "User" / "mcp.json"
