"""Terminal output helpers for install/uninstall/status."""

from __future__ import annotations

from pulsar.cli.install.artifact import State

OK = "✓"
FAILED = "✗"
CONFLICT = "!"
ABSENT = "○"
PENDING = "…"
REMOVED = "−"
INCOMPLETE = "▲"

_GLYPHS = {
    State.ACTIVE: OK,
    State.INCOMPLETE: INCOMPLETE,
    State.CONFLICT: CONFLICT,
    State.ABSENT: ABSENT,
}


def header(title: str, dry_run: bool) -> None:
    suffix = " (dry run)" if dry_run else ""
    print(f"\n{title}{suffix}")
    print("│")


def footer(message: str) -> None:
    print(f"└─ {message}\n")


def harness_line(name: str) -> None:
    print(f"│  {name}")


def detail(glyph: str, message: str) -> None:
    print(f"│    {glyph} {message}")


def note(message: str) -> None:
    print(f"│      note: {message}")


def glyph(state: State) -> str:
    return _GLYPHS[state]


def label(state: State) -> str:
    return state.value
