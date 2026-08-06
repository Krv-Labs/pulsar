"""Terminal output helpers for install/uninstall/status."""

from __future__ import annotations

from pulsar.cli.install.artifact import State


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


def ok() -> str:
    return "✓"


def failed() -> str:
    return "✗"


def conflict() -> str:
    return "!"


def absent() -> str:
    return "○"


def pending() -> str:
    return "…"


def removed() -> str:
    return "−"


def glyph(state: State) -> str:
    if state == State.ACTIVE:
        return ok()
    if state == State.INCOMPLETE:
        return "▲"
    if state == State.CONFLICT:
        return conflict()
    return absent()


def label(state: State) -> str:
    return state.value
