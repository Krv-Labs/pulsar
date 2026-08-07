"""Menu renderer + console-capability tests for the install harness.

The pyte-backed test replays frames through a terminal emulator to assert the
prompt leaves no stale option rows and does not erase scrollback above the
frame — the exact failure mode described in the CHANGELOG.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from pulsar.cli.install.menu import (
    HintStyle,
    MenuOption,
    _Framer,
    build_menu_frame,
    handle_key,
)
from pulsar.cli.install.terminal import (
    Tier,
    open_console,
    supports_color,
    visual_rows,
)

pyte = pytest.importorskip("pyte")


def _options() -> list[MenuOption]:
    return [
        MenuOption(
            id="cursor",
            name="Cursor",
            hint="detected",
            hint_style=HintStyle.PLAIN,
            checked=True,
            is_active=False,
        ),
        MenuOption(
            id="claude",
            name="Claude Code",
            hint="not configured",
            hint_style=HintStyle.PLAIN,
            checked=False,
            is_active=False,
        ),
    ]


class _CaptureConsole:
    tier = Tier.FULL

    def __init__(self, columns: int = 80) -> None:
        self._columns = columns
        self.written: list[str] = []

    def write(self, text: str) -> None:
        self.written.append(text)

    def columns(self) -> int:
        return self._columns


def test_visual_rows_counts_soft_wrapped_lines() -> None:
    assert visual_rows(["hello"], 80) == 1
    assert visual_rows([""], 80) == 1
    assert visual_rows(["a" * 80], 80) == 1
    assert visual_rows(["a" * 81], 80) == 2
    assert visual_rows(["short", "a" * 160], 80) == 1 + 2


def test_visual_rows_strips_ansi_before_measuring() -> None:
    colored = "\033[32m" + ("x" * 80) + "\033[0m"
    assert visual_rows([colored], 80) == 1
    assert visual_rows(["\033[1m" + ("y" * 81) + "\033[0m"], 80) == 2


def test_supports_color_respects_no_color_and_force(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = _CaptureConsole()
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert supports_color(console) is True

    monkeypatch.setenv("NO_COLOR", "1")
    assert supports_color(console) is False

    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("FORCE_COLOR", "1")
    line = _CaptureConsole()
    line.tier = Tier.LINE
    assert supports_color(line) is True


def test_handle_key_uses_abstract_names() -> None:
    options = _options()
    cursor, done, selected = handle_key(options, 0, "DOWN")
    assert (cursor, done, selected) == (1, False, None)
    cursor, done, selected = handle_key(options, 1, "SPACE")
    assert options[1].checked is True
    assert done is False
    cursor, done, selected = handle_key(options, 1, "ENTER")
    assert done is True
    assert selected == ["cursor", "claude"]


def test_menu_redraw_does_not_eat_scrollback_or_leave_stale_rows() -> None:
    """Feed successive frames through pyte; prior rows must not linger."""
    screen = pyte.Screen(80, 24)
    stream = pyte.Stream(screen)

    # Seed scrollback-visible content above where the menu will draw.
    marker = "KEEP_SCROLLBACK_MARKER"
    stream.feed(marker + "\r\n")

    console = _CaptureConsole(columns=80)
    framer = _Framer(console)
    options = _options()
    title = "Which agent integrations do you want to configure?"

    for cursor in (0, 1, 0):
        framer.draw(build_menu_frame(title, options, cursor, "active", color=False))
    # Collapse to the submit summary — the real exit path.
    framer.draw(build_menu_frame(title, options, 0, "submit", color=False))

    stream.feed("".join(console.written))
    rendered = "\n".join(line.rstrip() for line in screen.display)

    assert marker in rendered
    # Active list rows must be gone after the submit summary.
    assert "Claude Code" not in rendered
    assert "Cursor" in rendered  # still named in the one-line summary
    assert "Cancelled" not in rendered


def test_narrow_terminal_wrap_count_matches_framer_erase() -> None:
    """A title that soft-wraps must advance the framer by more than one row."""
    console = _CaptureConsole(columns=20)
    framer = _Framer(console)
    options = [replace(opt) for opt in _options()]
    title = "Which agent integrations do you want to configure?"
    lines = build_menu_frame(title, options, 0, "active", color=False)
    framer.draw(lines)
    # Second draw issues cursor-up by the previous height; height must match
    # visual_rows, not len(lines).
    assert framer._height == visual_rows(lines, 20)
    assert framer._height > len(lines)


def test_open_console_returns_line_tier_when_term_dumb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TERM", "dumb")
    console = open_console()
    if console is None:
        pytest.skip("no controlling console in this environment")
    try:
        assert console.tier is Tier.LINE
    finally:
        console.close()


def test_open_console_none_without_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.name == "nt":
        pytest.skip("Windows console detachment is not simulated here")

    # Point /dev/tty open at a failing path via os.open monkeypatch.
    real_open = os.open

    def _fake_open(path, flags, *args, **kwargs):
        if path == "/dev/tty":
            raise OSError("no tty")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", _fake_open)
    assert open_console() is None
