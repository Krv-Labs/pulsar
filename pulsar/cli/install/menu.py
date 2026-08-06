"""Interactive multi-select menu for harness install/uninstall."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from enum import Enum

from pulsar.cli.install.artifact import State

try:
    import termios
    import tty
except ImportError:  # Windows: no termios, so no raw-mode prompting
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]


def can_prompt() -> bool:
    """True when we can render a menu and read a keypress."""
    return termios is not None and sys.stderr.isatty() and sys.stdin.isatty()


class HintStyle(Enum):
    ACTIVE = "active"
    REPAIR = "repair"
    PLAIN = "plain"


@dataclass
class MenuOption:
    id: str
    name: str
    hint: str
    hint_style: HintStyle
    checked: bool
    is_active: bool


def run_menu(title: str, options: list[MenuOption]) -> list[str] | None:
    if not options:
        return []
    if not can_prompt():
        return None

    working = [replace(option) for option in options]
    cursor = 0
    print(title, file=sys.stderr)
    _render(working, cursor)
    while True:
        key = _read_key()
        if key in ("\x1b[A", "k"):  # up
            cursor = (cursor - 1) % len(working)
        elif key in ("\x1b[B", "j"):  # down
            cursor = (cursor + 1) % len(working)
        elif key == " ":
            working[cursor].checked = not working[cursor].checked
        elif key == "a":
            if all(option.checked for option in working):
                for option in working:
                    option.checked = False
            else:
                for option in working:
                    option.checked = True
        elif key in ("\r", "\n"):
            selected = [option.id for option in working if option.checked]
            print(file=sys.stderr)
            return selected
        elif key in ("\x03", "q"):
            print(file=sys.stderr)
            return None
        _render(working, cursor)


def run_confirm(title: str, plan: list[str]) -> bool:
    if not can_prompt():
        return False
    print(title, file=sys.stderr)
    for line in plan:
        print(f"  {line}", file=sys.stderr)
    print("  No / Yes (default No)", file=sys.stderr)
    while True:
        key = _read_key()
        if key in ("y", "Y"):
            print(file=sys.stderr)
            return True
        if key in ("\r", "\n", "n", "N", "\x03", "q"):
            print(file=sys.stderr)
            return False


def menu_hint(
    *,
    install: bool,
    state: State,
    detected: bool,
) -> tuple[str, HintStyle, bool]:
    if state == State.ACTIVE:
        return "active", HintStyle.ACTIVE, True
    if state == State.INCOMPLETE:
        return "needs repair", HintStyle.REPAIR, True
    if state == State.CONFLICT:
        return "conflict — inspect by hand", HintStyle.PLAIN, False
    if install and detected:
        return "detected", HintStyle.PLAIN, True
    return "not configured", HintStyle.PLAIN, False


def _render(options: list[MenuOption], cursor: int) -> None:
    sys.stderr.write("\033[J")
    for index, option in enumerate(options):
        marker = "●" if option.checked else "○"
        if index == cursor:
            marker = f">{marker}"
        hint = option.hint
        if option.hint_style == HintStyle.ACTIVE:
            hint = f"✓ {hint}"
        elif option.hint_style == HintStyle.REPAIR:
            hint = f"▲ {hint}"
        line = f"  {marker} {option.name} ({hint})"
        sys.stderr.write("\033[2K\r" + line + "\n")
    sys.stderr.write("\033[{0}A".format(len(options) + 1))
    sys.stderr.flush()


def _read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch
