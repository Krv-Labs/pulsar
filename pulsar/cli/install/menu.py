"""Interactive selection prompts for harness install/uninstall.

The renderer holds one invariant, and every bug it replaced was a violation of
it: **the cursor-up count must equal the rows the frame just advanced, and
every line the prompt owns — title, hints, options — must live inside the
redrawn frame.** A title printed once outside the frame cannot be counted, so
the erase drifts one row per redraw and eats the scrollback above it.

Frames end in a collapsed one-line summary so whatever prints next (see
`report.py`, which continues the same `│` rail) starts on a clean row instead
of overwriting the list.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from pulsar.cli.install.artifact import State
from pulsar.cli.install.terminal import (
    Console,
    Tier,
    open_console,
    supports_color,
    visual_rows,
)

# Raw mode turns off ONLCR, so a bare "\n" moves down without returning to
# column 0 and the frame walks diagonally off the screen. The renderer holds
# raw mode across the whole loop, so every line break has to be explicit.
_EOL = "\r\n"

_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"

_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "red": "\033[31m",
    "cyan": "\033[36m",
    "blue": "\033[38;5;39m",
}


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


def can_prompt() -> bool:
    """True when a console exists to render a prompt on.

    Deliberately not `sys.stdin.isatty()`: the prompts talk to the controlling
    console, so `pulsar install | tee log` and other redirections still get a
    working menu. Only a genuinely consoleless environment (CI, systemd) is
    false here, and that is the case the CLI turns into an error telling the
    user to pass explicit harness names or --all.
    """
    console = open_console()
    if console is None:
        return False
    console.close()
    return True


def run_menu(title: str, options: list[MenuOption]) -> list[str] | None:
    """Select harness ids. None means the user cancelled or there is no console."""
    if not options:
        return []
    console = open_console()
    if console is None:
        return None
    with console:
        if console.tier is Tier.FULL:
            return _menu_full(console, title, options)
        return _menu_line(console, title, options)


def run_confirm(title: str, plan: list[str]) -> bool:
    console = open_console()
    if console is None:
        # Not `return False`: the caller checked can_prompt() a moment ago, so
        # reaching here means the console vanished in between. Returning False
        # would be indistinguishable from the user declining, and would report
        # "Cancelled." for an uninstall they never got to answer.
        raise RuntimeError("the console went away before the prompt could be shown")
    with console:
        if console.tier is Tier.FULL:
            return _confirm_full(console, title, plan)
        return _confirm_line(console, title, plan)


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


# --- pure logic (unit-testable on any platform) ---------------------------


def handle_key(
    options: list[MenuOption], cursor: int, key: str
) -> tuple[int, bool, list[str] | None]:
    """Apply one abstract key. Returns (cursor, done, selected-or-None).

    Matches abstract key names rather than raw byte sequences: Windows reports
    arrows as a scancode pair rather than CSI, so matching on "\\x1b[A" here
    would make every arrow key a no-op there.

    CANCEL (Ctrl-C) is deliberately absent: raw mode clears ISIG, so an
    interrupt arrives here as an ordinary key, and treating it as a plain
    cancel would exit 0 in the TUI while the same keystroke exits 130 in the
    line-mode prompt, where SIGINT is still live. The drivers intercept it.
    """
    movement = {"UP": -1, "k": -1, "DOWN": 1, "j": 1}.get(key)
    if movement is not None:
        return (cursor + movement) % len(options), False, None
    if key == "SPACE":
        options[cursor].checked = not options[cursor].checked
    elif key == "a":
        checked = not all(option.checked for option in options)
        for option in options:
            option.checked = checked
    elif key == "ENTER":
        return cursor, True, [option.id for option in options if option.checked]
    elif key in ("ESC", "q"):
        return cursor, True, None
    return cursor, False, None


def parse_selection(text: str, options: list[MenuOption]) -> list[str] | None | str:
    """Interpret a typed line from the no-ANSI prompt.

    Returns the chosen ids, None to cancel, or an error string to re-ask with.
    """
    entry = text.strip().lower()
    if entry in ("q", "quit", "cancel"):
        return None
    if entry == "":
        return [option.id for option in options if option.checked]
    if entry == "all":
        return [option.id for option in options]
    if entry == "none":
        return []
    chosen: list[str] = []
    for token in entry.replace(" ", ",").split(","):
        if not token:
            continue
        if not token.isdigit():
            return f"'{token}' is not a number"
        index = int(token) - 1
        if not 0 <= index < len(options):
            return f"{token} is out of range (1-{len(options)})"
        if options[index].id not in chosen:
            chosen.append(options[index].id)
    return chosen


def parse_confirm(text: str) -> bool:
    """Interpret a y/N answer. Anything unrecognised means no.

    Uninstall deletes configuration, so an ambiguous answer must never be
    read as consent.
    """
    return text.strip().lower() in ("y", "yes")


def build_menu_frame(
    title: str,
    options: list[MenuOption],
    cursor: int,
    state: str,
    color: bool,
) -> list[str]:
    """The complete frame, one string per logical line."""
    paint = _painter(color)
    if state == "submit":
        picked = [option.name for option in options if option.checked]
        summary = ", ".join(picked) if picked else "(none)"
        return [
            f"{paint('◇', 'green')}  {paint(title, 'bold')} {paint(summary, 'cyan')}",
            paint("│", "dim"),
        ]
    if state == "cancel":
        return [
            f"{paint('■', 'red')}  {paint(title, 'bold')} {paint('Cancelled', 'dim')}",
            paint("│", "dim"),
        ]

    rail = paint("│", "dim")
    lines = [
        f"{paint('┌', 'dim')}  {paint(title, 'bold')}",
        rail,
        f"{rail}  {paint('↑↓ move, space toggle, a all, enter confirm, q cancel', 'dim')}",
        rail,
    ]
    for index, option in enumerate(options):
        lines.append(f"{rail} {_option_line(option, index == cursor, paint)}")
    lines.append(paint("└", "dim"))
    return lines


def build_confirm_frame(
    title: str, plan: list[str], yes: bool, state: str, color: bool
) -> list[str]:
    paint = _painter(color)
    if state == "submit":
        answer = "Yes" if yes else "No"
        return [
            f"{paint('◇', 'green')}  {paint(title, 'bold')} {paint(answer, 'cyan')}",
            paint("│", "dim"),
        ]
    if state == "cancel":
        return [
            f"{paint('■', 'red')}  {paint(title, 'bold')} {paint('Cancelled', 'dim')}",
            paint("│", "dim"),
        ]

    rail = paint("│", "dim")
    lines = [f"{paint('┌', 'dim')}  {paint(title, 'bold')}", rail]
    lines.extend(f"{rail}    {paint(entry, 'dim')}" for entry in plan)
    lines.append(rail)
    for label, selected in (("No", not yes), ("Yes", yes)):
        radio = paint("●", "green") if selected else paint("○", "dim")
        marker = paint("❯", "cyan") if selected else " "
        text = paint(label, "bold") if selected else label
        lines.append(f"{rail} {marker} {radio} {text}")
    lines.append(paint("└", "dim"))
    return lines


def _option_line(option: MenuOption, at_cursor: bool, paint) -> str:
    if option.checked:
        radio = paint("●", "blue" if option.is_active else "green")
    else:
        radio = paint("○", "dim")
    hint = option.hint
    if option.hint_style == HintStyle.ACTIVE:
        hint = f"✓ {hint}"
    elif option.hint_style == HintStyle.REPAIR:
        hint = f"▲ {hint}"
    marker = paint("❯", "cyan") if at_cursor else " "
    name = paint(option.name, "bold") if at_cursor else option.name
    return f"{marker} {radio} {name} {paint(f'({hint})', 'dim')}"


def _painter(color: bool):
    if not color:
        return lambda text, _style=None: text
    return lambda text, style=None: (
        f"{_CODES[style]}{text}{_CODES['reset']}" if style else text
    )


# --- drivers --------------------------------------------------------------


class _Framer:
    """Erases exactly the rows the previous frame occupied, then draws."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._height = 0

    def draw(self, lines: list[str]) -> None:
        clear = f"\033[{self._height}A\033[J" if self._height else ""
        self._console.write(clear + _EOL.join(lines) + _EOL)
        # Wrapped rows, not len(lines): a label that soft-wraps occupies two
        # physical rows, and undercounting here strands the cursor mid-frame.
        self._height = visual_rows(lines, self._console.columns())


def _menu_full(
    console: Console, title: str, options: list[MenuOption]
) -> list[str] | None:
    working = [replace(option) for option in options]
    cursor = 0
    color = supports_color(console)
    framer = _Framer(console)
    console.write(_HIDE_CURSOR)
    try:
        with console.raw():
            framer.draw(build_menu_frame(title, working, cursor, "active", color))
            while True:
                key = console.read_key()
                if key == "CANCEL":
                    framer.draw(
                        build_menu_frame(title, working, cursor, "cancel", color)
                    )
                    raise KeyboardInterrupt
                cursor, done, selected = handle_key(working, cursor, key)
                state = (
                    "active"
                    if not done
                    else ("submit" if selected is not None else "cancel")
                )
                framer.draw(build_menu_frame(title, working, cursor, state, color))
                if done:
                    return selected
    finally:
        # Runs for KeyboardInterrupt and any renderer fault too: leaving the
        # cursor hidden outlives the process and breaks the user's shell.
        console.write(_SHOW_CURSOR)


def _menu_line(
    console: Console, title: str, options: list[MenuOption]
) -> list[str] | None:
    working = [replace(option) for option in options]
    console.write(f"\n{title}\n")
    for index, option in enumerate(working, start=1):
        mark = "*" if option.checked else " "
        console.write(f"  {mark} {index}) {option.name} ({option.hint})\n")
    console.write("\n  (* = selected by default)\n")
    while True:
        console.write(
            "  Numbers to select (e.g. 1,3), 'all', 'none', "
            "enter to accept defaults, 'q' to cancel: "
        )
        result = parse_selection(console.read_line(), working)
        if isinstance(result, str):
            console.write(f"  {result}\n")
            continue
        console.write("\n")
        return result


def _confirm_full(console: Console, title: str, plan: list[str]) -> bool:
    yes = False
    color = supports_color(console)
    framer = _Framer(console)
    console.write(_HIDE_CURSOR)
    try:
        with console.raw():
            framer.draw(build_confirm_frame(title, plan, yes, "active", color))
            while True:
                key = console.read_key()
                if key in ("UP", "DOWN", "k", "j", "SPACE"):
                    yes = not yes
                elif key in ("y", "Y"):
                    yes = True
                elif key in ("n", "N"):
                    yes = False
                elif key == "ENTER":
                    framer.draw(build_confirm_frame(title, plan, yes, "submit", color))
                    return yes
                elif key in ("ESC", "q"):
                    framer.draw(build_confirm_frame(title, plan, yes, "cancel", color))
                    return False
                elif key == "CANCEL":
                    # See handle_key: Ctrl-C must exit 130 here just as it does
                    # in the line-mode prompt, where SIGINT is still live.
                    framer.draw(build_confirm_frame(title, plan, yes, "cancel", color))
                    raise KeyboardInterrupt
                framer.draw(build_confirm_frame(title, plan, yes, "active", color))
    finally:
        console.write(_SHOW_CURSOR)


def _confirm_line(console: Console, title: str, plan: list[str]) -> bool:
    console.write(f"\n{title}\n")
    for entry in plan:
        console.write(f"    {entry}\n")
    console.write("\n  Proceed? [y/N]: ")
    answer = parse_confirm(console.read_line())
    console.write("\n")
    return answer
