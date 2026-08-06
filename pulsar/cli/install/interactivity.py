"""TTY interactivity detection for install/uninstall commands."""

from __future__ import annotations

import sys
from enum import Enum


class Interactivity(Enum):
    INTERACTIVE = "interactive"
    HEADLESS = "headless"
    AMBIGUOUS = "ambiguous"


def detect_interactivity() -> Interactivity:
    stderr_tty = sys.stderr.isatty()
    stdout_tty = sys.stdout.isatty()
    stdin_tty = sys.stdin.isatty()
    if stderr_tty:
        return Interactivity.INTERACTIVE
    if not stderr_tty and not stdout_tty and not stdin_tty:
        return Interactivity.HEADLESS
    return Interactivity.AMBIGUOUS
