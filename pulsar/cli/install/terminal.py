"""Controlling-console access for interactive install/uninstall prompts.

Capability ladder (see CHANGELOG):

1. ``Tier.FULL`` — redrawn TUI with ANSI. POSIX: ``/dev/tty`` + termios.
   Windows: ``CONIN$``/``CONOUT$`` + ``msvcrt`` with
   ``ENABLE_VIRTUAL_TERMINAL_PROCESSING``.
2. ``Tier.LINE`` — numbered ASCII prompt when ANSI will not render
   (``TERM=dumb``, or a Windows console whose VT mode cannot be set).
3. ``None`` from :func:`open_console` — no controlling console at all.

Key reads are normalized to abstract names (``UP``, ``ENTER``, ``CANCEL``, …)
because Windows reports arrows as a scancode pair rather than CSI.
"""

from __future__ import annotations

import os
import re
import sys
from contextlib import contextmanager
from enum import Enum
from typing import Iterator, TextIO


class Tier(Enum):
    FULL = "full"
    LINE = "line"


# CSI / OSC / simple single-char ESC sequences that do not advance the cursor.
_ANSI_RE = re.compile(
    r"\033\[[0-9;?]*[A-Za-z]"  # CSI
    r"|\033\][^\033\x07]*(?:\x07|\033\\)"  # OSC
    r"|\033[=>]"  # keypad modes, etc.
)


def visual_rows(lines: list[str], columns: int) -> int:
    """Physical rows occupied by ``lines`` after ANSI strip + soft-wrap.

    Used by the menu framer so cursor-up equals rows just advanced — counting
    logical lines alone undercounts wrapped labels and strands the cursor.
    """
    width = columns if columns > 0 else 80
    total = 0
    for line in lines:
        plain = _ANSI_RE.sub("", line).replace("\r", "")
        if not plain:
            total += 1
            continue
        total += max(1, (len(plain) + width - 1) // width)
    return total


def supports_color(console: Console) -> bool:
    """Whether the menu should emit SGR color codes for this console."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    return console.tier is Tier.FULL


class Console:
    """A handle on the controlling console (not necessarily stdin/stdout)."""

    def __init__(
        self,
        *,
        writer: TextIO,
        tier: Tier,
        owns_writer: bool,
        fd: int | None = None,
        reader: TextIO | None = None,
        owns_reader: bool = False,
    ) -> None:
        self.tier = tier
        self._writer = writer
        self._owns_writer = owns_writer
        self._fd = fd
        self._reader = reader
        self._owns_reader = owns_reader
        self._raw_depth = 0
        self._saved_termios: list | None = None

    def write(self, text: str) -> None:
        self._writer.write(text)
        self._writer.flush()

    def columns(self) -> int:
        try:
            return os.get_terminal_size(self._writer.fileno()).columns
        except (OSError, AttributeError, ValueError):
            try:
                return os.get_terminal_size().columns
            except OSError:
                return 80

    def read_line(self) -> str:
        if self._fd is not None:
            return _read_line_fd(self._fd)
        assert self._reader is not None
        return self._reader.readline().rstrip("\r\n")

    def read_key(self) -> str:
        if sys.platform == "win32":
            return _read_key_windows()
        assert self._fd is not None
        return _read_key_posix(self._fd)

    @contextmanager
    def raw(self) -> Iterator[None]:
        """Enter raw/cbreak input. Nested calls are reference-counted."""
        if sys.platform == "win32":
            # msvcrt.getwch already returns unbuffered console input.
            yield
            return
        import termios
        import tty

        assert self._fd is not None
        fd = self._fd
        if self._raw_depth == 0:
            self._saved_termios = termios.tcgetattr(fd)
            tty.setraw(fd, when=termios.TCSANOW)
        self._raw_depth += 1
        try:
            yield
        finally:
            self._raw_depth -= 1
            if self._raw_depth == 0 and self._saved_termios is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, self._saved_termios)
                self._saved_termios = None

    def close(self) -> None:
        if self._raw_depth and self._saved_termios is not None and self._fd is not None:
            try:
                import termios

                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_termios)
            except Exception:
                pass
            self._raw_depth = 0
            self._saved_termios = None
        if self._owns_reader and self._reader is not None:
            try:
                self._reader.close()
            except OSError:
                pass
        if self._owns_writer:
            try:
                self._writer.close()
            except OSError:
                pass
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def __enter__(self) -> Console:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def open_console() -> Console | None:
    """Open the controlling console, or ``None`` when there is none."""
    if sys.platform == "win32":
        return _open_windows()
    return _open_posix()


# --- POSIX -----------------------------------------------------------------


def _open_posix() -> Console | None:
    try:
        fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
    except OSError:
        return None
    try:
        # Separate writer fd so TextIO buffering cannot steal input bytes
        # from the os.read path used by read_key / read_line.
        writer = open(os.dup(fd), "w", encoding="utf-8", closefd=True)  # noqa: SIM115
    except OSError:
        os.close(fd)
        return None

    tier = Tier.LINE if _term_is_dumb() else Tier.FULL
    return Console(
        writer=writer,
        tier=tier,
        owns_writer=True,
        fd=fd,
    )


def _read_line_fd(fd: int) -> str:
    buf = bytearray()
    while True:
        chunk = os.read(fd, 1)
        if not chunk or chunk in (b"\n", b"\r"):
            break
        buf.extend(chunk)
    return buf.decode("utf-8", errors="replace")


def _read_key_posix(fd: int) -> str:
    import select

    ch = _read_byte(fd)
    if ch == "\x03":
        return "CANCEL"
    if ch in ("\r", "\n"):
        return "ENTER"
    if ch == " ":
        return "SPACE"
    if ch != "\x1b":
        return ch

    # Timed read for the rest of an escape sequence. A blocking fixed-length
    # read hung forever on a bare Esc (and ate the *next* keypress on arrows).
    suffix: list[str] = []
    while True:
        ready, _, _ = select.select([fd], [], [], 0.05)
        if not ready:
            break
        nxt = _read_byte(fd)
        if not nxt:
            break
        suffix.append(nxt)
        if nxt.isalpha() or nxt == "~":
            break

    seq = "".join(suffix)
    if seq == "[A":
        return "UP"
    if seq == "[B":
        return "DOWN"
    if seq == "[C":
        return "RIGHT"
    if seq == "[D":
        return "LEFT"
    if seq == "":
        return "ESC"
    # Unknown CSI — ignore as a no-op key rather than injecting garbage.
    return ""


def _read_byte(fd: int) -> str:
    data = os.read(fd, 1)
    if not data:
        return ""
    return data.decode("utf-8", errors="replace")


# --- Windows ---------------------------------------------------------------


def _open_windows() -> Console | None:
    try:
        # CONIN$/CONOUT$ reach the controlling console even when stdin/stdout
        # are redirected (the `pulsar install | tee log` case).
        reader = open("CONIN$", "r", encoding="utf-8", errors="replace")  # noqa: SIM115
        writer = open("CONOUT$", "w", encoding="utf-8", errors="replace")  # noqa: SIM115
    except OSError:
        return None

    vt_ok = _enable_virtual_terminal(writer)
    tier = Tier.LINE if (_term_is_dumb() or not vt_ok) else Tier.FULL
    return Console(
        writer=writer,
        tier=tier,
        owns_writer=True,
        reader=reader,
        owns_reader=True,
    )


def _enable_virtual_terminal(writer: TextIO) -> bool:
    """Turn on VT processing on CONOUT$; False if the console cannot do ANSI."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    enable_vt = 0x0004
    enable_processed = 0x0001

    try:
        import msvcrt

        handle = msvcrt.get_osfhandle(writer.fileno())
    except (OSError, AttributeError):
        return False

    mode = wintypes.DWORD()
    if not kernel32.GetConsoleMode(wintypes.HANDLE(handle), ctypes.byref(mode)):
        return False
    new_mode = mode.value | enable_vt | enable_processed
    if new_mode == mode.value:
        return True
    return bool(kernel32.SetConsoleMode(wintypes.HANDLE(handle), new_mode))


def _read_key_windows() -> str:
    import msvcrt

    ch = msvcrt.getwch()
    if ch == "\x03":
        return "CANCEL"
    if ch in ("\r", "\n"):
        return "ENTER"
    if ch == " ":
        return "SPACE"
    if ch == "\x1b":
        return "ESC"
    # Arrow / function keys arrive as a lead byte plus a scancode.
    if ch in ("\x00", "\xe0"):
        code = msvcrt.getwch()
        return {
            "H": "UP",
            "P": "DOWN",
            "K": "LEFT",
            "M": "RIGHT",
        }.get(code, "")
    return ch


def _term_is_dumb() -> bool:
    return os.environ.get("TERM", "") == "dumb"
