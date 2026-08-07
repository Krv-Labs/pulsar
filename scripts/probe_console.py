#!/usr/bin/env python3
"""Report controlling-console detection for CI.

Windows CI runs this before the install suite so a regression in
``CONIN$``/``CONOUT$`` / VT enablement fails the job with a short, obvious
message instead of a buried pytest import error.
"""

from __future__ import annotations

from pulsar.cli.install.terminal import open_console, supports_color


def main() -> int:
    console = open_console()
    if console is None:
        print("console: none")
        return 0
    try:
        print(
            f"console: tier={console.tier.name} "
            f"columns={console.columns()} "
            f"color={supports_color(console)}"
        )
    finally:
        console.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
