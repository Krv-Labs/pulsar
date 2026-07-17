#!/usr/bin/env python3
"""Fail if any published version string diverges from Cargo.toml."""

from __future__ import annotations

import re
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The ``[package]`` table of Cargo.toml, up to the next table header or EOF.
_PACKAGE_TABLE_RE = re.compile(
    r"^\[package\]\s*$(?P<body>.*?)(?=^\[|\Z)",
    re.MULTILINE | re.DOTALL,
)
# ``version = "x.y.z"`` on its own line within that table.
_VERSION_RE = re.compile(r'^\s*version\s*=\s*"([^"]+)"', re.MULTILINE)


def cargo_version() -> str:
    text = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    table = _PACKAGE_TABLE_RE.search(text)
    if table is None:
        raise ValueError("no [package] table found in Cargo.toml")
    match = _VERSION_RE.search(table.group("body"))
    if match is None:
        raise ValueError("no version key in the [package] table of Cargo.toml")
    return match.group(1)


def python_source_version() -> str:
    version_globals = runpy.run_path(str(ROOT / "pulsar/_version.py"))
    return version_globals["_cargo_version"]()


def main() -> int:
    expected = cargo_version()
    errors: list[str] = []

    python_version = python_source_version()
    if python_version != expected:
        errors.append(
            f"pulsar._version._cargo_version() is {python_version!r}, "
            f"expected {expected!r}"
        )

    if errors:
        for message in errors:
            print(message, file=sys.stderr)
        return 1

    print(f"version check passed ({expected})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
