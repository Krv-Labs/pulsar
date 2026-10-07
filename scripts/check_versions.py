#!/usr/bin/env python3
"""Fail if any published version string diverges from the workspace Cargo.toml."""

from __future__ import annotations

import re
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Member crates whose versions must stay in sync with the workspace version.
MEMBER_MANIFESTS = (
    ROOT / "crates/pulsar-core/Cargo.toml",
    ROOT / "crates/pulsar-python/Cargo.toml",
)

# The ``[workspace.package]`` (or legacy ``[package]``) table of the root
# Cargo.toml, up to the next table header or EOF.
_PACKAGE_TABLE_RE = re.compile(
    r"^\[(?:workspace\.)?package\]\s*$(?P<body>.*?)(?=^\[|\Z)",
    re.MULTILINE | re.DOTALL,
)
# ``version = "x.y.z"`` on its own line within that table.
_VERSION_RE = re.compile(r'^\s*version\s*=\s*"([^"]+)"', re.MULTILINE)
# ``version.workspace = true`` — a member inheriting the workspace version.
_WORKSPACE_INHERIT_RE = re.compile(r"^\s*version\.workspace\s*=\s*true", re.MULTILINE)


def cargo_version() -> str:
    text = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    table = _PACKAGE_TABLE_RE.search(text)
    if table is None:
        raise ValueError("no [workspace.package] or [package] table found in Cargo.toml")
    match = _VERSION_RE.search(table.group("body"))
    if match is None:
        raise ValueError("no version key in the package table of Cargo.toml")
    return match.group(1)


def member_version_errors(expected: str) -> list[str]:
    """Each member must inherit the workspace version or restate it exactly."""
    errors: list[str] = []
    for manifest in MEMBER_MANIFESTS:
        text = manifest.read_text(encoding="utf-8")
        if _WORKSPACE_INHERIT_RE.search(text):
            continue
        match = _VERSION_RE.search(text)
        found = match.group(1) if match else None
        if found != expected:
            errors.append(
                f"{manifest.relative_to(ROOT)} version is {found!r}; expected "
                f"version.workspace = true or {expected!r}"
            )
    return errors


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

    errors.extend(member_version_errors(expected))

    if errors:
        for message in errors:
            print(message, file=sys.stderr)
        return 1

    print(f"version check passed ({expected})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
