"""Package version.

Installed wheels/sdists get their version from maturin, which reads
``[package].version`` in ``Cargo.toml``. Editable checkouts fall back to
that same file so every distribution channel shares one source of truth.

The fallback extracts the version with a small regex rather than a TOML
library, so it resolves with zero third-party dependencies -- including
under ``python -S`` and inside frozen bundles where ``tomllib`` (absent
before 3.11) and ``tomli`` may both be unavailable.
"""

from __future__ import annotations

import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_PACKAGE = "thema-pulsar"
_FALLBACK_VERSION = "0.0.0+unknown"

# The ``[package]`` table of Cargo.toml, up to the next table header or EOF.
_PACKAGE_TABLE_RE = re.compile(
    r"^\[package\]\s*$(?P<body>.*?)(?=^\[|\Z)",
    re.MULTILINE | re.DOTALL,
)
# ``version = "x.y.z"`` on its own line within that table.
_VERSION_RE = re.compile(r'^\s*version\s*=\s*"([^"]+)"', re.MULTILINE)


def _cargo_toml_candidates() -> list[Path]:
    candidates = [Path(__file__).resolve().parent.parent / "Cargo.toml"]
    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir:
        candidates.append(Path(bundle_dir) / "Cargo.toml")
    return candidates


def _parse_package_version(text: str) -> str | None:
    table = _PACKAGE_TABLE_RE.search(text)
    if table is None:
        return None
    match = _VERSION_RE.search(table.group("body"))
    return match.group(1) if match else None


def _cargo_version() -> str | None:
    for cargo_toml in _cargo_toml_candidates():
        try:
            text = cargo_toml.read_text(encoding="utf-8")
        except OSError:
            continue
        parsed = _parse_package_version(text)
        if parsed is not None:
            return parsed
    return None


def get_version() -> str:
    try:
        return version(_PACKAGE)
    except PackageNotFoundError:
        pass
    return _cargo_version() or _FALLBACK_VERSION


__version__ = get_version()
