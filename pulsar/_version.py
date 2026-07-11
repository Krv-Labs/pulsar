"""Package version.

Installed wheels/sdists get their version from maturin, which reads
``[package].version`` in ``Cargo.toml``. Editable checkouts fall back to
that same file so every distribution channel shares one source of truth.
"""

from __future__ import annotations

import sys
try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_PACKAGE = "thema-pulsar"
_FALLBACK_VERSION = "0.0.0+unknown"


def _cargo_toml_candidates() -> list[Path]:
    candidates = [Path(__file__).resolve().parent.parent / "Cargo.toml"]
    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir:
        candidates.append(Path(bundle_dir) / "Cargo.toml")
    return candidates


def _cargo_version() -> str | None:
    for cargo_toml in _cargo_toml_candidates():
        try:
            with cargo_toml.open("rb") as f:
                return tomllib.load(f)["package"]["version"]
        except (OSError, KeyError, tomllib.TOMLDecodeError):
            continue
    return None


def get_version() -> str:
    try:
        return version(_PACKAGE)
    except PackageNotFoundError:
        pass
    return _cargo_version() or _FALLBACK_VERSION


__version__ = get_version()
