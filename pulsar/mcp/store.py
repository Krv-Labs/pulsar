"""Object store for persisted derived artifacts (Stage 1, build-spec §2/§3.1).

Local-first: ``FsObjectStore`` writes blobs to a directory tree keyed by
``{user_id}/{dataset_id}/{config_hash}/...``. The prod swap (GCS) implements the
same ``put``/``get``/``exists`` surface — selected by ``DEPLOY_ENV`` upstream.

This is intentionally tiny and dependency-free: ``pulsar/artifacts.py`` only needs
a duck-typed object with ``put(key, bytes)`` / ``get(key) -> bytes`` / ``exists(key)``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol


class ObjectStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...


class FsObjectStore:
    """Local filesystem object store. Keys are POSIX-style relative paths."""

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Reject path traversal; keys are always relative.
        p = (self.root / key).resolve()
        if not str(p).startswith(str(self.root)):
            raise ValueError(f"object key escapes store root: {key!r}")
        return p

    def put(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()


def artifact_prefix(user_id: str, dataset_id: str, config_hash: str) -> str:
    """The object-store prefix for one artifact (build-spec §3.1)."""
    return f"{user_id}/{dataset_id}/{config_hash}"


def get_object_store() -> FsObjectStore:
    """Local object store from ``OBJECT_STORE_DIR`` (default ``./.localstore``)."""
    root = os.environ.get("OBJECT_STORE_DIR", "./.localstore")
    return FsObjectStore(root)
