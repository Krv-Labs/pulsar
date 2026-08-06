"""Install ownership ledger for leave-no-trace uninstall."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pulsar.cli.install import paths
from pulsar.cli.install.fsops import read_json_object, write_json_object

HARNESSES_KEY = "harnesses"
CREATED_FILES_KEY = "createdFiles"
CREATED_DIRS_KEY = "createdDirs"
STATE_FILE = "install.json"

NEVER_PRUNE = {".local", ".config", "Library"}


def state_dir(home: Path) -> Path:
    if sys.platform == "win32":
        return paths.app_data(home) / "pulsar"
    return home / ".local" / "state" / "pulsar"


def state_file_path(home: Path) -> Path:
    return state_dir(home) / STATE_FILE


def record_created_file(home: Path, harness_id: str, path: Path) -> None:
    ledger = _load(home)
    files = _files_of(ledger, harness_id)
    key = str(path)
    if key not in files:
        files.append(key)
    ledger["harnesses"][harness_id] = {CREATED_FILES_KEY: files}
    _save(home, ledger)


def was_created_by_install(home: Path, harness_id: str, path: Path) -> bool:
    return str(path) in _files_of(_load(home), harness_id)


def record_created_dirs(home: Path, dirs: list[Path]) -> None:
    ledger = _load(home)
    existing = ledger.get(CREATED_DIRS_KEY, [])
    for directory in dirs:
        key = str(directory)
        if key in existing:
            continue
        # Never prune the home directory or the shared roots we may have had
        # to create on the way to a config file. Test the directory itself —
        # testing every ancestor part matches home.name and skips everything.
        if directory == home or directory.name in NEVER_PRUNE:
            continue
        existing.append(key)
    ledger[CREATED_DIRS_KEY] = existing
    _save(home, ledger)


def created_dirs(home: Path) -> list[Path]:
    return [Path(value) for value in _load(home).get(CREATED_DIRS_KEY, [])]


def clear_created_files(home: Path, harness_id: str) -> None:
    ledger = _load(home)
    harnesses = ledger.get(HARNESSES_KEY, {})
    harnesses.pop(harness_id, None)
    if harnesses:
        ledger[HARNESSES_KEY] = harnesses
        _save(home, ledger)
        return
    _remove_state_file(home)


def _load(home: Path) -> dict:
    path = state_file_path(home)
    if not path.is_file():
        return {HARNESSES_KEY: {}, CREATED_DIRS_KEY: []}
    try:
        data = read_json_object(path)
    except (ValueError, json.JSONDecodeError):
        return {HARNESSES_KEY: {}, CREATED_DIRS_KEY: []}

    if HARNESSES_KEY in data:
        harnesses = data.get(HARNESSES_KEY, {})
        dirs = data.get(CREATED_DIRS_KEY, [])
    else:
        harnesses = {
            key: value
            for key, value in data.items()
            if key != CREATED_DIRS_KEY and isinstance(value, dict)
        }
        dirs = data.get(CREATED_DIRS_KEY, [])

    if not isinstance(harnesses, dict):
        harnesses = {}
    if not isinstance(dirs, list):
        dirs = []
    return {HARNESSES_KEY: harnesses, CREATED_DIRS_KEY: dirs}


def _save(home: Path, ledger: dict) -> None:
    path = state_file_path(home)
    payload = {
        HARNESSES_KEY: ledger.get(HARNESSES_KEY, {}),
        CREATED_DIRS_KEY: ledger.get(CREATED_DIRS_KEY, []),
    }
    write_json_object(path, payload, backup=False)


def _files_of(ledger: dict, harness_id: str) -> list[str]:
    harnesses = ledger.get(HARNESSES_KEY, {})
    record = harnesses.get(harness_id, {})
    files = record.get(CREATED_FILES_KEY, [])
    if isinstance(files, list):
        return [str(item) for item in files]
    return []


def _remove_state_file(home: Path) -> None:
    path = state_file_path(home)
    if path.is_file():
        path.unlink()
    directory = state_dir(home)
    if directory.is_dir():
        try:
            directory.rmdir()
        except OSError:
            pass
