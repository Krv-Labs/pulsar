"""Filesystem primitives for atomic writes, JSON/JSONC parsing, and pruning."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

BACKUP_SUFFIX = ".pulsar.backup"
TMP_SUFFIX = ".pulsar.tmp"


@dataclass
class WriteOutcome:
    created_dirs: list[Path]
    created_file: bool


def backup_path(path: Path) -> Path:
    return path.with_name(path.name + BACKUP_SUFFIX)


def _tmp_path(path: Path) -> Path:
    return path.with_name(path.name + TMP_SUFFIX)


def is_dangling_symlink(path: Path) -> bool:
    return path.is_symlink() and not path.exists()


def resolve_symlink(path: Path) -> Path:
    if path.is_symlink():
        try:
            return path.resolve()
        except OSError:
            return path
    return path


def atomic_write(path: Path, contents: str, backup: bool) -> WriteOutcome:
    if is_dangling_symlink(path):
        raise RuntimeError(f"{path} is a dangling symlink; refusing to replace it")
    created_dirs = _create_parents(path)
    target = resolve_symlink(path)
    created_file = not target.exists()
    if backup and not created_file:
        shutil.copy2(target, backup_path(target))
    tmp = _tmp_path(target)
    tmp.write_text(contents, encoding="utf-8")
    if not created_file:
        try:
            shutil.copystat(target, tmp)
        except OSError:
            pass
    os.replace(tmp, target)
    return WriteOutcome(created_dirs=created_dirs, created_file=created_file)


def _create_parents(path: Path) -> list[Path]:
    parent = path.parent
    if parent == Path():
        return []
    missing: list[Path] = []
    current = parent
    while not current.exists() and current != current.parent:
        missing.append(current)
        current = current.parent
    missing.reverse()
    for directory in missing:
        try:
            directory.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            pass
    return missing


def read_json_object(path: Path) -> dict:
    text = _read_optional(path)
    if text is None:
        return {}
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} top-level value must be an object")
    return data


def read_jsonc_object(path: Path) -> tuple[dict, bool]:
    text = _read_optional(path)
    if text is None:
        return {}, False
    stripped, had_comments = strip_jsonc(text)
    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError(f"{path} top-level value must be an object")
    return data, had_comments


def write_json_object(path: Path, data: dict, backup: bool) -> WriteOutcome:
    contents = json.dumps(data, indent=2) + "\n"
    return atomic_write(path, contents, backup)


def _read_optional(path: Path) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return None
    return text


def strip_jsonc(text: str) -> tuple[str, bool]:
    """Strip comments and trailing commas without touching string contents."""
    out: list[str] = []
    commas: list[int] = []
    index = 0
    had_comments = False
    in_string = False
    escape = False
    while index < len(text):
        ch = text[index]
        if in_string:
            out.append(ch)
            in_string, escape = _string_state(ch, escape)
            index += 1
            continue
        if ch == '"':
            in_string = True
        else:
            comment = _consume_comment(text, index)
            if comment is not None:
                index, replacement = comment
                out.extend(replacement)
                had_comments = True
                continue
            if ch == ",":
                commas.append(len(out))
        out.append(ch)
        index += 1
    return _drop_trailing_commas(out, commas), had_comments


def _string_state(ch: str, escape: bool) -> tuple[bool, bool]:
    if escape:
        return True, False
    if ch == "\\":
        return True, True
    return ch != '"', False


def _consume_comment(text: str, index: int) -> tuple[int, list[str]] | None:
    if text.startswith("//", index):
        end = text.find("\n", index)
        end = len(text) if end == -1 else end
        return end, [" "] * (end - index)
    if not text.startswith("/*", index):
        return None
    end = text.find("*/", index + 2)
    end = len(text) if end == -1 else end + 2
    replacement = ["\n" if ch == "\n" else " " for ch in text[index:end]]
    return end, replacement


def _drop_trailing_commas(chars: list[str], commas: list[int]) -> str:
    """Blank commas that precede a closing brace/bracket.

    Only the positions collected outside string literals are considered — a
    regex over the whole document would also eat a `, }` inside a string value.
    """
    for start in commas:
        index = start + 1
        while index < len(chars) and chars[index].isspace():
            index += 1
        if index < len(chars) and chars[index] in "}]":
            chars[start] = " "
    return "".join(chars)


def prune_dirs(dirs: list[Path]) -> None:
    for directory in reversed(dirs):
        try:
            directory.rmdir()
        except OSError:
            pass


def is_empty_json_config(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return not read_json_object(path)
    except (ValueError, json.JSONDecodeError):
        text = path.read_text(encoding="utf-8")
        return not text.strip()
