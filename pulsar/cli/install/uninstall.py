"""Remove Pulsar-owned MCP registrations and leave no trace."""

from __future__ import annotations

from pathlib import Path

from pulsar.cli.install import report, state
from pulsar.cli.install.artifact import State
from pulsar.cli.install.fsops import (
    backup_path,
    is_empty_json_config,
    prune_dirs,
    resolve_symlink,
)
from pulsar.cli.install.harness import HARNESSES, HarnessSpec, get_harness


def plan(home: Path, selected: list[str], purge_backups: bool) -> list[str]:
    actions: list[str] = []
    for harness_id in selected:
        spec = get_harness(harness_id)
        if spec is None:
            actions.append(f"unknown harness: {harness_id}")
            continue
        path = spec.config_path(home)
        inspection = spec.artifact.inspect_loose(path)
        if inspection.state == State.ABSENT:
            summary = f"{spec.name} — already clear"
        elif inspection.state == State.CONFLICT:
            detail = inspection.detail or "conflict"
            summary = f"{spec.name} — left untouched ({detail})"
        else:
            summary = (
                f"{spec.name} — remove MCP entry from "
                f"{_display_path(home, path)}"
            )
        actions.append(summary)
        if purge_backups:
            backup = backup_path(resolve_symlink(path))
            if backup.is_file():
                actions.append(
                    f"{spec.name} — delete backup {_display_path(home, backup)}"
                )
    return actions


def run(
    home: Path,
    selected: list[str],
    dry_run: bool,
    purge_backups: bool,
) -> None:
    report.header("Pulsar Harness Uninstall", dry_run)

    outcomes = [_remove_one(home, harness_id, dry_run) for harness_id in selected]
    _clean_up(home, selected, dry_run, purge_backups)

    if not all(outcomes):
        report.footer("Incomplete — review the entries above.")
        raise SystemExit(1)

    message = (
        "Done. (dry run — re-run without --dry-run to apply)"
        if dry_run
        else "Done. Restart any running agent for the change to take effect."
    )
    report.footer(message)


def _remove_one(home: Path, harness_id: str, dry_run: bool) -> bool:
    spec = get_harness(harness_id)
    if spec is None:
        report.detail(report.FAILED, f"unknown harness: {harness_id}")
        return False

    report.harness_line(spec.name)
    success = _remove_spec(home, spec, dry_run)
    print("│")
    return success


def _remove_spec(home: Path, spec: HarnessSpec, dry_run: bool) -> bool:
    path = spec.config_path(home)
    inspection = spec.artifact.inspect_loose(path)

    if inspection.state == State.ABSENT:
        report.detail(report.ABSENT, spec.absent_msg)
        return True

    if inspection.state == State.CONFLICT:
        detail = inspection.detail or f"{path} left untouched"
        report.detail(report.CONFLICT, detail)
        return True

    if dry_run:
        report.detail(
            report.REMOVED,
            f"would remove the MCP server entry from {_display_path(home, path)}",
        )
        return True

    try:
        spec.artifact.remove(path, dry_run=False)
    except RuntimeError as exc:
        report.detail(report.FAILED, str(exc))
        return False

    _delete_if_emptied_and_ours(home, spec.id, path)
    report.detail(report.REMOVED, f"removed the MCP server entry from {path}")
    return True


def _delete_if_emptied_and_ours(home: Path, harness_id: str, path: Path) -> None:
    if not state.was_created_by_install(home, harness_id, path):
        return
    if path.suffix == ".toml":
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        if text.strip():
            return
    elif not is_empty_json_config(path):
        return
    path.unlink(missing_ok=True)


def _clean_up(
    home: Path, selected: list[str], dry_run: bool, purge_backups: bool
) -> None:
    dirs = state.created_dirs(home)
    for harness_id in selected:
        state.clear_created_files(home, harness_id)
    if not dry_run:
        prune_dirs(dirs)
        if purge_backups:
            for spec in HARNESSES:
                if spec.id not in selected:
                    continue
                backup = backup_path(resolve_symlink(spec.config_path(home)))
                backup.unlink(missing_ok=True)
        prune_dirs([state.state_dir(home)])


def _display_path(home: Path, path: Path) -> str:
    try:
        rel = path.relative_to(home)
        return f"~/{rel.as_posix()}"
    except ValueError:
        return str(path)
