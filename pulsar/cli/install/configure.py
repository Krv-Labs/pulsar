"""Apply Pulsar MCP registration to selected harnesses."""

from __future__ import annotations

from pathlib import Path

from pulsar.cli.install.artifact import State
from pulsar.cli.install.command import LaunchSpec
from pulsar.cli.install.harness import HarnessSpec, get_harness
from pulsar.cli.install import report, state


def run(
    home: Path,
    launch: LaunchSpec,
    selected: list[str],
    dry_run: bool,
) -> None:
    report.header("Pulsar Harness Install", dry_run)
    print(f"│  Using {launch.command} {' '.join(launch.args)}")
    print("│")

    outcomes: list[bool] = []
    for harness_id in selected:
        outcomes.append(_configure_one(home, launch, harness_id, dry_run))

    if not all(outcomes):
        report.footer(
            "Incomplete — existing files were preserved; review the entries above."
        )
        raise SystemExit(1)

    message = (
        "Done. (dry run — re-run without --dry-run to apply)"
        if dry_run
        else "Done. Restart any running agent for it to pick up the new server."
    )
    report.footer(message)


def _configure_one(
    home: Path, launch: LaunchSpec, harness_id: str, dry_run: bool
) -> bool:
    spec = get_harness(harness_id)
    if spec is None:
        report.detail(report.failed(), f"unknown harness: {harness_id}")
        return False

    report.harness_line(spec.name)
    success = _configure_spec(home, launch, spec, dry_run)
    if note := spec.note(home):
        report.note(note)
    print("│")
    return success


def _configure_spec(
    home: Path, launch: LaunchSpec, spec: HarnessSpec, dry_run: bool
) -> bool:
    path = spec.config_path(home)
    inspection = spec.artifact.inspect(path, launch)

    if inspection.state == State.ACTIVE:
        report.detail(report.ok(), spec.active_msg)
        return True

    if inspection.state == State.CONFLICT:
        detail = inspection.detail or f"{path} needs manual attention"
        report.detail(report.conflict(), detail)
        return False

    if dry_run:
        if inspection.detail:
            report.detail(
                report.pending(),
                f"[dry run] would repair {path}: {inspection.detail}",
            )
        else:
            report.detail(
                report.pending(),
                f"[dry run] would register the MCP server in {path}",
            )
        return True

    repair = inspection.state == State.INCOMPLETE
    try:
        outcome = spec.artifact.apply(path, launch)
    except RuntimeError as exc:
        report.detail(report.failed(), str(exc))
        return False

    wrote = outcome is not None
    if outcome is not None:
        if outcome.created_file:
            state.record_created_file(home, spec.id, path)
        if outcome.created_dirs:
            state.record_created_dirs(home, outcome.created_dirs)

    report.detail(report.ok(), _applied_message(spec, repair, wrote))
    return True


def _applied_message(spec: HarnessSpec, repair: bool, wrote: bool) -> str:
    if not wrote:
        return f"{spec.active_msg} (unchanged)"
    if repair:
        return f"repaired — {spec.active_msg}"
    return spec.active_msg
