"""Show which harnesses have Pulsar MCP registrations."""

from __future__ import annotations

import json
from pathlib import Path

from pulsar.cli.install.artifact import Inspection, State
from pulsar.cli.install.harness import HARNESSES, HarnessSpec
from pulsar.cli.install import report


def run(home: Path, json_output: bool) -> None:
    rows = [
        (spec, spec.artifact.inspect_loose(spec.config_path(home)))
        for spec in HARNESSES
    ]
    if json_output:
        _print_json(home, rows)
        return
    _print_human(home, rows)


def _print_human(home: Path, rows: list[tuple[HarnessSpec, Inspection]]) -> None:
    report.header("Pulsar Harness Status", dry_run=False)
    print("│  Launch: validated from each registration")
    print("│")

    for spec, inspection in _sorted(rows):
        report.harness_line(spec.name)
        report.detail(report.glyph(inspection.state), _describe(spec, inspection))
        if note := spec.note(home):
            report.note(note)
        print("│")

    active = sum(1 for _, inspection in rows if inspection.state == State.ACTIVE)
    report.footer(f"{active}/{len(rows)} harness integrations active.")


def _sorted(
    rows: list[tuple[HarnessSpec, Inspection]],
) -> list[tuple[HarnessSpec, Inspection]]:
    order = {
        State.ACTIVE: 0,
        State.INCOMPLETE: 1,
        State.CONFLICT: 2,
        State.ABSENT: 3,
    }
    return sorted(rows, key=lambda item: order[item[1].state])


def _describe(spec: HarnessSpec, inspection: Inspection) -> str:
    if inspection.state == State.ACTIVE:
        return spec.active_msg
    if inspection.state == State.ABSENT:
        return spec.absent_msg
    if inspection.state == State.INCOMPLETE:
        reason = inspection.detail or "needs repair"
        return f"{reason} — run `pulsar install {spec.id}`"
    return inspection.detail or "needs manual attention"


def _print_json(home: Path, rows: list[tuple[HarnessSpec, Inspection]]) -> None:
    active = sum(1 for _, inspection in rows if inspection.state == State.ACTIVE)
    payload = {
        "launch": None,
        "active": active,
        "total": len(rows),
        "harnesses": [
            {
                "id": spec.id,
                "name": spec.name,
                "state": report.label(inspection.state),
                "config": str(spec.config_path(home)),
                "detail": inspection.detail,
                "note": spec.note(home),
            }
            for spec, inspection in rows
        ],
    }
    print(json.dumps(payload, indent=2))
