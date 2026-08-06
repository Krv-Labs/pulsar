"""Pulsar CLI — install/uninstall/status for MCP agent harnesses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pulsar.cli.install import configure, status, uninstall
from pulsar.cli.install.artifact import State
from pulsar.cli.install.command import (
    LaunchSpec,
    resolve_launch_spec,
    resolve_launch_spec_optional,
)
from pulsar.cli.install.harness import HARNESSES, get_harness, harness_ids
from pulsar.cli.install.menu import (
    MenuOption,
    can_prompt,
    menu_hint,
    run_confirm,
    run_menu,
)
from pulsar.cli.install.paths import home_dir


def _is_headless() -> bool:
    """No terminal anywhere: safe to act without asking for confirmation."""
    return not any(
        stream.isatty() for stream in (sys.stdin, sys.stdout, sys.stderr)
    )


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "install":
            _run_install(args)
        elif args.command == "uninstall":
            _run_uninstall(args)
        elif args.command == "status":
            _run_status(args)
        else:
            parser.print_help()
            raise SystemExit(2)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pulsar", description="Pulsar CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    install = sub.add_parser("install", help="Configure agent harnesses to use Pulsar")
    install.add_argument("harnesses", nargs="*", help="Harness ids to configure")
    install.add_argument("--all", action="store_true", help="Configure every harness")
    install.add_argument("--dry-run", action="store_true", help="Preview without writing")
    install.add_argument(
        "--mode",
        choices=("uvx", "pipx"),
        default="uvx",
        help="Launch mode written into MCP configs (default: uvx)",
    )
    install.add_argument(
        "--pin-version",
        action="store_true",
        help="Pin thema-pulsar to the installed package version in uvx args",
    )

    uninstall_parser = sub.add_parser(
        "uninstall", help="Remove Pulsar from agent harnesses"
    )
    uninstall_parser.add_argument("harnesses", nargs="*", help="Harness ids to remove")
    uninstall_parser.add_argument("--all", action="store_true")
    uninstall_parser.add_argument("--dry-run", action="store_true")
    uninstall_parser.add_argument("-y", "--yes", action="store_true")
    uninstall_parser.add_argument("--purge-backups", action="store_true")

    status_parser = sub.add_parser("status", help="Show harness configuration status")
    status_parser.add_argument("--json", action="store_true")

    return parser


def _run_install(args: argparse.Namespace) -> None:
    home = home_dir()
    launch = resolve_launch_spec(mode=args.mode, pin_version=args.pin_version)
    selected = _select_install_targets(args, home, launch)
    if selected is None:
        print("Cancelled.")
        return
    if not selected:
        print("No integrations selected.")
        return
    configure.run(home, launch, selected, dry_run=args.dry_run)


def _run_uninstall(args: argparse.Namespace) -> None:
    home = home_dir()
    # Removal never needs a launcher — uninstalling must stay possible after
    # uv itself is gone.
    selected = _select_uninstall_targets(args, home)
    if selected is None:
        print("Cancelled.")
        return
    if not selected:
        print("No integrations selected for removal.")
        return

    if args.dry_run:
        uninstall.run(home, selected, True, args.purge_backups)
        return

    if args.yes or _is_headless():
        uninstall.run(home, selected, False, args.purge_backups)
        return

    if can_prompt():
        plan = uninstall.plan(home, selected, args.purge_backups)
        if run_confirm("Uninstall Pulsar from these agents?", plan):
            uninstall.run(home, selected, False, args.purge_backups)
        else:
            print("Cancelled.")
        return

    raise RuntimeError(
        "there is nowhere to confirm (stdin and stderr must both be a "
        "terminal) — re-run with --yes to apply, or --dry-run to preview"
    )


def _run_status(args: argparse.Namespace) -> None:
    home = home_dir()
    status.run(home, resolve_launch_spec_optional(), json_output=args.json)


def _select_install_targets(
    args: argparse.Namespace, home: Path, launch: LaunchSpec
) -> list[str] | None:
    if args.harnesses or args.all:
        return _validate_ids(args.harnesses, args.all)
    if not can_prompt():
        raise RuntimeError(
            "non-interactive shells must pass explicit harness names or --all "
            "(see `pulsar install --help`)"
        )
    return _interactive_select(home, launch, install=True)


def _select_uninstall_targets(
    args: argparse.Namespace,
    home: Path,
) -> list[str] | None:
    if args.harnesses or args.all:
        return _validate_ids(args.harnesses, args.all)
    if can_prompt():
        return _interactive_select(home, None, install=False)
    return [
        spec.id
        for spec in HARNESSES
        if spec.artifact.inspect_loose(spec.config_path(home)).state != State.ABSENT
    ]


def _interactive_select(
    home: Path, launch: LaunchSpec | None, *, install: bool
) -> list[str] | None:
    title = (
        "Which agent integrations do you want to configure?"
        if install
        else "Which agent integrations do you want to uninstall?"
    )
    options: list[MenuOption] = []
    for spec in HARNESSES:
        if install and launch is not None:
            inspection = spec.artifact.inspect(spec.config_path(home), launch)
        else:
            inspection = spec.artifact.inspect_loose(spec.config_path(home))
        hint, hint_style, checked = menu_hint(
            install=install,
            state=inspection.state,
            detected=spec.detect(home),
        )
        options.append(
            MenuOption(
                id=spec.id,
                name=spec.name,
                hint=hint,
                hint_style=hint_style,
                checked=checked,
                is_active=inspection.state == State.ACTIVE,
            )
        )
    return run_menu(title, options)


def _validate_ids(requested: list[str], all_harnesses: bool) -> list[str]:
    if all_harnesses:
        return harness_ids()
    unknown: list[str] = []
    selected: list[str] = []
    for harness_id in requested:
        lowered = harness_id.lower()
        if get_harness(lowered) is None:
            unknown.append(harness_id)
        elif lowered not in selected:
            selected.append(lowered)
    if unknown:
        supported = ", ".join(harness_ids())
        raise RuntimeError(
            f"unknown harness(es): {', '.join(unknown)} (supported: {supported})"
        )
    return selected


if __name__ == "__main__":
    main()
