"""End-to-end tests for the Pulsar MCP install harness."""

from __future__ import annotations

import json
import stat
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from pulsar.cli.install.artifact import State
from pulsar.cli.install.command import LaunchSpec, uvx_args
from pulsar.cli.install.fsops import backup_path, strip_jsonc
from pulsar.cli.install.harness import get_harness
from pulsar.cli.install.terminal import Tier
from pulsar.cli.install import configure, paths, state, uninstall
from pulsar.cli.main import main


@pytest.fixture
def fake_uvx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    uvx = tmp_path / "bin" / "uvx"
    uvx.parent.mkdir(parents=True)
    uvx.write_text("#!/bin/sh\n", encoding="utf-8")
    uvx.chmod(uvx.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(uvx.parent))
    return uvx


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    if sys.platform == "win32":
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


def _launch(uvx: Path) -> LaunchSpec:
    return LaunchSpec(
        command=uvx,
        args=("--from", "thema-pulsar[mcp]", "pulsar-mcp"),
        mode="uvx",
    )


def test_install_and_uninstall_cursor_round_trip(home: Path, fake_uvx: Path) -> None:
    launch = _launch(fake_uvx)
    (home / ".cursor").mkdir()

    configure.run(home, launch, ["cursor"], dry_run=False)

    config = home / ".cursor" / "mcp.json"
    assert config.is_file()
    data = json.loads(config.read_text(encoding="utf-8"))
    assert data["mcpServers"]["pulsar"]["command"] == str(fake_uvx.resolve())
    assert data["mcpServers"]["pulsar"]["args"] == list(launch.args)

    spec = get_harness("cursor")
    assert spec is not None
    assert spec.artifact.inspect(config, launch).state == State.ACTIVE

    uninstall.run(home, ["cursor"], dry_run=False, purge_backups=False)
    assert spec.artifact.inspect_loose(config).state == State.ABSENT


def test_install_creates_backup_on_first_write(home: Path, fake_uvx: Path) -> None:
    launch = _launch(fake_uvx)
    (home / ".cursor").mkdir()
    config = home / ".cursor" / "mcp.json"
    config.write_text('{"other": true}\n', encoding="utf-8")
    pristine = config.read_text(encoding="utf-8")

    configure.run(home, launch, ["cursor"], dry_run=False)

    backup = backup_path(config)
    assert backup.is_file()
    assert backup.read_text(encoding="utf-8") == pristine


def test_install_repairs_relative_uvx_path(home: Path, fake_uvx: Path) -> None:
    launch = _launch(fake_uvx)
    (home / ".cursor").mkdir()
    config = home / ".cursor" / "mcp.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "pulsar": {
                        "command": "uvx",
                        "args": list(launch.args),
                    }
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    spec = get_harness("cursor")
    assert spec is not None
    assert spec.artifact.inspect(config, launch).state == State.INCOMPLETE

    configure.run(home, launch, ["cursor"], dry_run=False)
    assert spec.artifact.inspect(config, launch).state == State.ACTIVE


def test_foreign_pulsar_entry_is_conflict(home: Path, fake_uvx: Path) -> None:
    launch = _launch(fake_uvx)
    (home / ".cursor").mkdir()
    config = home / ".cursor" / "mcp.json"
    config.write_text(
        json.dumps(
            {"mcpServers": {"pulsar": {"command": "uvx", "args": ["other-mcp"]}}}
        )
        + "\n",
        encoding="utf-8",
    )

    spec = get_harness("cursor")
    assert spec is not None
    assert spec.artifact.inspect(config, launch).state == State.CONFLICT


def test_codex_toml_round_trip(home: Path, fake_uvx: Path) -> None:
    launch = _launch(fake_uvx)
    (home / ".codex").mkdir()

    configure.run(home, launch, ["codex"], dry_run=False)

    config = home / ".codex" / "config.toml"
    assert config.is_file()
    assert "pulsar" in config.read_text(encoding="utf-8")

    uninstall.run(home, ["codex"], dry_run=False, purge_backups=False)
    spec = get_harness("codex")
    assert spec is not None
    assert spec.artifact.inspect_loose(config).state == State.ABSENT


def test_vscode_jsonc_writes_stdio_type(home: Path, fake_uvx: Path) -> None:
    launch = _launch(fake_uvx)
    config = paths.vscode_config(home)
    config.parent.mkdir(parents=True)

    configure.run(home, launch, ["vscode"], dry_run=False)

    data = json.loads(config.read_text(encoding="utf-8"))
    assert data["servers"]["pulsar"]["type"] == "stdio"


def test_consoleless_install_requires_harness_ids(
    home: Path, fake_uvx: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    # No controlling console at all (CI, systemd) — there is nowhere to draw a
    # menu, so the CLI must say so rather than silently configure nothing.
    monkeypatch.setattr("pulsar.cli.install.menu.open_console", lambda: None)

    with pytest.raises(SystemExit) as exc:
        main(["install"])
    assert exc.value.code == 1
    assert "no console available" in capsys.readouterr().err


def test_install_all_headless(home: Path, fake_uvx: Path) -> None:
    launch = _launch(fake_uvx)
    (home / ".cursor").mkdir()
    (home / ".claude").mkdir()

    configure.run(home, launch, ["cursor", "claude"], dry_run=False)

    assert (home / ".cursor" / "mcp.json").is_file()
    assert (home / ".claude.json").is_file()


def test_validate_unknown_harness(home: Path, fake_uvx: Path, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["install", "nope"])
    assert exc.value.code == 1
    assert "unknown harness" in capsys.readouterr().err


class FakeConsole:
    """A scripted stand-in for the controlling console."""

    tier = Tier.FULL

    def __init__(self, keys: list[str]) -> None:
        self._keys = list(keys)
        self.written: list[str] = []

    def write(self, text: str) -> None:
        self.written.append(text)

    def columns(self) -> int:
        return 80

    def read_key(self) -> str:
        return self._keys.pop(0)

    def read_line(self) -> str:
        return ""

    @contextmanager
    def raw(self):
        yield

    def close(self) -> None:
        pass

    def __enter__(self) -> FakeConsole:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def test_install_with_piped_stdin_still_prompts(
    home: Path, fake_uvx: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Deliberate reversal of 094f776, which errored out whenever stdin was not
    # a TTY. The prompts now talk to the controlling console rather than to
    # stdin, so `pulsar install | tee log` gets a real menu. That commit's
    # intent — never silently no-op — is better served by prompting than by
    # refusing, and the consoleless case above still errors.
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    console = FakeConsole(["ENTER"])
    monkeypatch.setattr("pulsar.cli.install.menu.open_console", lambda: console)

    main(["install", "--dry-run"])

    frames = "".join(console.written)
    assert "Which agent integrations do you want to configure?" in frames


def test_gemini_directory_does_not_imply_antigravity(home: Path) -> None:
    (home / ".gemini").mkdir()
    spec = get_harness("antigravity")
    assert spec is not None
    assert spec.detect(home) is False


def test_state_tracks_created_files(home: Path, fake_uvx: Path) -> None:
    launch = _launch(fake_uvx)
    (home / ".cursor").mkdir()
    config = home / ".cursor" / "mcp.json"

    configure.run(home, launch, ["cursor"], dry_run=False)

    assert state.was_created_by_install(home, "cursor", config)


def test_cli_status_json(home: Path, fake_uvx: Path, capsys) -> None:
    launch = _launch(fake_uvx)
    (home / ".cursor").mkdir()
    config = home / ".cursor" / "mcp.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "pulsar": {
                        "command": str(fake_uvx.resolve()),
                        "args": list(launch.args),
                    }
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    main(["status", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["active"] >= 1
    assert any(row["id"] == "cursor" for row in payload["harnesses"])


def _write_cursor_entry(home: Path, command: str, args: list[str]) -> Path:
    (home / ".cursor").mkdir(exist_ok=True)
    config = home / ".cursor" / "mcp.json"
    config.write_text(
        json.dumps({"mcpServers": {"pulsar": {"command": command, "args": args}}})
        + "\n",
        encoding="utf-8",
    )
    return config


def test_install_repairs_arg_drift_instead_of_conflicting(
    home: Path, fake_uvx: Path
) -> None:
    launch = _launch(fake_uvx)
    config = _write_cursor_entry(
        home,
        str(fake_uvx.resolve()),
        ["--from", "thema-pulsar[mcp]==0.1.0", "pulsar-mcp"],
    )
    spec = get_harness("cursor")
    assert spec is not None
    assert spec.artifact.inspect(config, launch).state == State.INCOMPLETE

    configure.run(home, launch, ["cursor"], dry_run=False)
    assert spec.artifact.inspect(config, launch).state == State.ACTIVE
    data = json.loads(config.read_text(encoding="utf-8"))
    assert data["mcpServers"]["pulsar"]["args"] == list(launch.args)


def test_install_repairs_pipx_entry_and_backs_it_up(home: Path, fake_uvx: Path) -> None:
    server = fake_uvx.parent / "pulsar-mcp"
    server.write_text("#!/bin/sh\n", encoding="utf-8")
    server.chmod(server.stat().st_mode | stat.S_IXUSR)

    launch = _launch(fake_uvx)
    config = _write_cursor_entry(home, str(server.resolve()), [])
    pristine = config.read_text(encoding="utf-8")

    spec = get_harness("cursor")
    assert spec is not None
    # A pipx-mode entry is ours, so it is repairable — not a conflict.
    assert spec.artifact.inspect(config, launch).state == State.INCOMPLETE

    configure.run(home, launch, ["cursor"], dry_run=False)
    assert spec.artifact.inspect(config, launch).state == State.ACTIVE
    assert backup_path(config).read_text(encoding="utf-8") == pristine


def test_uninstall_removes_a_pinned_entry_install_would_repair(
    home: Path, fake_uvx: Path
) -> None:
    config = _write_cursor_entry(
        home,
        str(fake_uvx.resolve()),
        ["--from", "thema-pulsar[mcp]==0.1.0", "pulsar-mcp"],
    )
    spec = get_harness("cursor")
    assert spec is not None

    uninstall.run(home, ["cursor"], dry_run=False, purge_backups=False)
    assert spec.artifact.inspect_loose(config).state == State.ABSENT


def test_vscode_config_with_comments_is_a_conflict(home: Path, fake_uvx: Path) -> None:
    launch = _launch(fake_uvx)
    config = paths.vscode_config(home)
    config.parent.mkdir(parents=True)
    config.write_text('{\n  // hand-maintained\n  "servers": {}\n}\n', encoding="utf-8")
    pristine = config.read_text(encoding="utf-8")

    spec = get_harness("vscode")
    assert spec is not None
    inspection = spec.artifact.inspect(config, launch)
    assert inspection.state == State.CONFLICT
    assert "comments" in (inspection.detail or "")

    with pytest.raises(SystemExit) as exc:
        configure.run(home, launch, ["vscode"], dry_run=False)
    assert exc.value.code == 1
    assert config.read_text(encoding="utf-8") == pristine


def test_purge_backups_deletes_the_backup(home: Path, fake_uvx: Path) -> None:
    launch = _launch(fake_uvx)
    (home / ".cursor").mkdir()
    config = home / ".cursor" / "mcp.json"
    config.write_text('{"other": true}\n', encoding="utf-8")

    configure.run(home, launch, ["cursor"], dry_run=False)
    assert backup_path(config).is_file()

    uninstall.run(home, ["cursor"], dry_run=False, purge_backups=True)
    assert not backup_path(config).exists()


def test_uninstall_and_status_work_without_a_launcher(
    home: Path, fake_uvx: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    launch = _launch(fake_uvx)
    (home / ".cursor").mkdir()
    configure.run(home, launch, ["cursor"], dry_run=False)
    config = home / ".cursor" / "mcp.json"

    # uv has been uninstalled since registration.
    monkeypatch.setenv("PATH", str(home / "empty"))
    capsys.readouterr()

    main(["status", "--json"])
    assert json.loads(capsys.readouterr().out)["launch"] is None

    main(["uninstall", "cursor", "--yes"])
    assert not config.exists()


def test_jsonc_strip_preserves_commas_inside_strings() -> None:
    source = '{"servers": {}, "note": "a, }", "trailing": [1, 2,],}'
    stripped, had_comments = strip_jsonc(source)
    assert had_comments is False
    assert json.loads(stripped) == {
        "servers": {},
        "note": "a, }",
        "trailing": [1, 2],
    }


def test_jsonc_strip_removes_comments_and_trailing_commas() -> None:
    source = '{\n  // lead\n  "servers": {}, /* mid */\n  "x": 1,\n}\n'
    stripped, had_comments = strip_jsonc(source)
    assert had_comments is True
    assert json.loads(stripped) == {"servers": {}, "x": 1}


def test_uninstall_prunes_directories_install_created(
    home: Path, fake_uvx: Path
) -> None:
    launch = _launch(fake_uvx)
    config = paths.claude_desktop_config(home)

    configure.run(home, launch, ["claude-desktop"], dry_run=False)
    assert config.is_file()
    assert state.created_dirs(home), "install recorded no directories"

    uninstall.run(home, ["claude-desktop"], dry_run=False, purge_backups=False)
    assert not config.exists()
    assert not config.parent.exists()
    assert home.is_dir()


def test_pin_version_emits_a_parseable_requirement() -> None:
    args = uvx_args(pin_version=True)
    assert args[0] == "--from"
    assert args[-1] == "pulsar-mcp"
    # extras precede the version specifier in PEP 508
    Requirement(args[1])


def test_uninstall_dry_run_preserves_ownership_state(
    home: Path, fake_uvx: Path
) -> None:
    launch = _launch(fake_uvx)
    (home / ".cursor").mkdir()
    config = home / ".cursor" / "mcp.json"
    configure.run(home, launch, ["cursor"], dry_run=False)

    uninstall.run(home, ["cursor"], dry_run=True, purge_backups=False)
    assert state.was_created_by_install(home, "cursor", config)

    uninstall.run(home, ["cursor"], dry_run=False, purge_backups=False)
    assert not config.exists()


def test_noninteractive_uninstall_requires_targets_and_yes(
    home: Path,
    fake_uvx: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    (home / ".cursor").mkdir()
    configure.run(home, _launch(fake_uvx), ["cursor"], dry_run=False)
    config = home / ".cursor" / "mcp.json"
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        monkeypatch.setattr(stream, "isatty", lambda: False)
    capsys.readouterr()

    with pytest.raises(SystemExit) as missing_targets:
        main(["uninstall", "--yes"])
    assert missing_targets.value.code == 1
    assert config.exists()

    with pytest.raises(SystemExit) as missing_confirmation:
        main(["uninstall", "cursor"])
    assert missing_confirmation.value.code == 1
    assert config.exists()


def test_uvx_entry_from_another_package_is_a_conflict(
    home: Path, fake_uvx: Path
) -> None:
    config = _write_cursor_entry(
        home,
        str(fake_uvx.resolve()),
        ["--from", "another-package[mcp]", "pulsar-mcp"],
    )
    spec = get_harness("cursor")
    assert spec is not None
    assert spec.artifact.inspect(config, _launch(fake_uvx)).state == State.CONFLICT
    assert spec.artifact.inspect_loose(config).state == State.CONFLICT


def test_install_refuses_a_dangling_config_symlink(home: Path, fake_uvx: Path) -> None:
    (home / ".cursor").mkdir()
    target = home / "missing-target.json"
    config = home / ".cursor" / "mcp.json"
    config.symlink_to(target)

    with pytest.raises(SystemExit) as exc:
        configure.run(home, _launch(fake_uvx), ["cursor"], dry_run=False)
    assert exc.value.code == 1
    assert config.is_symlink()
    assert not target.exists()


def test_status_accepts_pipx_registration_when_uvx_is_also_available(
    home: Path, fake_uvx: Path, capsys
) -> None:
    server = fake_uvx.parent / "pulsar-mcp"
    server.write_text("#!/bin/sh\n", encoding="utf-8")
    server.chmod(server.stat().st_mode | stat.S_IXUSR)
    _write_cursor_entry(home, str(server.resolve()), [])

    main(["status", "--json"])
    payload = json.loads(capsys.readouterr().out)
    cursor = next(row for row in payload["harnesses"] if row["id"] == "cursor")
    assert cursor["state"] == "active"


def test_codex_toml_comments_survive_install_and_uninstall(
    home: Path, fake_uvx: Path
) -> None:
    (home / ".codex").mkdir()
    config = home / ".codex" / "config.toml"
    config.write_text('# keep this comment\nmodel = "gpt-5"\n', encoding="utf-8")

    configure.run(home, _launch(fake_uvx), ["codex"], dry_run=False)
    assert "# keep this comment" in config.read_text(encoding="utf-8")

    uninstall.run(home, ["codex"], dry_run=False, purge_backups=False)
    assert "# keep this comment" in config.read_text(encoding="utf-8")


def test_windows_home_prefers_userprofile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    native_home = tmp_path / "native"
    git_home = tmp_path / "git-style"
    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setenv("USERPROFILE", str(native_home))
    monkeypatch.setenv("HOME", str(git_home))
    assert paths.home_dir() == native_home
