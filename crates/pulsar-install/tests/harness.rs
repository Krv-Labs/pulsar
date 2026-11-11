//! Behaviour of install / uninstall / status against a throwaway home.

use std::fs;
use std::path::{Path, PathBuf};

use pulsar_install::fsops::backup_path;
use pulsar_install::harness::{self, HARNESSES};
use pulsar_install::{install, status, uninstall, LaunchSpec, Mode};
use serde_json::{json, Value};

struct Env {
    _tmp: tempfile::TempDir,
    home: PathBuf,
    launch: LaunchSpec,
}

fn fake_exe(dir: &Path, name: &str) -> PathBuf {
    fs::create_dir_all(dir).unwrap();
    let path = dir.join(name);
    fs::write(&path, "#!/bin/sh\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    }
    path.canonicalize().unwrap()
}

fn env() -> Env {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    fs::create_dir_all(&home).unwrap();
    let exe = fake_exe(
        &tmp.path().join("bin"),
        if cfg!(windows) {
            "pulsar.exe"
        } else {
            "pulsar"
        },
    );
    let launch = LaunchSpec {
        command: exe,
        args: vec!["mcp".into()],
        mode: Mode::Binary,
    };
    Env {
        home,
        launch,
        _tmp: tmp,
    }
}

fn ids(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

fn path_of(env: &Env, id: &str) -> PathBuf {
    (harness::get(id).unwrap().config_path)(&env.home)
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn install_writes_each_harness_in_its_own_format() {
    let e = env();
    let all: Vec<String> = harness::ids().iter().map(|s| s.to_string()).collect();
    let report = install(&e.home, &e.launch, &all, false);
    assert!(report.ok, "{}", report.render());

    let command = e.launch.command_str();
    for h in HARNESSES.iter().filter(|h| h.id != "codex") {
        let entry = &read_json(&path_of(&e, h.id))[h.kind.container_key()]["pulsar"];
        assert_eq!(entry["command"], command.as_str(), "{}", h.id);
        assert_eq!(entry["args"], json!(["mcp"]), "{}", h.id);
        let wants_type = h.id == "vscode";
        assert_eq!(
            entry.get("type") == Some(&json!("stdio")),
            wants_type,
            "{} type",
            h.id
        );
    }
    let toml = fs::read_to_string(path_of(&e, "codex")).unwrap();
    assert!(toml.contains("[mcp_servers.pulsar]"), "{toml}");
    assert!(toml.contains("args = [\"mcp\"]"), "{toml}");
    assert_eq!(status(&e.home).active(), HARNESSES.len());
}

#[test]
fn install_is_idempotent_and_writes_nothing_the_second_time() {
    let e = env();
    install(&e.home, &e.launch, &ids(&["cursor"]), false);
    let path = path_of(&e, "cursor");
    let before = fs::read_to_string(&path).unwrap();
    let again = install(&e.home, &e.launch, &ids(&["cursor"]), false);
    assert!(
        again.ok
            && again
                .render()
                .contains("MCP server registered in ~/.cursor/mcp.json"),
        "{}",
        again.render()
    );
    assert_eq!(fs::read_to_string(&path).unwrap(), before);
    assert!(
        !backup_path(&path).exists(),
        "an unchanged entry needs no backup"
    );
}

#[test]
fn existing_user_config_and_key_order_survive_and_a_backup_is_kept() {
    let e = env();
    let path = path_of(&e, "cursor");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let original = "{\n  \"zeta\": true,\n  \"mcpServers\": {\n    \"other\": {\"command\": \"x\"}\n  },\n  \"alpha\": 1\n}\n";
    fs::write(&path, original).unwrap();

    assert!(install(&e.home, &e.launch, &ids(&["cursor"]), false).ok);
    let text = fs::read_to_string(&path).unwrap();
    assert!(text.find("zeta").unwrap() < text.find("mcpServers").unwrap());
    assert!(
        text.find("mcpServers").unwrap() < text.find("alpha").unwrap(),
        "{text}"
    );
    let v = read_json(&path);
    assert_eq!(v["mcpServers"]["other"]["command"], "x");
    assert_eq!(v["mcpServers"]["pulsar"]["args"], json!(["mcp"]));
    assert_eq!(fs::read_to_string(backup_path(&path)).unwrap(), original);
}

#[test]
fn reinstalling_with_a_new_launch_spec_keeps_the_original_backup() {
    let e = env();
    let path = path_of(&e, "cursor");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let original = "{\"mcpServers\": {\"other\": {\"command\": \"x\"}}}\n";
    fs::write(&path, original).unwrap();

    assert!(install(&e.home, &e.launch, &ids(&["cursor"]), false).ok);
    let mut changed = e.launch.clone();
    changed.args = vec!["serve".into()];
    assert!(install(&e.home, &changed, &ids(&["cursor"]), false).ok);
    assert_eq!(
        read_json(&path)["mcpServers"]["pulsar"]["args"],
        json!(["serve"])
    );
    assert_eq!(fs::read_to_string(backup_path(&path)).unwrap(), original);
}

#[test]
fn drifted_entries_are_repaired_not_duplicated() {
    let e = env();
    let path = path_of(&e, "gemini");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    // Ours (shape matches) but pointing at a binary that no longer exists.
    fs::write(
        &path,
        json!({"mcpServers": {"pulsar": {"command": "/gone/pulsar", "args": ["mcp"]}}}).to_string(),
    )
    .unwrap();

    let before = status(&e.home);
    let row = before.rows.iter().find(|(h, _)| h.id == "gemini").unwrap();
    assert_eq!(row.1.state.as_str(), "incomplete");

    let report = install(&e.home, &e.launch, &ids(&["gemini"]), false);
    assert!(
        report.ok && report.render().contains("repaired"),
        "{}",
        report.render()
    );
    let v = read_json(&path);
    assert_eq!(
        v["mcpServers"]["pulsar"]["command"],
        e.launch.command_str().as_str()
    );
    assert_eq!(v["mcpServers"].as_object().unwrap().len(), 1);
}

#[test]
fn python_era_entries_are_recognized_and_upgraded() {
    let e = env();
    let path = path_of(&e, "claude");
    let legacy = json!({"mcpServers": {"pulsar": {
        "command": "/usr/local/bin/uvx",
        "args": ["--from", "thema-pulsar[mcp]", "pulsar-mcp"]
    }}});
    fs::write(&path, legacy.to_string()).unwrap();
    let report = install(&e.home, &e.launch, &ids(&["claude"]), false);
    assert!(report.ok, "{}", report.render());
    assert_eq!(
        read_json(&path)["mcpServers"]["pulsar"]["args"],
        json!(["mcp"])
    );
}

#[test]
fn a_foreign_pulsar_entry_is_never_touched() {
    let e = env();
    let path = path_of(&e, "cursor");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let theirs =
        json!({"mcpServers": {"pulsar": {"command": "/opt/other-tool", "args": ["--serve"]}}})
            .to_string();
    fs::write(&path, &theirs).unwrap();

    let report = install(&e.home, &e.launch, &ids(&["cursor"]), false);
    assert!(!report.ok);
    assert!(
        report.render().contains("did not write"),
        "{}",
        report.render()
    );
    assert_eq!(fs::read_to_string(&path).unwrap(), theirs);

    let out = uninstall(&e.home, &ids(&["cursor"]), false, false);
    assert!(out.ok, "leaving a foreign entry alone is not a failure");
    assert_eq!(fs::read_to_string(&path).unwrap(), theirs);
}

#[test]
fn a_jsonc_file_with_comments_is_refused_with_the_snippet_to_add() {
    let e = env();
    let path = path_of(&e, "vscode");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let original = "{\n  // my servers\n  \"servers\": {},\n}\n";
    fs::write(&path, original).unwrap();

    let report = install(&e.home, &e.launch, &ids(&["vscode"]), false);
    assert!(!report.ok);
    let text = report.render();
    assert!(
        text.contains("contains comments") && text.contains("\"type\":\"stdio\""),
        "{text}"
    );
    assert_eq!(fs::read_to_string(&path).unwrap(), original);
}

#[test]
fn a_jsonc_file_without_comments_tolerates_trailing_commas() {
    let e = env();
    let path = path_of(&e, "vscode");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, "{\"servers\": {\"keep\": {\"command\": \"k\"},},}").unwrap();
    assert!(install(&e.home, &e.launch, &ids(&["vscode"]), false).ok);
    let v = read_json(&path);
    assert_eq!(v["servers"]["keep"]["command"], "k");
    assert_eq!(v["servers"]["pulsar"]["type"], "stdio");
}

#[test]
fn toml_keeps_comments_and_other_tables() {
    let e = env();
    let path = path_of(&e, "codex");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(
        &path,
        "# my codex config\nmodel = \"x\"\n\n[mcp_servers.other]\ncommand = \"o\"\n",
    )
    .unwrap();
    assert!(install(&e.home, &e.launch, &ids(&["codex"]), false).ok);
    let text = fs::read_to_string(&path).unwrap();
    assert!(
        text.contains("# my codex config") && text.contains("model = \"x\""),
        "{text}"
    );
    assert!(
        text.contains("[mcp_servers.other]") && text.contains("[mcp_servers.pulsar]"),
        "{text}"
    );

    assert!(uninstall(&e.home, &ids(&["codex"]), false, false).ok);
    let after = fs::read_to_string(&path).unwrap();
    assert!(
        after.contains("[mcp_servers.other]") && !after.contains("pulsar"),
        "{after}"
    );
}

#[test]
fn malformed_config_is_a_conflict_and_left_as_is() {
    let e = env();
    let path = path_of(&e, "cursor");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, "{ not json").unwrap();
    let report = install(&e.home, &e.launch, &ids(&["cursor"]), false);
    assert!(
        !report.ok && report.render().contains("parsing"),
        "{}",
        report.render()
    );
    assert_eq!(fs::read_to_string(&path).unwrap(), "{ not json");
}

#[test]
fn dry_run_touches_nothing() {
    let e = env();
    let report = install(&e.home, &e.launch, &ids(&["cursor", "codex"]), true);
    assert!(
        report.ok && report.render().contains("[dry run] would register"),
        "{}",
        report.render()
    );
    assert!(!path_of(&e, "cursor").exists() && !path_of(&e, "codex").exists());
    assert!(
        fs::read_dir(&e.home).unwrap().next().is_none(),
        "home must stay empty"
    );
}

#[test]
fn unknown_harness_fails_and_names_the_valid_ones() {
    let e = env();
    let report = install(&e.home, &e.launch, &ids(&["emacs"]), false);
    assert!(!report.ok);
    assert!(
        report
            .render()
            .contains("unknown harness: emacs (known: claude,"),
        "{}",
        report.render()
    );
}

#[test]
fn uninstall_removes_what_install_created_and_leaves_no_trace() {
    let e = env();
    assert!(install(&e.home, &e.launch, &ids(&["cursor", "codex"]), false).ok);
    assert!(path_of(&e, "cursor").exists());
    let report = uninstall(&e.home, &ids(&["cursor", "codex"]), false, false);
    assert!(report.ok, "{}", report.render());
    assert!(
        fs::read_dir(&e.home).unwrap().next().is_none(),
        "files, directories and state are all gone"
    );
}

#[test]
fn uninstall_keeps_a_file_the_user_created_even_when_it_ends_up_empty() {
    let e = env();
    let path = path_of(&e, "cursor");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, "{}").unwrap();
    assert!(install(&e.home, &e.launch, &ids(&["cursor"]), false).ok);
    assert!(uninstall(&e.home, &ids(&["cursor"]), false, true).ok);
    assert!(path.exists(), "the user's file stays");
    assert_eq!(read_json(&path), json!({}));
    assert!(
        !backup_path(&path).exists(),
        "--purge-backups removes our backup"
    );
}

#[test]
fn status_reports_every_state_and_serializes_to_json() {
    let e = env();
    install(&e.home, &e.launch, &ids(&["claude"]), false);
    let foreign = path_of(&e, "cursor");
    fs::create_dir_all(foreign.parent().unwrap()).unwrap();
    fs::write(
        &foreign,
        json!({"mcpServers": {"pulsar": {"command": "/x/other"}}}).to_string(),
    )
    .unwrap();

    let s = status(&e.home);
    assert_eq!(s.active(), 1);
    let v = s.to_json(&e.home);
    assert_eq!(
        (v["active"].clone(), v["total"].clone()),
        (json!(1), json!(HARNESSES.len()))
    );
    let state_of = |id: &str| {
        v["harnesses"]
            .as_array()
            .unwrap()
            .iter()
            .find(|h| h["id"] == id)
            .unwrap()["state"]
            .clone()
    };
    assert_eq!(state_of("claude"), "active");
    assert_eq!(state_of("cursor"), "conflict");
    assert_eq!(state_of("codex"), "absent");
    assert!(s
        .render(&e.home)
        .contains("1/8 harness integrations active."));
}

#[test]
fn detection_finds_only_harnesses_present_on_the_machine() {
    let e = env();
    assert!(
        harness::detected(&e.home)
            .iter()
            .all(|h| h.id == "claude-desktop" || h.id == "vscode")
            || harness::detected(&e.home).is_empty()
    );
    fs::create_dir_all(e.home.join(".cursor")).unwrap();
    fs::create_dir_all(e.home.join(".codex")).unwrap();
    let found: Vec<&str> = harness::detected(&e.home).iter().map(|h| h.id).collect();
    assert!(
        found.contains(&"cursor") && found.contains(&"codex") && !found.contains(&"claude"),
        "{found:?}"
    );
}

#[cfg(unix)]
#[test]
fn a_dangling_symlink_config_is_a_conflict() {
    let e = env();
    let path = path_of(&e, "cursor");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::os::unix::fs::symlink(e.home.join("missing"), &path).unwrap();
    let report = install(&e.home, &e.launch, &ids(&["cursor"]), false);
    assert!(
        !report.ok && report.render().contains("dangling symlink"),
        "{}",
        report.render()
    );
}
