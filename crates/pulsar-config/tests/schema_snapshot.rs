//! The committed schema must match the generated one. Regenerate with
//! `PULSAR_UPDATE_SCHEMA=1 cargo test -p pulsar-config --test schema_snapshot`.

use std::path::PathBuf;

#[test]
fn schema_matches_snapshot() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config.schema.json");
    let generated = serde_json::to_string_pretty(&pulsar_config::json_schema()).unwrap() + "\n";
    if std::env::var_os("PULSAR_UPDATE_SCHEMA").is_some() {
        std::fs::write(&path, &generated).unwrap();
        return;
    }
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    assert!(
        committed == generated,
        "config.schema.json is stale; rerun with PULSAR_UPDATE_SCHEMA=1"
    );
}
