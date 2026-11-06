//! create_config's feature-expansion warning and its `detail='summary'` mode.

mod common;

use std::path::PathBuf;

use common::{connect, ok_pairs as ok};
use rmcp::service::{RoleClient, RunningService};
use serde_json::{json, Value};

type Client = RunningService<RoleClient, ()>;

fn csv(dir: &std::path::Path, name: &str, header: &str, rows: &[String]) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, format!("{header}\n{}\n", rows.join("\n"))).unwrap();
    path
}

/// 150 rows, three numeric columns.
fn numeric_rows() -> Vec<String> {
    (0..150u32)
        .map(|i| {
            let f = f64::from(i);
            format!(
                "{:.3},{:.3},{:.3}",
                (f * 0.37).sin(),
                (f * 0.11).cos(),
                (f * 0.07).sin() * 2.0
            )
        })
        .collect()
}

/// Same, plus a text column `site` with 12 levels (under sqrt(150) = 12.2) and a
/// 3-level `arm`: 5 raw columns become more than 1.5x as many processed features.
fn wide_rows() -> Vec<String> {
    numeric_rows()
        .into_iter()
        .enumerate()
        .map(|(i, r)| format!("{r},s{},a{}", i % 12, i % 3))
        .collect()
}

async fn ingest(client: &Client, path: &std::path::Path) -> String {
    let v = ok(
        client,
        "ingest_dataset",
        &[("path", json!(path.display().to_string()))],
    )
    .await;
    v["dataset_id"].as_str().unwrap().to_owned()
}

fn temp(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("pulsar-cc-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn codes(created: &Value) -> Vec<String> {
    created["warnings"]
        .as_array()
        .map(|w| {
            w.iter()
                .map(|x| x["code"].as_str().unwrap().to_owned())
                .collect()
        })
        .unwrap_or_default()
}

#[tokio::test]
async fn expansion_warning_fires_above_one_and_a_half_times_and_names_the_widest_columns() {
    let dir = temp("wide");
    let client = connect(&dir.join("store")).await;
    let path = csv(&dir, "wide.csv", "a,b,c,site,arm", &wide_rows());
    let id = ingest(&client, &path).await;
    let created = ok(&client, "create_config", &[("dataset_id", json!(id))]).await;
    assert!(
        created["raw_to_processed_expansion_ratio"]
            .as_f64()
            .unwrap()
            > 1.5,
        "{created}"
    );
    let warning = created["warnings"]
        .as_array()
        .and_then(|w| {
            w.iter()
                .find(|w| w["code"] == "PROCESSED_FEATURES_EXPANDED")
        })
        .unwrap_or_else(|| panic!("no expansion warning: {:?}", created["warnings"]));
    let message = warning["message"].as_str().unwrap();
    let (site, arm) = (message.find("`site`"), message.find("`arm`"));
    assert!(site.is_some() && arm.is_some(), "{message}");
    assert!(site < arm, "widest column first: {message}");
    assert!(message.contains("levels"), "{message}");
    client.cancel().await.unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn expansion_warning_is_silent_below_the_ratio() {
    let dir = temp("narrow");
    let client = connect(&dir.join("store")).await;
    let path = csv(&dir, "narrow.csv", "a,b,c", &numeric_rows());
    let id = ingest(&client, &path).await;
    let created = ok(&client, "create_config", &[("dataset_id", json!(id))]).await;
    assert!(
        created["raw_to_processed_expansion_ratio"]
            .as_f64()
            .unwrap()
            <= 1.5,
        "{created}"
    );
    assert!(
        !codes(&created).contains(&"PROCESSED_FEATURES_EXPANDED".to_owned()),
        "{created}"
    );
    client.cancel().await.unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn summary_detail_omits_the_yaml_and_the_sweep_uses_the_active_config() {
    let dir = temp("summary");
    let client = connect(&dir.join("store")).await;
    let path = csv(&dir, "narrow.csv", "a,b,c", &numeric_rows());
    let id = ingest(&client, &path).await;

    let full = ok(&client, "create_config", &[("dataset_id", json!(id))]).await;
    let yaml = full["config_yaml"].as_str().expect("full is the default");
    assert!(full.get("config_yaml_note").is_none());

    let summary = ok(
        &client,
        "create_config",
        &[("dataset_id", json!(id)), ("detail", json!("summary"))],
    )
    .await;
    assert!(summary.get("config_yaml").is_none(), "{summary}");
    assert_eq!(
        summary["config_yaml_chars"].as_u64().unwrap() as usize,
        yaml.chars().count()
    );
    assert!(summary.get("config_yaml_note").is_none());
    let context = ok(&client, "get_runtime_context", &[]).await;
    let plan_id = summary["preprocessing_plan"]["plan_id"].as_str().unwrap();
    let active = context["active_config_yaml"].as_str().expect("stored");
    assert!(
        active.contains(plan_id),
        "the summary call's config is the active one"
    );
    assert!(!yaml.contains(plan_id));

    let swept = ok(
        &client,
        "run_topological_sweep",
        &[
            ("dataset_id", json!(id)),
            ("response_format", json!("json")),
        ],
    )
    .await;
    assert_eq!(swept["status"], "ok", "{swept}");
    assert!(swept["run_id"].is_string(), "{swept}");
    client.cancel().await.unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}
