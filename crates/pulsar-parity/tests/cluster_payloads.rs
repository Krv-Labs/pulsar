//! Parity for the cluster-facing payload builders: the Rust port must reproduce
//! what `generate_cluster_payloads.py` (Python generator, see `tests/fixtures/golden/README.md`) froze from
//! Python (`pulsar/mcp/payloads.py`).
//!
//! Covered: the cluster header (sizes, fragmentation, every readiness status),
//! the compact header, the selection disclosure, the dossier and summary
//! evidence payloads at several detail and cap settings, the Markdown views of
//! summaries, profiles and feature signals, and the threshold surface. Whole
//! payloads are compared (numbers within 1e-7); Markdown byte for byte.

use pulsar_analysis::cluster_payloads::*;
use pulsar_analysis::dossier::build_dossier;
use pulsar_analysis::dossier_payloads::{cluster_profile_payload, feature_signal_payload};
use pulsar_analysis::feature_evidence::{build_feature_evidence_index, FeatureEvidenceIndex};
use pulsar_data::{read_csv_path, Table};
use pulsar_parity::{assert_json_close, golden_dir};
use serde_json::{json, Value};

fn load(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(golden_dir().join(name)).unwrap()).unwrap()
}

struct Case {
    table: Table,
    labels: Vec<i64>,
    edges: Vec<(usize, usize, f64)>,
    index: FeatureEvidenceIndex,
}

fn case_for(c: &Value) -> Case {
    let table = read_csv_path(golden_dir().join(c["csv"].as_str().unwrap())).unwrap();
    let labels: Vec<i64> = c["labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let edges: Vec<(usize, usize, f64)> = c["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            (
                e[0].as_u64().unwrap() as usize,
                e[1].as_u64().unwrap() as usize,
                e[2].as_f64().unwrap(),
            )
        })
        .collect();
    let exclude: Vec<String> = c["exclude"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    let index = build_feature_evidence_index(&table, &labels, &edges, &exclude, None).unwrap();
    Case {
        table,
        labels,
        edges,
        index,
    }
}

fn max_of(v: &Value) -> Option<usize> {
    v.as_u64().map(|m| m as usize)
}

#[test]
fn cluster_result_headers_match_python() {
    let fx = load("dossier/cluster_payloads.json");
    let results = fx["results"].as_object().unwrap();
    assert_eq!(results.len(), 24);
    let mut statuses = std::collections::BTreeSet::new();
    for (name, r) in results {
        let labels: Vec<i64> = r["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let n_clusters = labels
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        let got = cluster_result_payload(
            &labels,
            r["method"].as_str().unwrap(),
            n_clusters,
            r["silhouette"].as_f64(),
            r["threshold"].as_f64().unwrap(),
            &r["plateaus"],
            r["failure"].as_str(),
        );
        assert_json_close(name, &got, &r["payload"], 1e-9);
        statuses.insert(
            got["interpretation_readiness"]["status"]
                .as_str()
                .unwrap()
                .to_owned(),
        );
    }
    assert_eq!(
        statuses.len(),
        3,
        "ready, caution and review_required are all exercised"
    );
}

#[test]
fn evidence_payloads_and_markdown_match_python() {
    let fx = load("dossier/cluster_payloads.json");
    let evidence = load("dossier/feature_evidence.json");
    let mut checked = 0;
    for (name, out) in fx["cases"].as_object().unwrap() {
        let c = case_for(&evidence[name]);
        let (meta, graph_metrics) = (&out["meta"], &out["graph_metrics"]);

        for full in [false, true] {
            let got = compact_cluster_result_payload(meta, full);
            assert_json_close(
                &format!("{name} compact {full}"),
                &got,
                &out["compact"][if full { "True" } else { "False" }],
                1e-9,
            );
            checked += 1;
        }
        for s in out["selection"].as_array().unwrap() {
            let shown: Vec<i64> = s["shown"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            let got = build_cluster_selection(meta, &shown, s["ranked_by"].as_str().unwrap());
            assert_json_close(&format!("{name} selection"), &got, &s["payload"], 1e-9);
            assert_eq!(
                cluster_selection_to_markdown(&got),
                s["markdown"].as_str().unwrap(),
                "{name} selection markdown"
            );
            checked += 1;
        }
        for e in out["evidence"].as_array().unwrap() {
            let detail = e["detail"].as_str().unwrap();
            let mut dossier = build_dossier(
                &c.index,
                &c.table,
                &c.labels,
                &c.edges,
                detail,
                graph_metrics.clone(),
            )
            .unwrap();
            dossier.global_stats["graph_metrics"] = graph_metrics.clone();
            let got = build_evidence_payload(
                &dossier,
                meta,
                detail,
                max_of(&e["max_clusters"]),
                e["preview"].as_u64().unwrap() as usize,
            );
            assert_json_close(
                &format!("{name} evidence {detail}"),
                &got,
                &e["payload"],
                1e-7,
            );
            checked += 1;
        }
        for s in out["summary_evidence"].as_array().unwrap() {
            let got = build_summary_evidence_payload(
                &c.index,
                meta,
                graph_metrics,
                max_of(&s["max_clusters"]),
                s["preview"].as_u64().unwrap() as usize,
                s["global_preview"].as_u64().unwrap() as usize,
            );
            assert_json_close(
                &format!("{name} summary evidence"),
                &got,
                &s["payload"],
                1e-7,
            );
            assert_eq!(
                summary_evidence_payload_to_markdown(&got),
                s["markdown"].as_str().unwrap(),
                "{name} summary markdown"
            );
            checked += 1;
        }
        for p in out["profile_markdown"].as_array().unwrap() {
            let cluster = cluster_profile_payload(
                &c.index,
                p["cluster_id"].as_i64().unwrap(),
                p["detail"].as_str().unwrap(),
                p["max_features"].as_u64().unwrap() as usize,
            )
            .unwrap();
            let wrapped = json!({"cluster": cluster, "detail": p["detail"]});
            assert_eq!(
                cluster_profile_payload_to_markdown(&wrapped),
                p["markdown"].as_str().unwrap(),
                "{name} profile {}",
                p["cluster_id"]
            );
            checked += 1;
        }
        for s in out["signal_markdown"].as_array().unwrap() {
            let names: Vec<String> = s["names"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_owned())
                .collect();
            let sig = feature_signal_payload(
                &c.index,
                &names,
                None,
                s["detail"].as_str().unwrap(),
                max_of(&s["max_clusters"]),
            );
            assert_eq!(
                feature_signal_payload_to_markdown(&sig),
                s["markdown"].as_str().unwrap(),
                "{name} signal {names:?}"
            );
            checked += 1;
        }
    }
    assert!(checked >= 90, "only {checked} comparisons");
}

#[test]
fn threshold_surfaces_and_notes_match_python() {
    let fx = load("dossier/cluster_payloads.json");
    let mut statuses = std::collections::BTreeSet::new();
    for t in fx["thresholds"].as_array().unwrap() {
        let (construction, interpretation) = (
            t["construction"].as_f64().unwrap(),
            t["interpretation"].as_f64().unwrap(),
        );
        let got = threshold_surface_payload(
            construction,
            interpretation,
            t["inherited"].as_bool().unwrap(),
            t["source"].as_str().unwrap(),
        );
        assert_json_close("threshold surface", &got, &t["payload"], 1e-12);
        assert_eq!(
            prepend_threshold_markdown("BODY", &got),
            t["markdown"].as_str().unwrap()
        );
        assert_eq!(
            two_lever_markdown_note(construction, interpretation),
            t["two_lever"].as_str().unwrap()
        );
        statuses.insert(got["status"].as_str().unwrap().to_owned());
    }
    assert!(statuses.len() >= 5, "{statuses:?}");
}
