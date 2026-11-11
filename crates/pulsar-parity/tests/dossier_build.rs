//! Parity for the cluster dossier: the Rust port must reproduce what
//! `generate_dossier_build.py` (Python generator, see `tests/fixtures/golden/README.md`) froze from Python
//! (`build_dossier`, `dossier_to_markdown`).
//!
//! Each case rebuilds the feature evidence index (under parity in
//! `dossier_evidence`), builds the dossier at every recorded detail and compares
//! it as one document: cluster profiles, the three central rows chosen by
//! PageRank, global statistics. The Markdown is compared byte for byte in four
//! renderings (all clusters or the two largest, all rows or a preview of three).
//! `diagnose_model` needs a fitted model, so its output is an opaque input here.

use pulsar_analysis::dossier::{build_dossier, dossier_to_markdown, pagerank};
use pulsar_analysis::feature_evidence::build_feature_evidence_index;
use pulsar_data::read_csv_path;
use pulsar_parity::{assert_json_close, golden_dir};
use serde_json::Value;

fn fixture() -> Value {
    let path = golden_dir().join("dossier/dossier.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn every_dossier_and_rendering_matches_python() {
    let fx = fixture();
    let mut renderings = 0;
    for (name, case) in fx.as_object().unwrap() {
        let table = read_csv_path(golden_dir().join(case["csv"].as_str().unwrap())).unwrap();
        let labels: Vec<i64> = case["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let edges: Vec<(usize, usize, f64)> = case["edges"]
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
        let exclude: Vec<String> = case["exclude"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect();
        let index = build_feature_evidence_index(&table, &labels, &edges, &exclude, None).unwrap();

        assert_eq!(
            build_dossier(&index, &table, &labels, &edges, "bogus", Value::Null)
                .unwrap_err()
                .to_string(),
            case["bad_detail_error"].as_str().unwrap(),
            "{name}"
        );
        for (detail, recorded) in case["details"].as_object().unwrap() {
            let want = &recorded["dossier"];
            let metrics = want["global_stats"]["graph_metrics"].clone();
            let dossier = build_dossier(&index, &table, &labels, &edges, detail, metrics).unwrap();
            assert_json_close(
                &format!("{name}/{detail}"),
                &serde_json::to_value(&dossier).unwrap(),
                want,
                1e-7,
            );
            for (key, want_md) in recorded["markdown"].as_object().unwrap() {
                let (max, preview) = key.split_once('/').unwrap();
                let parse = |s: &str| (s != "None").then(|| s.parse::<usize>().unwrap());
                let md = dossier_to_markdown(&dossier, parse(max), parse(preview));
                assert_eq!(
                    md,
                    want_md.as_str().unwrap(),
                    "{name}/{detail} markdown {key}"
                );
                renderings += 1;
            }
        }
    }
    assert!(renderings >= 40, "only {renderings} renderings compared");
}

#[test]
fn the_fixtures_exercise_the_warning_the_omission_and_central_rows() {
    let fx = fixture();
    let markdown = |name: &str, detail: &str, key: &str| {
        fx[name]["details"][detail]["markdown"][key]
            .as_str()
            .unwrap()
            .to_owned()
    };
    assert!(markdown("wide", "full", "None/None").contains("> [!WARNING]"));
    assert!(!markdown("synth", "full", "None/None").contains("> [!WARNING]"));
    assert!(markdown("synth", "summary", "2/None").contains("smaller cluster(s) omitted"));
    assert!(markdown("synth", "summary", "None/None").contains("gated as high-cardinality noise"));
    let rows = |name: &str| {
        fx[name]["details"]["summary"]["dossier"]["clusters"]
            .as_array()
            .unwrap()
            .clone()
    };
    assert!(rows("synth")
        .iter()
        .all(|c| !c["central_rows"].as_array().unwrap().is_empty()));
    // A single-row cluster has exactly one central row.
    assert!(rows("synth")
        .iter()
        .any(|c| c["central_rows"].as_array().unwrap().len() == 1));
}

#[test]
fn pagerank_matches_the_closed_form_on_small_graphs() {
    // A path 0 - 1 - 2 with unit weights: the middle node ranks highest.
    let scores = pagerank(3, &[(0, 1, 1.0), (1, 2, 1.0)]).unwrap();
    assert!(scores[1] > scores[0] && (scores[0] - scores[2]).abs() < 1e-12);
    assert!((scores.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    // No edges: every node is dangling and the ranks are uniform.
    let uniform = pagerank(4, &[]).unwrap();
    assert!(uniform.iter().all(|s| (s - 0.25).abs() < 1e-12));
    assert_eq!(pagerank(0, &[]), Some(vec![]));
}
