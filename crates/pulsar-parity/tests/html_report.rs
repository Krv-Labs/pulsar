//! Parity for the HTML report: rendering the dossier frozen by
//! `generate_html_report.py` (Python generator, see `tests/fixtures/golden/README.md`) against a Rust fit
//! of the same case must reproduce Python's `dossier_to_html` byte for byte,
//! including the force-directed topology figure and the threshold map.

use pulsar_analysis::dossier::TopologicalDossier;
use pulsar_parity::golden_dir;
use pulsar_pipeline::{fit, FitOptions};
use pulsar_report::{dossier_to_html, ReportGraph};
use serde_json::Value;

#[test]
fn html_report_matches_python_byte_for_byte() {
    let fx: Value = serde_json::from_str(
        &std::fs::read_to_string(golden_dir().join("export/html_report.json")).unwrap(),
    )
    .unwrap();
    for case in fx["cases"].as_array().unwrap() {
        let name = case["case"].as_str().unwrap();
        let dir = golden_dir().join(name);
        let cfg = pulsar_config::from_path(dir.join("config.yaml")).unwrap();
        let table = pulsar_data::read_csv_path(dir.join("input.csv")).unwrap();
        let model = fit(&cfg, &table, FitOptions::default(), None).unwrap();
        let dossier: TopologicalDossier = serde_json::from_value(case["dossier"].clone()).unwrap();
        let labels: Vec<i64> = serde_json::from_value(case["labels"].clone()).unwrap();
        let affinity = model.weighted_edges(Some(0.0));
        let graph = ReportGraph {
            graph: &model.graph,
            affinity: &affinity,
            stability: model.stability.as_ref(),
            resolved_threshold: model.construction_threshold,
        };
        let got = dossier_to_html(
            &dossier,
            &labels,
            Some(&graph),
            case["config_yaml"].as_str(),
        );
        let want = case["html"].as_str().unwrap();
        if got != want {
            let k = got
                .bytes()
                .zip(want.bytes())
                .position(|(a, b)| a != b)
                .unwrap_or(got.len().min(want.len()));
            let ctx = |s: &str| {
                let lo = s.floor_char_boundary(k.saturating_sub(160));
                let hi = s.ceil_char_boundary((k + 160).min(s.len()));
                s[lo..hi].to_owned()
            };
            panic!(
                "{name}: first difference at byte {k} (lengths {} vs {})\n got: …{}…\nwant: …{}…",
                got.len(),
                want.len(),
                ctx(&got),
                ctx(want)
            );
        }
    }
}
