//! Parity for the spectral-sparsifier graph artifact
//! (`_build_sparsifier_artifact_worker`): on the blobs and penguins golden
//! fits, under two parameter sets, the normalized sparsified edges and the
//! artifact's metrics must match what
//! `generate_graph_artifact.py` (Python generator, see `tests/fixtures/golden/README.md`) froze.

use pulsar_analysis::artifact::spectral_sparsifier_graph;
use pulsar_parity::{assert_close, golden_dir, Manifest};
use pulsar_pipeline::{fit, FitOptions};
use serde_json::Value;

fn close(label: &str, got: &Value, want: &Value, rtol: f64) {
    match (got, want) {
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            assert!(
                (a - b).abs() <= 1e-12_f64.max(rtol * b.abs()),
                "{label}: got {a}, want {b}"
            );
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "{label}: length");
            for (k, (x, y)) in a.iter().zip(b).enumerate() {
                close(&format!("{label}[{k}]"), x, y, rtol);
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            for (k, y) in b {
                close(
                    &format!("{label}.{k}"),
                    a.get(k).unwrap_or(&Value::Null),
                    y,
                    rtol,
                );
            }
        }
        _ => assert_eq!(got, want, "{label}"),
    }
}

#[test]
fn sparsifier_artifacts_match_python() {
    let rtol = Manifest::load().rtol("graph_artifact");
    let fx: Value = serde_json::from_str(
        &std::fs::read_to_string(golden_dir().join("export/graph_artifact.json")).unwrap(),
    )
    .unwrap();
    for case in fx["cases"].as_array().unwrap() {
        let name = case["case"].as_str().unwrap();
        let dir = golden_dir().join(name);
        let cfg = pulsar_config::from_path(dir.join("config.yaml")).unwrap();
        let table = pulsar_data::read_csv_path(dir.join("input.csv")).unwrap();
        let model = fit(&cfg, &table, FitOptions::default(), None).unwrap();
        let p = &case["params"];
        let (graph, metrics) = spectral_sparsifier_graph(
            &model.base_cosmic,
            p["epsilon"].as_f64().unwrap(),
            p["seed"].as_u64().unwrap(),
            p["sketch_dim"].as_u64().map(|v| v as usize),
            p["sample_count"].as_u64().map(|v| v as usize),
            p["pcg_tol"].as_f64().unwrap(),
            p["max_iter"].as_u64().unwrap() as usize,
        )
        .unwrap();
        let label = format!("{name} eps={}", p["epsilon"]);
        let mut edges = graph.edges.clone();
        edges.sort_by_key(|&(i, j, _)| (i, j));
        let want = case["edges"].as_array().unwrap();
        assert_eq!(
            edges.len(),
            case["artifact_edge_count"].as_u64().unwrap() as usize,
            "{label}: edge count"
        );
        assert_eq!(edges.len(), want.len(), "{label}: edges");
        for (g, w) in edges.iter().zip(want) {
            assert_eq!(
                (g.0 as u64, g.1 as u64),
                (w[0].as_u64().unwrap(), w[1].as_u64().unwrap()),
                "{label}: endpoints"
            );
        }
        let gw: Vec<f64> = edges.iter().map(|e| e.2).collect();
        let ww: Vec<f64> = want.iter().map(|e| e[2].as_f64().unwrap()).collect();
        assert_close(&format!("{label} weights"), &gw, &ww, rtol);
        close(
            &format!("{label} metrics"),
            &serde_json::to_value(&metrics).unwrap(),
            &case["metrics"],
            rtol,
        );
    }
}
