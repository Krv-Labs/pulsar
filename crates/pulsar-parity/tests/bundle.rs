//! Parity for `export_dataset_bundle` (`pulsar/exports.py`): each case of
//! `generate_bundle.py` (Python generator, see `tests/fixtures/golden/README.md`) is fitted again in Rust
//! and exported with the same options; every Parquet file is read back and
//! compared with what pyarrow read from Python's bundle (Arrow schema in
//! pyarrow's spelling, then column values), and the manifests are compared
//! except `config_hash` (a documented divergence).
//!
//! The spectral layout is compared as what it is: per coordinate column, an
//! eigenvector of the graph Laplacian, with the same eigenvalues as Python's.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use arrow::array::{Array, AsArray};
use arrow::datatypes::{DataType, Float32Type, Float64Type, Int64Type, UInt32Type};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pulsar_parity::{golden_dir, Manifest};
use pulsar_pipeline::{fit, FitOptions, Fitted, Graph};
use pulsar_report::{export_dataset_bundle, BundleOptions, Layout};
use serde_json::{json, Value};

struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn pyarrow_type(dt: &DataType) -> String {
    match dt {
        DataType::UInt32 => "uint32".into(),
        DataType::Int64 => "int64".into(),
        DataType::Float32 => "float".into(),
        DataType::Float64 => "double".into(),
        DataType::Boolean => "bool".into(),
        DataType::Utf8 => "string".into(),
        DataType::LargeUtf8 => "large_string".into(),
        DataType::List(f) => format!("list<element: {}>", pyarrow_type(f.data_type())),
        other => format!("{other:?}"),
    }
}

/// `{schema: [[name, type, nullable]], columns: {name: [values]}}`.
fn read_parquet(path: &Path) -> Value {
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap())
        .unwrap()
        .build()
        .unwrap();
    let batches: Vec<_> = reader.map(|b| b.unwrap()).collect();
    let schema = batches[0].schema();
    let mut columns = serde_json::Map::new();
    for (k, field) in schema.fields().iter().enumerate() {
        let mut values = vec![];
        for batch in &batches {
            let col = batch.column(k);
            for i in 0..col.len() {
                values.push(if col.is_null(i) {
                    Value::Null
                } else {
                    match field.data_type() {
                        DataType::UInt32 => json!(col.as_primitive::<UInt32Type>().value(i)),
                        DataType::Int64 => json!(col.as_primitive::<Int64Type>().value(i)),
                        DataType::Float32 => {
                            json!(col.as_primitive::<Float32Type>().value(i) as f64)
                        }
                        DataType::Float64 => json!(col.as_primitive::<Float64Type>().value(i)),
                        DataType::Boolean => json!(col.as_boolean().value(i)),
                        DataType::Utf8 => json!(col.as_string::<i32>().value(i)),
                        DataType::LargeUtf8 => json!(col.as_string::<i64>().value(i)),
                        DataType::List(_) => {
                            let inner = col.as_list::<i32>().value(i);
                            json!(inner.as_primitive::<UInt32Type>().values().to_vec())
                        }
                        other => panic!("unexpected type {other:?}"),
                    }
                });
            }
        }
        columns.insert(field.name().clone(), Value::Array(values));
    }
    let schema: Vec<Value> = schema
        .fields()
        .iter()
        .map(|f| json!([f.name(), pyarrow_type(f.data_type()), f.is_nullable()]))
        .collect();
    json!({"schema": schema, "columns": columns})
}

fn close(label: &str, got: &Value, want: &Value, rtol: f64) {
    match (got, want) {
        (Value::Array(g), Value::Array(w)) => {
            assert_eq!(g.len(), w.len(), "{label}: length");
            for (k, (a, b)) in g.iter().zip(w).enumerate() {
                close(&format!("{label}[{k}]"), a, b, rtol);
            }
        }
        (Value::Number(a), Value::Number(b)) if a.is_f64() || b.is_f64() => {
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            assert!(
                (a - b).abs() <= 1e-9_f64.max(rtol * b.abs()),
                "{label}: got {a}, want {b}"
            );
        }
        _ => assert_eq!(got, want, "{label}"),
    }
}

/// Rayleigh quotient and relative residual of `v` for the Laplacian of `g`.
fn eigen_check(g: &Graph, v: &[f64]) -> (f64, f64) {
    let mut lv = vec![0.0; g.n];
    for &(i, j, w) in &g.edges {
        lv[i] += w * (v[i] - v[j]);
        lv[j] += w * (v[j] - v[i]);
    }
    let vv: f64 = v.iter().map(|x| x * x).sum();
    let lambda = v.iter().zip(&lv).map(|(a, b)| a * b).sum::<f64>() / vv;
    let resid: f64 = lv
        .iter()
        .zip(v)
        .map(|(l, x)| (l - lambda * x).powi(2))
        .sum::<f64>()
        .sqrt();
    (lambda, resid / vv.sqrt())
}

fn spectral_eigenvalues(g: &Graph, nodes: &Value) -> Vec<f64> {
    let mut out = vec![];
    for axis in ["ex", "ey", "ez"] {
        let v: Vec<f64> = nodes["columns"][axis]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        if v.iter().all(|x| x.abs() < 1e-12) {
            continue;
        }
        let (lambda, resid) = eigen_check(g, &v);
        assert!(
            resid < 1e-3 * lambda.abs().max(1.0),
            "{axis}: not an eigenvector (residual {resid})"
        );
        out.push(lambda);
    }
    out.sort_by(f64::total_cmp);
    out
}

fn options_names(v: &Value) -> BTreeMap<i64, String> {
    v.as_object()
        .map(|m| {
            m.iter()
                .map(|(k, s)| (k.parse().unwrap(), s.as_str().unwrap().to_owned()))
                .collect()
        })
        .unwrap_or_default()
}

fn fitted(case: &Value) -> (Fitted, pulsar_data::Table) {
    let cfg = pulsar_config::from_yaml_str(case["config_yaml"].as_str().unwrap()).unwrap();
    let table =
        pulsar_data::read_csv(case["input_csv"].as_str().unwrap().as_bytes(), "input.csv").unwrap();
    (
        fit(&cfg, &table, FitOptions::default(), None).unwrap(),
        table,
    )
}

#[test]
fn bundles_match_python() {
    let rtol = Manifest::load().rtol("bundle");
    let fx: Value = serde_json::from_str(
        &std::fs::read_to_string(golden_dir().join("export/bundle.json")).unwrap(),
    )
    .unwrap();
    let mut compared = 0;
    for case in fx["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (model, raw) = fitted(case);
        let labels: Vec<i64> = serde_json::from_value(case["labels"].clone()).unwrap();
        for (k, export) in case["exports"].as_array().unwrap().iter().enumerate() {
            let o = &export["options"];
            let layout = match o["layout"].as_str().unwrap() {
                "zeros" => Layout::Zeros,
                "spectral" => Layout::Spectral,
                _ => Layout::Projection,
            };
            let (names, descs) = (
                options_names(&o["cluster_names"]),
                options_names(&o["cluster_descriptions"]),
            );
            let opts = BundleOptions {
                labels: &labels,
                cluster_names: &names,
                cluster_descriptions: &descs,
                edges_threshold: o["edges_threshold"].as_f64(),
                layout,
                include_clean: o["include_clean"].as_bool().unwrap_or(false),
                write_manifest: o["write_manifest"].as_bool().unwrap_or(true),
                pulsar_version: "0.6.0",
            };
            let dir = TempDir(
                std::env::temp_dir()
                    .join(format!("pulsar-bundle-{name}-{k}-{}", std::process::id())),
            );
            let manifest = export_dataset_bundle(&model, &raw, &dir.0, "bundle", &opts).unwrap();
            let label = format!("{name} export {k}");

            let strip_hash = |m: &Value| {
                let mut m = m.clone();
                if let Some(h) = m.get("config_hash") {
                    let h = h.as_str().unwrap();
                    assert!(
                        h.len() == 64 && h.chars().all(|c| c.is_ascii_hexdigit()),
                        "{label}: hash {h}"
                    );
                    m["config_hash"] = Value::Null;
                }
                m
            };
            let want_manifest = &export["files"]["export_manifest.json"];
            assert_eq!(
                strip_hash(&manifest),
                strip_hash(&export["returned_manifest"]),
                "{label}: returned manifest"
            );
            let written = dir.0.join("bundle/export_manifest.json");
            if want_manifest.is_null() {
                assert!(!written.exists(), "{label}: manifest written");
            } else {
                let got: Value =
                    serde_json::from_str(&std::fs::read_to_string(&written).unwrap()).unwrap();
                close(
                    &format!("{label} manifest"),
                    &strip_hash(&got),
                    &strip_hash(want_manifest),
                    1e-12,
                );
            }

            for (rel, want) in export["files"].as_object().unwrap() {
                if rel.ends_with(".json") {
                    continue;
                }
                let path = dir.0.join("bundle").join(rel);
                if want.is_null() {
                    assert!(!path.exists(), "{label}: {rel} should not exist");
                    continue;
                }
                let got = read_parquet(&path);
                assert_eq!(got["schema"], want["schema"], "{label}: {rel} schema");
                for (col, values) in want["columns"].as_object().unwrap() {
                    if layout == Layout::Spectral
                        && rel == "graph/nodes.parquet"
                        && ["ex", "ey", "ez"].contains(&col.as_str())
                    {
                        continue;
                    }
                    close(
                        &format!("{label}: {rel}.{col}"),
                        &got["columns"][col],
                        values,
                        rtol,
                    );
                    compared += 1;
                }
                if layout == Layout::Spectral && rel == "graph/nodes.parquet" {
                    let ours = spectral_eigenvalues(&model.graph, &got);
                    let theirs = spectral_eigenvalues(&model.graph, want);
                    assert_eq!(ours.len(), theirs.len(), "{label}: spectral columns");
                    assert!(
                        !ours.is_empty(),
                        "{label}: no spectral coordinates were compared"
                    );
                    for (a, b) in ours.iter().zip(&theirs) {
                        assert!(
                            (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                            "{label}: eigenvalue {a} vs {b}"
                        );
                    }
                }
            }
        }
    }
    assert!(compared > 150, "only {compared} columns compared");
}
