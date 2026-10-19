//! Parity harness for the v1.0 Rust port (#60).
//!
//! Golden fixtures in `tests/fixtures/golden/` are frozen from Pulsar v0.5
//! (Python) by `scripts/golden/generate.py`. Every stage ported from Python
//! must reproduce them within the tolerance its artifact declares in
//! `manifest.json`. Do not regenerate fixtures to silence a failure.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use ndarray::Array2;
use serde::de::DeserializeOwned;
use serde::Deserialize;

/// Absolute floor for relative comparisons, so exact zeros compare sanely.
const ABS_FLOOR: f64 = 1e-12;

/// `tests/fixtures/golden/` at the repo root.
pub fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/golden")
}

/// Deserialize `<case>/<artifact>.json` (or `manifest.json` when `case` is empty).
pub fn load<T: DeserializeOwned>(case: &str, artifact: &str) -> T {
    let path = golden_dir().join(case).join(format!("{artifact}.json"));
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading golden fixture {}: {e}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("parsing golden fixture {}: {e}", path.display()))
}

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub tolerances: BTreeMap<String, Tolerance>,
    pub cases: BTreeMap<String, serde_json::Value>,
    /// Configs frozen under `config/` (resolved by Python `load_config`).
    #[serde(default)]
    pub config_cases: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum Tolerance {
    Exact,
    Rtol { rtol: f64 },
    Partition { min_ari: f64 },
    Info,
}

impl Manifest {
    pub fn load() -> Self {
        load("", "manifest")
    }

    /// Relative tolerance for a float artifact; panics if it is not `rtol`.
    pub fn rtol(&self, artifact: &str) -> f64 {
        match self.tolerances.get(artifact) {
            Some(Tolerance::Rtol { rtol }) => *rtol,
            other => panic!("artifact {artifact:?} has no rtol tolerance: {other:?}"),
        }
    }

    pub fn case_names(&self) -> Vec<String> {
        self.cases.keys().cloned().collect()
    }
}

/// Row-major dense matrix as written by the generator.
#[derive(Debug, Deserialize)]
pub struct Matrix {
    pub shape: [usize; 2],
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn to_array(&self) -> Array2<f64> {
        Array2::from_shape_vec((self.shape[0], self.shape[1]), self.data.clone())
            .expect("fixture shape matches data length")
    }
}

/// Assert `got` matches `want` elementwise within `rtol` (with a 1e-12 absolute floor).
pub fn assert_close(label: &str, got: &[f64], want: &[f64], rtol: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = ABS_FLOOR.max(rtol * w.abs());
        assert!(
            (g - w).abs() <= tol,
            "{label}[{k}]: got {g:e}, want {w:e} (|diff| {:e} > tol {tol:e})",
            (g - w).abs()
        );
    }
}

/// Assert two `[i, j, w]` edge lists match: identical endpoints, weights within `rtol`.
pub fn assert_edges_close(label: &str, got: &[(usize, usize, f64)], want: &[[f64; 3]], rtol: f64) {
    let mut got = got.to_vec();
    got.sort_by_key(|&(i, j, _)| (i, j));
    assert_eq!(got.len(), want.len(), "{label}: edge count mismatch");
    for (k, (&(i, j, w), exp)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            (i, j),
            (exp[0] as usize, exp[1] as usize),
            "{label}: edge {k} endpoints differ"
        );
        assert_close(&format!("{label} edge ({i},{j})"), &[w], &[exp[2]], rtol);
    }
}

/// Assert two JSON documents match: identical structure and keys, numbers
/// within `rtol` (with the absolute floor), everything else equal.
pub fn assert_json_close(
    label: &str,
    got: &serde_json::Value,
    want: &serde_json::Value,
    rtol: f64,
) {
    use serde_json::Value;
    match (got, want) {
        (Value::Number(g), Value::Number(w)) => {
            let (g, w) = (
                g.as_f64().unwrap_or(f64::NAN),
                w.as_f64().unwrap_or(f64::NAN),
            );
            assert_close(label, &[g], &[w], rtol);
        }
        (Value::Array(g), Value::Array(w)) => {
            assert_eq!(g.len(), w.len(), "{label}: array length mismatch");
            for (k, (gi, wi)) in g.iter().zip(w).enumerate() {
                assert_json_close(&format!("{label}[{k}]"), gi, wi, rtol);
            }
        }
        (Value::Object(g), Value::Object(w)) => {
            let gk: Vec<&String> = g.keys().collect();
            let wk: Vec<&String> = w.keys().collect();
            let mut gs = gk.clone();
            let mut ws = wk.clone();
            gs.sort();
            ws.sort();
            assert_eq!(gs, ws, "{label}: key sets differ");
            for (key, wv) in w {
                assert_json_close(&format!("{label}.{key}"), &g[key], wv, rtol);
            }
        }
        _ => assert_eq!(got, want, "{label}: values differ"),
    }
}

/// Adjusted Rand index between two labelings of the same points.
pub fn adjusted_rand_index(a: &[usize], b: &[usize]) -> f64 {
    assert_eq!(a.len(), b.len(), "labelings must have the same length");
    let n = a.len();
    if n == 0 {
        return 1.0;
    }
    let ka = a.iter().copied().max().map_or(0, |m| m + 1);
    let kb = b.iter().copied().max().map_or(0, |m| m + 1);
    let mut table = vec![vec![0u64; kb]; ka];
    for (&x, &y) in a.iter().zip(b) {
        table[x][y] += 1;
    }
    let c2 = |x: u64| (x * x.saturating_sub(1) / 2) as f64;
    let sum_ij: f64 = table.iter().flatten().map(|&v| c2(v)).sum();
    let sum_a: f64 = table.iter().map(|row| c2(row.iter().sum())).sum();
    let sum_b: f64 = (0..kb)
        .map(|j| c2(table.iter().map(|row| row[j]).sum()))
        .sum();
    let total = c2(n as u64);
    let expected = sum_a * sum_b / total;
    let max_index = (sum_a + sum_b) / 2.0;
    if (max_index - expected).abs() < f64::EPSILON {
        return 1.0;
    }
    (sum_ij - expected) / (max_index - expected)
}
