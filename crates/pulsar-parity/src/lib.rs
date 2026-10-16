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
