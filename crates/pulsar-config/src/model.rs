//! The resolved config: every default applied, every sweep parameter
//! expanded, every cross-field rule checked.

use indexmap::IndexMap;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The config format version this build reads and writes.
pub const CONFIG_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PulsarConfig {
    pub version: u32,
    pub run_name: String,
    /// Data path; empty when the caller supplies data directly.
    pub data: String,
    pub preprocessing: Preprocessing,
    pub projection: Projection,
    pub ball_mapper: BallMapper,
    pub cosmic_graph: CosmicGraph,
    pub n_reps: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Preprocessing {
    pub drop_columns: Vec<String>,
    pub impute: IndexMap<String, Impute>,
    pub encode: IndexMap<String, Encode>,
    pub missing_indicator_groups: IndexMap<String, Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Impute {
    pub method: ImputeMethod,
    pub seed: u64,
    /// Set exactly when `method` is `fill_value`.
    pub value: Option<FillValue>,
    pub indicator: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Encode {
    pub method: EncodeMethod,
    pub max_categories: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Projection {
    pub method: ProjectionMethod,
    pub dimensions: Vec<usize>,
    pub seeds: Vec<u64>,
    pub center: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BallMapper {
    pub epsilons: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CosmicGraph {
    pub construction_threshold: Threshold,
    pub neighborhood: String,
    pub sparsify: bool,
    pub sparsify_epsilon: f64,
    pub sparsify_seed: u64,
    pub sparsify_sketch_dim: Option<u64>,
    pub sparsify_sample_count: Option<u64>,
    pub sparsify_pcg_tol: f64,
    pub sparsify_max_iter: u64,
    pub construction: Construction,
    pub minhash_d: u64,
    pub minhash_seed: u64,
}

/// Serialized as `"auto"` or the number.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Threshold {
    Auto,
    Value(f64),
}

impl Serialize for Threshold {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Auto => s.serialize_str("auto"),
            Self::Value(v) => s.serialize_f64(*v),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ImputeMethod {
    SampleNormal,
    SampleCategorical,
    FillMean,
    FillMedian,
    FillMode,
    /// Fill with the constant in `value`.
    FillValue,
}

impl ImputeMethod {
    pub const ALL: [&'static str; 6] = [
        "sample_normal",
        "sample_categorical",
        "fill_mean",
        "fill_median",
        "fill_mode",
        "fill_value",
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EncodeMethod {
    OneHot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionMethod {
    /// Johnson–Lindenstrauss random projection.
    Jl,
    /// Randomized PCA (legacy).
    Pca,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Construction {
    Minhash,
    Exact,
}

/// A `fill_value` constant: an integer, a finite float, or a string.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum FillValue {
    Integer(i64),
    Float(f64),
    Text(String),
}
