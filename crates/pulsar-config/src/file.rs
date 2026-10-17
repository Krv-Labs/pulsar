//! The on-disk `params.yaml` format.
//!
//! These types are the single source of truth for the JSON Schema
//! ([`crate::json_schema`]), for unknown-key detection, and for
//! deserialization. Optional fields default as documented; the resolved,
//! fully-defaulted form is [`crate::PulsarConfig`].

use indexmap::IndexMap;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Number;

use crate::model::{Construction, EncodeMethod, FillValue, ImputeMethod, ProjectionMethod};

/// A Pulsar run configuration (`params.yaml`).
///
/// Sweep parameters (`dimensions`, `seed`, `epsilon`) accept a scalar, a
/// list, `{values: [...]}`, `{range: {min, max, steps}}` (inclusive
/// linspace), or `{distribution: {type: uniform, min, max}}`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(title = "Pulsar config")]
pub struct ConfigFile {
    /// Config format version. Default: 1 (the only version this build reads).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<u32>,
    #[serde(default)]
    pub run: RunFile,
    /// Legacy location of the data path. Prefer `run.data`; used only when
    /// `run.data` is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(default)]
    pub preprocessing: PreprocessingFile,
    #[serde(default)]
    pub sweep: SweepFile,
    #[serde(default)]
    pub cosmic_graph: CosmicGraphFile,
    #[serde(default)]
    pub output: OutputFile,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunFile {
    /// Run name, used in reports and artifact names. Default: "".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Path to the input table (CSV or Parquet). May be omitted when the
    /// caller supplies the data directly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
}

/// Column-level preprocessing applied before scaling and projection.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreprocessingFile {
    /// Columns removed before anything else.
    #[serde(default)]
    pub drop_columns: Vec<ColumnName>,
    /// Shared missingness markers: `{group: [columns]}` adds one
    /// `<group>_was_missing` column per group (1 where any listed column was
    /// blank) instead of one marker per column. A column may be in at most
    /// one group.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub missing_indicator_groups: IndexMap<String, Vec<ColumnName>>,
    /// Per-column imputation, keyed by column name.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub impute: IndexMap<String, ImputeFile>,
    /// Per-column categorical encoding, keyed by column name.
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub encode: IndexMap<String, EncodeFile>,
}

/// A column name. Integer names (e.g. a year column `2010`) are accepted and
/// treated as their decimal string.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum ColumnName {
    Name(String),
    Integer(i64),
}

impl ColumnName {
    pub fn into_string(self) -> String {
        match self {
            Self::Name(s) => s,
            Self::Integer(i) => i.to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImputeFile {
    pub method: ImputeMethod,
    /// RNG seed for the sampling methods. Default: 42.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// The constant used by `fill_value` (required there, rejected
    /// elsewhere). A finite number or a string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<FillValue>,
    /// Add a `<column>_was_missing` marker column. Default: true. Ignored
    /// for columns covered by `missing_indicator_groups`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub indicator: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EncodeFile {
    pub method: EncodeMethod,
    /// Fail if the column has more categories than this. Default: no hard
    /// limit (a warning above 50).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_categories: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SweepFile {
    /// Random projection grid. Takes precedence over the legacy `pca` block.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection: Option<ProjectionFile>,
    /// Legacy projection grid (`dimensions`, `seed`), read as a JL projection
    /// only when `projection` is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pca: Option<LegacyPcaFile>,
    #[serde(default)]
    pub ball_mapper: BallMapperFile,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProjectionFile {
    /// Default: `jl`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<ProjectionMethod>,
    /// Target dimensions (integers >= 1). Default: [2].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<ParamSpec>,
    /// Projection seeds (non-negative integers). Default: [42].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<ParamSpec>,
    /// Center columns before projecting. Default: true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub center: Option<bool>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LegacyPcaFile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<ParamSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<ParamSpec>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BallMapperFile {
    /// Ball radii (finite, > 0). Default: [0.5].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epsilon: Option<ParamSpec>,
}

/// A sweep parameter specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged, deny_unknown_fields)]
pub enum ParamSpec {
    /// A literal list of values.
    List(Vec<Number>),
    /// `{values: [...]}`.
    Values { values: Vec<Number> },
    /// `{range: {min, max, steps}}`: `steps` evenly spaced values from `min`
    /// to `max` inclusive (numpy `linspace`).
    Range { range: RangeSpec },
    /// `{distribution: {type: uniform, min, max}}`: expands to the bounds
    /// `[min, max]`; reserved for random search.
    Distribution { distribution: DistributionSpec },
    /// A single value.
    Scalar(Number),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RangeSpec {
    pub min: f64,
    pub max: f64,
    /// Number of values (>= 1). Default: 10.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DistributionSpec {
    /// Distribution family. Default and only supported value: `uniform`.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    pub min: f64,
    pub max: f64,
}

/// Cosmic-graph construction and optional spectral sparsification.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CosmicGraphFile {
    /// Edge-weight cutoff in [0, 1], or `auto` to pick a stable plateau with
    /// H0 persistence. Default: `auto` (the safe default for unknown data;
    /// `0.0` usually yields a structureless, fully connected graph).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub construction_threshold: Option<ThresholdFile>,
    /// Reserved. Default: `node`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub neighborhood: Option<String>,
    /// Opt-in spectral sparsifier; slow on large N. Default: false.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparsify: Option<bool>,
    /// Sparsifier accuracy (> 0). Default: 1.0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparsify_epsilon: Option<f64>,
    /// Default: 42.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparsify_seed: Option<u64>,
    /// JL sketch dimension (>= 1). Default: chosen from N.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparsify_sketch_dim: Option<u64>,
    /// Edge samples (>= 1). Default: chosen from N.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparsify_sample_count: Option<u64>,
    /// PCG tolerance (> 0). Default: 1e-6.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparsify_pcg_tol: Option<f64>,
    /// PCG iteration cap (>= 1). Default: 1000.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparsify_max_iter: Option<u64>,
    /// `minhash` (approximate, constant memory) or `exact`. Default: `minhash`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub construction: Option<Construction>,
    /// MinHash signature depth (>= 1); weight variance is J(1-J)/d. Default: 256.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minhash_d: Option<u64>,
    /// Default: 42.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minhash_seed: Option<u64>,
}

/// `auto` or a number in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum ThresholdFile {
    Auto(AutoLiteral),
    Value(#[schemars(range(min = 0.0, max = 1.0))] f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum AutoLiteral {
    #[serde(rename = "auto")]
    Auto,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OutputFile {
    /// Number of representative ball maps to select (>= 1). Default: 4.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_reps: Option<u64>,
}
