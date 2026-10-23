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

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Preprocessing {
    pub drop_columns: Vec<String>,
    pub impute: IndexMap<String, Impute>,
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub impute_candidates: IndexMap<String, Vec<Impute>>,
    pub encode: IndexMap<String, Encode>,
    pub missing_indicator_groups: IndexMap<String, Vec<String>>,
    pub n_imputations: u32,
}

impl Default for Preprocessing {
    fn default() -> Self {
        Self {
            drop_columns: Vec::new(),
            impute: IndexMap::new(),
            impute_candidates: IndexMap::new(),
            encode: IndexMap::new(),
            missing_indicator_groups: IndexMap::new(),
            n_imputations: 1,
        }
    }
}

impl Preprocessing {
    /// Candidate specs for `col`, falling back to `impute[col]` when
    /// `impute_candidates` has no entry.
    pub fn candidates_for(&self, col: &str) -> Vec<Impute> {
        if let Some(cands) = self.impute_candidates.get(col) {
            if !cands.is_empty() {
                return cands.clone();
            }
        }
        self.impute.get(col).cloned().into_iter().collect()
    }

    /// All columns with an imputation policy (`impute` union non-empty
    /// `impute_candidates`), preserving insertion order.
    pub fn impute_columns(&self) -> Vec<&String> {
        let mut cols: Vec<&String> = self.impute.keys().collect();
        for (col, cands) in &self.impute_candidates {
            if !cands.is_empty() && !self.impute.contains_key(col) {
                cols.push(col);
            }
        }
        cols
    }
}

impl PulsarConfig {
    /// Expand multi-imputation candidates and `n_imputations` into concrete
    /// single-imputation variant configs, deduplicating identical deterministic
    /// combinations and auto-incrementing seeds for repeated stochastic draws.
    pub fn preprocessing_variants(&self) -> Vec<PulsarConfig> {
        let pre = &self.preprocessing;
        let candidates_map: IndexMap<String, Vec<Impute>> = pre
            .impute_columns()
            .into_iter()
            .map(|col| (col.clone(), pre.candidates_for(col)))
            .collect();
        let max_candidates = candidates_map.values().map(Vec::len).max().unwrap_or(1);
        let target_variants = (pre.n_imputations.max(1) as usize).max(max_candidates);
        if candidates_map.is_empty() {
            return vec![self.clone()];
        }

        let mut used_stochastic_seeds: std::collections::HashMap<
            (String, ImputeMethod, bool),
            std::collections::HashSet<u64>,
        > = std::collections::HashMap::new();
        let mut seen_signatures: Vec<Vec<(String, Impute)>> = Vec::new();
        let mut variants = Vec::new();

        for k in 0..target_variants {
            let mut variant_impute = IndexMap::new();
            let mut variant_candidates = IndexMap::new();
            let mut sig_items = Vec::with_capacity(candidates_map.len());

            for (col, specs) in &candidates_map {
                let mut chosen = if k < specs.len() {
                    specs[k].clone()
                } else {
                    let stochastic: Vec<&Impute> =
                        specs.iter().filter(|s| s.method.is_stochastic()).collect();
                    if !stochastic.is_empty() {
                        let offset = k - specs.len();
                        stochastic[offset % stochastic.len()].clone()
                    } else {
                        specs[k % specs.len()].clone()
                    }
                };

                if chosen.method.is_stochastic() {
                    let key = (col.clone(), chosen.method, chosen.indicator);
                    let seen_seeds = used_stochastic_seeds.entry(key).or_default();
                    let mut seed = chosen.seed;
                    while seen_seeds.contains(&seed) {
                        seed += 1;
                    }
                    seen_seeds.insert(seed);
                    chosen.seed = seed;
                }

                let mut sig_spec = chosen.clone();
                if !sig_spec.method.is_stochastic() {
                    sig_spec.seed = 0;
                }
                sig_items.push((col.clone(), sig_spec));
                variant_candidates.insert(col.clone(), vec![chosen.clone()]);
                variant_impute.insert(col.clone(), chosen);
            }

            if seen_signatures.contains(&sig_items) {
                continue;
            }
            seen_signatures.push(sig_items);

            let mut variant = self.clone();
            variant.preprocessing.impute = variant_impute;
            variant.preprocessing.impute_candidates = variant_candidates;
            variant.preprocessing.n_imputations = 1;
            variants.push(variant);
        }

        if variants.is_empty() {
            vec![self.clone()]
        } else {
            variants
        }
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
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

    pub fn is_stochastic(self) -> bool {
        matches!(self, Self::SampleNormal | Self::SampleCategorical)
    }
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
