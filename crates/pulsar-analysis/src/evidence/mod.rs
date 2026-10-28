//! Empirical preprocessing evidence: per-column statistics, shared missing
//! masks, mask dependencies and inter-column relations (#plan-port, PR 1).
//!
//! Port of `pulsar/analysis/preprocessing_evidence.py`. Pure measurement:
//! nothing here decides what to do with a column.
//!
//! Where the port is not bit-for-bit with pandas/numpy:
//!
//! * Typing follows [`pulsar_data::Column`], so there are no `datetime64` or
//!   `timedelta64` columns; dates are only ever recognised in text
//!   ([`kind`] documents the accepted formats, a conservative subset of what
//!   `pd.to_datetime(format="mixed")` takes).
//! * Relations are measured on a row sample when the table is larger than
//!   `sample_rows`. numpy's PCG64 `choice` is not reproduced, so the sampled
//!   rows differ from Python's; tables at or under `sample_rows` are exact.
//! * Floating-point sums use plain left-to-right accumulation instead of
//!   numpy's pairwise summation, so values agree to ~1e-12 rather than
//!   bit-for-bit.

mod columns;
pub(crate) mod kind;
pub(crate) mod knn;
mod relations;

use std::collections::HashMap;
use std::time::Instant;

use pulsar_data::Table;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use knn::neighborhood_overlap;

/// What a column looks like to the plan, independent of how it was stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColumnKind {
    Numeric,
    Boolean,
    DatetimeLike,
    NumericText,
    Text,
}

impl ColumnKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Numeric => "numeric",
            Self::Boolean => "boolean",
            Self::DatetimeLike => "datetime_like",
            Self::NumericText => "numeric_text",
            Self::Text => "text",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnEvidence {
    pub name: String,
    pub kind: ColumnKind,
    pub n_rows: usize,
    pub n_missing: usize,
    pub missing_fraction: f64,
    pub n_unique: usize,
    pub unique_ratio: f64,
    pub is_constant: bool,
    pub n_zero: Option<usize>,
    pub observed_min: Option<f64>,
    pub observed_max: Option<f64>,
    pub integer_valued: Option<bool>,
    pub nonnegative: Option<bool>,
    pub minority_fraction: Option<f64>,
    pub max_abs_z: Option<f64>,
    pub missing_group: Option<String>,
    pub mask_dependency: Option<MaskDependency>,
    /// Up to five most frequent non-numeric tokens of a `numeric_text` column.
    pub non_numeric_values: Option<Vec<(String, usize)>>,
    pub n_non_numeric: Option<usize>,
}

/// Columns whose partial missing masks are identical.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MissingGroup {
    pub group_id: String,
    pub columns: Vec<String>,
    pub n_missing: usize,
    pub zero_free_columns: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskRelation {
    MissingWhenPartnerZero,
    MissingWhenPartnerNonzero,
}

impl MaskRelation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MissingWhenPartnerZero => "missing_when_partner_zero",
            Self::MissingWhenPartnerNonzero => "missing_when_partner_nonzero",
        }
    }
}

/// A column that is missing exactly where a complete 0/1-ish partner is
/// zero (or nonzero).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaskDependency {
    pub column: String,
    pub partner: String,
    pub relation: MaskRelation,
    pub n_other_partners: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationKind {
    Duplicate,
    SumOf,
    StrongCorrelation,
}

impl RelationKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Duplicate => "duplicate",
            Self::SumOf => "sum_of",
            Self::StrongCorrelation => "strong_correlation",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnRelation {
    pub kind: RelationKind,
    pub target: String,
    pub sources: Vec<String>,
    /// 0 for duplicates, the max absolute residual for sums, |r| for
    /// correlations.
    pub value: f64,
    pub rows_used: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkippedStep {
    pub step: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceCost {
    pub rows_sampled: usize,
    pub numeric_columns_considered: usize,
    pub steps_skipped: Vec<SkippedStep>,
    pub elapsed_ms: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreprocessingEvidence {
    pub n_rows: usize,
    pub n_columns: usize,
    pub columns: Vec<ColumnEvidence>,
    pub missing_groups: Vec<MissingGroup>,
    pub mask_dependencies: Vec<MaskDependency>,
    pub relations: Vec<ColumnRelation>,
    pub cost: EvidenceCost,
}

#[derive(Debug, Clone)]
pub struct EvidenceOptions {
    /// Rows used for relation measurement (at least 1).
    pub sample_rows: usize,
    /// Seed of the relation row sample.
    pub seed: u64,
    pub max_sum_sources: usize,
    pub max_correlation_pairs: usize,
    /// Above this many numeric/boolean columns relations are skipped.
    pub max_relation_columns: usize,
    pub exclude_columns: Vec<String>,
}

impl Default for EvidenceOptions {
    fn default() -> Self {
        Self {
            sample_rows: 5000,
            seed: 0,
            max_sum_sources: 6,
            max_correlation_pairs: 20,
            max_relation_columns: 2000,
            exclude_columns: Vec::new(),
        }
    }
}

#[derive(Debug, Error)]
pub enum EvidenceError {
    #[error("dataset has no rows and no columns")]
    Empty,
    #[error("dataset has no rows")]
    NoRows,
    #[error("dataset has no columns")]
    NoColumns,
    #[error("dataset has duplicate column names: {0:?}")]
    DuplicateColumns(Vec<String>),
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

impl EvidenceError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Empty | Self::NoRows | Self::NoColumns => "EMPTY_DATASET",
            Self::DuplicateColumns(_) => "DUPLICATE_COLUMNS",
            Self::InvalidArgument(_) => "INVALID_ARGUMENT",
        }
    }
}

/// Bit-pack a boolean mask; equal masks of equal length pack to equal keys.
fn pack(mask: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; mask.len().div_ceil(8)];
    for (i, &b) in mask.iter().enumerate() {
        if b {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

/// Measure per-column statistics, missing groups, mask dependencies and
/// relations.
pub fn measure_preprocessing_evidence(
    table: &Table,
    opts: &EvidenceOptions,
) -> Result<PreprocessingEvidence, EvidenceError> {
    let started = Instant::now();

    let keep: Vec<usize> = (0..table.n_cols())
        .filter(|&i| !opts.exclude_columns.contains(&table.names[i]))
        .collect();
    let n_rows = table.n_rows();
    let n_cols = keep.len();
    match (n_rows, n_cols) {
        (0, 0) => return Err(EvidenceError::Empty),
        (0, _) => return Err(EvidenceError::NoRows),
        (_, 0) => return Err(EvidenceError::NoColumns),
        _ => {}
    }

    let mut seen = std::collections::HashSet::new();
    let mut dups: Vec<String> = Vec::new();
    for &i in &keep {
        let name = &table.names[i];
        if !seen.insert(name.as_str()) && !dups.contains(name) {
            dups.push(name.clone());
        }
    }
    if !dups.is_empty() {
        return Err(EvidenceError::DuplicateColumns(dups));
    }

    let mut raw: Vec<ColumnEvidence> = Vec::with_capacity(n_cols);
    // (column position, packed partial missing mask), in column order.
    let mut partial: Vec<(usize, Vec<u8>)> = Vec::new();
    let mut numeric_names: Vec<String> = Vec::new();
    let mut numeric_arrays: Vec<Vec<f64>> = Vec::new();
    let mut zero_partners: HashMap<Vec<u8>, Vec<usize>> = HashMap::new();
    let mut nonzero_partners: HashMap<Vec<u8>, Vec<usize>> = HashMap::new();

    for (pos, &src) in keep.iter().enumerate() {
        let name = &table.names[src];
        let profile = columns::profile_column(name, &table.columns[src], n_rows);
        if let Some(mask) = &profile.partial_mask {
            partial.push((pos, pack(mask)));
        }
        if let Some(values) = profile.values {
            let ev = &profile.evidence;
            if ev.n_missing == 0 && ev.n_zero.is_some_and(|z| z > 0 && z < n_rows) {
                let zero_mask: Vec<bool> = values.iter().map(|&v| v == 0.0).collect();
                let nonzero_mask: Vec<bool> = zero_mask.iter().map(|&z| !z).collect();
                zero_partners.entry(pack(&zero_mask)).or_default().push(pos);
                nonzero_partners
                    .entry(pack(&nonzero_mask))
                    .or_default()
                    .push(pos);
            }
            numeric_names.push(name.clone());
            numeric_arrays.push(values);
        }
        raw.push(profile.evidence);
    }

    let (missing_groups, col_group) = build_missing_groups(&raw, &partial);
    let (mask_dependencies, col_dep) =
        build_mask_dependencies(&raw, &partial, &zero_partners, &nonzero_partners);

    let columns: Vec<ColumnEvidence> = raw
        .into_iter()
        .enumerate()
        .map(|(pos, mut r)| {
            r.missing_group = col_group.get(&pos).cloned();
            r.mask_dependency = col_dep.get(&pos).cloned();
            r
        })
        .collect();

    let (relations, rows_sampled, numeric_considered, steps_skipped) =
        relations::measure_relations(n_rows, &numeric_names, &numeric_arrays, opts);

    Ok(PreprocessingEvidence {
        n_rows,
        n_columns: n_cols,
        columns,
        missing_groups,
        mask_dependencies,
        relations,
        cost: EvidenceCost {
            rows_sampled,
            numeric_columns_considered: numeric_considered,
            steps_skipped,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        },
    })
}

fn build_missing_groups(
    raw: &[ColumnEvidence],
    partial: &[(usize, Vec<u8>)],
) -> (Vec<MissingGroup>, HashMap<usize, String>) {
    let mut buckets: HashMap<&[u8], Vec<usize>> = HashMap::new();
    for (pos, packed) in partial {
        buckets.entry(packed.as_slice()).or_default().push(*pos);
    }
    let mut multi: Vec<Vec<usize>> = buckets.into_values().filter(|c| c.len() >= 2).collect();
    multi.sort_by_key(|cols| (std::cmp::Reverse(raw[cols[0]].n_missing), cols[0]));

    let mut groups = Vec::with_capacity(multi.len());
    let mut col_group = HashMap::new();
    for (g, cols) in multi.iter().enumerate() {
        let gid = format!("mg{}", g + 1);
        let zero_free = cols
            .iter()
            .filter(|&&c| {
                let r = &raw[c];
                r.kind == ColumnKind::Numeric && r.nonnegative == Some(true) && r.n_zero == Some(0)
            })
            .map(|&c| raw[c].name.clone())
            .collect();
        groups.push(MissingGroup {
            group_id: gid.clone(),
            columns: cols.iter().map(|&c| raw[c].name.clone()).collect(),
            n_missing: raw[cols[0]].n_missing,
            zero_free_columns: zero_free,
        });
        for &c in cols {
            col_group.insert(c, gid.clone());
        }
    }
    (groups, col_group)
}

fn build_mask_dependencies(
    raw: &[ColumnEvidence],
    partial: &[(usize, Vec<u8>)],
    zero_partners: &HashMap<Vec<u8>, Vec<usize>>,
    nonzero_partners: &HashMap<Vec<u8>, Vec<usize>>,
) -> (Vec<MaskDependency>, HashMap<usize, MaskDependency>) {
    let mut deps = Vec::new();
    let mut by_col = HashMap::new();
    // `partial` is in column order, matching the Python loop over profiles.
    for (pos, packed) in partial {
        let z = zero_partners.get(packed).map(Vec::as_slice).unwrap_or(&[]);
        let nz = nonzero_partners
            .get(packed)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        if z.is_empty() && nz.is_empty() {
            continue;
        }
        let dep = if !z.is_empty() && (nz.is_empty() || z[0] <= nz[0]) {
            MaskDependency {
                column: raw[*pos].name.clone(),
                partner: raw[z[0]].name.clone(),
                relation: MaskRelation::MissingWhenPartnerZero,
                n_other_partners: z.len() - 1,
            }
        } else {
            MaskDependency {
                column: raw[*pos].name.clone(),
                partner: raw[nz[0]].name.clone(),
                relation: MaskRelation::MissingWhenPartnerNonzero,
                n_other_partners: nz.len() - 1,
            }
        };
        deps.push(dep.clone());
        by_col.insert(*pos, dep);
    }
    (deps, by_col)
}
