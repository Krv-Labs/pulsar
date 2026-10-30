//! Processed feature-space dry run and target-proxy overlap measurement
//! (`preprocessing_plan/proxies.py`).

use std::collections::HashSet;

use pulsar_data::{preprocess, Column, Table};
use serde::Serialize;
use serde_json::json;

use super::model::Role;
use super::plan::{PreprocessingPlan, TargetProxy};
use super::render::plan_to_preprocessing_dict;
use crate::evidence::{kind::to_numeric, ColumnKind};
use crate::pyround::round_to;

/// Size of the feature space a plan would produce.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProcessedSpace {
    pub feature_count: usize,
    pub missing_marker_count: usize,
    pub dummy_column_count: usize,
}

/// Whether the plan could be applied to the data at all.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DryRun {
    pub valid: bool,
    pub error: Option<String>,
}

/// Rows where `column` is truthy: true, non-zero, or (for text with no
/// numeric reading) anything outside a small set of "no"-like spellings.
fn target_non_zero_mask(column: &Column) -> Vec<bool> {
    match column {
        Column::Bool(v) => v.iter().map(|b| *b == Some(true)).collect(),
        Column::Int(v) => v.iter().map(|&x| x != 0).collect(),
        Column::Float(v) => v.iter().map(|&x| !x.is_nan() && x != 0.0).collect(),
        Column::Text(v) => {
            let numeric: Vec<Option<f64>> = v
                .iter()
                .map(|s| s.as_deref().and_then(|s| to_numeric(s.trim())))
                .collect();
            if numeric.iter().any(Option::is_some) {
                return numeric
                    .iter()
                    .map(|x| x.is_some_and(|x| x != 0.0))
                    .collect();
            }
            const NO: [&str; 9] = ["", "0", "0.0", "false", "no", "f", "n", "none", "nan"];
            v.iter()
                .map(|s| {
                    s.as_deref()
                        .is_some_and(|s| !NO.contains(&s.trim().to_lowercase().as_str()))
                })
                .collect()
        }
    }
}

/// Explicit target columns, falling back to boolean mask-dependency partners.
fn resolve_proxy_targets(table: &Table, plan: &PreprocessingPlan) -> Vec<String> {
    let mut targets: Vec<String> = plan
        .entries
        .iter()
        .filter(|(c, e)| e.role == Role::Target && table.index_of(c).is_some())
        .map(|(c, _)| c.clone())
        .collect();
    if targets.is_empty() {
        for md in &plan.evidence.mask_dependencies {
            let p = &md.partner;
            let boolean = plan
                .evidence
                .columns
                .iter()
                .any(|c| &c.name == p && c.kind == ColumnKind::Boolean);
            if !targets.contains(p) && table.index_of(p).is_some() && boolean {
                targets.push(p.clone());
            }
        }
    }
    targets
}

fn measure_target_proxies(
    table: &Table,
    matrix: &ndarray::Array2<f64>,
    names: &[String],
    encode_prefixes: &[String],
    plan: &PreprocessingPlan,
) -> Vec<TargetProxy> {
    let n_rows = table.n_rows();
    if n_rows == 0 || matrix.nrows() != n_rows {
        return vec![];
    }
    let targets = resolve_proxy_targets(table, plan);
    let inputs: HashSet<&str> = table.names.iter().map(String::as_str).collect();
    let candidates: Vec<usize> = names
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            let marker = c.ends_with("_was_missing");
            let dummy = !marker
                && !inputs.contains(c.as_str())
                && encode_prefixes.iter().any(|p| c.starts_with(p.as_str()));
            marker || dummy
        })
        .map(|(i, _)| i)
        .collect();
    if targets.is_empty() || candidates.is_empty() {
        return vec![];
    }

    let mut proxies = Vec::new();
    for tgt in &targets {
        let mask = target_non_zero_mask(table.column(tgt).expect("resolved target exists"));
        let n_target = mask.iter().filter(|&&b| b).count();
        let base_rate = round_to(n_target as f64 / n_rows as f64, 4);
        for &j in &candidates {
            let col = &names[j];
            let arr = matrix.column(j);
            let count = |value: f64| {
                let rows = arr.iter().filter(|&&x| x == value).count();
                let both = arr
                    .iter()
                    .zip(&mask)
                    .filter(|(&x, &t)| x == value && t)
                    .count();
                (rows, both)
            };
            let (m1_rows, both_1) = count(1.0);
            if col.ends_with("_was_missing") && both_1 == 0 {
                let (m0_rows, both_0) = count(0.0);
                if m0_rows > 0 && both_0 == m0_rows {
                    proxies.push(TargetProxy {
                        column: col.clone(),
                        target: tgt.clone(),
                        marker_rows: m0_rows,
                        share_with_target: 1.0,
                        target_base_rate: base_rate,
                    });
                    continue;
                }
            }
            if m1_rows == 0 {
                continue;
            }
            let share = if both_1 == m1_rows {
                1.0
            } else {
                round_to(both_1 as f64 / m1_rows as f64, 4).min(0.9999)
            };
            proxies.push(TargetProxy {
                column: col.clone(),
                target: tgt.clone(),
                marker_rows: m1_rows,
                share_with_target: share,
                target_base_rate: base_rate,
            });
        }
    }
    proxies.sort_by(|a, b| {
        b.share_with_target
            .total_cmp(&a.share_with_target)
            .then(b.marker_rows.cmp(&a.marker_rows))
            .then_with(|| a.column.cmp(&b.column))
            .then_with(|| a.target.cmp(&b.target))
    });
    proxies
}

/// The plan as a resolved config, validated like any `params.yaml`.
pub(super) fn plan_config(
    plan: &PreprocessingPlan,
    run_name: &str,
) -> Result<pulsar_config::PulsarConfig, String> {
    pulsar_config::from_value(json!({
        "run": {"name": run_name, "data": "in_memory"},
        "preprocessing": plan_to_preprocessing_dict(plan),
    }))
    .map_err(|e| e.to_string())
}

/// Whether `pre` is exactly the preprocessing `plan` renders to, so a run
/// whose config carries `pre` applied this plan revision.
pub fn plan_matches_preprocessing(
    plan: &PreprocessingPlan,
    pre: &pulsar_config::Preprocessing,
) -> bool {
    plan_config(plan, "match").is_ok_and(|config| config.preprocessing == *pre)
}

/// Why processed-space calibration fell back to raw numeric columns.
#[derive(Debug, Clone, PartialEq)]
pub enum CalibrationFallback {
    /// The plan does not render to a valid preprocessing config.
    PlanInvalid(String),
    /// Preprocessing the table under the plan failed.
    Preprocess { code: &'static str, message: String },
    /// Preprocessing left fewer than two feature columns.
    TooFewFeatures { found: usize },
    /// Preprocessing left no rows.
    NoRows,
    /// The processed matrix still holds a non-finite value.
    NonFinite,
    /// The processed matrix could not be profiled.
    Profile(String),
}

impl std::fmt::Display for CalibrationFallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PlanInvalid(reason) => {
                write!(f, "the preprocessing plan does not render to a valid config ({reason})")
            }
            Self::Preprocess { code, message } => {
                write!(f, "preprocessing under the plan failed with {code}: {message}")
            }
            Self::TooFewFeatures { found } => write!(
                f,
                "preprocessing under the plan left {found} feature column(s); geometry needs at least 2"
            ),
            Self::NoRows => write!(f, "preprocessing under the plan left no rows"),
            Self::NonFinite => write!(
                f,
                "the processed matrix still holds a non-finite value"
            ),
            Self::Profile(reason) => {
                write!(f, "the processed matrix could not be profiled ({reason})")
            }
        }
    }
}

/// Run the plan's preprocessing (first imputation variant), z-score the
/// result and profile it (`_calibrate_processed_space`). An error names why
/// the plan cannot be applied, too few features or rows remain, or the
/// matrix cannot be profiled; the caller then falls back to raw-space
/// geometry. Python lets a profiling failure escape as an error; a missing
/// calibration is the more useful answer for an agent. `probes` sets where
/// the variance curve is measured.
pub fn calibrate_processed_space(
    table: &Table,
    plan: &PreprocessingPlan,
    probes: crate::characterization::PcaProbes<'_>,
) -> Result<crate::characterization::NumericProfile, CalibrationFallback> {
    let config = plan_config(plan, "calibrate").map_err(CalibrationFallback::PlanInvalid)?;
    let variants = config.preprocessing_variants();
    let variant = variants.first().ok_or_else(|| {
        CalibrationFallback::PlanInvalid("it yields no preprocessing variant".into())
    })?;
    let processed = preprocess(table, &variant.preprocessing, None, None).map_err(|e| {
        CalibrationFallback::Preprocess {
            code: e.code(),
            message: e.to_string(),
        }
    })?;
    let (n, k) = processed.matrix.dim();
    if k < 2 {
        return Err(CalibrationFallback::TooFewFeatures { found: k });
    }
    if n == 0 {
        return Err(CalibrationFallback::NoRows);
    }
    if processed.matrix.iter().any(|v| !v.is_finite()) {
        return Err(CalibrationFallback::NonFinite);
    }
    // StandardScaler: population standard deviation, zero variance left at 0.
    let mut x = processed.matrix;
    for mut col in x.columns_mut() {
        let mean = col.sum() / n as f64;
        let sd = (col.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64).sqrt();
        let scale = if sd > 0.0 { sd } else { 1.0 };
        col.mapv_inplace(|v| (v - mean) / scale);
    }
    crate::characterization::profile_numeric_matrix(x.view(), 1000, 42, probes)
        .map_err(|e| CalibrationFallback::Profile(e.to_string()))
}

/// The processed feature names the plan's first imputation variant yields,
/// or why the plan cannot be applied.
pub fn processed_feature_names(
    table: &Table,
    plan: &PreprocessingPlan,
) -> Result<Vec<String>, String> {
    let config = plan_config(plan, "summary")?;
    let variants = config.preprocessing_variants();
    let variant = variants
        .first()
        .ok_or("the plan yields no preprocessing variant")?;
    let processed =
        preprocess(table, &variant.preprocessing, None, None).map_err(|e| e.to_string())?;
    Ok(processed.layout.feature_names)
}

fn run(table: &Table, plan: &mut PreprocessingPlan) -> Result<ProcessedSpace, String> {
    let config = plan_config(plan, "dry_run")?;

    let variants = config.preprocessing_variants();
    let first =
        preprocess(table, &variants[0].preprocessing, None, None).map_err(|e| e.to_string())?;
    // Every variant must be applicable, not just the first.
    for variant in &variants[1..] {
        preprocess(table, &variant.preprocessing, None, None).map_err(|e| e.to_string())?;
    }

    let names = first.feature_names();
    let prefixes: Vec<String> = config
        .preprocessing
        .encode
        .keys()
        .map(|c| format!("{c}_"))
        .collect();
    let inputs: HashSet<&str> = table.names.iter().map(String::as_str).collect();
    let markers = names.iter().filter(|c| c.ends_with("_was_missing")).count();
    let dummies = if prefixes.is_empty() {
        0
    } else {
        names
            .iter()
            .filter(|c| {
                !c.ends_with("_was_missing")
                    && !inputs.contains(c.as_str())
                    && prefixes.iter().any(|p| c.starts_with(p.as_str()))
            })
            .count()
    };
    plan.target_proxies = measure_target_proxies(table, &first.matrix, names, &prefixes, plan);
    Ok(ProcessedSpace {
        feature_count: first.matrix.ncols(),
        missing_marker_count: markers,
        dummy_column_count: dummies,
    })
}

/// Apply the plan to the data without raising: the processed-space size and
/// whether it was valid. Also refreshes `plan.target_proxies`.
pub fn dry_run_processed_space(
    table: &Table,
    plan: &mut PreprocessingPlan,
) -> (Option<ProcessedSpace>, DryRun) {
    match run(table, plan) {
        Ok(space) => (
            Some(space),
            DryRun {
                valid: true,
                error: None,
            },
        ),
        Err(error) => {
            plan.target_proxies = vec![];
            (
                None,
                DryRun {
                    valid: false,
                    error: Some(error),
                },
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::characterization::PcaProbes;
    use crate::plan::{draft_plan, DraftOptions};

    fn table(columns: Vec<(&str, Column)>) -> Table {
        let (names, cols) = columns.into_iter().map(|(n, c)| (n.to_owned(), c)).unzip();
        Table::new(names, cols).unwrap()
    }

    #[test]
    fn calibration_fallback_names_its_cause() {
        let n = 30;
        let a: Vec<f64> = (0..n).map(f64::from).collect();
        let b: Vec<f64> = (0..n).map(|i| f64::from(i % 4)).collect();
        let ok = table(vec![
            ("a", Column::Float(a.clone())),
            ("b", Column::Float(b.clone())),
        ]);
        let plan = draft_plan(&ok, "ds_aaaaaaaaaaaa", &DraftOptions::default()).unwrap();
        assert!(calibrate_processed_space(&ok, &plan, PcaProbes::DataRelative).is_ok());

        let one = table(vec![("a", Column::Float(a.clone()))]);
        let plan = draft_plan(&one, "ds_aaaaaaaaaaaa", &DraftOptions::default()).unwrap();
        let err = calibrate_processed_space(&one, &plan, PcaProbes::DataRelative).unwrap_err();
        assert_eq!(err, CalibrationFallback::TooFewFeatures { found: 1 });

        let mut spiked = b;
        spiked[3] = f64::INFINITY;
        let spiked = table(vec![("a", Column::Float(a)), ("b", Column::Float(spiked))]);
        let plan = draft_plan(&spiked, "ds_aaaaaaaaaaaa", &DraftOptions::default()).unwrap();
        let err = calibrate_processed_space(&spiked, &plan, PcaProbes::DataRelative).unwrap_err();
        assert!(
            matches!(
                err,
                CalibrationFallback::Preprocess {
                    code: "NON_FINITE_VALUE",
                    ..
                }
            ),
            "{err:?}"
        );
        assert!(
            err.to_string()
                .starts_with("preprocessing under the plan failed with NON_FINITE_VALUE"),
            "{err}"
        );
    }
}
