//! The long-format -> panel pivot and the checks around it.
//!
//! Known divergences from Python:
//! - `TIME_COLUMN_NOT_ORDERABLE` cannot occur: a typed column always sorts.
//! - Text cells coerced for imputation use the shared `pd.to_numeric` token
//!   rule (`evidence::kind::to_numeric`).

use std::collections::{HashMap, HashSet};

use ndarray::{concatenate, Array2, ArrayView2, Axis};
use pulsar_config::{ImputeMethod, Preprocessing, PulsarConfig};
use pulsar_core::{impute_column_inplace, StandardScalerInner};
use pulsar_data::{Column, Key, Table};
use serde_json::{json, Map, Value};

use crate::characterization::profile_numeric_matrix;
use crate::evidence::kind::to_numeric;
use crate::payloads::bounded_list;
use crate::plan::py_repr_list;
use crate::pyround::round_to;

pub const PIVOT_POLICIES: [&str; 3] = ["drop_entity", "forward_fill", "allow_ragged"];
pub const REPRESENTATIONS: [&str; 3] = ["trajectory", "temporal", "both"];

/// Ceiling on the peak temporal working set; the dense `(n, n, T)` float64
/// tensor uses one third of it.
const DEFAULT_MAX_TENSOR_BYTES: usize = 2 * 1024 * 1024 * 1024;
/// The build keeps three tensor-sized buffers alive during normalization.
const TEMPORAL_PEAK_TENSOR_MULTIPLIER: usize = 3;
const MB: f64 = 1024.0 * 1024.0;

/// A pivot or panel failure with what an MCP error envelope needs.
#[derive(Debug, Clone, PartialEq)]
pub struct PanelError {
    pub reason: String,
    pub error_code: &'static str,
    pub agent_action: String,
    pub details: Box<Map<String, Value>>,
}

impl std::fmt::Display for PanelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.reason)
    }
}

impl std::error::Error for PanelError {}

fn panel_error(
    reason: impl Into<String>,
    error_code: &'static str,
    agent_action: impl Into<String>,
    details: Value,
) -> PanelError {
    PanelError {
        reason: reason.into(),
        error_code,
        agent_action: agent_action.into(),
        details: Box::new(match details {
            Value::Object(map) => map,
            _ => Map::new(),
        }),
    }
}

/// A pivoted longitudinal panel ready for `from_snapshots`.
#[derive(Debug, Clone, PartialEq)]
pub struct Panel {
    /// One `(n_t, F)` matrix per time step.
    pub snapshots: Vec<Array2<f64>>,
    /// Entities kept, in first-appearance order.
    pub entity_ids: Vec<Key>,
    /// Distinct time values, ascending.
    pub times: Vec<Key>,
    pub feature_columns: Vec<String>,
    pub report: Value,
    /// The entities of each snapshot's rows.
    pub snapshot_entity_ids: Vec<Vec<Key>>,
}

impl Panel {
    pub fn n_entities(&self) -> usize {
        self.entity_ids.len()
    }

    pub fn n_times(&self) -> usize {
        self.times.len()
    }

    /// Every snapshot has the same entities in the same row order.
    pub fn is_aligned(&self) -> bool {
        self.snapshot_entity_ids
            .split_first()
            .is_some_and(|(first, rest)| rest.iter().all(|ids| ids == first))
    }

    fn observation_count(&self) -> usize {
        self.snapshots.iter().map(Array2::nrows).sum()
    }
}

/// `PULSAR_MCP_MAX_TENSOR_BYTES`, or 2 GiB when unset or not a positive integer.
pub fn max_tensor_bytes() -> usize {
    std::env::var("PULSAR_MCP_MAX_TENSOR_BYTES")
        .ok()
        .and_then(|raw| raw.trim().parse::<i64>().ok())
        .filter(|&v| v > 0)
        .map_or(DEFAULT_MAX_TENSOR_BYTES, |v| v as usize)
}

/// pandas `is_numeric_dtype`.
fn is_numeric(col: &Column) -> bool {
    !matches!(col, Column::Text(_))
}

fn key_at(col: &Column, i: usize) -> Option<Key> {
    match col {
        Column::Float(v) => (!v[i].is_nan()).then_some(Key::Float(v[i])),
        Column::Int(v) => Some(Key::Int(v[i])),
        Column::Bool(v) => v[i].map(Key::Bool),
        Column::Text(v) => v[i].clone().map(Key::Text),
    }
}

fn num_at(col: &Column, i: usize) -> f64 {
    match col {
        Column::Float(v) => v[i],
        Column::Int(v) => v[i] as f64,
        Column::Bool(v) => v[i].map_or(f64::NAN, |b| f64::from(u8::from(b))),
        Column::Text(_) => f64::NAN,
    }
}

/// Sorted copies of `names` (Python's `sorted(set & ...)` before dedup).
fn sorted_names<'a>(names: impl IntoIterator<Item = &'a String>) -> Vec<String> {
    let mut out: Vec<String> = names.into_iter().cloned().collect();
    out.sort();
    out
}

/// Numeric columns other than the two keys, in table order.
fn numeric_feature_columns(table: &Table, entity: &str, time: &str) -> Vec<String> {
    table
        .names
        .iter()
        .zip(&table.columns)
        .filter(|(n, c)| n.as_str() != entity && n.as_str() != time && is_numeric(c))
        .map(|(n, _)| n.clone())
        .collect()
}

/// Apply the numeric preprocessing rules (drops, imputation) without
/// touching the entity and time keys.
pub fn prepare_panel_frame(
    table: &Table,
    pre: &Preprocessing,
    entity_column: &str,
    time_column: &str,
    feature_columns: &[String],
) -> Result<Table, PanelError> {
    let keys: HashSet<&str> = [entity_column, time_column].into();
    let drop: HashSet<&str> = pre
        .drop_columns
        .iter()
        .map(String::as_str)
        .filter(|c| !keys.contains(c))
        .collect();
    let mut selected_keys =
        sorted_names(feature_columns.iter().filter(|c| keys.contains(c.as_str())));
    selected_keys.dedup();
    if !selected_keys.is_empty() {
        return Err(panel_error(
            format!(
                "Explicit feature selection includes panel key column(s) {}.",
                py_repr_list(&selected_keys)
            ),
            "PANEL_FEATURE_KEY_SELECTED",
            "Remove entity_column and time_column from feature_columns; they are preserved as panel keys and cannot be geometry features.",
            json!({"columns": selected_keys}),
        ));
    }
    let imputed_keys = sorted_names(pre.impute.keys().filter(|c| keys.contains(c.as_str())));
    if !imputed_keys.is_empty() {
        return Err(panel_error(
            format!(
                "Imputation rules target panel key column(s) {}.",
                py_repr_list(&imputed_keys)
            ),
            "PANEL_PREPROCESSING_KEY_IMPUTE",
            "Remove entity_column and time_column from preprocessing.impute; panel keys are preserved as identifiers.",
            json!({"columns": imputed_keys}),
        ));
    }

    let (names, mut columns): (Vec<String>, Vec<Column>) = table
        .names
        .iter()
        .zip(&table.columns)
        .filter(|(n, _)| !drop.contains(n.as_str()))
        .map(|(n, c)| (n.clone(), c.clone()))
        .unzip();
    let selected: HashSet<String> = if !feature_columns.is_empty() {
        feature_columns.iter().cloned().collect()
    } else {
        let mut set: HashSet<String> = names
            .iter()
            .zip(&columns)
            .filter(|(n, c)| {
                n.as_str() != entity_column && n.as_str() != time_column && is_numeric(c)
            })
            .map(|(n, _)| n.clone())
            .collect();
        set.extend(pre.impute.keys().filter(|c| names.contains(c)).cloned());
        set
    };

    let encoded = sorted_names(pre.encode.keys().filter(|c| {
        !drop.contains(c.as_str())
            && !keys.contains(c.as_str())
            && table.index_of(c).is_some()
            && selected.contains(c.as_str())
    }));
    if !encoded.is_empty() {
        return Err(panel_error(
            "Categorical encoding is not supported for longitudinal features.",
            "PANEL_CATEGORICAL_UNSUPPORTED",
            "Use numeric feature columns, or remove preprocessing.encode rules for this longitudinal build.",
            json!({"columns": encoded}),
        ));
    }
    if !feature_columns.is_empty() {
        let mut conflicting =
            sorted_names(feature_columns.iter().filter(|c| drop.contains(c.as_str())));
        conflicting.dedup();
        if !conflicting.is_empty() {
            return Err(panel_error(
                format!(
                    "Explicit feature selection includes dropped column(s) {}.",
                    py_repr_list(&conflicting)
                ),
                "PANEL_FEATURE_DROPPED",
                "Remove these columns from feature_columns or from preprocessing.drop_columns so the selection is unambiguous.",
                json!({"columns": conflicting}),
            ));
        }
    }

    for (column, spec) in &pre.impute {
        let Some(k) = names.iter().position(|n| n == column) else {
            continue;
        };
        if !selected.contains(column) {
            continue;
        }
        let method = serde_json::to_value(spec.method)
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_default();
        if spec.method == ImputeMethod::FillValue {
            return Err(panel_error(
                format!("Imputation method '{method}' for '{column}' is not numeric."),
                "PANEL_IMPUTE_METHOD_UNSUPPORTED",
                "Longitudinal preprocessing supports numeric imputation only; use a method supported for numeric values.",
                json!({"column": column, "method": method}),
            ));
        }
        let mut values: Vec<f64> = match &columns[k] {
            Column::Text(cells) => {
                let mut invalid = 0;
                let coerced = cells
                    .iter()
                    .map(|cell| match cell {
                        None => f64::NAN,
                        Some(s) => to_numeric(s).unwrap_or_else(|| {
                            invalid += 1;
                            f64::NAN
                        }),
                    })
                    .collect();
                if invalid > 0 {
                    return Err(panel_error(
                        format!("Imputation target '{column}' contains non-numeric values."),
                        "PANEL_IMPUTE_NON_NUMERIC_VALUES",
                        "Clean the column or drop it. Redaction and other text values are not treated as ordinary missing data.",
                        json!({"column": column, "invalid_value_count": invalid}),
                    ));
                }
                coerced
            }
            col => (0..col.len()).map(|i| num_at(col, i)).collect(),
        };
        let missing = values.iter().filter(|v| v.is_nan()).count();
        if matches!(columns[k], Column::Text(_)) {
            columns[k] = Column::Float(values.clone());
        }
        if missing == 0 {
            continue;
        }
        if missing == values.len() {
            return Err(panel_error(
                format!("Feature column '{column}' is all-missing."),
                "PANEL_FEATURE_ALL_MISSING",
                "Drop the all-missing column or provide observed numeric values before imputing it.",
                json!({"column": column}),
            ));
        }
        impute_column_inplace(&mut values, &method, spec.seed).map_err(|e| {
            panel_error(
                e.to_string(),
                "PANEL_IMPUTE_FAILED",
                "Check the imputation rule for this column.",
                json!({"column": column, "method": method}),
            )
        })?;
        columns[k] = Column::Float(values);
    }
    Table::new(names, columns).map_err(|e| {
        panel_error(
            e.to_string(),
            "PANEL_EMPTY",
            "Check that the dataset keeps at least one column after preprocessing.drop_columns.",
            json!({}),
        )
    })
}

/// Pivot a long-format table into per-time snapshots with aligned rows.
///
/// `time_column` must already be discrete and orderable; no binning is
/// applied. Rows are keyed by `(entity, time)`; duplicates are an error, not
/// a silent aggregation. An empty `feature_columns` selects every numeric
/// column other than the keys.
pub fn pivot_panel(
    table: &Table,
    entity_column: &str,
    time_column: &str,
    feature_columns: &[String],
    on_missing: &str,
) -> Result<Panel, PanelError> {
    if !PIVOT_POLICIES.contains(&on_missing) {
        return Err(panel_error(
            format!("Unknown on_missing policy '{on_missing}'."),
            "INVALID_PIVOT_POLICY",
            format!("Use one of {}.", py_repr_list(&PIVOT_POLICIES)),
            json!({"on_missing": on_missing, "supported": PIVOT_POLICIES}),
        ));
    }
    let available = || bounded_list(&table.names, 20);
    let missing_keys: Vec<&str> = [entity_column, time_column]
        .into_iter()
        .filter(|c| table.index_of(c).is_none())
        .collect();
    if !missing_keys.is_empty() {
        return Err(panel_error(
            format!(
                "Column(s) {} not present in the dataset.",
                py_repr_list(&missing_keys)
            ),
            "PANEL_COLUMN_NOT_FOUND",
            "Call probe_columns or characterize_dataset to list available columns, then pass the correct entity_column and time_column.",
            json!({"missing": missing_keys, "available_columns": available()}),
        ));
    }
    let features: Vec<String> = if !feature_columns.is_empty() {
        let unknown: Vec<&String> = feature_columns
            .iter()
            .filter(|c| table.index_of(c).is_none())
            .collect();
        if !unknown.is_empty() {
            return Err(panel_error(
                format!(
                    "Feature column(s) {} not present in the dataset.",
                    py_repr_list(&unknown)
                ),
                "PANEL_COLUMN_NOT_FOUND",
                "Pass feature_columns that exist in the dataset.",
                json!({"missing": unknown, "available_columns": available()}),
            ));
        }
        let non_numeric: Vec<&String> = feature_columns
            .iter()
            .filter(|c| table.column(c).is_some_and(|col| !is_numeric(col)))
            .collect();
        if !non_numeric.is_empty() {
            return Err(panel_error(
                format!(
                    "Feature column(s) {} are not numeric.",
                    py_repr_list(&non_numeric)
                ),
                "PANEL_FEATURES_NOT_NUMERIC",
                "Longitudinal geometry needs numeric features. Encode or drop these columns before building.",
                json!({"non_numeric": non_numeric}),
            ));
        }
        feature_columns.to_vec()
    } else {
        numeric_feature_columns(table, entity_column, time_column)
    };
    if features.is_empty() {
        return Err(panel_error(
            "No numeric feature columns remain after excluding entity and time.",
            "PANEL_NO_FEATURES",
            "Pass feature_columns explicitly, or encode categorical columns first.",
            json!({"entity_column": entity_column, "time_column": time_column}),
        ));
    }

    let col = |name: &str| table.column(name).expect("checked above");
    let (ent_col, time_col) = (col(entity_column), col(time_column));
    let feat_cols: Vec<&Column> = features.iter().map(|f| col(f)).collect();
    let n_rows = table.n_rows();

    let all_missing: Vec<&String> = features
        .iter()
        .zip(&feat_cols)
        .filter(|(_, c)| (0..n_rows).all(|i| num_at(c, i).is_nan()))
        .map(|(f, _)| f)
        .collect();
    if !all_missing.is_empty() {
        return Err(panel_error(
            format!(
                "Feature column(s) {} are all-missing.",
                py_repr_list(&all_missing)
            ),
            "PANEL_FEATURE_ALL_MISSING",
            "Drop all-missing columns or provide observed numeric values before building the panel.",
            json!({"columns": all_missing}),
        ));
    }

    let row_keys: Vec<(Option<Key>, Option<Key>)> = (0..n_rows)
        .map(|i| (key_at(ent_col, i), key_at(time_col, i)))
        .collect();
    let mut seen = HashSet::new();
    let duplicates = row_keys.iter().filter(|k| !seen.insert(*k)).count();
    if duplicates > 0 {
        return Err(panel_error(
            format!("{duplicates} duplicate (entity, time) rows found."),
            "PANEL_DUPLICATE_OBSERVATIONS",
            "Aggregate the duplicates to one row per (entity, time) before ingesting; Pulsar will not pick an aggregator for you.",
            json!({
                "duplicate_row_count": duplicates,
                "entity_column": entity_column,
                "time_column": time_column,
            }),
        ));
    }

    let mut times: Vec<Key> = row_keys.iter().filter_map(|(_, t)| t.clone()).collect();
    times.sort();
    times.dedup();
    if times.is_empty() {
        return Err(panel_error(
            format!("Time column '{time_column}' has no non-null values."),
            "PANEL_EMPTY",
            "Check the time column for an all-null or empty dataset.",
            json!({"time_column": time_column}),
        ));
    }
    if times.len() < 2 {
        return Err(panel_error(
            format!("Time column '{time_column}' has only one distinct value."),
            "PANEL_SINGLE_TIME_STEP",
            "A longitudinal build needs at least 2 time steps. Use run_topological_sweep for single-snapshot analysis.",
            json!({"time_column": time_column, "n_times": times.len()}),
        ));
    }

    let mut entities: Vec<Key> = Vec::new();
    let mut entity_index: HashMap<Key, usize> = HashMap::new();
    for (e, _) in &row_keys {
        if let Some(e) = e {
            if !entity_index.contains_key(e) {
                entity_index.insert(e.clone(), entities.len());
                entities.push(e.clone());
            }
        }
    }
    let time_index: HashMap<&Key, usize> = times.iter().enumerate().map(|(k, t)| (t, k)).collect();
    let (n_e, n_t, n_f) = (entities.len(), times.len(), features.len());

    // cube[e][t][f], NaN where (e, t) is absent.
    let mut present = vec![vec![false; n_t]; n_e];
    let mut cube = vec![vec![vec![f64::NAN; n_f]; n_t]; n_e];
    for (i, (e, t)) in row_keys.iter().enumerate() {
        let (Some(e), Some(t)) = (e, t) else { continue };
        let (ei, ti) = (entity_index[e], time_index[t]);
        present[ei][ti] = true;
        for (f, c) in feat_cols.iter().enumerate() {
            let value = num_at(c, i);
            // NaN is a missing cell handled by the policy below; an infinity
            // would turn its whole column to NaN once the panel is scaled.
            if value.is_infinite() {
                return Err(panel_error(
                    format!(
                        "Feature column '{}' holds the non-finite value {value} for entity {e} at time {t} (row {i}).",
                        features[f]
                    ),
                    "PANEL_NON_FINITE_VALUE",
                    "Replace or remove infinite values in this column before building the panel; Pulsar does not choose a substitute.",
                    json!({
                        "column": features[f],
                        "entity": e.to_string(),
                        "time": t.to_string(),
                        "row": i,
                        "value": value.to_string(),
                    }),
                ));
            }
            cube[ei][ti][f] = value;
        }
    }
    let cells = (n_e * n_t) as f64;
    let present_cells = present.iter().flatten().filter(|&&p| p).count() as f64;
    let cells_missing_fraction = if cells > 0.0 {
        1.0 - present_cells / cells
    } else {
        0.0
    };

    let mut dropped: Vec<&Key> = Vec::new();
    let (snapshots, snapshot_entity_ids, kept_entities) = if on_missing == "allow_ragged" {
        let mut snapshots = Vec::with_capacity(n_t);
        let mut ids = Vec::with_capacity(n_t);
        for t in 0..n_t {
            let rows: Vec<usize> = (0..n_e).filter(|&e| present[e][t]).collect();
            let data: Vec<f64> = rows.iter().flat_map(|&e| cube[e][t].clone()).collect();
            snapshots
                .push(Array2::from_shape_vec((rows.len(), n_f), data).expect("rows x features"));
            ids.push(
                rows.iter()
                    .map(|&e| entities[e].clone())
                    .collect::<Vec<_>>(),
            );
        }
        (snapshots, ids, entities.clone())
    } else {
        if on_missing == "forward_fill" {
            for slab in &mut cube {
                for t in 1..n_t {
                    let (before, after) = slab.split_at_mut(t);
                    for (cell, prev) in after[0].iter_mut().zip(&before[t - 1]) {
                        if cell.is_nan() {
                            *cell = *prev;
                        }
                    }
                }
            }
        }
        let has_nan = |e: usize| cube[e].iter().flatten().any(|v| v.is_nan());
        let complete: Vec<bool> = (0..n_e)
            .map(|e| match on_missing {
                "forward_fill" => !has_nan(e),
                _ => present[e].iter().all(|&p| p) && !has_nan(e),
            })
            .collect();
        let kept: Vec<usize> = (0..n_e).filter(|&e| complete[e]).collect();
        dropped = (0..n_e)
            .filter(|&e| !complete[e])
            .map(|e| &entities[e])
            .collect();
        if kept.is_empty() {
            return Err(panel_error(
                "No entity is observed at every time step.",
                "PANEL_NO_COMPLETE_ENTITIES",
                "Try on_missing='forward_fill', or on_missing='allow_ragged' with representation='trajectory'.",
                json!({
                    "n_entities_input": n_e,
                    "n_times": n_t,
                    "cells_missing_fraction": round_to(cells_missing_fraction, 4),
                }),
            ));
        }
        let snapshots = (0..n_t)
            .map(|t| {
                let data: Vec<f64> = kept.iter().flat_map(|&e| cube[e][t].clone()).collect();
                Array2::from_shape_vec((kept.len(), n_f), data).expect("rows x features")
            })
            .collect();
        let kept_ids: Vec<Key> = kept.iter().map(|&e| entities[e].clone()).collect();
        (snapshots, vec![kept_ids.clone(); n_t], kept_ids)
    };

    let mut nan_counts = vec![0_usize; n_f];
    for snap in &snapshots {
        for row in snap.rows() {
            for (f, v) in row.iter().enumerate() {
                nan_counts[f] += usize::from(v.is_nan());
            }
        }
    }
    let remaining: Map<String, Value> = features
        .iter()
        .zip(&nan_counts)
        .filter(|(_, &c)| c > 0)
        .map(|(f, &c)| (f.clone(), json!(c)))
        .collect();
    if !remaining.is_empty() {
        return Err(panel_error(
            "Missing values remain in longitudinal feature columns.",
            "PANEL_NAN_REMAINS",
            "Add numeric imputation rules in preprocessing.impute or remove the affected columns with preprocessing.drop_columns.",
            json!({"missing_by_column": remaining}),
        ));
    }

    let dropped_text: Vec<String> = dropped.iter().map(|e| e.to_string()).collect();
    let mut panel = Panel {
        snapshots,
        entity_ids: kept_entities,
        times,
        feature_columns: features,
        report: Value::Null,
        snapshot_entity_ids,
    };
    panel.report = json!({
        "policy_applied": on_missing,
        "n_entities_input": n_e,
        "n_entities_kept": panel.n_entities(),
        "dropped_entities": bounded_list(&dropped_text, 20),
        "n_times": n_t,
        "n_features": n_f,
        "cells_missing_fraction": round_to(cells_missing_fraction, 4),
        "alignment": if on_missing == "allow_ragged" { "ragged" } else { "complete" },
        "observation_count": panel.observation_count(),
    });
    Ok(panel)
}

/// Fit the projection grid and the epsilon grid to the panel.
///
/// `create_config` calibrates against the raw long-format file, whose column
/// count includes the keys and therefore exceeds the panel's feature count;
/// dimensions above it are dropped and disclosed. The notes mirror Python,
/// which kept a `pca` copy of the projection grid and reported it too.
pub fn adapt_config_to_panel(
    config: &PulsarConfig,
    panel: &Panel,
) -> Result<(PulsarConfig, Map<String, Value>), PanelError> {
    let n_features = panel.feature_columns.len();
    if n_features < 2 {
        return Err(panel_error(
            format!("Panel has {n_features} feature column(s); need at least 2."),
            "PANEL_TOO_FEW_FEATURES",
            "Pass more feature_columns, or check that entity_column and time_column are not consuming the only numeric columns.",
            json!({"n_features": n_features, "feature_columns": panel.feature_columns}),
        ));
    }
    let mut adapted = config.clone();
    let mut notes = Map::new();
    notes.insert("panel_feature_count".into(), json!(n_features));
    let requested = adapted.projection.dimensions.clone();
    let mut keep: Vec<usize> = requested
        .iter()
        .copied()
        .filter(|&d| d <= n_features)
        .collect();
    let dropped: Vec<usize> = requested
        .iter()
        .copied()
        .filter(|&d| d > n_features)
        .collect();
    if keep.is_empty() {
        keep = vec![n_features];
    }
    adapted.projection.dimensions = keep.clone();
    if !dropped.is_empty() {
        for spec in ["projection", "pca"] {
            notes.insert(format!("{spec}_dimensions_requested"), json!(requested));
            notes.insert(format!("{spec}_dimensions_applied"), json!(keep));
            notes.insert(format!("{spec}_dimensions_dropped"), json!(dropped));
        }
        notes.insert(
            "projection_reason".into(),
            json!("Projection dimensions above the panel feature count were dropped; config was calibrated against the raw file, which also counts the entity and time columns."),
        );
    }
    notes.extend(calibrate_epsilon_to_panel(&mut adapted, panel)?);
    Ok((adapted, notes))
}

/// Check the epsilon grid against the scaled panel's k-NN domain
/// `[knn_p5, knn_p95]`; recalibrate only when no epsilon falls inside it.
fn calibrate_epsilon_to_panel(
    config: &mut PulsarConfig,
    panel: &Panel,
) -> Result<Map<String, Value>, PanelError> {
    let views: Vec<ArrayView2<f64>> = panel.snapshots.iter().map(Array2::view).collect();
    let pooled = concatenate(Axis(0), &views).map_err(|e| {
        panel_error(
            e.to_string(),
            "PANEL_EMPTY",
            "Check the panel's snapshots.",
            json!({}),
        )
    })?;
    let (scaled, _) = StandardScalerInner::fit_transform(&pooled);
    let profile = profile_numeric_matrix(scaled.view(), 1000, 42, None).map_err(|e| {
        panel_error(
            e.to_string(),
            "PANEL_PROFILE_FAILED",
            "The panel is too small to profile; add entities or time steps.",
            json!({}),
        )
    })?;
    let (low, high) = (profile.knn_p5, profile.knn_p95);
    let requested = config.ball_mapper.epsilons.clone();
    let in_domain = requested.iter().filter(|&&e| low <= e && e <= high).count();
    let mut notes = Map::new();
    notes.insert(
        "panel_epsilon_domain".into(),
        json!({"knn_p5": round_to(low, 4), "knn_p95": round_to(high, 4)}),
    );
    notes.insert("epsilons_in_domain".into(), json!(in_domain));
    if in_domain > 0 || high <= 0.0 {
        return Ok(notes);
    }
    let steps = requested.len().max(2);
    let applied: Vec<f64> = (0..steps)
        .map(|k| {
            // np.linspace: start + k * step, with the endpoint set exactly.
            let v = if k == steps - 1 {
                high
            } else {
                low + k as f64 * ((high - low) / (steps - 1) as f64)
            };
            round_to(v, 6)
        })
        .collect();
    config.ball_mapper.epsilons = applied.clone();
    notes.insert("epsilons_requested".into(), bounded_list(&requested, 5));
    notes.insert("epsilons_applied".into(), bounded_list(&applied, 5));
    notes.insert(
        "epsilon_reason".into(),
        json!("No requested epsilon fell inside the panel's k-NN domain, which would produce a degenerate single-ball cover. Recalibrated onto [knn_p5, knn_p95] of the scaled panel. Pass config_yaml with your own epsilons to override."),
    );
    Ok(notes)
}

fn tensor_bytes(panel: &Panel) -> usize {
    panel.n_entities() * panel.n_entities() * panel.n_times() * 8
}

fn mb(bytes: usize) -> f64 {
    round_to(bytes as f64 / MB, 2)
}

/// Visible cost context so the agent chooses a representation deliberately.
pub fn estimate_costs(panel: &Panel, config: &PulsarConfig) -> Value {
    let tensor = tensor_bytes(panel);
    let peak = tensor * TEMPORAL_PEAK_TENSOR_MULTIPLIER;
    let p = &config.projection;
    let eps = &config.ball_mapper.epsilons;
    let grid = p.dimensions.len() * p.seeds.len() * eps.len();
    json!({
        "n_observations": panel.report["observation_count"],
        "n_entities": panel.n_entities(),
        "n_times": panel.n_times(),
        "temporal_tensor_bytes": tensor,
        "temporal_tensor_mb": mb(tensor),
        "temporal_peak_bytes": peak,
        "temporal_peak_mb": mb(peak),
        "temporal_tensor_limit_mb": mb(max_tensor_bytes()),
        "estimated_ball_maps_trajectory": grid,
        "estimated_ball_maps_temporal": grid * panel.n_times(),
        "projection_dimensions": p.dimensions,
        "epsilons": eps,
    })
}

/// Refuse a representation the panel cannot support: unknown names, ragged
/// panels for the entity tensor, and tensors over the memory ceiling.
pub fn guard_representation(panel: &Panel, representation: &str) -> Result<(), PanelError> {
    if !REPRESENTATIONS.contains(&representation) {
        return Err(panel_error(
            format!("Unknown representation '{representation}'."),
            "INVALID_REPRESENTATION",
            format!("Use one of {}.", py_repr_list(&REPRESENTATIONS)),
            json!({"supported": REPRESENTATIONS}),
        ));
    }
    if representation == "trajectory" {
        return Ok(());
    }
    if !panel.is_aligned() {
        let counts: Vec<usize> = panel.snapshots.iter().map(Array2::nrows).collect();
        return Err(panel_error(
            "TemporalCosmicGraph requires the same entities in the same order at every time step.",
            "RAGGED_PANEL_NOT_SUPPORTED",
            "Use representation='trajectory' (tolerates ragged panels), or rebuild with on_missing='drop_entity' or 'forward_fill'.",
            json!({
                "snapshot_row_counts": bounded_list(&counts, 20),
                "policy_applied": panel.report.get("policy_applied"),
            }),
        ));
    }
    let limit = max_tensor_bytes();
    let tensor = tensor_bytes(panel);
    let peak = tensor * TEMPORAL_PEAK_TENSOR_MULTIPLIER;
    if peak > limit {
        return Err(panel_error(
            "Temporal tensor exceeds the configured memory ceiling.",
            "TENSOR_TOO_LARGE",
            "Use representation='trajectory' — it stays sparse and never allocates an (n, n, T) tensor. Otherwise reduce entities or time steps, or raise the peak-working-set limit PULSAR_MCP_MAX_TENSOR_BYTES.",
            json!({
                "tensor_shape": [panel.n_entities(), panel.n_entities(), panel.n_times()],
                "tensor_bytes": tensor,
                "tensor_mb": mb(tensor),
                "peak_required_bytes": peak,
                "peak_required_mb": mb(peak),
                "limit_bytes": limit,
                "limit_mb": mb(limit),
            }),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aligned(n: usize, t: usize) -> Panel {
        let ids: Vec<Key> = (0..n as i64).map(Key::Int).collect();
        Panel {
            snapshots: vec![Array2::zeros((n, 2)); t],
            entity_ids: ids.clone(),
            times: (0..t as i64).map(Key::Int).collect(),
            feature_columns: vec!["a".into(), "b".into()],
            report: json!({"policy_applied": "drop_entity", "observation_count": n * t}),
            snapshot_entity_ids: vec![ids; t],
        }
    }

    #[test]
    fn a_tensor_over_the_ceiling_is_refused_but_a_trajectory_is_not() {
        // 5000^2 * 4 * 8 bytes = 800 MB; x3 peak exceeds the 2 GiB default.
        let big = aligned(5000, 4);
        assert_eq!(max_tensor_bytes(), DEFAULT_MAX_TENSOR_BYTES);
        assert!(guard_representation(&big, "trajectory").is_ok());
        let err = guard_representation(&big, "both").unwrap_err();
        assert_eq!(err.error_code, "TENSOR_TOO_LARGE");
        assert_eq!(err.details["tensor_shape"], json!([5000, 5000, 4]));
        assert_eq!(err.details["tensor_bytes"], json!(800_000_000));
        assert_eq!(err.details["peak_required_mb"], json!(2288.82));
        assert!(guard_representation(&aligned(100, 4), "temporal").is_ok());
    }

    #[test]
    fn an_infinite_cell_is_refused_with_its_entity_time_and_row() {
        let table = Table::new(
            vec!["e".into(), "t".into(), "x".into()],
            vec![
                Column::Int(vec![1, 1, 2, 2]),
                Column::Int(vec![0, 1, 0, 1]),
                Column::Float(vec![0.5, 1.0, f64::INFINITY, 2.0]),
            ],
        )
        .unwrap();
        let err = pivot_panel(&table, "e", "t", &[], "drop_entity").unwrap_err();
        assert_eq!(err.error_code, "PANEL_NON_FINITE_VALUE");
        assert!(err.reason.contains("'x'"), "{}", err.reason);
        assert!(err.reason.contains("inf"), "{}", err.reason);
        assert_eq!(err.details["entity"], json!("2"));
        assert_eq!(err.details["time"], json!("0"));
        assert_eq!(err.details["row"], json!(2));
    }

    #[test]
    fn alignment_needs_identical_ids_in_every_snapshot() {
        let mut p = aligned(3, 2);
        assert!(p.is_aligned());
        p.snapshot_entity_ids[1].swap(0, 1);
        assert!(!p.is_aligned());
        p.snapshot_entity_ids.clear();
        assert!(!p.is_aligned(), "Python: bool([]) is False");
    }
}
