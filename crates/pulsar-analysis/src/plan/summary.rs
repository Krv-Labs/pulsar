//! A short account of what a plan does to the data: the `inputs` part of
//! `io.phirewall/summary` v1, for a reviewer who knows the domain.
//!
//! It names columns and counts, never values: a category level or a fill
//! value can be patient-level data, so neither is read here. Drop reasons are
//! codes derived from the entry's role, provenance and evidence, not the
//! entry's prose `reason`, which can quote a value. Every column is named by
//! its `feature` id, its position in the plan (the table's column order),
//! beside a label cut by [`label`].

use std::collections::HashMap;

use serde_json::{json, Value};

use super::model::{DecidedBy, PlanEntry, QuestionStatus, Role};
use super::plan::PreprocessingPlan;
use crate::disclosure::{label, row_fraction, RowCount};
use crate::evidence::{ColumnEvidence, ColumnKind};

/// Every code [`drop_reason`] can return.
pub const DROP_REASONS: [&str; 10] = [
    "designated",
    "user",
    "agent",
    "missing",
    "constant",
    "identifier",
    "datetime",
    "conditional",
    "high_cardinality",
    "other",
];

/// Why a non-feature column is left out, as one of [`DROP_REASONS`]: a
/// designated entity, time or target column; a user's or agent's decision;
/// otherwise the default rule that dropped it: blank in every row or in more
/// than half of them (`missing`), one value, unique text per row, date/time
/// text, blank exactly where a 0/1 partner has one value (`conditional`), or
/// more text categories than `sqrt(rows)`.
fn drop_reason(
    entry: &PlanEntry,
    ev_by_col: &HashMap<&str, &ColumnEvidence>,
    cat_threshold: f64,
) -> &'static str {
    if matches!(entry.role, Role::Entity | Role::Time | Role::Target) {
        return "designated";
    }
    match entry.decided_by {
        DecidedBy::User => return "user",
        DecidedBy::Agent => return "agent",
        DecidedBy::Default => {}
    }
    if entry.role == Role::Id {
        return "identifier";
    }
    let Some(ev) = ev_by_col.get(entry.column.as_str()) else {
        return "other";
    };
    let conditional = ev.mask_dependency.as_ref().is_some_and(|md| {
        ev_by_col
            .get(md.partner.as_str())
            .is_some_and(|p| p.kind == ColumnKind::Boolean)
    });
    if ev.n_rows > 0 && ev.n_missing >= ev.n_rows {
        "missing"
    } else if ev.is_constant {
        "constant"
    } else if ev.kind == ColumnKind::DatetimeLike {
        "datetime"
    } else if conditional {
        "conditional"
    } else if ev.kind == ColumnKind::Text && ev.n_unique as f64 > cat_threshold {
        "high_cardinality"
    } else if ev.n_missing > ev.n_rows - ev.n_missing {
        "missing"
    } else {
        "other"
    }
}

/// Processed features each encoded column added: names `<column>_<level>`,
/// each credited to the longest encoded column name it starts with, leaving
/// out blank markers and names that are input columns themselves.
fn features_added<'a>(
    plan: &PreprocessingPlan,
    encoded: &[&'a str],
    feature_names: &[String],
) -> HashMap<&'a str, usize> {
    let mut added: HashMap<&str, usize> = encoded.iter().map(|c| (*c, 0)).collect();
    for name in feature_names {
        if name.ends_with("_was_missing") || plan.entries.contains_key(name) {
            continue;
        }
        let owner = encoded
            .iter()
            .filter(|c| {
                name.len() > c.len() + 1 && name.starts_with(*c) && name[c.len()..].starts_with('_')
            })
            .max_by_key(|c| c.len());
        if let Some(c) = owner {
            *added.get_mut(c).expect("every encoded column is counted") += 1;
        }
    }
    added
}

/// `{feature, column, truncated?}` for the plan's column `col`, or `None`
/// for a name the plan does not have.
fn column(plan: &PreprocessingPlan, col: &str) -> Option<Value> {
    let feature = plan.entries.get_index_of(col)?;
    let (column, truncated) = label(col);
    let mut out = json!({"feature": feature, "column": column});
    if truncated {
        out["truncated"] = json!(true);
    }
    Some(out)
}

/// `column` with `fields` added.
fn entry(plan: &PreprocessingPlan, col: &str, fields: Value) -> Value {
    let mut out = column(plan, col).expect("an entry of the plan");
    if let (Value::Object(out), Value::Object(fields)) = (&mut out, fields) {
        out.extend(fields);
    }
    out
}

/// A blank marker as a code: `none`, `own`, or `group` for `group:<gid>`.
fn indicator_code(indicator: &str) -> &str {
    if indicator.starts_with("group:") {
        "group"
    } else {
        indicator
    }
}

/// `{feature, column, field, to}` for each column whose role, imputation
/// method, encoding or blank marker differs from `previous`. Imputation is
/// compared by method only, so a fill value never appears.
fn changes(previous: &PreprocessingPlan, plan: &PreprocessingPlan) -> Vec<Value> {
    let fields = |e: &PlanEntry| {
        [
            ("role", json!(e.role)),
            ("impute", json!(e.impute_method())),
            ("encode", json!(e.encode)),
            ("indicator", json!(indicator_code(&e.indicator))),
        ]
    };
    let mut out = Vec::new();
    for (col, old) in &previous.entries {
        let Some(new) = plan.entries.get(col) else {
            continue;
        };
        for ((field, from), (_, to)) in fields(old).into_iter().zip(fields(new)) {
            if from != to {
                out.push(entry(plan, col, json!({"field": field, "to": to})));
            }
        }
    }
    out
}

/// The `inputs` part for `plan`.
///
/// `counts` agree with the plan's `preprocessing:` block: `kept` columns are
/// features, `dropped` are the block's `drop_columns` (every non-feature
/// role), and `encoded` and `imputed` count features with an `encode` or
/// `impute` entry. One column can be both encoded and imputed (a text column
/// whose blanks become their own category), so those two can overlap.
///
/// `feature_names` are the processed features when the plan applies to the
/// data; without them `output_features` and `levels` are null.
/// `changes_from_previous` compares against `previous`, the revision this
/// one replaced, and is empty without one. Every list is whole, so each
/// `*_omitted` is 0; an open question's `rows_affected` is a row count
/// disclosed under `row_floor`.
pub fn build_preprocessing_summary(
    plan: &PreprocessingPlan,
    previous: Option<&PreprocessingPlan>,
    feature_names: Option<&[String]>,
    row_floor: usize,
) -> Value {
    let ev_by_col: HashMap<&str, &ColumnEvidence> = plan
        .evidence
        .columns
        .iter()
        .map(|c| (c.name.as_str(), c))
        .collect();
    let cat_threshold = (plan.evidence.n_rows.max(1) as f64).sqrt();
    let features: Vec<&PlanEntry> = plan
        .entries
        .values()
        .filter(|e| e.role == Role::Feature)
        .collect();
    let encoded_cols: Vec<&str> = features
        .iter()
        .filter(|e| e.encode.is_some())
        .map(|e| e.column.as_str())
        .collect();
    let added = feature_names.map(|names| features_added(plan, &encoded_cols, names));

    let dropped: Vec<Value> = plan
        .entries
        .values()
        .filter(|e| e.role != Role::Feature)
        .map(|e| {
            let reason = drop_reason(e, &ev_by_col, cat_threshold);
            entry(plan, &e.column, json!({"reason": reason}))
        })
        .collect();
    let encoded: Vec<Value> = features
        .iter()
        .filter_map(|e| Some((e, e.encode.as_ref()?)))
        .map(|(e, method)| {
            // One-hot adds one feature per level.
            let n = added.as_ref().map(|a| a[e.column.as_str()]);
            entry(plan, &e.column, json!({"method": method, "levels": n}))
        })
        .collect();
    let imputed: Vec<Value> = features
        .iter()
        .filter_map(|e| Some((e, e.impute_method()?)))
        .map(|(e, method)| {
            let fraction = ev_by_col
                .get(e.column.as_str())
                .map(|ev| row_fraction(ev.missing_fraction));
            entry(
                plan,
                &e.column,
                json!({"missing_fraction": fraction, "method": method}),
            )
        })
        .collect();
    let open_questions: Vec<Value> = plan
        .questions
        .values()
        .filter(|q| q.status == QuestionStatus::Open)
        .map(|q| {
            let columns: Vec<Value> = q
                .applies_to
                .iter()
                .filter_map(|c| column(plan, c))
                .collect();
            json!({
                "code": q.topic,
                "columns": columns,
                "columns_omitted": 0,
                "rows_affected": RowCount::of(q.rows_affected, row_floor),
            })
        })
        .collect();

    let changes_from_previous = previous.map(|p| changes(p, plan)).unwrap_or_default();
    json!({
        "revision": plan.revision,
        "input_columns": plan.entries.len(),
        "output_features": feature_names.map(<[String]>::len),
        "counts": {
            "kept": features.len(),
            "dropped": dropped.len(),
            "encoded": encoded.len(),
            "imputed": imputed.len(),
        },
        "dropped": dropped,
        "dropped_omitted": 0,
        "encoded": encoded,
        "encoded_omitted": 0,
        "imputed": imputed,
        "imputed_omitted": 0,
        "open_questions": open_questions,
        "open_questions_omitted": 0,
        "changes_from_previous": changes_from_previous,
        "changes_from_previous_omitted": 0,
    })
}

#[cfg(test)]
mod tests {
    use pulsar_data::{Column, Table};

    use super::*;
    use crate::plan::{draft_plan, processed_feature_names, DraftOptions};

    fn text(values: &[Option<&str>]) -> Column {
        Column::Text(values.iter().map(|v| v.map(String::from)).collect())
    }

    /// 36 rows: an id, two encoded text columns whose names overlap, a
    /// constant, a numeric column with blanks and one blank in most rows.
    fn table() -> Table {
        let n = 36;
        let ids: Vec<String> = (0..n).map(|i| format!("r{i}")).collect();
        let ward = ["north", "south", "east"];
        let cols = vec![
            (
                "pid",
                text(&ids.iter().map(|s| Some(s.as_str())).collect::<Vec<_>>()),
            ),
            (
                "ward",
                text(
                    &(0..n)
                        .map(|i| (i % 7 != 0).then_some(ward[i % 3]))
                        .collect::<Vec<_>>(),
                ),
            ),
            (
                "ward_type",
                text(
                    &(0..n)
                        .map(|i| Some(["acute", "rehab"][i % 2]))
                        .collect::<Vec<_>>(),
                ),
            ),
            ("site", Column::Int(vec![1; n])),
            (
                "score",
                Column::Float(
                    (0..n)
                        .map(|i| if i % 5 == 0 { f64::NAN } else { i as f64 })
                        .collect(),
                ),
            ),
            (
                "age",
                Column::Float((0..n).map(|i| (i * 7 % 50) as f64).collect()),
            ),
            (
                "rare",
                Column::Float(
                    (0..n)
                        .map(|i| if i % 6 == 0 { i as f64 } else { f64::NAN })
                        .collect(),
                ),
            ),
        ];
        let (names, cols) = cols.into_iter().map(|(n, c)| (n.to_owned(), c)).unzip();
        Table::new(names, cols).unwrap()
    }

    #[test]
    fn the_summary_counts_columns_and_never_shows_a_level() {
        let t = table();
        let plan = draft_plan(&t, "ds_aaaaaaaaaaaa", &DraftOptions::default()).unwrap();
        let names = processed_feature_names(&t, &plan).unwrap();
        let s = build_preprocessing_summary(&plan, None, Some(&names), 8);

        assert_eq!(
            s["counts"],
            json!({"kept": 4, "dropped": 3, "encoded": 2, "imputed": 2})
        );
        assert_eq!(
            s["dropped"],
            json!([
                {"feature": 0, "column": "pid", "reason": "identifier"},
                {"feature": 3, "column": "site", "reason": "constant"},
                {"feature": 6, "column": "rare", "reason": "missing"},
            ])
        );
        // ward: three levels plus its blank category; ward_type's dummies are
        // not credited to ward although they start with "ward_".
        assert_eq!(
            s["encoded"],
            json!([
                {"feature": 1, "column": "ward", "method": "one_hot", "levels": 4},
                {"feature": 2, "column": "ward_type", "method": "one_hot", "levels": 2},
            ])
        );
        assert_eq!(s["output_features"], json!(names.len()));
        // score is blank in 8 of 36 rows: 0.22 to 2 dp.
        let score = s["imputed"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["column"] == "score")
            .unwrap();
        assert_eq!(score["feature"], 4);
        assert_eq!(score["missing_fraction"], 0.22);
        assert_eq!(s["changes_from_previous"], json!([]));
        let text = s.to_string();
        for level in ["north", "south", "east", "acute", "rehab", "(blank)"] {
            assert!(!text.contains(level), "{level} in {text}");
        }
    }

    #[test]
    fn the_default_plan_drops_a_row_index_as_an_identifier() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/golden/penguins/input.csv");
        let t = pulsar_data::read_table(&path).unwrap();
        let plan = draft_plan(&t, "ds_aaaaaaaaaaaa", &DraftOptions::default()).unwrap();
        assert_eq!(plan.entries["rowid"].role, Role::Id);
        assert_eq!(plan.entries["rowid"].decided_by, DecidedBy::Default);
        let s = build_preprocessing_summary(&plan, None, None, 8);
        assert!(
            s["dropped"]
                .as_array()
                .unwrap()
                .contains(&json!({"feature": 0, "column": "rowid", "reason": "identifier"})),
            "{}",
            s["dropped"]
        );
        assert_eq!(plan.entries["bill_length_mm"].role, Role::Feature);
    }

    #[test]
    fn every_drop_reason_is_a_listed_code() {
        let t = table();
        let mut plan = draft_plan(&t, "ds_aaaaaaaaaaaa", &DraftOptions::default()).unwrap();
        let previous = plan.clone();
        let age = plan.entries.get_mut("age").unwrap();
        age.role = Role::Drop;
        age.decided_by = DecidedBy::User;
        let s = build_preprocessing_summary(&plan, Some(&previous), None, 8);
        for d in s["dropped"].as_array().unwrap() {
            assert!(DROP_REASONS.contains(&d["reason"].as_str().unwrap()), "{d}");
        }
        assert_eq!(
            s["dropped"][2],
            json!({"feature": 5, "column": "age", "reason": "user"})
        );
        assert_eq!(
            s["changes_from_previous"],
            json!([{"feature": 5, "column": "age", "field": "role", "to": "drop"}])
        );
        assert!(s["output_features"].is_null());
    }
}
