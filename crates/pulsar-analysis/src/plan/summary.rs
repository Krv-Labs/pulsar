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

use super::model::{DecidedBy, EntryStatus, PlanEntry, QuestionStatus, Role};
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
fn summary_decided_by(plan: &PreprocessingPlan, e: &PlanEntry) -> &'static str {
    match e.decided_by {
        DecidedBy::Default => "default",
        DecidedBy::Agent => "agent",
        DecidedBy::User => "user",
        DecidedBy::Phirewall => {
            if e.status == EntryStatus::Open {
                "default"
            } else if plan
                .imported
                .as_ref()
                .and_then(|imp| imp.column(&e.column))
                .is_some_and(|c| c.decided_by == "clinician")
            {
                "user"
            } else {
                "agent"
            }
        }
    }
}

fn drop_reason(
    plan: &PreprocessingPlan,
    entry: &PlanEntry,
    ev_by_col: &HashMap<&str, &ColumnEvidence>,
    cat_threshold: f64,
) -> &'static str {
    if matches!(entry.role, Role::Entity | Role::Time | Role::Target) {
        return "designated";
    }
    if entry.role == Role::Id && entry.decided_by == DecidedBy::Phirewall {
        return "identifier";
    }
    match entry.decided_by {
        DecidedBy::User => return "user",
        DecidedBy::Agent => return "agent",
        DecidedBy::Phirewall if entry.status == EntryStatus::Resolved => {
            return summary_decided_by(plan, entry);
        }
        DecidedBy::Phirewall => {
            if plan
                .imported
                .as_ref()
                .and_then(|imp| imp.column(&entry.column))
                .is_some_and(|c| {
                    c.meaning.as_deref() == Some("date") || c.op.as_deref() == Some("date_format")
                })
            {
                return "datetime";
            }
        }
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

/// The share of rows where `col` is blank, or `None` without evidence.
/// Every blank share the summary reports comes through here, so a
/// disclosure rule for shares (share × rows is a row count) applies in one
/// place.
fn blank_share(ev_by_col: &HashMap<&str, &ColumnEvidence>, col: &str) -> Option<f64> {
    ev_by_col
        .get(col)
        .map(|ev| row_fraction(ev.missing_fraction))
}

/// `{partner, relation}` when `col` is blank exactly where a 0/1 partner
/// column has one value: the partner named as a plan column, and the
/// relation as its code (`missing_when_partner_zero` or `_nonzero`).
fn partner(
    plan: &PreprocessingPlan,
    ev_by_col: &HashMap<&str, &ColumnEvidence>,
    col: &str,
) -> Option<Value> {
    let md = ev_by_col.get(col)?.mask_dependency.as_ref()?;
    let is_flag = ev_by_col
        .get(md.partner.as_str())
        .is_some_and(|p| p.kind == ColumnKind::Boolean);
    if !is_flag {
        return None;
    }
    Some(json!({"partner": column(plan, &md.partner)?, "relation": md.relation.as_str()}))
}

/// The open question `e` belongs to, by id, if it is still open.
fn open_question<'a>(plan: &'a PreprocessingPlan, e: &PlanEntry) -> Option<&'a str> {
    let id = e.question_id.as_deref()?;
    plan.questions
        .values()
        .find(|q| q.question_id == id && q.status == QuestionStatus::Open)
        .map(|q| q.question_id.as_str())
}

/// The fields every column's line carries beside its own: its blank share,
/// who decided its treatment, its open question, and for a column blank
/// where a 0/1 partner has one value, that partner.
fn common(
    plan: &PreprocessingPlan,
    ev_by_col: &HashMap<&str, &ColumnEvidence>,
    e: &PlanEntry,
) -> Value {
    let mut out = json!({
        "missing_fraction": blank_share(ev_by_col, &e.column),
        "decided_by": summary_decided_by(plan, e),
    });
    if let Some(decision) = plan
        .imported
        .as_ref()
        .and_then(|imp| imp.summary_decision(&e.column))
    {
        out["decision"] = decision;
    }
    if let Some(id) = open_question(plan, e) {
        out["question"] = json!(id);
    }
    if let Some(Value::Object(p)) = partner(plan, ev_by_col, &e.column) {
        out.as_object_mut().expect("an object").extend(p);
    }
    out
}

/// True iff a column's [`common`] fields say nothing beyond the default
/// rules: decided by them, with no settled decision, open question or 0/1
/// partner. Such a line repeats what `counts` already says.
fn is_default(fields: &Value) -> bool {
    fields["decided_by"] == "default"
        && ["decision", "question", "partner"]
            .iter()
            .all(|k| fields.get(k).is_none())
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
/// compared by method only, so a fill value never appears. Markers are
/// compared as written (`group:a` to `group:b` is a change) but shown as
/// their code.
fn changes(previous: &PreprocessingPlan, plan: &PreprocessingPlan) -> Vec<Value> {
    // (field, value compared, value shown)
    let fields = |e: &PlanEntry| {
        [
            ("role", json!(e.role), json!(e.role)),
            ("impute", json!(e.impute_method()), json!(e.impute_method())),
            ("encode", json!(e.encode), json!(e.encode)),
            (
                "indicator",
                json!(e.indicator),
                json!(indicator_code(&e.indicator)),
            ),
        ]
    };
    let mut out = Vec::new();
    for (col, old) in &previous.entries {
        let Some(new) = plan.entries.get(col) else {
            continue;
        };
        for ((field, from, _), (_, to, shown)) in fields(old).into_iter().zip(fields(new)) {
            if from != to {
                out.push(entry(
                    plan,
                    col,
                    json!({"field": field, "to": shown, "by": summary_decided_by(plan, new)}),
                ));
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
/// one replaced, and is empty without one. `kept` and `imputed` list only
/// columns that differ from the default treatment (see [`is_default`]);
/// `kept_omitted` and `imputed_omitted` count the rest, and `counts` stay
/// exact. A wide table otherwise sends one line per column twice, which
/// alone can fill an answer. The other lists are whole, so their
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
            let reason = drop_reason(plan, e, &ev_by_col, cat_threshold);
            let mut fields = common(plan, &ev_by_col, e);
            fields["reason"] = json!(reason);
            fields["role"] = json!(e.role);
            entry(plan, &e.column, fields)
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
    let imputed_all: Vec<(&&PlanEntry, &str)> = features
        .iter()
        .filter_map(|e| Some((e, e.impute_method()?)))
        .collect();
    let imputed_count = imputed_all.len();
    let imputed: Vec<Value> = imputed_all
        .into_iter()
        .filter(|(e, _)| !is_default(&common(plan, &ev_by_col, e)))
        .map(|(e, method)| {
            let fraction = blank_share(&ev_by_col, &e.column);
            entry(
                plan,
                &e.column,
                json!({"missing_fraction": fraction, "method": method}),
            )
        })
        .collect();
    // Features treated otherwise than by default, with their blank share and
    // blank marker.
    let kept: Vec<Value> = features
        .iter()
        .filter_map(|e| {
            let mut fields = common(plan, &ev_by_col, e);
            if is_default(&fields) {
                return None;
            }
            fields["indicator"] = json!(indicator_code(&e.indicator));
            Some(entry(plan, &e.column, fields))
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
                "id": q.question_id,
                "default": q.default_option_id,
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
            "imputed": imputed_count,
        },
        "kept_omitted": features.len() - kept.len(),
        "kept": kept,
        "dropped": dropped,
        "dropped_omitted": 0,
        "encoded": encoded,
        "encoded_omitted": 0,
        "imputed_omitted": imputed_count - imputed.len(),
        "imputed": imputed,
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
                {"feature": 0, "column": "pid", "reason": "identifier", "role": "id",
                 "missing_fraction": 0.0, "decided_by": "default"},
                {"feature": 3, "column": "site", "reason": "constant", "role": "drop",
                 "missing_fraction": 0.0, "decided_by": "default"},
                {"feature": 6, "column": "rare", "reason": "missing", "role": "drop",
                 "missing_fraction": 0.83, "decided_by": "default", "question": "q2"},
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
        // Only score, which an open question decides, differs from the
        // default rules; the omitted counts carry the rest.
        assert_eq!(s["kept_omitted"], 3);
        assert_eq!(s["imputed_omitted"], 1);
        // score is blank in 8 of 36 rows: 0.22 to 2 dp.
        let score = s["imputed"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["column"] == "score")
            .unwrap();
        assert_eq!(score["feature"], 4);
        assert_eq!(score["missing_fraction"], 0.22);
        // Each kept line carries its blank share and marker.
        let kept = s["kept"].as_array().unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0]["question"], "q1");
        let kept_score = kept.iter().find(|e| e["column"] == "score").unwrap();
        assert_eq!(kept_score["missing_fraction"], 0.22);
        assert!(kept_score["indicator"].is_string(), "{kept_score}");
        for q in s["open_questions"].as_array().unwrap() {
            assert!(q["id"].is_string() && q["default"].is_string(), "{q}");
        }
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
            s["dropped"].as_array().unwrap().contains(
                &json!({"feature": 0, "column": "rowid", "reason": "identifier",
                    "role": "id", "missing_fraction": 0.0, "decided_by": "default"})
            ),
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
        assert_eq!(s["dropped"][2]["reason"], "user");
        assert_eq!(s["dropped"][2]["decided_by"], "user");
        assert_eq!(
            s["changes_from_previous"],
            json!([{"feature": 5, "column": "age", "field": "role", "to": "drop", "by": "user"}])
        );
        assert!(s["output_features"].is_null());
    }

    #[test]
    fn a_marker_moved_between_groups_is_a_change_shown_as_its_code() {
        let t = table();
        let mut previous = draft_plan(&t, "ds_aaaaaaaaaaaa", &DraftOptions::default()).unwrap();
        previous.entries.get_mut("score").unwrap().indicator = "group:mg1".into();
        let mut plan = previous.clone();
        plan.entries.get_mut("score").unwrap().indicator = "group:mg2".into();
        let s = build_preprocessing_summary(&plan, Some(&previous), None, 8);
        assert_eq!(
            s["changes_from_previous"],
            json!([{"feature": 4, "column": "score", "field": "indicator", "to": "group",
                    "by": "default"}])
        );
        assert!(!s.to_string().contains("mg"), "{s}");
    }

    /// A column blank exactly where a 0/1 flag is false names the flag and
    /// how the two relate, whoever dropped it.
    #[test]
    fn a_column_blank_unless_a_flag_is_set_names_the_flag() {
        let n = 40;
        let died: Vec<Option<bool>> = (0..n).map(|i| Some(i % 4 == 0)).collect();
        let death_age = (0..n)
            .map(|i| {
                if i % 4 == 0 {
                    60.0 + i as f64
                } else {
                    f64::NAN
                }
            })
            .collect();
        let score = (0..n).map(|i| i as f64).collect();
        let t = Table::new(
            vec!["died".into(), "death_age".into(), "score".into()],
            vec![
                Column::Bool(died),
                Column::Float(death_age),
                Column::Float(score),
            ],
        )
        .unwrap();
        let plan = draft_plan(&t, "ds_aaaaaaaaaaaa", &DraftOptions::default()).unwrap();
        let s = build_preprocessing_summary(&plan, None, None, 8);
        let lines: Vec<&Value> = s["kept"]
            .as_array()
            .unwrap()
            .iter()
            .chain(s["dropped"].as_array().unwrap())
            .collect();
        let death_age = lines.iter().find(|e| e["column"] == "death_age").unwrap();
        assert_eq!(death_age["partner"]["column"], "died", "{death_age}");
        assert_eq!(death_age["relation"], "missing_when_partner_zero");
        // died and score follow the default rules: counted, not listed.
        assert!(lines.iter().all(|e| e["column"] != "score"));
        assert_eq!(s["kept_omitted"], 2);
    }
}
