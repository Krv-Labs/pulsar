//! Per-column default entries, Rules 1-8 of `draft_plan`
//! (`preprocessing_plan/defaults.py` and `context.py`).

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;
use pulsar_data::{Column, Table};
use serde_json::{json, Map, Value};

use super::model::{DecidedBy, EntryStatus, ImputeSpec, PlanEditError, PlanEntry, Role};
use super::text::{condition, format_missing_reason, plural, py_repr, py_repr_list, thousands};
use crate::evidence::{ColumnEvidence, ColumnKind, PreprocessingEvidence};

/// Shared state the default rules fill in and the question builders read.
pub struct DraftContext<'a> {
    /// The raw table; only a few question builders read values from it.
    pub table: &'a Table,
    pub evidence: &'a PreprocessingEvidence,
    pub ev_by_col: HashMap<&'a str, &'a ColumnEvidence>,
    /// `mg1 -> [columns]` in group order.
    pub missing_groups: IndexMap<String, Vec<String>>,
    pub target_set: HashSet<String>,
    pub entity_col: String,
    pub time_col: String,
    pub n_rows: usize,
    /// `sqrt(max(1, n_rows))`: most distinct text values a column may have
    /// before one-hot encoding stops being reasonable.
    pub cat_threshold: f64,
    pub entries: IndexMap<String, PlanEntry>,
    pub date_cols: Vec<String>,
    pub mixed_value_cols: Vec<String>,
    pub conditional_by_partner: IndexMap<String, Vec<String>>,
    pub rule7b_cols: Vec<String>,
    pub grouped_7b: IndexMap<String, Vec<String>>,
    /// Columns in a blank group asked about by `blank_meaning`; later
    /// builders skip them.
    pub blank_group_cols: HashSet<String>,
    /// Give row-index columns ([`row_index_like`]) the identifier role.
    /// Python kept them as features; on by default.
    pub row_index_ids: bool,
}

impl<'a> DraftContext<'a> {
    pub fn new(
        table: &'a Table,
        evidence: &'a PreprocessingEvidence,
        target_columns: &[String],
        entity_col: &str,
        time_col: &str,
    ) -> Self {
        Self {
            table,
            evidence,
            blank_group_cols: HashSet::new(),
            ev_by_col: evidence
                .columns
                .iter()
                .map(|c| (c.name.as_str(), c))
                .collect(),
            missing_groups: evidence
                .missing_groups
                .iter()
                .map(|g| (g.group_id.clone(), g.columns.clone()))
                .collect(),
            target_set: target_columns.iter().cloned().collect(),
            entity_col: entity_col.into(),
            time_col: time_col.into(),
            n_rows: evidence.n_rows,
            cat_threshold: (evidence.n_rows.max(1) as f64).sqrt(),
            entries: IndexMap::new(),
            date_cols: Vec::new(),
            mixed_value_cols: Vec::new(),
            conditional_by_partner: IndexMap::new(),
            rule7b_cols: Vec::new(),
            grouped_7b: IndexMap::new(),
            row_index_ids: true,
        }
    }
}

fn invalid_argument(message: String, entries: Vec<(&str, Value)>) -> PlanEditError {
    let details: Map<String, Value> = entries
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    PlanEditError::new("INVALID_ARGUMENT", message, details)
}

/// Check the role arguments against the column names and normalize them:
/// `(targets, entity, time)`.
pub fn validate_role_designations(
    columns: &[String],
    target_columns: &[String],
    entity_column: &str,
    time_column: &str,
) -> Result<(Vec<String>, String, String), PlanEditError> {
    let available: Vec<&String> = columns.iter().take(20).collect();
    let known: HashSet<&str> = columns.iter().map(String::as_str).collect();
    let targets: Vec<String> = target_columns
        .iter()
        .filter(|c| !c.is_empty())
        .cloned()
        .collect();
    let entity = entity_column.trim().to_string();
    let time = time_column.trim().to_string();

    for (label, name) in [("entity_column", &entity), ("time_column", &time)] {
        if !name.is_empty() && !known.contains(name.as_str()) {
            return Err(invalid_argument(
                format!("Unknown {label}: {}.", py_repr(name)),
                vec![
                    (label, json!(name)),
                    ("available_columns", json!(available)),
                ],
            ));
        }
    }
    let unknown: Vec<&String> = targets
        .iter()
        .filter(|c| !known.contains(c.as_str()))
        .collect();
    if !unknown.is_empty() {
        return Err(invalid_argument(
            format!("Unknown target_columns: {}.", py_repr_list(&unknown)),
            vec![
                ("unknown_target_columns", json!(unknown)),
                ("available_columns", json!(available)),
            ],
        ));
    }
    Ok((targets, entity, time))
}

fn resolved(column: &str, role: Role, reason: String) -> PlanEntry {
    PlanEntry::new(column, role, EntryStatus::Resolved, reason)
}

fn designated(column: &str, role: Role, what: &str) -> PlanEntry {
    let mut e = resolved(column, role, format!("designated as {what} column"));
    e.decided_by = DecidedBy::Agent;
    e
}

/// Indicator for a column with blanks: its missing group, else its own.
fn blank_indicator(ev: &ColumnEvidence) -> String {
    match &ev.missing_group {
        Some(g) => format!("group:{g}"),
        None => "own".into(),
    }
}

fn blank_word(n: usize) -> &'static str {
    plural(n, "blank", "blanks")
}

/// Rules 5 and 7: a column blank exactly where a 0/1 partner has one value
/// is a condition of that partner, so it is held out until asked about.
fn conditional_entry(col: &str, ev: &ColumnEvidence, ctx: &mut DraftContext) -> Option<PlanEntry> {
    let md = ev.mask_dependency.as_ref()?;
    if ctx.ev_by_col.get(md.partner.as_str())?.kind != ColumnKind::Boolean {
        return None;
    }
    ctx.conditional_by_partner
        .entry(md.partner.clone())
        .or_default()
        .push(col.to_string());
    Some(PlanEntry::new(
        col,
        Role::Drop,
        EntryStatus::Open,
        format!(
            "{} {}; blank on the exact {} where {}",
            thousands(ev.n_missing),
            blank_word(ev.n_missing),
            plural(ev.n_missing, "row", "rows"),
            condition(&md.partner, md.relation.as_str())
        ),
    ))
}

/// Rule 5a: numeric text with non-numeric tokens is never silently coerced.
fn mixed_numeric_text_entry(
    col: &str,
    ev: &ColumnEvidence,
    ctx: &mut DraftContext,
    n_non_num: usize,
) -> PlanEntry {
    let top = ev
        .non_numeric_values
        .as_ref()
        .and_then(|v| v.first())
        .map_or("non-numeric text", |(t, _)| t.as_str());
    let rows = plural(n_non_num, "row", "rows");
    let described = format!(
        "mostly numbers with {} ({} {rows}; {} {})",
        py_repr(top),
        thousands(n_non_num),
        thousands(ev.n_missing),
        blank_word(ev.n_missing)
    );
    ctx.mixed_value_cols.push(col.to_string());

    if ev.n_unique as f64 <= ctx.cat_threshold {
        let mut e = PlanEntry::new(
            col,
            Role::Feature,
            EntryStatus::Open,
            format!("{described}; kept as {} categories", thousands(ev.n_unique)),
        );
        e.impute = (ev.n_missing > 0).then(|| ImputeSpec::fill_value("(blank)"));
        e.encode = Some("one_hot".into());
        return e;
    }
    let mut e = PlanEntry::new(
        col,
        Role::Feature,
        EntryStatus::Open,
        format!("{described}; non-numeric text treated as blank with median fill and blank marker"),
    );
    e.impute = Some(ImputeSpec::method("fill_median"));
    e.indicator = blank_indicator(ev);
    e
}

/// Rule 5: numeric text.
fn numeric_text_entry(col: &str, ev: &ColumnEvidence, ctx: &mut DraftContext) -> PlanEntry {
    let n_non_num = ev.n_non_numeric.unwrap_or(0);
    if n_non_num > 0 {
        return mixed_numeric_text_entry(col, ev, ctx, n_non_num);
    }
    if let Some(entry) = conditional_entry(col, ev, ctx) {
        return entry;
    }
    if ev.n_missing > 0 {
        ctx.rule7b_cols.push(col.to_string());
    }
    let mut e = resolved(
        col,
        Role::Feature,
        format!(
            "numeric text ({} {}); coerced to numbers with median fill",
            thousands(ev.n_missing),
            blank_word(ev.n_missing)
        ),
    );
    e.impute = Some(ImputeSpec::method("fill_median"));
    if ev.n_missing > 0 {
        e.indicator = blank_indicator(ev);
    }
    e
}

/// Rule 6: text categorical.
fn text_entry(col: &str, ev: &ColumnEvidence, ctx: &DraftContext) -> PlanEntry {
    if ev.n_unique as f64 > ctx.cat_threshold {
        return resolved(
            col,
            Role::Drop,
            format!(
                "{} distinct text categories (exceeds sqrt({}) = {}); would add {} columns",
                thousands(ev.n_unique),
                thousands(ctx.n_rows),
                ctx.cat_threshold as usize,
                thousands(ev.n_unique)
            ),
        );
    }
    let reason = if ev.n_missing > 0 {
        format!(
            "text column with {} categories and {} {}; blanks become '(blank)' category",
            thousands(ev.n_unique),
            thousands(ev.n_missing),
            blank_word(ev.n_missing)
        )
    } else {
        format!(
            "complete text column with {} categories",
            thousands(ev.n_unique)
        )
    };
    let mut e = resolved(col, Role::Feature, reason);
    e.impute = (ev.n_missing > 0).then(|| ImputeSpec::fill_value("(blank)"));
    e.encode = Some("one_hot".into());
    e
}

/// Rules 7 and 8: numeric or boolean.
fn numeric_or_boolean_entry(col: &str, ev: &ColumnEvidence, ctx: &mut DraftContext) -> PlanEntry {
    if ev.n_missing == 0 {
        return resolved(
            col,
            Role::Feature,
            format!(
                "complete {} column ({} distinct values)",
                ev.kind.as_str(),
                thousands(ev.n_unique)
            ),
        );
    }
    if let Some(entry) = conditional_entry(col, ev, ctx) {
        return entry;
    }
    ctx.rule7b_cols.push(col.to_string());
    let mut e = resolved(
        col,
        Role::Feature,
        format_missing_reason(ev, &ctx.missing_groups, &ctx.ev_by_col),
    );
    e.impute = Some(ImputeSpec::method("fill_median"));
    e.indicator = blank_indicator(ev);
    e
}

/// Rules 1-2: designated roles, all-blank columns and constants.
fn explicit_or_constant_entry(
    col: &str,
    ev: &ColumnEvidence,
    ctx: &DraftContext,
) -> Option<PlanEntry> {
    if col == ctx.entity_col {
        return Some(designated(col, Role::Entity, "entity"));
    }
    if col == ctx.time_col {
        return Some(designated(col, Role::Time, "time"));
    }
    if ctx.target_set.contains(col) {
        return Some(designated(col, Role::Target, "target"));
    }
    if ev.n_missing >= ev.n_rows && ev.n_rows > 0 {
        return Some(resolved(
            col,
            Role::Drop,
            format!("all {} rows are blank", thousands(ev.n_rows)),
        ));
    }
    if ev.is_constant {
        return Some(resolved(
            col,
            Role::Drop,
            "one distinct value; adds no distance information".into(),
        ));
    }
    None
}

/// Whether a column is a row index rather than a measurement: a row-id name
/// (`rowid`, `row_id`, `index`, `Unnamed: 0`, blank) or integers counting
/// the rows from 0 or 1.
pub fn row_index_like(name: &str, column: &Column) -> bool {
    let key: String = name
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect::<String>()
        .to_ascii_lowercase();
    if matches!(
        key.as_str(),
        "" | "rowid" | "rownum" | "rownumber" | "index" | "unnamed0"
    ) {
        return true;
    }
    let Column::Int(values) = column else {
        return false;
    };
    let counts_from = |start: i64| (start..).zip(values).all(|(i, &v)| i == v);
    values.len() > 1 && (counts_from(0) || counts_from(1))
}

fn default_entry(col: &str, ev: &ColumnEvidence, ctx: &mut DraftContext) -> PlanEntry {
    if let Some(entry) = explicit_or_constant_entry(col, ev, ctx) {
        return entry;
    }
    if ctx.row_index_ids
        && ctx
            .table
            .column(col)
            .is_some_and(|c| row_index_like(col, c))
    {
        return resolved(col, Role::Id, "row index, not a measurement".into());
    }
    let n_observed = ev.n_rows - ev.n_missing;
    if ev.kind == ColumnKind::Text
        && ev.unique_ratio == 1.0
        && ev.n_unique == n_observed
        && n_observed > 1
    {
        return resolved(
            col,
            Role::Id,
            format!(
                "unique text value per observed row ({} distinct values)",
                thousands(ev.n_unique)
            ),
        );
    }
    match ev.kind {
        ColumnKind::DatetimeLike => {
            ctx.date_cols.push(col.to_string());
            PlanEntry::new(
                col,
                Role::Drop,
                EntryStatus::Open,
                format!(
                    "date/time text ({} distinct values; {} {})",
                    thousands(ev.n_unique),
                    thousands(ev.n_missing),
                    blank_word(ev.n_missing)
                ),
            )
        }
        ColumnKind::NumericText => numeric_text_entry(col, ev, ctx),
        ColumnKind::Text => text_entry(col, ev, ctx),
        ColumnKind::Numeric | ColumnKind::Boolean => numeric_or_boolean_entry(col, ev, ctx),
    }
}

/// Fill `ctx.entries` and the rule-specific column lists for every measured
/// column, in column order.
pub fn populate_default_entries(ctx: &mut DraftContext) {
    let evidence = ctx.evidence;
    for ev in &evidence.columns {
        let entry = default_entry(&ev.name, ev, ctx);
        ctx.entries.insert(ev.name.clone(), entry);
    }
    for col in ctx.rule7b_cols.clone() {
        let gid = ctx.ev_by_col[col.as_str()]
            .missing_group
            .clone()
            .unwrap_or_else(|| format!("singleton:{col}"));
        ctx.grouped_7b.entry(gid).or_default().push(col);
    }
}
