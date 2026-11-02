//! `group_summary`: a compact, data-agnostic orientation report on a
//! grouping. It names the few columns that separate the groups most and how
//! the largest groups differ from all rows on them, as standardized
//! differences and counts only.
//!
//! Definitions:
//!
//! * **sd**, one standardization for every kind of column: (group mean −
//!   all-rows mean) / all-rows population SD (ddof 0). A categorical or
//!   boolean level is its 0/1 indicator, so its mean is the level's
//!   proportion `p` and its SD is `sqrt(p(1−p))`. Missing cells (`NaN`,
//!   infinite numbers, empty or whitespace-only text) are excluded from every mean, SD and
//!   proportion; there is no "missing" level. All-rows statistics use every
//!   row of the table, whatever its group.
//! * **separation**: η², the share of the column's variance that lies
//!   between groups: the between-group sum of squares of the groups of at
//!   least `min_rows` rows over the column's total sum of squares across all
//!   rows. Smaller groups' rows count toward the total but not between, so a
//!   shattered grouping (every row its own group) separates nothing instead
//!   of everything. A categorical column scores as its most separating level
//!   (the max over its levels' indicators), so it counts once. Only columns
//!   used for grouping can be defining features, and only with a separation
//!   above 0 at three decimals; a constant column scores 0.
//! * **kind**: integer and float columns are `numeric`, boolean and text
//!   columns `categorical` (ordinal text is categorical); either is `binary`
//!   when it holds exactly two distinct values. A numeric binary column is
//!   standardized as a number and names no level.
//!
//! Privacy: a categorical level backed by fewer than `min_rows` rows over
//! all rows is never emitted nor scored, and a cell's level is only emitted
//! when at least `min_rows` of the group's rows hold it, so a level a group
//! lacks is withheld and the cell falls to its next strongest level. A
//! numeric cell needs `min_rows` observed values in the group. Groups under
//! `min_rows` rows are only counted. No raw value is emitted, and no
//! non-finite number is serialized.
//!
//! Row counts (each shown group's `size`, `rows` and `small_groups.rows`)
//! are rounded to the nearest 5, halves up, as Phirewall rounds every count
//! it discloses (`COUNT_ROUNDING`, `round_count`), so subtracting one from
//! another cannot recover a small group's size exactly. A nonzero count
//! under `min_rows` is not rounded but withheld as `"under <min_rows>"`.
//! Counts of groups are exact. Column names are never interpreted.
//!
//! Size: the summary is as large as the thresholds asked for. Entries
//! follow them (`max_groups` × `max_features` cells, `max_traits` per
//! group, at most 20 unused names) and there is no cap on them here:
//! `max_groups` is applied as at most the group count and `max_features` to the
//! number of grouping columns, which is just "all of them". Every string
//! emitted (column names, levels, the method) is cut to [`MAX_LABEL_CHARS`]
//! characters on a char boundary, ending in `…`; the entry holding a cut
//! name or level is marked `truncated` (a name in `not_used_for_grouping`
//! is only cut). So a string serializes to at most 485 bytes (80
//! characters escaped as `\u00XX`, the ellipsis and quotes), and the
//! worst case is proportional to the requested counts: about 66 KB at the
//! defaults, growing with `max_groups` × `max_features` cells and
//! `max_traits` per group; plain-text names and levels make it a fraction
//! of that. `size_chars` reports the actual serialized length, so a caller
//! can see what drove an oversized answer.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};

use pulsar_data::{Column, Table};
use serde::Serialize;

use crate::pyround::round_to;

/// The report's bounds. The defaults are [`GROUP_SUMMARY_DEFAULTS`]; the
/// thresholds applied are echoed in the output.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct GroupSummaryThresholds {
    /// Groups listed, largest first; the rest are one `small_groups` count.
    pub max_groups: usize,
    /// Defining features, strongest first: the only columns with cells.
    pub max_features: usize,
    /// |sd| below this is not called a difference.
    pub min_sd: f64,
    /// Groups and categorical levels backed by fewer rows are withheld.
    pub min_rows: usize,
    /// Strongest differences listed per group.
    pub max_traits: usize,
}

pub const GROUP_SUMMARY_DEFAULTS: GroupSummaryThresholds = GroupSummaryThresholds {
    max_groups: 6,
    max_features: 5,
    min_sd: 0.5,
    min_rows: 8,
    max_traits: 3,
};

/// Columns named in `not_used_for_grouping`; the rest are only counted.
const MAX_LISTED_UNUSED: usize = 20;

/// Longest column name or level emitted, in characters, before the `…`.
pub const MAX_LABEL_CHARS: usize = 80;

/// `s` cut to [`MAX_LABEL_CHARS`] characters plus `…`, and whether it was.
fn label(s: &str) -> (String, bool) {
    match s.char_indices().nth(MAX_LABEL_CHARS) {
        Some((end, _)) => (format!("{}…", &s[..end]), true),
        None => (s.to_owned(), false),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureKind {
    Numeric,
    Categorical,
    Binary,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Trait {
    pub column: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<String>,
    pub sd: f64,
    /// The column name or level was cut to `MAX_LABEL_CHARS`.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GroupEntry {
    pub id: i64,
    /// Rounded to the nearest 5.
    pub size: usize,
    pub traits: Vec<Trait>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SmallGroups {
    pub count: usize,
    pub rows: RowCount,
}

/// A disclosed row count: 0, or rounded to the nearest 5, or withheld
/// (`"under <min_rows>"`) when under `min_rows`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum RowCount {
    Rounded(usize),
    Withheld(String),
}

impl RowCount {
    fn of(rows: usize, min_rows: usize) -> Self {
        if rows == 0 || rows >= min_rows {
            Self::Rounded(round_count(rows))
        } else {
            Self::Withheld(format!("under {min_rows}"))
        }
    }
}

/// `count` rounded to the nearest 5, halves up: Phirewall's `round_count`.
fn round_count(count: usize) -> usize {
    (count + 2) / 5 * 5
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DefiningFeature {
    pub column: String,
    pub kind: FeatureKind,
    pub separation: f64,
    /// The column name was cut to `MAX_LABEL_CHARS`.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Cell {
    pub group: i64,
    pub column: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<String>,
    pub sd: f64,
    /// The column name or level was cut to `MAX_LABEL_CHARS`.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GroupSummary {
    pub thresholds: GroupSummaryThresholds,
    pub method: String,
    pub cut_off: Option<f64>,
    pub group_count: usize,
    pub rows: RowCount,
    pub groups: Vec<GroupEntry>,
    pub small_groups: SmallGroups,
    pub defining_features: Vec<DefiningFeature>,
    pub cells: Vec<Cell>,
    pub not_used_for_grouping: Vec<String>,
    pub not_used_for_grouping_omitted: usize,
    /// The length in characters of this summary serialized as JSON
    /// (`serde_json::to_string`), this field included.
    pub size_chars: usize,
}

/// One standardized axis of a column: the column itself (numeric) or one
/// level's 0/1 indicator.
struct Axis {
    level: Option<String>,
    /// All-rows mean in the units of `group_sum`: 0 for a numeric axis,
    /// whose sums are of deviations from the mean (which keeps the
    /// between-group sum of squares free of cancellation), `p` for a level.
    mean: f64,
    sd: f64,
    /// Per group: the sum of deviations (numeric) or the level's count.
    group_sum: HashMap<i64, f64>,
}

struct ColumnStats<'a> {
    name: &'a str,
    kind: FeatureKind,
    /// Observed rows over all rows.
    n: usize,
    /// Observed rows per group.
    group_n: HashMap<i64, usize>,
    axes: Vec<Axis>,
}

impl ColumnStats<'_> {
    fn sd_of(&self, axis: &Axis, group: i64) -> Option<f64> {
        let n = *self.group_n.get(&group)? as f64;
        let sum = axis.group_sum.get(&group).copied().unwrap_or(0.0);
        let sd = (sum / n - axis.mean) / axis.sd;
        sd.is_finite().then_some(sd)
    }

    /// Rows behind `axis` in `group`: observed values, or rows holding the level.
    fn backing(&self, axis: &Axis, group: i64) -> usize {
        match axis.level {
            None => self.group_n.get(&group).copied().unwrap_or(0),
            Some(_) => axis.group_sum.get(&group).copied().unwrap_or(0.0) as usize,
        }
    }

    /// η² of one axis over the `eligible` groups. With `s` a group's sum and
    /// `m` the mean, Σ n·(s/n − m)² = Σ s²/n − 2m·Σ s + m²·Σ n, which only
    /// needs the groups where the axis has a nonzero sum.
    /// `eligible_n` is the column's observed rows in the eligible groups.
    fn separation(&self, axis: &Axis, eligible: &HashSet<i64>, eligible_n: usize) -> f64 {
        let (mut squares, mut sums) = (0.0, 0.0);
        for (g, &s) in &axis.group_sum {
            if eligible.contains(g) {
                squares += s * s / self.group_n[g] as f64;
                sums += s;
            }
        }
        let m = axis.mean;
        let between = squares - 2.0 * m * sums + m * m * eligible_n as f64;
        let total = self.n as f64 * axis.sd * axis.sd;
        let eta2 = (between / total).clamp(0.0, 1.0);
        if eta2.is_finite() {
            eta2
        } else {
            0.0
        }
    }

    /// Drop the per-group statistics of every group not in `groups`.
    fn keep_groups(&mut self, groups: &HashSet<i64>) {
        self.group_n.retain(|g, _| groups.contains(g));
        for axis in &mut self.axes {
            axis.group_sum.retain(|g, _| groups.contains(g));
        }
    }

    /// The strongest emittable difference of `group`, positive on a tie.
    fn cell(&self, group: i64, min_rows: usize) -> Option<(Option<&str>, f64)> {
        self.axes
            .iter()
            .filter(|a| self.backing(a, group) >= min_rows)
            .filter_map(|a| Some((a.level.as_deref(), self.sd_of(a, group)?)))
            .max_by(|(_, x), (_, y)| x.abs().total_cmp(&y.abs()).then(x.total_cmp(y)))
    }
}

/// A scored column, ordered by separation, then the earlier column first.
struct Ranked<'a> {
    separation: f64,
    index: usize,
    stats: ColumnStats<'a>,
}

impl Ord for Ranked<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.separation
            .total_cmp(&other.separation)
            .then(other.index.cmp(&self.index))
    }
}

impl PartialOrd for Ranked<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Ranked<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Ranked<'_> {}

fn numeric_stats<'a>(name: &'a str, values: &[f64], labels: &[i64]) -> Option<ColumnStats<'a>> {
    let observed = || {
        values
            .iter()
            .zip(labels)
            .filter(|(x, _)| x.is_finite())
            .map(|(&x, &g)| (x, g))
    };
    // A constant column has no spread: say so exactly, before any rounding
    // error in the mean can make up one.
    let (first, _) = observed().next()?;
    if observed().all(|(x, _)| x == first) {
        return None;
    }
    let n = observed().count();
    // Two passes: the second corrects the first mean by the mean residual,
    // so a large offset does not swamp the spread.
    let rough = observed().map(|(x, _)| x).sum::<f64>() / n as f64;
    let mean = rough + observed().map(|(x, _)| x - rough).sum::<f64>() / n as f64;
    let sd = (observed().map(|(x, _)| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    if !(sd.is_finite() && sd > 0.0) {
        return None;
    }
    let (mut group_n, mut group_sum) = (HashMap::new(), HashMap::new());
    let mut distinct: Vec<f64> = Vec::with_capacity(3);
    for (x, g) in observed() {
        *group_n.entry(g).or_insert(0) += 1;
        *group_sum.entry(g).or_insert(0.0) += x - mean;
        if distinct.len() < 3 && !distinct.contains(&x) {
            distinct.push(x);
        }
    }
    Some(ColumnStats {
        name,
        kind: if distinct.len() == 2 {
            FeatureKind::Binary
        } else {
            FeatureKind::Numeric
        },
        n,
        group_n,
        axes: vec![Axis {
            level: None,
            mean: 0.0,
            sd,
            group_sum,
        }],
    })
}

fn categorical_stats<'a>(
    name: &'a str,
    values: &[Option<&str>],
    labels: &[i64],
    min_rows: usize,
) -> Option<ColumnStats<'a>> {
    let mut index: HashMap<&str, usize> = HashMap::new();
    let mut counts: Vec<usize> = vec![];
    let mut group_n: HashMap<i64, usize> = HashMap::new();
    let mut coded: Vec<Option<(usize, i64)>> = Vec::with_capacity(values.len());
    for (v, &g) in values.iter().zip(labels) {
        coded.push(v.map(|v| {
            let next = index.len();
            let i = *index.entry(v).or_insert(next);
            if i == counts.len() {
                counts.push(0);
            }
            counts[i] += 1;
            *group_n.entry(g).or_insert(0) += 1;
            (i, g)
        }));
    }
    let kind = match index.len() {
        0 | 1 => return None,
        2 => FeatureKind::Binary,
        _ => FeatureKind::Categorical,
    };
    let n: usize = counts.iter().sum();
    // Only levels backed by min_rows rows overall are scored or emitted.
    let mut axis_of: Vec<Option<usize>> = vec![None; counts.len()];
    let mut levels: Vec<(&str, usize)> = index.into_iter().collect();
    levels.sort_by_key(|&(_, i)| i);
    let mut axes = vec![];
    for (level, i) in levels {
        if counts[i] < min_rows {
            continue;
        }
        let p = counts[i] as f64 / n as f64;
        axis_of[i] = Some(axes.len());
        axes.push(Axis {
            level: Some(level.to_owned()),
            mean: p,
            sd: (p * (1.0 - p)).sqrt(),
            group_sum: HashMap::new(),
        });
    }
    for (i, g) in coded.into_iter().flatten() {
        if let Some(a) = axis_of[i] {
            *axes[a].group_sum.entry(g).or_insert(0.0) += 1.0;
        }
    }
    (!axes.is_empty()).then_some(ColumnStats {
        name,
        kind,
        n,
        group_n,
        axes,
    })
}

fn column_stats<'a>(
    name: &'a str,
    column: &'a Column,
    labels: &[i64],
    min_rows: usize,
) -> Option<ColumnStats<'a>> {
    match column {
        Column::Float(v) => numeric_stats(name, v, labels),
        Column::Int(v) => {
            let v: Vec<f64> = v.iter().map(|&x| x as f64).collect();
            numeric_stats(name, &v, labels)
        }
        Column::Bool(v) => {
            let v: Vec<Option<&str>> = v
                .iter()
                .map(|b| b.map(|b| if b { "True" } else { "False" }))
                .collect();
            categorical_stats(name, &v, labels, min_rows)
        }
        Column::Text(v) => {
            // Blank text is missing whatever the source: a CSV reads an
            // empty cell as null, a Parquet file can hold "".
            let v: Vec<Option<&str>> = v
                .iter()
                .map(|s| s.as_deref().filter(|s| !s.trim().is_empty()))
                .collect();
            categorical_stats(name, &v, labels, min_rows)
        }
    }
}

/// Summarize the grouping `labels` (one per table row) of `table`.
///
/// `exclude_columns` are never profiled (identifiers, hidden columns);
/// `not_used_for_grouping` are profiled columns the grouping did not use:
/// they are listed but cannot be defining features. `method` and `cut_off`
/// describe the grouping and are echoed.
///
/// Panics if `labels` is not one label per table row.
pub fn group_summary(
    table: &Table,
    labels: &[i64],
    exclude_columns: &[String],
    not_used_for_grouping: &[String],
    method: &str,
    cut_off: f64,
    thresholds: &GroupSummaryThresholds,
) -> GroupSummary {
    let rows = table.n_rows();
    assert_eq!(labels.len(), rows, "one label per table row");

    let mut sizes: HashMap<i64, usize> = HashMap::new();
    for &g in labels {
        *sizes.entry(g).or_insert(0) += 1;
    }
    // Asking for more groups or features than exist is asking for all of
    // them: clamp (the request is still what is echoed).
    let grouping_columns = table
        .names
        .iter()
        .filter(|n| !exclude_columns.contains(n) && !not_used_for_grouping.contains(n))
        .count();
    let clamped = GroupSummaryThresholds {
        max_groups: thresholds.max_groups.min(sizes.len()),
        max_features: thresholds.max_features.min(grouping_columns),
        ..*thresholds
    };
    let t = &clamped;
    let mut by_size: Vec<(i64, usize)> = sizes.into_iter().collect();
    by_size.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let eligible: HashSet<i64> = by_size
        .iter()
        .filter(|(_, n)| *n >= t.min_rows)
        .map(|(g, _)| *g)
        .collect();
    let shown: Vec<(i64, usize)> = by_size
        .iter()
        .copied()
        .filter(|(g, _)| eligible.contains(g))
        .take(t.max_groups)
        .collect();
    let shown_rows: usize = shown.iter().map(|(_, n)| n).sum();

    let excluded: HashSet<&str> = exclude_columns.iter().map(String::as_str).collect();
    let unused: HashSet<&str> = not_used_for_grouping.iter().map(String::as_str).collect();
    let profiled = table
        .names
        .iter()
        .zip(&table.columns)
        .filter(|(n, _)| !excluded.contains(n.as_str()));
    let unused_listed: Vec<&String> = profiled
        .clone()
        .filter(|(n, _)| unused.contains(n.as_str()))
        .map(|(n, _)| n)
        .collect();

    // The max_features strongest columns, kept in a bounded min-heap: a
    // column that loses is dropped as soon as it is scored, and a kept one
    // holds per-group statistics of the shown groups only.
    let shown_ids: HashSet<i64> = shown.iter().map(|&(g, _)| g).collect();
    let mut top: BinaryHeap<Reverse<Ranked>> = BinaryHeap::with_capacity(t.max_features + 1);
    let candidates = profiled
        .filter(|(n, _)| !unused.contains(n.as_str()))
        .filter_map(|(n, c)| column_stats(n, c, labels, t.min_rows));
    for (index, mut stats) in candidates.enumerate() {
        let eligible_n: usize = stats
            .group_n
            .iter()
            .filter(|(g, _)| eligible.contains(g))
            .map(|(_, &n)| n)
            .sum();
        let best = stats
            .axes
            .iter()
            .map(|a| stats.separation(a, &eligible, eligible_n))
            .fold(0.0, f64::max);
        let separation = round_to(best, 3);
        if separation > 0.0 {
            stats.keep_groups(&shown_ids);
            top.push(Reverse(Ranked {
                separation,
                index,
                stats,
            }));
            if top.len() > t.max_features {
                top.pop();
            }
        }
    }
    debug_assert!(top.len() <= t.max_features);
    debug_assert!(top.iter().all(|Reverse(r)| {
        r.stats.group_n.len() <= shown.len()
            && r.stats
                .axes
                .iter()
                .all(|a| a.group_sum.len() <= shown.len())
    }));
    // Strongest first, the earlier column on ties.
    let scored: Vec<(ColumnStats, f64)> = top
        .into_sorted_vec()
        .into_iter()
        .map(|Reverse(r)| (r.stats, r.separation))
        .collect();

    let mut cells = vec![];
    let groups = shown
        .iter()
        .map(|&(id, size)| {
            let mut traits: Vec<(Trait, f64)> = scored
                .iter()
                .filter_map(|(stats, _)| {
                    let (level, sd) = stats.cell(id, t.min_rows)?;
                    (sd.abs() >= t.min_sd).then(|| {
                        let (column, cut_column) = label(stats.name);
                        let (level, cut_level) = level.map(label).unzip();
                        let truncated = cut_column || cut_level == Some(true);
                        cells.push(Cell {
                            group: id,
                            column: column.clone(),
                            level: level.clone(),
                            sd: round_to(sd, 2),
                            truncated,
                        });
                        (
                            Trait {
                                column,
                                level,
                                sd: round_to(sd, 2),
                                truncated,
                            },
                            sd.abs(),
                        )
                    })
                })
                .collect();
            traits.sort_by(|a, b| b.1.total_cmp(&a.1));
            GroupEntry {
                id,
                size: round_count(size),
                traits: traits
                    .into_iter()
                    .take(t.max_traits)
                    .map(|(tr, _)| tr)
                    .collect(),
            }
        })
        .collect();

    let mut summary = GroupSummary {
        thresholds: *thresholds,
        method: label(method).0,
        cut_off: cut_off.is_finite().then_some(cut_off),
        group_count: by_size.len(),
        rows: RowCount::of(rows, t.min_rows),
        groups,
        small_groups: SmallGroups {
            count: by_size.len() - shown.len(),
            rows: RowCount::of(rows - shown_rows, t.min_rows),
        },
        defining_features: scored
            .iter()
            .map(|(stats, separation)| {
                let (column, truncated) = label(stats.name);
                DefiningFeature {
                    column,
                    kind: stats.kind,
                    separation: *separation,
                    truncated,
                }
            })
            .collect(),
        cells,
        not_used_for_grouping: unused_listed
            .iter()
            .take(MAX_LISTED_UNUSED)
            .map(|n| label(n).0)
            .collect(),
        not_used_for_grouping_omitted: unused_listed.len().saturating_sub(MAX_LISTED_UNUSED),
        size_chars: 0,
    };
    // The count is part of what it counts: repeat until its digits settle.
    loop {
        let n = serde_json::to_string(&summary)
            .expect("a summary serializes")
            .chars()
            .count();
        if n == summary.size_chars {
            break;
        }
        summary.size_chars = n;
    }
    summary
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use serde_json::Value;

    use super::*;

    const T: GroupSummaryThresholds = GROUP_SUMMARY_DEFAULTS;

    fn table(columns: Vec<(&str, Column)>) -> Table {
        let (names, columns) = columns.into_iter().map(|(n, c)| (n.to_owned(), c)).unzip();
        Table::new(names, columns).unwrap()
    }

    fn summarize(table: &Table, labels: &[i64]) -> GroupSummary {
        group_summary(table, labels, &[], &[], "components", 0.25, &T)
    }

    fn json_of(s: &GroupSummary) -> Value {
        serde_json::to_value(s).unwrap()
    }

    /// Every number in `v` is finite and no string spells a non-finite one.
    fn assert_finite(v: &Value) {
        match v {
            Value::Number(n) => assert!(n.as_f64().unwrap().is_finite(), "{n}"),
            Value::Array(a) => a.iter().for_each(assert_finite),
            Value::Object(m) => m.values().for_each(assert_finite),
            _ => {}
        }
        let text = v.to_string();
        assert!(!text.contains("NaN") && !text.contains("inf"), "{text}");
    }

    fn small(count: usize, rows: usize) -> SmallGroups {
        SmallGroups {
            count,
            rows: RowCount::Rounded(rows),
        }
    }

    /// Group `g` of `k` near-equal groups of `rows` rows.
    fn labels(rows: usize, k: usize) -> Vec<i64> {
        (0..rows).map(|i| (i % k) as i64).collect()
    }

    /// Numbers shifted by `shift * group`, with noise.
    fn shifted(rng: &mut StdRng, labels: &[i64], shift: f64) -> Column {
        Column::Float(
            labels
                .iter()
                .map(|&g| g as f64 * shift + rng.gen_range(-1.0..1.0))
                .collect(),
        )
    }

    #[test]
    fn sd_and_separation_are_the_standardized_mean_gap_and_eta_squared() {
        // Two groups of 10: values 0 and 2, so mean 1 and population SD 1.
        let labels = labels(20, 2);
        let x = labels.iter().map(|&g| 2.0 * g as f64).collect();
        let s = summarize(&table(vec![("x", Column::Float(x))]), &labels);
        assert_eq!(s.defining_features[0].separation, 1.0);
        assert_eq!(s.defining_features[0].kind, FeatureKind::Binary);
        let sds: Vec<(i64, f64)> = s.cells.iter().map(|c| (c.group, c.sd)).collect();
        assert_eq!(sds, vec![(0, -1.0), (1, 1.0)]);

        // A level held by 6 of 10 rows in one group and 2 of 10 in the
        // other: p = 0.4, SD sqrt(0.24), gaps ±0.2.
        let v = (0..20)
            .map(|i| {
                Some(
                    if (i % 2 == 0 && i < 12) || (i % 2 == 1 && i < 4) {
                        "a"
                    } else {
                        "b"
                    }
                    .into(),
                )
            })
            .collect();
        let s = summarize(&table(vec![("c", Column::Text(v))]), &labels);
        let eta2 = 20.0 * 0.2f64.powi(2) / (20.0 * 0.24);
        assert_eq!(s.defining_features[0].separation, round_to(eta2, 3));
        assert_eq!(s.defining_features[0].kind, FeatureKind::Binary);
    }

    #[test]
    fn numeric_binary_categorical_and_ordinal_columns_are_standardized() {
        let mut rng = StdRng::seed_from_u64(1);
        let labels = labels(300, 3);
        let ordinal = ["low", "mid", "high"];
        let s = summarize(
            &table(vec![
                ("num", shifted(&mut rng, &labels, 2.0)),
                (
                    "int01",
                    Column::Int(labels.iter().map(|&g| i64::from(g == 2)).collect()),
                ),
                (
                    "flag",
                    Column::Bool(labels.iter().map(|&g| Some(g == 0)).collect()),
                ),
                (
                    "cat",
                    Column::Text(
                        labels
                            .iter()
                            .map(|&g| Some(["x", "y", "z"][g as usize].into()))
                            .collect(),
                    ),
                ),
                (
                    "ord",
                    Column::Text(
                        labels
                            .iter()
                            .map(|&g| Some(ordinal[g as usize].into()))
                            .collect(),
                    ),
                ),
            ]),
            &labels,
        );
        let kinds: HashMap<&str, FeatureKind> = s
            .defining_features
            .iter()
            .map(|f| (f.column.as_str(), f.kind))
            .collect();
        assert_eq!(kinds["num"], FeatureKind::Numeric);
        assert_eq!(kinds["int01"], FeatureKind::Binary);
        assert_eq!(kinds["flag"], FeatureKind::Binary);
        assert_eq!(kinds["cat"], FeatureKind::Categorical);
        assert_eq!(kinds["ord"], FeatureKind::Categorical);
        // Each group is all "x", "y" or "z": the level it holds is its cell.
        let level = |g: i64| {
            s.cells
                .iter()
                .find(|c| c.group == g && c.column == "cat")
                .and_then(|c| c.level.clone())
        };
        assert_eq!(level(1).as_deref(), Some("y"));
        assert!(s
            .cells
            .iter()
            .filter(|c| c.column == "int01")
            .all(|c| c.level.is_none()));
    }

    #[test]
    fn missing_values_are_excluded_and_constant_columns_never_define() {
        let mut rng = StdRng::seed_from_u64(2);
        let labels = labels(400, 2);
        let mostly_missing = labels
            .iter()
            .enumerate()
            .map(|(i, &g)| {
                if i % 10 < 2 {
                    g as f64 * 5.0 + i as f64 * 1e-3
                } else {
                    f64::NAN
                }
            })
            .collect();
        let s = summarize(
            &table(vec![
                ("sparse", Column::Float(mostly_missing)),
                ("constant", Column::Float(vec![3.0; 400])),
                ("one_level", Column::Text(vec![Some("a".into()); 400])),
                ("all_missing", Column::Float(vec![f64::NAN; 400])),
                ("noise", shifted(&mut rng, &labels, 0.0)),
            ]),
            &labels,
        );
        let names: Vec<&str> = s
            .defining_features
            .iter()
            .map(|f| f.column.as_str())
            .collect();
        // 80% missing: the 80 observed values still split the groups.
        assert_eq!(names[0], "sparse");
        assert!(s.defining_features[0].separation > 0.99);
        assert_eq!(s.cells.iter().filter(|c| c.column == "sparse").count(), 2);
        for c in ["constant", "one_level", "all_missing"] {
            assert!(!names.contains(&c), "{names:?}");
        }
    }

    #[test]
    fn constant_decimals_and_large_offsets_do_not_separate_from_rounding_error() {
        // 0.1 is not exact in binary: its mean drifts off 0.1 by an ulp.
        let constant = || Column::Float(vec![0.1; 20]);
        for labels in [labels(20, 2), vec![0; 20]] {
            let s = summarize(&table(vec![("c", constant())]), &labels);
            assert!(
                s.defining_features.is_empty() && s.cells.is_empty(),
                "{s:?}"
            );
            assert!(s.groups.iter().all(|g| g.traits.is_empty()));
        }

        // The same noise with and without a 1e9 offset scores the same.
        let labels = labels(200, 2);
        let mut rng = StdRng::seed_from_u64(10);
        let Column::Float(x) = shifted(&mut rng, &labels, 1.0) else {
            unreachable!()
        };
        let offset = x.iter().map(|v| v + 1e9).collect();
        let a = summarize(&table(vec![("x", Column::Float(x))]), &labels);
        let b = summarize(&table(vec![("x", Column::Float(offset))]), &labels);
        assert!(!a.defining_features.is_empty());
        assert_eq!(a.defining_features, b.defining_features);
        assert_eq!(a.cells, b.cells);

        // A single group of 1e9 plus small noise separates nothing.
        let one = vec![0; 200];
        let x = (0..200).map(|_| 1e9 + rng.gen_range(-1e-3..1e-3)).collect();
        let s = summarize(&table(vec![("x", Column::Float(x))]), &one);
        assert!(
            s.defining_features.is_empty() && s.cells.is_empty(),
            "{s:?}"
        );
    }

    #[test]
    fn blank_text_is_missing_in_csv_and_parquet_alike() {
        // Group 0 holds "" or "  ", group 1 "A": only blanks tell them apart.
        let labels = labels(40, 2);
        let blank = |i: usize, g: i64| match (g, i % 4) {
            (1, _) => "A",
            (_, 0) => "  ",
            _ => "",
        };
        let values: Vec<Option<String>> = labels
            .iter()
            .enumerate()
            .map(|(i, &g)| Some(blank(i, g).to_owned()))
            .collect();
        let check = |t: &Table| {
            let s = summarize(t, &labels);
            assert!(
                s.defining_features.is_empty() && s.cells.is_empty(),
                "{s:?}"
            );
        };

        let path = std::env::temp_dir().join(format!(
            "pulsar-group-summary-blanks-{}.parquet",
            std::process::id()
        ));
        pulsar_data::write_parquet(&table(vec![("c", Column::Text(values.clone()))]), &path)
            .unwrap();
        let parquet = pulsar_data::read_parquet(&path);
        let _ = std::fs::remove_file(&path);
        let parquet = parquet.unwrap();
        assert!(
            matches!(&parquet.columns[0], Column::Text(v) if v.iter().any(|s| s.as_deref() == Some(""))),
            "Parquet keeps the empty strings"
        );
        check(&parquet);

        // A second, constant column keeps a blank row from being skipped.
        let csv: String = std::iter::once("c,k\n".to_owned())
            .chain(
                values
                    .iter()
                    .map(|v| format!("{},1\n", v.as_deref().unwrap())),
            )
            .collect();
        check(&pulsar_data::read_csv(csv.as_bytes(), "inline").unwrap());
    }

    #[test]
    fn a_single_group_separates_nothing() {
        let mut rng = StdRng::seed_from_u64(3);
        let labels = vec![7; 50];
        let s = summarize(
            &table(vec![("x", shifted(&mut rng, &labels, 1.0))]),
            &labels,
        );
        assert_eq!((s.group_count, &s.rows), (1, &RowCount::Rounded(50)));
        assert_eq!(s.groups.len(), 1);
        assert_eq!((s.groups[0].id, s.groups[0].size), (7, 50));
        assert!(s.groups[0].traits.is_empty());
        assert!(s.defining_features.is_empty() && s.cells.is_empty());
        assert_eq!(s.small_groups, small(0, 0));
    }

    #[test]
    fn when_every_group_is_small_only_counts_are_given() {
        let mut rng = StdRng::seed_from_u64(4);
        let labels = labels(70, 10);
        let s = summarize(
            &table(vec![("x", shifted(&mut rng, &labels, 3.0))]),
            &labels,
        );
        assert!(s.groups.is_empty() && s.defining_features.is_empty() && s.cells.is_empty());
        assert_eq!(s.small_groups, small(10, 70));
    }

    #[test]
    fn thousands_of_groups_stay_bounded() {
        let mut rng = StdRng::seed_from_u64(5);
        // 1500 singletons beside four large groups.
        let mut labels: Vec<i64> = (0..400).map(|i| i % 4).collect();
        labels.extend(100..1600);
        let s = summarize(
            &table(vec![("x", shifted(&mut rng, &labels, 0.01))]),
            &labels,
        );
        assert_eq!(s.group_count, 1504);
        assert_eq!(s.groups.len(), 4);
        assert_eq!(s.small_groups, small(1500, 1500));
        assert!(s.cells.len() <= T.max_groups * T.max_features);

        // A shattered grouping: separation over singletons would be 1.
        let labels: Vec<i64> = (0..2000).collect();
        let s = summarize(
            &table(vec![("x", shifted(&mut rng, &labels, 1.0))]),
            &labels,
        );
        assert!(s.groups.is_empty() && s.defining_features.is_empty());
        assert_eq!(s.small_groups.count, 2000);
    }

    /// 250 columns: every 25th numeric one shifted by group, the rest noise
    /// or categorical; 30 are not used for grouping.
    fn wide(rows: usize) -> (Table, Vec<i64>, Vec<String>) {
        let mut rng = StdRng::seed_from_u64(6);
        let labels = labels(rows, 9);
        let columns: Vec<(String, Column)> = (0..250)
            .map(|j| {
                let column = if j % 3 == 0 {
                    Column::Text(
                        labels
                            .iter()
                            .map(|_| Some(format!("L{}", rng.gen_range(0..12))))
                            .collect(),
                    )
                } else {
                    shifted(&mut rng, &labels, if j % 25 == 1 { 1.5 } else { 0.0 })
                };
                (format!("column_{j}"), column)
            })
            .collect();
        let unused = (200..230).map(|j| format!("column_{j}")).collect();
        let (names, columns) = columns.into_iter().unzip();
        (Table::new(names, columns).unwrap(), labels, unused)
    }

    #[test]
    fn wide_data_stays_small_and_bounded() {
        let (table, labels, unused) = wide(3000);
        let s = group_summary(&table, &labels, &[], &unused, "spectral", 0.1, &T);
        assert_eq!(s.groups.len(), T.max_groups);
        assert_eq!(s.defining_features.len(), T.max_features);
        assert!(s.groups.iter().all(|g| g.traits.len() <= T.max_traits));
        assert!(s
            .defining_features
            .iter()
            .all(|f| !unused.contains(&f.column)));
        let shifted_column = |c: &str| c["column_".len()..].parse::<usize>().unwrap() % 25 == 1;
        assert!(
            s.defining_features
                .iter()
                .all(|f| shifted_column(&f.column)),
            "{:?}",
            s.defining_features
        );
        assert_eq!(s.not_used_for_grouping.len(), MAX_LISTED_UNUSED);
        assert_eq!(s.not_used_for_grouping_omitted, 10);
        let bytes = serde_json::to_string(&s).unwrap().len();
        assert!(bytes < 20_000, "{bytes} bytes");
    }

    #[test]
    fn all_categorical_and_all_numeric_tables_work() {
        let mut rng = StdRng::seed_from_u64(7);
        let labels = labels(200, 4);
        let cat = |g: &i64| Some(format!("v{}", g % 2));
        let s = summarize(
            &table(vec![
                ("a", Column::Text(labels.iter().map(cat).collect())),
                (
                    "b",
                    Column::Bool(labels.iter().map(|&g| Some(g < 1)).collect()),
                ),
            ]),
            &labels,
        );
        assert_eq!(s.defining_features.len(), 2);
        assert!(s.cells.iter().all(|c| c.level.is_some()));

        let s = summarize(
            &table(vec![
                ("a", shifted(&mut rng, &labels, 1.0)),
                ("b", Column::Int(labels.iter().map(|&g| g * 3).collect())),
            ]),
            &labels,
        );
        assert_eq!(s.defining_features.len(), 2);
        assert!(s.cells.iter().all(|c| c.level.is_none()));
    }

    #[test]
    fn column_names_are_kept_and_never_interpreted() {
        let mut rng = StdRng::seed_from_u64(8);
        let labels = labels(100, 2);
        let x = shifted(&mut rng, &labels, 4.0);
        let long = "ß→测试🐧 ".repeat(40);
        let names = [long.as_str(), "id", "rowid", "species", "cluster", "label"];
        let pick = |names: &[&str]| {
            let s = summarize(
                &table(names.iter().map(|&n| (n, x.clone())).collect()),
                &labels,
            );
            let mut v = json_of(&s);
            v["defining_features"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .for_each(|f| {
                    f["column"] = Value::Null;
                    f.as_object_mut().unwrap().remove("truncated");
                });
            (s, v)
        };
        let (s, a) = pick(&names);
        let cut: String = long.chars().take(MAX_LABEL_CHARS).collect();
        assert_eq!(s.defining_features[0].column, cut + "…");
        assert!(s.defining_features[0].truncated);
        assert!(!s.defining_features[1].truncated);
        assert_eq!(s.defining_features.len(), 5, "no name is special-cased");
        let (_, b) = pick(&["p", "q", "r", "s", "t", "u"]);
        assert_eq!(a["defining_features"], b["defining_features"]);
        assert_eq!(
            a["groups"].as_array().unwrap().len(),
            b["groups"].as_array().unwrap().len()
        );
    }

    #[test]
    fn non_finite_numbers_are_never_serialized() {
        let labels = labels(100, 2);
        let x = labels
            .iter()
            .enumerate()
            .map(|(i, &g)| match i % 7 {
                0 => f64::INFINITY,
                1 => f64::NEG_INFINITY,
                2 => f64::NAN,
                _ => g as f64,
            })
            .collect();
        let huge = (0..100)
            .map(|i| if i % 2 == 0 { f64::MAX } else { -f64::MAX })
            .collect();
        let s = group_summary(
            &table(vec![("x", Column::Float(x)), ("huge", Column::Float(huge))]),
            &labels,
            &[],
            &[],
            "components",
            f64::NAN,
            &T,
        );
        let v = json_of(&s);
        assert_finite(&v);
        assert!(v["cut_off"].is_null());
        assert_eq!(s.defining_features.len(), 1, "{v}");
    }

    #[test]
    fn rare_levels_and_small_groups_are_withheld() {
        // Groups 0 and 1 of 40 rows, group 2 of 5 rows.
        let mut labels: Vec<i64> = (0..80).map(|i| i % 2).collect();
        labels.extend([2; 5]);
        let values: Vec<Option<String>> = labels
            .iter()
            .enumerate()
            .map(|(i, &g)| {
                Some(match (g, i) {
                    (2, _) => "only_small".into(),
                    (_, i) if i < 7 => "rare".into(),
                    (0, _) => "zero".into(),
                    _ => "one".into(),
                })
            })
            .collect();
        let s = summarize(&table(vec![("c", Column::Text(values))]), &labels);
        let text = serde_json::to_string(&s).unwrap();
        assert!(
            !text.contains("rare") && !text.contains("only_small"),
            "{text}"
        );
        assert!(s.cells.iter().all(|c| c.group != 2));
        assert_eq!(s.small_groups.rows, RowCount::Withheld("under 8".into()));
        assert_eq!(s.small_groups.count, 1);
        // Group 0 lacks "one" (strongly under-represented, 0 rows): the
        // cell falls to the level it holds.
        let cell = s.cells.iter().find(|c| c.group == 0).unwrap();
        assert_eq!(cell.level.as_deref(), Some("zero"));
        assert!(cell.sd > 0.0);
    }

    #[test]
    fn row_counts_are_rounded_so_a_small_group_cannot_be_recovered() {
        let mut rng = StdRng::seed_from_u64(11);
        // Groups of 12 and 13 rows and a singleton.
        let mut labels = vec![0; 12];
        labels.extend([1; 13]);
        labels.push(2);
        let s = summarize(
            &table(vec![("x", shifted(&mut rng, &labels, 3.0))]),
            &labels,
        );
        let sizes: Vec<usize> = s.groups.iter().map(|g| g.size).collect();
        assert_eq!(sizes, [15, 10]);
        assert_eq!(s.rows, RowCount::Rounded(25));
        assert_eq!(s.small_groups.count, 1);
        let text = serde_json::to_string(&s.small_groups).unwrap();
        assert_eq!(text, r#"{"count":1,"rows":"under 8"}"#);

        // A remainder of min_rows or more is rounded too.
        let mut labels = labels.clone();
        labels.extend((3..12).map(|g| g as i64));
        let s = summarize(
            &table(vec![("x", shifted(&mut rng, &labels, 3.0))]),
            &labels,
        );
        assert_eq!(s.small_groups, small(10, 10));
        assert_eq!(s.rows, RowCount::Rounded(35));
        assert_eq!(round_count(2), 0);
        assert_eq!(round_count(3), 5);
    }

    #[test]
    fn many_groups_and_columns_keep_the_ranking_of_a_full_sort() {
        // 2000 groups of 8 and 40 columns shifted by different amounts.
        let mut rng = StdRng::seed_from_u64(12);
        let labels = labels(16_000, 2000);
        let columns: Vec<(String, Column)> = (0..40)
            .map(|j| {
                (
                    format!("f{j}"),
                    shifted(&mut rng, &labels, 0.001 * j as f64),
                )
            })
            .collect();
        let (names, columns): (Vec<String>, Vec<Column>) = columns.into_iter().unzip();
        let table = Table::new(names.clone(), columns).unwrap();
        let s = summarize(&table, &labels);
        assert_eq!(s.groups.len(), T.max_groups);
        assert_eq!(s.small_groups.count, 2000 - T.max_groups);

        // The same ranking, by a full stable sort of every column alone.
        let mut all: Vec<(String, f64)> = names
            .iter()
            .zip(&table.columns)
            .filter_map(|(n, c)| {
                let one = summarize(
                    &Table::new(vec![n.clone()], vec![c.clone()]).unwrap(),
                    &labels,
                );
                one.defining_features
                    .first()
                    .map(|f| (n.clone(), f.separation))
            })
            .collect();
        all.sort_by(|a, b| b.1.total_cmp(&a.1));
        all.truncate(T.max_features);
        let top: Vec<(String, f64)> = s
            .defining_features
            .iter()
            .map(|f| (f.column.clone(), f.separation))
            .collect();
        assert_eq!(top, all);
    }

    /// Every string in `v` is at most a cut label long.
    fn assert_short_strings(v: &Value) {
        match v {
            Value::String(s) => assert!(s.chars().count() <= MAX_LABEL_CHARS + 1, "{s:.100}"),
            Value::Array(a) => a.iter().for_each(assert_short_strings),
            Value::Object(m) => m.values().for_each(assert_short_strings),
            _ => {}
        }
    }

    #[test]
    fn long_levels_and_names_are_cut_and_the_bytes_bounded() {
        // Two groups of 8, each holding its own 100k-character level.
        let labels = labels(16, 2);
        let long = |g: i64| Some(format!("{g}{}", "测".repeat(100_000)));
        let name = "ß".repeat(100_000);
        let s = summarize(
            &table(vec![(
                &name,
                Column::Text(labels.iter().map(|&g| long(g)).collect()),
            )]),
            &labels,
        );
        let v = json_of(&s);
        assert_short_strings(&v);
        assert_eq!(s.cells.len(), 2);
        assert!(s.cells.iter().all(|c| c.truncated));
        assert!(s
            .groups
            .iter()
            .all(|g| g.traits.iter().all(|t| t.truncated)));
        assert!(s.defining_features[0].truncated);
        // Cut levels stay apart: they differ before the cut.
        assert_ne!(s.cells[0].level, s.cells[1].level);
    }

    /// The largest serialized summary under `t`: every entry at its cap,
    /// every string at its longest, every number at its widest.
    fn size_bound(t: &GroupSummaryThresholds) -> usize {
        // 80 characters escaped to `\u00XX`, the ellipsis, the quotes.
        let string = 6 * MAX_LABEL_CHARS + "…".len() + 2;
        let number = 24; // an f64 or an i64, as serde_json writes them
        let (g, f, tr) = (t.max_groups, t.max_features, t.max_traits);
        let entry = |template: &str, strings: usize, numbers: usize| {
            template.len() + 1 + strings * string + numbers * number
        };
        let cell = entry(
            r#"{"group":,"column":,"level":,"sd":,"truncated":true}"#,
            2,
            2,
        );
        let trait_ = entry(r#"{"column":,"level":,"sd":,"truncated":true}"#, 2, 1);
        let group = entry(r#"{"id":,"size":,"traits":[]}"#, 0, 2);
        let feature = entry(
            r#"{"column":,"kind":"categorical","separation":,"truncated":true}"#,
            1,
            1,
        );
        let fixed = entry(
            r#"{"thresholds":{"max_groups":,"max_features":,"min_sd":,"min_rows":,"max_traits":},"method":,"cut_off":,"group_count":,"rows":,"groups":[],"small_groups":{"count":,"rows":},"defining_features":[],"cells":[],"not_used_for_grouping":[],"not_used_for_grouping_omitted":,"size_chars":}"#,
            3,
            12,
        );
        fixed
            + g * f * cell
            + g * (group + tr.min(f) * trait_)
            + f * feature
            + MAX_LISTED_UNUSED * (string + 1)
    }

    #[test]
    fn at_large_overrides_the_size_stays_under_its_bound() {
        let max = GroupSummaryThresholds {
            max_groups: 20,
            max_features: 20,
            min_sd: 0.0,
            min_rows: 8,
            max_traits: 20,
        };
        // 20 groups of 8; 25 grouping columns where each group holds its
        // own level, and 25 unused ones. Names and levels escape to 6
        // bytes a character after a distinct prefix.
        let labels = labels(160, 20);
        let long = |prefix: usize, chars: usize| format!("{prefix:02}{}", "\u{1}".repeat(chars));
        let levels = || {
            Column::Text(
                labels
                    .iter()
                    .map(|&g| Some(long(g as usize, 200)))
                    .collect(),
            )
        };
        let mut columns: Vec<(String, Column)> =
            (0..25).map(|j| (long(j, 100_000), levels())).collect();
        let unused: Vec<String> = (30..55).map(|j| long(j, 100_000)).collect();
        columns.extend(unused.iter().map(|n| (n.clone(), levels())));
        let (names, columns) = columns.into_iter().unzip();
        let table = Table::new(names, columns).unwrap();
        let s = group_summary(&table, &labels, &[], &unused, "components", 0.25, &max);

        assert_eq!(s.defining_features.len(), 20);
        assert_eq!(s.cells.len(), 20 * 20);
        assert!(s.groups.iter().all(|g| g.traits.len() == 20));
        assert!(s.cells.iter().all(|c| c.truncated));
        let v = json_of(&s);
        assert_short_strings(&v);
        let (bytes, bound) = (v.to_string().len(), size_bound(&max));
        // The strings are at their longest, so the bound is nearly met.
        assert!(
            bytes <= bound && bytes > bound * 9 / 10,
            "{bytes} of {bound}"
        );
        // The figure in the module documentation.
        assert!(size_bound(&T) < 66_000, "{}", size_bound(&T));
    }

    #[test]
    fn large_requests_are_honoured_and_clamped_to_what_exists() {
        let (table, labels, unused) = wide(3000);
        let ask = |n: usize| GroupSummaryThresholds {
            max_groups: n,
            max_features: n,
            min_sd: 0.0,
            min_rows: 8,
            max_traits: n,
        };
        // Past the old cap of 20: all 9 groups, and 40 features of the 220
        // grouping columns.
        let s = group_summary(&table, &labels, &[], &unused, "spectral", 0.1, &ask(40));
        assert_eq!(s.groups.len(), 9);
        assert_eq!(s.defining_features.len(), 40);
        assert!(
            s.defining_features.len() > 20,
            "{}",
            s.defining_features.len()
        );
        assert!(s.groups.iter().any(|g| g.traits.len() > 3));
        // More than exists is all of it, echoed as applied.
        let s = group_summary(
            &table,
            &labels,
            &[],
            &unused,
            "spectral",
            0.1,
            &ask(100_000),
        );
        assert_eq!(s.thresholds, ask(100_000));
        assert_eq!(s.groups.len(), 9);
        assert!(s.defining_features.len() <= 220);
    }

    #[test]
    fn size_chars_is_the_serialized_length() {
        let (table, labels, unused) = wide(3000);
        for t in [
            T,
            GroupSummaryThresholds {
                max_groups: 50,
                max_features: 50,
                ..T
            },
        ] {
            let s = group_summary(&table, &labels, &[], &unused, "spectral", 0.1, &t);
            let text = serde_json::to_string(&s).unwrap();
            assert_eq!(s.size_chars, text.chars().count());
            assert!(s.size_chars > 1000);
        }
        // Non-ASCII names count as characters, not bytes.
        let t = table_of_unicode();
        let s = summarize(&t.0, &t.1);
        assert_eq!(
            s.size_chars,
            serde_json::to_string(&s).unwrap().chars().count()
        );
    }

    fn table_of_unicode() -> (Table, Vec<i64>) {
        let mut rng = StdRng::seed_from_u64(11);
        let labels = labels(40, 2);
        let x = shifted(&mut rng, &labels, 2.0);
        (table(vec![("größe→测试", x)]), labels)
    }

    #[test]
    fn output_stays_small_whatever_the_input_size() {
        let mut rng = StdRng::seed_from_u64(9);
        let rows = 50_000;
        let labels: Vec<i64> = (0..rows).map(|i| (i % 3000) as i64).collect();
        let columns: Vec<(String, Column)> = (0..60)
            .map(|j| {
                (
                    format!("feature_{j}"),
                    shifted(&mut rng, &labels, 0.001 * j as f64),
                )
            })
            .collect();
        let (names, columns) = columns.into_iter().unzip();
        let s = summarize(&Table::new(names, columns).unwrap(), &labels);
        let bytes = serde_json::to_string(&s).unwrap().len();
        assert!(bytes < 20_000, "{bytes} bytes");
        assert_eq!(s.groups.len(), T.max_groups);
    }
}
