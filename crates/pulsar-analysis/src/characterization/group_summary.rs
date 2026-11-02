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
//!   infinite numbers, blanks) are excluded from every mean, SD and
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
//! non-finite number is serialized. Column names are kept as-is and never
//! interpreted.

use std::collections::{HashMap, HashSet};

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
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GroupEntry {
    pub id: i64,
    pub size: usize,
    pub traits: Vec<Trait>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SmallGroups {
    pub count: usize,
    pub rows: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DefiningFeature {
    pub column: String,
    pub kind: FeatureKind,
    pub separation: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Cell {
    pub group: i64,
    pub column: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<String>,
    pub sd: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GroupSummary {
    pub thresholds: GroupSummaryThresholds,
    pub method: String,
    pub cut_off: Option<f64>,
    pub group_count: usize,
    pub rows: usize,
    pub groups: Vec<GroupEntry>,
    pub small_groups: SmallGroups,
    pub defining_features: Vec<DefiningFeature>,
    pub cells: Vec<Cell>,
    pub not_used_for_grouping: Vec<String>,
    pub not_used_for_grouping_omitted: usize,
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

    /// The strongest emittable difference of `group`, positive on a tie.
    fn cell(&self, group: i64, min_rows: usize) -> Option<(Option<&str>, f64)> {
        self.axes
            .iter()
            .filter(|a| self.backing(a, group) >= min_rows)
            .filter_map(|a| Some((a.level.as_deref(), self.sd_of(a, group)?)))
            .max_by(|(_, x), (_, y)| x.abs().total_cmp(&y.abs()).then(x.total_cmp(y)))
    }
}

fn numeric_stats<'a>(name: &'a str, values: &[f64], labels: &[i64]) -> Option<ColumnStats<'a>> {
    let observed = || {
        values
            .iter()
            .zip(labels)
            .filter(|(x, _)| x.is_finite())
            .map(|(&x, &g)| (x, g))
    };
    let n = observed().count();
    if n == 0 {
        return None;
    }
    let mean = observed().map(|(x, _)| x).sum::<f64>() / n as f64;
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
            let v: Vec<Option<&str>> = v.iter().map(Option::as_deref).collect();
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
    let t = thresholds;

    let mut sizes: HashMap<i64, usize> = HashMap::new();
    for &g in labels {
        *sizes.entry(g).or_insert(0) += 1;
    }
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

    let mut scored: Vec<(ColumnStats, f64)> = profiled
        .filter(|(n, _)| !unused.contains(n.as_str()))
        .filter_map(|(n, c)| column_stats(n, c, labels, t.min_rows))
        .filter_map(|stats| {
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
            (separation > 0.0).then_some((stats, separation))
        })
        .collect();
    // A stable sort keeps table order on ties.
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(t.max_features);

    let mut cells = vec![];
    let groups = shown
        .iter()
        .map(|&(id, size)| {
            let mut traits: Vec<(Trait, f64)> = scored
                .iter()
                .filter_map(|(stats, _)| {
                    let (level, sd) = stats.cell(id, t.min_rows)?;
                    (sd.abs() >= t.min_sd).then(|| {
                        let level = level.map(str::to_owned);
                        cells.push(Cell {
                            group: id,
                            column: stats.name.to_owned(),
                            level: level.clone(),
                            sd: round_to(sd, 2),
                        });
                        let column = stats.name.to_owned();
                        (
                            Trait {
                                column,
                                level,
                                sd: round_to(sd, 2),
                            },
                            sd.abs(),
                        )
                    })
                })
                .collect();
            traits.sort_by(|a, b| b.1.total_cmp(&a.1));
            GroupEntry {
                id,
                size,
                traits: traits
                    .into_iter()
                    .take(t.max_traits)
                    .map(|(tr, _)| tr)
                    .collect(),
            }
        })
        .collect();

    GroupSummary {
        thresholds: *t,
        method: method.to_owned(),
        cut_off: cut_off.is_finite().then_some(cut_off),
        group_count: by_size.len(),
        rows,
        groups,
        small_groups: SmallGroups {
            count: by_size.len() - shown.len(),
            rows: rows - shown_rows,
        },
        defining_features: scored
            .iter()
            .map(|(stats, separation)| DefiningFeature {
                column: stats.name.to_owned(),
                kind: stats.kind,
                separation: *separation,
            })
            .collect(),
        cells,
        not_used_for_grouping: unused_listed
            .iter()
            .take(MAX_LISTED_UNUSED)
            .map(|n| (*n).clone())
            .collect(),
        not_used_for_grouping_omitted: unused_listed.len().saturating_sub(MAX_LISTED_UNUSED),
    }
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
    fn a_single_group_separates_nothing() {
        let mut rng = StdRng::seed_from_u64(3);
        let labels = vec![7; 50];
        let s = summarize(
            &table(vec![("x", shifted(&mut rng, &labels, 1.0))]),
            &labels,
        );
        assert_eq!((s.group_count, s.rows), (1, 50));
        assert_eq!(s.groups.len(), 1);
        assert_eq!((s.groups[0].id, s.groups[0].size), (7, 50));
        assert!(s.groups[0].traits.is_empty());
        assert!(s.defining_features.is_empty() && s.cells.is_empty());
        assert_eq!((s.small_groups.count, s.small_groups.rows), (0, 0));
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
        assert_eq!((s.small_groups.count, s.small_groups.rows), (10, 70));
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
        assert_eq!((s.small_groups.count, s.small_groups.rows), (1500, 1500));
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
                .for_each(|f| f["column"] = Value::Null);
            (s, v)
        };
        let (s, a) = pick(&names);
        assert_eq!(s.defining_features[0].column, long);
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
        assert_eq!((s.small_groups.count, s.small_groups.rows), (1, 5));
        // Group 0 lacks "one" (strongly under-represented, 0 rows): the
        // cell falls to the level it holds.
        let cell = s.cells.iter().find(|c| c.group == 0).unwrap();
        assert_eq!(cell.level.as_deref(), Some("zero"));
        assert!(cell.sd > 0.0);
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
