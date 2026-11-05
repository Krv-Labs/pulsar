//! [`ReleaseBuilder`]: the block's groups (twin row sets, encoded as
//! ranges) and its labels.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use pulsar_data::{Column, Table};
use pulsar_store::RunId;
use serde_json::{json, Value};

use super::capability::released_dataset_path;
use crate::state::{LiveRun, ServerState};

/// Group identifier for a cluster id.
pub fn group_id(cid: i64) -> String {
    format!("c{cid}")
}

/// A JSON Pointer reference token for `s` (RFC 6901 escaping).
pub(super) fn escape_token(s: &str) -> String {
    s.replace('~', "~0").replace('/', "~1")
}

/// The value evidence uses for a missing categorical cell.
const MISSING_VALUE: &str = "__MISSING__";

/// `{enc: "ranges", v: [[start, end), ...]}` from ascending row indices.
fn rows_json(rows: &[usize]) -> Value {
    let mut ranges: Vec<[usize; 2]> = Vec::new();
    for &r in rows {
        if let Some(last) = ranges.last_mut() {
            if last[1] == r {
                last[1] = r + 1;
                continue;
            }
        }
        ranges.push([r, r + 1]);
    }
    json!({
        "enc": "ranges",
        "v": ranges,
    })
}

/// Group entries (`{id, rows: {enc: "ranges", v: [[start, end), ...]}}`) from
/// a per-row cluster label slice where node index == twin row index.
pub fn groups_of(labels: &[i64]) -> Vec<Value> {
    let mut by_cluster: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (row, &cid) in labels.iter().enumerate() {
        by_cluster.entry(cid).or_default().push(row);
    }
    by_cluster
        .into_iter()
        .filter(|(_, rows)| !rows.is_empty())
        .map(|(cid, rows)| {
            json!({
                "id": group_id(cid),
                "rows": rows_json(&rows),
            })
        })
        .collect()
}

/// A categorical cell's value encoded as the evidence rows encode it.
fn encoded_value(column: &Column, row: usize) -> Option<String> {
    match column {
        Column::Bool(v) => v.get(row).map(|b| match b {
            Some(true) => "True".to_owned(),
            Some(false) => "False".to_owned(),
            None => MISSING_VALUE.to_owned(),
        }),
        Column::Text(v) => v
            .get(row)
            .map(|s| s.clone().unwrap_or_else(|| MISSING_VALUE.to_owned())),
        _ => None,
    }
}

enum ReleaseTable {
    Live(Arc<LiveRun>),
    #[cfg(test)]
    Owned(Table),
}

impl ReleaseTable {
    fn table(&self) -> &Table {
        match self {
            Self::Live(live) => &live.table,
            #[cfg(test)]
            Self::Owned(table) => table,
        }
    }
}

/// Builder for an `io.phirewall/release` v1 block.
pub struct ReleaseBuilder {
    dataset_path: String,
    groups: Vec<Value>,
    labels: Vec<Value>,
    cluster_labels: Vec<i64>,
    table: Option<ReleaseTable>,
    group_ids: BTreeSet<String>,
    levels: BTreeMap<String, BTreeMap<String, usize>>,
}

impl ReleaseBuilder {
    pub fn new(state: &ServerState, run_id: &RunId, cluster_labels: &[i64]) -> Option<Self> {
        let dataset_path = released_dataset_path(state, run_id)?;
        let table = state.cache.get(run_id).map(ReleaseTable::Live);
        Self::from_source(dataset_path, cluster_labels, table)
    }

    fn from_source(
        dataset_path: String,
        cluster_labels: &[i64],
        table: Option<ReleaseTable>,
    ) -> Option<Self> {
        let groups = groups_of(cluster_labels);
        if groups.is_empty() {
            return None;
        }
        let group_ids = groups
            .iter()
            .filter_map(|g| g["id"].as_str().map(str::to_owned))
            .collect();
        Some(Self {
            dataset_path,
            groups,
            labels: Vec::new(),
            cluster_labels: cluster_labels.to_vec(),
            table,
            group_ids,
            levels: BTreeMap::new(),
        })
    }

    #[cfg(test)]
    pub(super) fn for_test(
        dataset_path: &str,
        cluster_labels: &[i64],
        table: Table,
    ) -> Option<Self> {
        Self::from_source(
            dataset_path.to_owned(),
            cluster_labels,
            Some(ReleaseTable::Owned(table)),
        )
    }

    fn table(&self) -> Option<&Table> {
        self.table.as_ref().map(ReleaseTable::table)
    }

    /// Index of `value` among its column's distinct levels (sorted, so the
    /// id never carries the raw value).
    fn level_index(&mut self, column: &str, value: &str) -> Option<usize> {
        if let Some(levels) = self.levels.get(column) {
            return levels.get(value).copied();
        }
        let table = self.table()?;
        let values = table.column(column)?;
        let levels: BTreeSet<String> = (0..values.len())
            .filter_map(|row| encoded_value(values, row))
            .collect();
        let index = levels.iter().position(|v| v == value);
        self.levels.insert(
            column.to_owned(),
            levels
                .into_iter()
                .enumerate()
                .map(|(i, v)| (v, i))
                .collect(),
        );
        index
    }

    /// Declare `rows` as group `id` (once), unless there are none.
    fn named_group(&mut self, id: String, rows: Vec<usize>) -> Option<String> {
        if rows.is_empty() {
            return None;
        }
        if self.group_ids.insert(id.clone()) {
            self.groups.push(json!({
                "id": id,
                "rows": rows_json(&rows),
            }));
        }
        Some(id)
    }

    /// Rows in `cid` whose `column` equals `value`, as a named sub-group.
    pub(super) fn subgroup(&mut self, cid: i64, column: &str, value: &str) -> Option<String> {
        let level = self.level_index(column, value)?;
        let id = format!("c{cid}:{column}={level}");
        let values = self.table()?.column(column)?;
        let rows: Vec<usize> = self
            .cluster_labels
            .iter()
            .enumerate()
            .filter(|&(row, &c)| c == cid && encoded_value(values, row).as_deref() == Some(value))
            .map(|(row, _)| row)
            .collect();
        self.named_group(id, rows)
    }

    /// Every row whose `column` equals `value`, as a named group.
    pub(super) fn value_group(&mut self, column: &str, value: &str) -> Option<String> {
        let level = self.level_index(column, value)?;
        let id = format!("v:{column}={level}");
        let values = self.table()?.column(column)?;
        let rows: Vec<usize> = (0..values.len())
            .filter(|&row| encoded_value(values, row).as_deref() == Some(value))
            .collect();
        self.named_group(id, rows)
    }

    pub fn count(&mut self, path: impl Into<String>, group: &str) {
        self.labels.push(json!({
            "path": path.into(),
            "kind": "count",
            "group": group,
        }));
    }

    pub fn share(&mut self, path: impl Into<String>, group: &str, of: &str) {
        self.labels.push(json!({
            "path": path.into(),
            "kind": "share",
            "group": group,
            "of": of,
        }));
    }

    pub fn statistic(&mut self, path: impl Into<String>, group: &str, vs: &str) {
        self.labels.push(json!({
            "path": path.into(),
            "kind": "statistic",
            "group": group,
            "vs": vs,
        }));
    }

    pub fn label(&mut self, path: impl Into<String>, group: &str, column: &str) {
        self.labels.push(json!({
            "path": path.into(),
            "kind": "label",
            "group": group,
            "column": column,
        }));
    }

    pub fn records(&mut self, path: impl Into<String>, group: &str) {
        self.labels.push(json!({
            "path": path.into(),
            "kind": "records",
            "group": group,
        }));
    }

    pub fn finish(self) -> Value {
        json!({
            "version": 1,
            "dataset_path": self.dataset_path,
            "groups": self.groups,
            "labels": self.labels,
        })
    }
}
