//! The `io.phirewall/release` v1 block: declares every group a cluster
//! payload speaks about as twin row ranges and labels every group-derived
//! count, share, statistic, category level and record list by RFC 6901 JSON
//! Pointer so the Phirewall gateway can verify disclosure centrally.
//!
//! Emitted only when the run's dataset record has `DatasetSource::Path`;
//! inline datasets have no host path for Phirewall to resolve against its
//! twin registry.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use pulsar_data::{Column, Table};
use pulsar_store::{DatasetSource, RunId};
use rmcp::model::{CallToolResult, ClientCapabilities};
use serde_json::{json, Value};

use crate::state::{ClusterState, LiveRun, ServerState};

/// The structured-content key.
pub const KEY: &str = "io.phirewall/release";

/// True iff the client declared experimental `io.phirewall/release` with `{"version": 1}`.
pub fn declared_by(capabilities: &ClientCapabilities) -> bool {
    capabilities
        .experimental
        .as_ref()
        .and_then(|exp| exp.get(KEY))
        .and_then(|v| v.get("version"))
        .and_then(Value::as_u64)
        == Some(1)
}

/// Group identifier for a cluster id.
pub fn group_id(cid: i64) -> String {
    format!("c{cid}")
}

fn escape_token(s: &str) -> String {
    s.replace('~', "~0").replace('/', "~1")
}

fn dataset_path_of(state: &ServerState, run_id: &RunId) -> Option<String> {
    let run = state.store.get_run(run_id).ok()?;
    let ds_id = run.dataset_id?;
    let ds = state.store.get_dataset(&ds_id).ok()?;
    if ds.source != DatasetSource::Path {
        return None;
    }
    Some(ds.path.display().to_string())
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
        if !state.phirewall_release() {
            return None;
        }
        let dataset_path = dataset_path_of(state, run_id)?;
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
    fn for_test(dataset_path: &str, cluster_labels: &[i64], table: Table) -> Option<Self> {
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
        let mut levels = BTreeSet::new();
        for row in 0..values.len() {
            if let Some(v) = encoded_value(values, row) {
                levels.insert(v);
            }
        }
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

    /// Rows in `cid` whose `column` equals `value`, as a named sub-group.
    fn subgroup(&mut self, cid: i64, column: &str, value: &str) -> Option<String> {
        let level = self.level_index(column, value)?;
        let id = format!("c{cid}:{column}={level}");
        let table = self.table()?;
        let values = table.column(column)?;
        let rows: Vec<usize> = self
            .cluster_labels
            .iter()
            .enumerate()
            .filter(|&(row, &c)| c == cid && encoded_value(values, row).as_deref() == Some(value))
            .map(|(row, _)| row)
            .collect();
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

    /// Every row whose `column` equals `value`, as a named group.
    fn value_group(&mut self, column: &str, value: &str) -> Option<String> {
        let level = self.level_index(column, value)?;
        let id = format!("v:{column}={level}");
        let table = self.table()?;
        let values = table.column(column)?;
        let rows: Vec<usize> = (0..values.len())
            .filter(|&row| encoded_value(values, row).as_deref() == Some(value))
            .collect();
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

    fn cluster_result(&mut self, payload: &Value) {
        let Some(sizes) = payload["cluster_result"]["cluster_sizes"].as_array() else {
            return;
        };
        for (i, entry) in sizes.iter().enumerate() {
            let Some(cid) = entry["cluster_id"].as_i64() else {
                continue;
            };
            let gid = group_id(cid);
            if entry.get("n").is_some_and(Value::is_number) {
                self.count(format!("/cluster_result/cluster_sizes/{i}/n"), &gid);
            }
            if entry.get("pct").is_some_and(Value::is_number) {
                self.share(
                    format!("/cluster_result/cluster_sizes/{i}/pct"),
                    &gid,
                    "all",
                );
            }
        }
    }

    fn feature_preview_row(&mut self, base: &str, row: &Value, cid: i64) {
        if row["kind"].as_str() != Some("categorical") {
            return;
        }
        let (Some(column), Some(value)) = (row["column"].as_str(), row["value"].as_str()) else {
            return;
        };
        let gid = group_id(cid);
        self.label(format!("{base}/value"), &gid, column);
        if row.get("prevalence_cluster").is_some_and(Value::is_number) {
            if let Some(sub) = self.subgroup(cid, column, value) {
                self.share(format!("{base}/prevalence_cluster"), &sub, &gid);
            }
        }
    }

    fn numeric_feature_row(&mut self, base: &str, row: &Value, gid: &str) {
        for k in [
            "z_score",
            "effect_mean_std",
            "effect_median_mad",
            "ks_stat",
            "wasserstein_norm",
            "variance_ratio_log",
            "one_vs_rest_p",
            "one_vs_rest_q",
        ] {
            if row.get(k).is_some_and(Value::is_number) {
                self.statistic(format!("{base}/{k}"), gid, "rest");
            }
        }
    }

    fn categorical_feature_row(&mut self, base: &str, row: &Value, cid: i64) {
        let gid = group_id(cid);
        let column = row["column"].as_str();
        let value = row["value"].as_str();
        if let (Some(column), Some(_)) = (column, value) {
            self.label(format!("{base}/value"), &gid, column);
        }
        let sub = match (column, value) {
            (Some(column), Some(value)) => self.subgroup(cid, column, value),
            _ => None,
        };
        if let Some(sub) = &sub {
            for k in ["count", "count_cluster"] {
                if row.get(k).is_some_and(Value::is_number) {
                    self.count(format!("{base}/{k}"), sub);
                }
            }
            for k in [
                "prevalence_cluster",
                "in_cluster_prevalence",
                "concentration",
            ] {
                if row.get(k).is_some_and(Value::is_number) {
                    self.share(format!("{base}/{k}"), sub, &gid);
                }
            }
            if row.get("global_recall").is_some_and(Value::is_number) {
                if let (Some(column), Some(value)) = (column, value) {
                    if let Some(all_with_value) = self.value_group(column, value) {
                        self.share(format!("{base}/global_recall"), sub, &all_with_value);
                    }
                }
            }
        }
        if row.get("cluster_size").is_some_and(Value::is_number) {
            self.count(format!("{base}/cluster_size"), &gid);
        }
        for k in ["fisher_p", "fisher_q"] {
            if row.get(k).is_some_and(Value::is_number) {
                self.statistic(format!("{base}/{k}"), &gid, "rest");
            }
        }
    }

    fn cluster_entry(&mut self, base: &str, cluster: &Value) {
        let Some(cid) = cluster["cluster_id"].as_i64() else {
            return;
        };
        let gid = group_id(cid);
        if cluster.get("size").is_some_and(Value::is_number) {
            self.count(format!("{base}/size"), &gid);
        }
        if cluster.get("cluster_size").is_some_and(Value::is_number) {
            self.count(format!("{base}/cluster_size"), &gid);
        }
        if cluster.get("size_pct").is_some_and(Value::is_number) {
            self.share(format!("{base}/size_pct"), &gid, "all");
        }
        if cluster.get("central_rows").is_some_and(Value::is_array) {
            self.records(format!("{base}/central_rows"), &gid);
        }
        if let Some(arr) = cluster["feature_preview"].as_array() {
            for (j, row) in arr.iter().enumerate() {
                self.feature_preview_row(&format!("{base}/feature_preview/{j}"), row, cid);
            }
        }
        if let Some(arr) = cluster["numeric_features"].as_array() {
            for (j, row) in arr.iter().enumerate() {
                self.numeric_feature_row(&format!("{base}/numeric_features/{j}"), row, &gid);
            }
        }
        if let Some(arr) = cluster["categorical_features"].as_array() {
            for (j, row) in arr.iter().enumerate() {
                self.categorical_feature_row(&format!("{base}/categorical_features/{j}"), row, cid);
            }
        }
        if let Some(tiers) = cluster["numeric_tiers"].as_object() {
            for (tier, val) in tiers {
                if let Some(arr) = val.as_array() {
                    let t = escape_token(tier);
                    for (j, row) in arr.iter().enumerate() {
                        self.numeric_feature_row(
                            &format!("{base}/numeric_tiers/{t}/{j}"),
                            row,
                            &gid,
                        );
                    }
                }
            }
        }
        if let Some(tiers) = cluster["categorical_tiers"].as_object() {
            for (tier, val) in tiers {
                if let Some(arr) = val.as_array() {
                    let t = escape_token(tier);
                    for (j, row) in arr.iter().enumerate() {
                        self.categorical_feature_row(
                            &format!("{base}/categorical_tiers/{t}/{j}"),
                            row,
                            cid,
                        );
                    }
                }
            }
        }
    }

    fn signal_matrix(&mut self, base: &str, sm: &Value) {
        if let Some(arr) = sm["omitted_clusters"].as_array() {
            for (i, c) in arr.iter().enumerate() {
                if let Some(cid) = c["cluster_id"].as_i64() {
                    if c.get("max_signal").is_some_and(Value::is_number) {
                        self.statistic(
                            format!("{base}/omitted_clusters/{i}/max_signal"),
                            &group_id(cid),
                            "rest",
                        );
                    }
                }
            }
        }
        if let Some(arr) = sm["numeric_rows"].as_array() {
            for (i, row) in arr.iter().enumerate() {
                let Some(cid) = row["cluster_id"].as_i64() else {
                    continue;
                };
                let gid = group_id(cid);
                if let Some(vals) = row["values"].as_object() {
                    for (col, val) in vals {
                        if val.is_number() {
                            self.statistic(
                                format!("{base}/numeric_rows/{i}/values/{}", escape_token(col)),
                                &gid,
                                "rest",
                            );
                        }
                    }
                }
            }
        }
        if let Some(arr) = sm["categorical_rows"].as_array() {
            for (i, row) in arr.iter().enumerate() {
                let Some(cid) = row["cluster_id"].as_i64() else {
                    continue;
                };
                let gid = group_id(cid);
                if let Some(vals) = row["values"].as_object() {
                    for (k, val) in vals {
                        let Some((column, value)) = k.split_once('=') else {
                            continue;
                        };
                        let Some(sub) = self.subgroup(cid, column, value) else {
                            continue;
                        };
                        let token = escape_token(k);
                        let path = format!("{base}/categorical_rows/{i}/values/{token}");
                        if val.is_number() {
                            self.share(path, &sub, &gid);
                        } else if let Some(cell) = val.as_object() {
                            for field in ["value", "prevalence"] {
                                if cell.get(field).is_some_and(Value::is_number) {
                                    self.share(format!("{path}/{field}"), &sub, &gid);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Attach `io.phirewall/release` to a `generate_cluster_dossier` tool result.
pub fn attach_dossier(state: &ServerState, result: &mut CallToolResult) {
    if !state.phirewall_release() {
        return;
    }
    let Some(clusters) = state.session().clusters.clone() else {
        return;
    };
    let Some(mut builder) = ReleaseBuilder::new(state, &clusters.run_id, &clusters.labels) else {
        return;
    };
    let Some(payload) = result.structured_content.as_mut() else {
        return;
    };
    builder.cluster_result(payload);
    if let Some(arr) = payload["cluster_index"]["clusters"].as_array() {
        for (i, entry) in arr.iter().enumerate() {
            if let Some(cid) = entry["cluster_id"].as_i64() {
                if entry.get("size").is_some_and(Value::is_number) {
                    builder.count(format!("/cluster_index/clusters/{i}/size"), &group_id(cid));
                }
            }
        }
    }
    if let Some(arr) = payload["clusters"].as_array() {
        for (i, c) in arr.iter().enumerate() {
            builder.cluster_entry(&format!("/clusters/{i}"), c);
        }
    }
    if payload.get("signal_matrix").is_some_and(Value::is_object) {
        builder.signal_matrix("/signal_matrix", &payload["signal_matrix"]);
    }
    payload[KEY] = builder.finish();
}

/// Attach `io.phirewall/release` to a `get_cluster_detail` payload.
pub fn attach_detail(state: &ServerState, run_id: &RunId, labels: &[i64], payload: &mut Value) {
    let Some(mut builder) = ReleaseBuilder::new(state, run_id, labels) else {
        return;
    };
    if let Some(arr) = payload["clusters"].as_array() {
        for (i, c) in arr.iter().enumerate() {
            builder.cluster_entry(&format!("/clusters/{i}"), c);
        }
    }
    payload[KEY] = builder.finish();
}

/// Attach `io.phirewall/release` to a `merge_small_groups` payload.
pub fn attach_merge(state: &ServerState, run_id: &RunId, labels: &[i64], payload: &mut Value) {
    let Some(mut builder) = ReleaseBuilder::new(state, run_id, labels) else {
        return;
    };
    if let Some(sizes) = payload["cluster_sizes"].as_array() {
        for (i, entry) in sizes.iter().enumerate() {
            if let Some(cid) = entry["cluster_id"].as_i64() {
                if entry.get("size").is_some_and(Value::is_number) {
                    builder.count(format!("/cluster_sizes/{i}/size"), &group_id(cid));
                }
            }
        }
    }
    payload[KEY] = builder.finish();
}

/// Attach `io.phirewall/release` to a `get_cluster_profile` payload.
pub fn attach_profile(state: &ServerState, clusters: &ClusterState, payload: &mut Value) {
    let Some(mut builder) = ReleaseBuilder::new(state, &clusters.run_id, &clusters.labels) else {
        return;
    };
    builder.cluster_result(payload);
    builder.cluster_entry("/cluster", &payload["cluster"]);
    payload[KEY] = builder.finish();
}

/// Attach `io.phirewall/release` to a `get_feature_signal` payload.
pub fn attach_signal(state: &ServerState, clusters: &ClusterState, payload: &mut Value) {
    let Some(mut builder) = ReleaseBuilder::new(state, &clusters.run_id, &clusters.labels) else {
        return;
    };
    builder.cluster_result(payload);
    if let Some(arr) = payload["signals"]["clusters"].as_array() {
        for (i, c) in arr.iter().enumerate() {
            builder.cluster_entry(&format!("/signals/clusters/{i}"), c);
        }
    }
    payload[KEY] = builder.finish();
}

/// Attach `io.phirewall/release` to a `get_cluster_signal_matrix` payload.
pub fn attach_matrix(state: &ServerState, clusters: &ClusterState, payload: &mut Value) {
    let Some(mut builder) = ReleaseBuilder::new(state, &clusters.run_id, &clusters.labels) else {
        return;
    };
    builder.cluster_result(payload);
    builder.signal_matrix("/signal_matrix", &payload["signal_matrix"]);
    payload[KEY] = builder.finish();
}

/// Attach `io.phirewall/release` to a `compare_clusters` payload.
pub fn attach_compare(
    state: &ServerState,
    clusters: &ClusterState,
    a: i64,
    b: i64,
    payload: &mut Value,
) {
    let Some(mut builder) = ReleaseBuilder::new(state, &clusters.run_id, &clusters.labels) else {
        return;
    };
    let ga = group_id(a);
    let gb = group_id(b);
    if let Some(rows) = payload["results"].as_array() {
        for (i, row) in rows.iter().enumerate() {
            if row.get("mean_a").is_some_and(Value::is_number) {
                builder.statistic(format!("/results/{i}/mean_a"), &ga, &gb);
            }
            if row.get("mean_b").is_some_and(Value::is_number) {
                builder.statistic(format!("/results/{i}/mean_b"), &gb, &ga);
            }
            for k in ["p_val_t", "p_val_ks", "cohens_d"] {
                if row.get(k).is_some_and(Value::is_number) {
                    builder.statistic(format!("/results/{i}/{k}"), &ga, &gb);
                }
            }
        }
    }
    payload[KEY] = builder.finish();
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn table() -> Table {
        let species = (0..30)
            .map(|row| Some(if row < 12 { "Adelie" } else { "Gentoo" }.to_owned()))
            .collect();
        Table::new(vec!["species".to_owned()], vec![Column::Text(species)]).unwrap()
    }

    fn cluster() -> Value {
        let adelie = json!({
            "column": "species",
            "value": "Adelie",
            "cluster_id": 0,
            "count": 12,
            "count_cluster": 12,
            "count_rest": 0,
            "cluster_size": 20,
            "global_count": 12,
            "prevalence_cluster": 60.0,
            "prevalence_rest": 0.0,
            "prevalence_global": 40.0,
            "lift": 1.5,
            "log_lift": 0.405_465_108_118_159_83,
            "specificity": 60.0,
            "global_recall": 100.0,
            "neighbor_specificity": 0.0,
            "mi_contrib": 0.1,
            "fisher_p": 0.03,
            "fisher_q": 0.04,
            "global_p": 0.05,
            "global_q": 0.06,
            "aggregate_score": 0.8,
            "percentile_score": 90.0,
            "signal_tier": "core",
            "in_cluster_prevalence": 60.0,
            "concentration": 60.0,
            "test_method": "chi2",
            "failure_reasons": [],
            "evidence_vector": {},
        });
        let gentoo = json!({
            "column": "species",
            "value": "Gentoo",
            "cluster_id": 0,
            "count": 8,
            "count_cluster": 8,
            "count_rest": 10,
            "cluster_size": 20,
            "global_count": 18,
            "prevalence_cluster": 40.0,
            "prevalence_rest": 100.0,
            "prevalence_global": 60.0,
            "lift": 0.666_666_666_666_666_6,
            "log_lift": -0.405_465_108_118_159_9,
            "specificity": -60.0,
            "global_recall": 44.44,
            "neighbor_specificity": -100.0,
            "mi_contrib": 0.2,
            "fisher_p": 0.01,
            "fisher_q": 0.02,
            "global_p": 0.05,
            "global_q": 0.06,
            "aggregate_score": 0.9,
            "percentile_score": 95.0,
            "signal_tier": "core",
            "in_cluster_prevalence": 40.0,
            "concentration": 40.0,
            "test_method": "fisher",
            "failure_reasons": [],
            "evidence_vector": {},
        });
        let numeric = json!({
            "column": "age",
            "cluster_id": 0,
            "direction": "High",
            "mean": 5.0,
            "mean_rest": 3.0,
            "median": 5.0,
            "median_rest": 3.0,
            "std_cluster": 1.0,
            "std_rest": 1.0,
            "mad_cluster": 1.0,
            "mad_rest": 1.0,
            "global_mean": 4.0,
            "global_mean_rest": 3.0,
            "z_score": 1.0,
            "homogeneity": 0.5,
            "effect_mean_std": 2.0,
            "effect_median_mad": 2.0,
            "ks_stat": 0.5,
            "wasserstein_norm": 0.4,
            "variance_ratio_log": 0.1,
            "concentration_gain": 0.2,
            "one_vs_rest_p": 0.01,
            "one_vs_rest_q": 0.02,
            "global_p": 0.03,
            "global_q": 0.04,
            "global_cluster_signal": 0.5,
            "neighbor_cluster_id": 1,
            "neighbor_effect": 0.3,
            "neighbor_p": 0.05,
            "specificity_score": 0.7,
            "aggregate_score": 0.8,
            "percentile_score": 90.0,
            "signal_tier": "core",
            "sparkline": "",
            "failure_reasons": [],
            "evidence_vector": {},
        });
        let preview = json!([
            {
                "kind": "numeric",
                "column": "age",
                "signal_tier": "core",
                "aggregate_score": 0.8,
                "direction": "High",
                "mean": 5.0,
                "mean_rest": 3.0,
            },
            {
                "kind": "categorical",
                "column": "species",
                "signal_tier": "core",
                "aggregate_score": 0.8,
                "value": "Adelie",
                "lift": 1.5,
                "prevalence_cluster": 60.0,
            },
        ]);
        let mut cluster = json!({
            "cluster_id": 0,
            "size": 20,
            "cluster_size": 20,
            "size_pct": 66.67,
            "central_rows": [],
        });
        cluster["feature_preview"] = preview;
        cluster["numeric_features"] = json!([numeric]);
        cluster["categorical_features"] = json!([adelie, gentoo]);
        cluster
    }

    fn group_rows(release: &Value) -> BTreeMap<String, BTreeSet<usize>> {
        let mut groups = BTreeMap::new();
        for group in release["groups"].as_array().unwrap() {
            let id = group["id"].as_str().unwrap().to_owned();
            let mut rows = BTreeSet::new();
            for range in group["rows"]["v"].as_array().unwrap() {
                let start = range[0].as_u64().unwrap() as usize;
                let end = range[1].as_u64().unwrap() as usize;
                rows.extend(start..end);
            }
            assert!(groups.insert(id, rows).is_none(), "duplicate group");
        }
        groups
    }

    #[test]
    fn labels_match_their_twin_rows() {
        let labels: Vec<i64> = (0..30).map(|row| if row < 20 { 0 } else { 1 }).collect();
        let mut builder = ReleaseBuilder::for_test("/tmp/twin.csv", &labels, table()).unwrap();
        let cluster = cluster();
        let matrix = json!({
            "numeric_rows": [{"cluster_id": 0, "values": {"age": 1.0}}],
            "categorical_rows": [
                {"cluster_id": 0, "values": {"species=Adelie": 60.0}},
                {"cluster_id": 0, "values": {"species=Gentoo": {
                    "value": 40.0,
                    "prevalence": 40.0,
                    "aggregate_score": 0.9,
                    "signal_tier": "core",
                }}},
            ],
        });
        builder.cluster_entry("/clusters/0", &cluster);
        builder.signal_matrix("/signal_matrix", &matrix);
        let release = builder.finish();
        let payload = json!({"clusters": [cluster], "signal_matrix": matrix});

        let groups = group_rows(&release);
        let all = labels.len();
        let mut paths = BTreeSet::new();
        for label in release["labels"].as_array().unwrap() {
            let path = label["path"].as_str().unwrap();
            paths.insert(path.to_owned());
            let target = payload
                .pointer(path)
                .unwrap_or_else(|| panic!("{path} does not resolve"));
            match label["kind"].as_str().unwrap() {
                "count" => {
                    let group = groups.get(label["group"].as_str().unwrap()).unwrap();
                    assert_eq!(
                        target.as_f64().unwrap(),
                        group.len() as f64,
                        "{path} does not count its group"
                    );
                }
                "share" => {
                    let group = groups.get(label["group"].as_str().unwrap()).unwrap();
                    let of = label["of"].as_str().unwrap();
                    let of_len = if of == "all" {
                        all
                    } else {
                        groups.get(of).unwrap().len()
                    };
                    let scale = if target.as_f64().unwrap() > 1.0 {
                        100.0
                    } else {
                        1.0
                    };
                    let expected =
                        (group.len() as f64 / of_len as f64 * scale * 100.0).round() / 100.0;
                    assert_eq!(
                        target.as_f64().unwrap(),
                        expected,
                        "{path} does not describe its twin-row share"
                    );
                }
                "label" => assert!(target.is_string(), "{path} is not a string"),
                _ => {}
            }
        }

        // Group ids name levels by index, never by raw value.
        for id in groups.keys() {
            assert!(!id.contains("Adelie"), "{id} carries a raw value");
            assert!(!id.contains("Gentoo"), "{id} carries a raw value");
        }
        for id in ["c0:species=0", "c0:species=1", "v:species=0", "v:species=1"] {
            assert!(groups.contains_key(id), "missing group {id}");
        }

        // Numeric rows label only c0-vs-rest statistics.
        for key in [
            "neighbor_cluster_id",
            "global_mean",
            "mean_rest",
            "median_rest",
            "neighbor_effect",
        ] {
            let path = format!("/clusters/0/numeric_features/0/{key}");
            assert!(!paths.contains(&path), "{path} must stay unlabelled");
        }
        for key in [
            "z_score",
            "effect_mean_std",
            "effect_median_mad",
            "ks_stat",
            "wasserstein_norm",
            "variance_ratio_log",
            "one_vs_rest_p",
            "one_vs_rest_q",
        ] {
            let path = format!("/clusters/0/numeric_features/0/{key}");
            assert!(paths.contains(&path), "{path} should be labelled");
        }

        // Rest/global preview values are never attributed to c0.
        for path in [
            "/clusters/0/feature_preview/0/mean_rest",
            "/clusters/0/feature_preview/1/lift",
            "/clusters/0/categorical_features/0/prevalence_rest",
            "/clusters/0/categorical_features/0/lift",
        ] {
            assert!(!paths.contains(path), "{path} must stay unlabelled");
        }
        assert!(paths.contains("/clusters/0/feature_preview/1/prevalence_cluster"));

        // Matrix cells are self-scoped shares of their categorical sub-group.
        assert!(paths.contains("/signal_matrix/categorical_rows/0/values/species=Adelie"));
        assert!(paths.contains("/signal_matrix/categorical_rows/1/values/species=Gentoo/value"));
    }

    #[test]
    fn neighbor_cluster_id_and_global_mean_carry_no_label() {
        let labels: Vec<i64> = (0..30).map(|row| if row < 20 { 0 } else { 1 }).collect();
        let mut builder = ReleaseBuilder::for_test("/tmp/twin.csv", &labels, table()).unwrap();
        let cluster = cluster();
        builder.cluster_entry("/clusters/0", &cluster);
        let release = builder.finish();
        let paths: BTreeSet<String> = release["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| l["path"].as_str().unwrap().to_owned())
            .collect();
        assert!(
            !paths.contains("/clusters/0/numeric_features/0/neighbor_cluster_id"),
            "neighbor_cluster_id must carry no label"
        );
        assert!(
            !paths.contains("/clusters/0/numeric_features/0/global_mean"),
            "global_mean must carry no label"
        );
    }

    #[test]
    fn group_ids_contain_no_raw_values() {
        let labels: Vec<i64> = (0..30).map(|row| if row < 20 { 0 } else { 1 }).collect();
        let mut builder = ReleaseBuilder::for_test("/tmp/twin.csv", &labels, table()).unwrap();
        let cluster = cluster();
        builder.cluster_entry("/clusters/0", &cluster);
        let release = builder.finish();
        for g in release["groups"].as_array().unwrap() {
            let id = g["id"].as_str().unwrap();
            assert!(
                !id.contains("Adelie"),
                "group id {id} contains raw value Adelie"
            );
            assert!(
                !id.contains("Gentoo"),
                "group id {id} contains raw value Gentoo"
            );
        }
    }
}
