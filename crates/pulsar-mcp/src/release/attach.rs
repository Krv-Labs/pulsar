//! Which payload paths of each cluster tool get which release labels, and
//! the `attach_*` entry point per tool.
//!
//! Labels are emitted in payload order (and sub-groups declared as they are
//! first met), so a block is a pure function of its payload.

use pulsar_store::RunId;
use rmcp::model::CallToolResult;
use serde_json::{Map, Value};

use super::builder::{escape_token, group_id, ReleaseBuilder};
use super::KEY;
use crate::state::{ClusterState, ServerState};

/// Numeric feature fields that compare a cluster with the rest.
const NUMERIC_STATISTICS: [&str; 8] = [
    "z_score",
    "effect_mean_std",
    "effect_median_mad",
    "ks_stat",
    "wasserstein_norm",
    "variance_ratio_log",
    "one_vs_rest_p",
    "one_vs_rest_q",
];

/// Categorical feature fields that share a value's sub-group of its cluster.
const CATEGORICAL_SHARES: [&str; 3] = [
    "prevalence_cluster",
    "in_cluster_prevalence",
    "concentration",
];

/// True iff `v[key]` is a number.
fn is_num(v: &Value, key: &str) -> bool {
    v.get(key).is_some_and(Value::is_number)
}

/// `(index, item)` of `v` if it is an array; nothing otherwise.
fn items(v: &Value) -> impl Iterator<Item = (usize, &Value)> {
    v.as_array().into_iter().flatten().enumerate()
}

/// `(key, value)` of `v` if it is an object; nothing otherwise.
fn entries(v: &Value) -> impl Iterator<Item = (&String, &Value)> {
    v.as_object().into_iter().flat_map(Map::iter)
}

/// `(escaped tier, index, row)` of a `{tier: [row, ...]}` object.
fn tier_rows(v: &Value) -> impl Iterator<Item = (String, usize, &Value)> {
    entries(v).flat_map(|(tier, rows)| {
        let t = escape_token(tier);
        items(rows).map(move |(j, row)| (t.clone(), j, row))
    })
}

impl ReleaseBuilder {
    pub(super) fn cluster_result(&mut self, payload: &Value) {
        let Some(sizes) = payload["cluster_result"]["cluster_sizes"].as_array() else {
            return;
        };
        for (i, entry) in sizes.iter().enumerate() {
            let Some(cid) = entry["cluster_id"].as_i64() else {
                continue;
            };
            let gid = group_id(cid);
            if is_num(entry, "n") {
                self.count(format!("/cluster_result/cluster_sizes/{i}/n"), &gid);
            }
            if is_num(entry, "pct") {
                self.share(
                    format!("/cluster_result/cluster_sizes/{i}/pct"),
                    &gid,
                    "all",
                );
            }
        }
    }

    /// Counts of `/{list}/{i}/{field}`, for entries that name a cluster.
    fn cluster_counts(&mut self, list: &str, field: &str, payload: &Value) {
        for (i, entry) in items(payload) {
            if let Some(cid) = entry["cluster_id"].as_i64() {
                if is_num(entry, field) {
                    self.count(format!("{list}/{i}/{field}"), &group_id(cid));
                }
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
        if is_num(row, "prevalence_cluster") {
            if let Some(sub) = self.subgroup(cid, column, value) {
                self.share(format!("{base}/prevalence_cluster"), &sub, &gid);
            }
        }
    }

    fn numeric_feature_row(&mut self, base: &str, row: &Value, gid: &str) {
        for k in NUMERIC_STATISTICS {
            if is_num(row, k) {
                self.statistic(format!("{base}/{k}"), gid, "rest");
            }
        }
    }

    fn categorical_feature_row(&mut self, base: &str, row: &Value, cid: i64) {
        let gid = group_id(cid);
        let cell = row["column"].as_str().zip(row["value"].as_str());
        if let Some((column, _)) = cell {
            self.label(format!("{base}/value"), &gid, column);
        }
        let sub = cell.and_then(|(column, value)| self.subgroup(cid, column, value));
        if let (Some(sub), Some((column, value))) = (&sub, cell) {
            self.value_shares(base, row, sub, &gid, (column, value));
        }
        if is_num(row, "cluster_size") {
            self.count(format!("{base}/cluster_size"), &gid);
        }
        for k in ["fisher_p", "fisher_q"] {
            if is_num(row, k) {
                self.statistic(format!("{base}/{k}"), &gid, "rest");
            }
        }
    }

    /// The counts and shares of a categorical row's value sub-group `sub`.
    fn value_shares(
        &mut self,
        base: &str,
        row: &Value,
        sub: &str,
        gid: &str,
        (column, value): (&str, &str),
    ) {
        for k in ["count", "count_cluster"] {
            if is_num(row, k) {
                self.count(format!("{base}/{k}"), sub);
            }
        }
        for k in CATEGORICAL_SHARES {
            if is_num(row, k) {
                self.share(format!("{base}/{k}"), sub, gid);
            }
        }
        if is_num(row, "global_recall") {
            if let Some(all_with_value) = self.value_group(column, value) {
                self.share(format!("{base}/global_recall"), sub, &all_with_value);
            }
        }
    }

    pub(super) fn cluster_entry(&mut self, base: &str, cluster: &Value) {
        let Some(cid) = cluster["cluster_id"].as_i64() else {
            return;
        };
        let gid = group_id(cid);
        self.cluster_size_fields(base, cluster, &gid);
        self.feature_lists(base, cluster, cid, &gid);
        self.feature_tiers(base, cluster, cid, &gid);
    }

    fn cluster_size_fields(&mut self, base: &str, cluster: &Value, gid: &str) {
        for k in ["size", "cluster_size"] {
            if is_num(cluster, k) {
                self.count(format!("{base}/{k}"), gid);
            }
        }
        if is_num(cluster, "size_pct") {
            self.share(format!("{base}/size_pct"), gid, "all");
        }
        if cluster.get("central_rows").is_some_and(Value::is_array) {
            self.records(format!("{base}/central_rows"), gid);
        }
    }

    fn feature_lists(&mut self, base: &str, cluster: &Value, cid: i64, gid: &str) {
        for (j, row) in items(&cluster["feature_preview"]) {
            self.feature_preview_row(&format!("{base}/feature_preview/{j}"), row, cid);
        }
        for (j, row) in items(&cluster["numeric_features"]) {
            self.numeric_feature_row(&format!("{base}/numeric_features/{j}"), row, gid);
        }
        for (j, row) in items(&cluster["categorical_features"]) {
            self.categorical_feature_row(&format!("{base}/categorical_features/{j}"), row, cid);
        }
    }

    fn feature_tiers(&mut self, base: &str, cluster: &Value, cid: i64, gid: &str) {
        for (t, j, row) in tier_rows(&cluster["numeric_tiers"]) {
            self.numeric_feature_row(&format!("{base}/numeric_tiers/{t}/{j}"), row, gid);
        }
        for (t, j, row) in tier_rows(&cluster["categorical_tiers"]) {
            self.categorical_feature_row(&format!("{base}/categorical_tiers/{t}/{j}"), row, cid);
        }
    }

    /// Every cluster entry of the array at `payload`, under `{list}/{i}`.
    fn cluster_entries(&mut self, list: &str, payload: &Value) {
        for (i, c) in items(payload) {
            self.cluster_entry(&format!("{list}/{i}"), c);
        }
    }

    pub(super) fn signal_matrix(&mut self, base: &str, sm: &Value) {
        self.omitted_clusters(base, &sm["omitted_clusters"]);
        self.numeric_matrix_rows(base, &sm["numeric_rows"]);
        self.categorical_matrix_rows(base, &sm["categorical_rows"]);
    }

    fn omitted_clusters(&mut self, base: &str, clusters: &Value) {
        for (i, c) in items(clusters) {
            if let Some(cid) = c["cluster_id"].as_i64() {
                if is_num(c, "max_signal") {
                    self.statistic(
                        format!("{base}/omitted_clusters/{i}/max_signal"),
                        &group_id(cid),
                        "rest",
                    );
                }
            }
        }
    }

    fn numeric_matrix_rows(&mut self, base: &str, rows: &Value) {
        for (i, row) in items(rows) {
            let Some(cid) = row["cluster_id"].as_i64() else {
                continue;
            };
            let gid = group_id(cid);
            for (col, _) in entries(&row["values"]).filter(|(_, v)| v.is_number()) {
                self.statistic(
                    format!("{base}/numeric_rows/{i}/values/{}", escape_token(col)),
                    &gid,
                    "rest",
                );
            }
        }
    }

    fn categorical_matrix_rows(&mut self, base: &str, rows: &Value) {
        for (i, row) in items(rows) {
            let Some(cid) = row["cluster_id"].as_i64() else {
                continue;
            };
            let gid = group_id(cid);
            let values = format!("{base}/categorical_rows/{i}/values");
            for (key, val) in entries(&row["values"]) {
                self.categorical_matrix_cell(&values, cid, &gid, key, val);
            }
        }
    }

    /// A `column=value` cell: a share of its sub-group of the cluster, as a
    /// number or as `{value, prevalence}`.
    fn categorical_matrix_cell(
        &mut self,
        values: &str,
        cid: i64,
        gid: &str,
        key: &str,
        val: &Value,
    ) {
        let Some((column, value)) = key.split_once('=') else {
            return;
        };
        let Some(sub) = self.subgroup(cid, column, value) else {
            return;
        };
        let path = format!("{values}/{}", escape_token(key));
        if val.is_number() {
            self.share(path, &sub, gid);
            return;
        }
        for field in ["value", "prevalence"] {
            if is_num(val, field) {
                self.share(format!("{path}/{field}"), &sub, gid);
            }
        }
    }

    /// One `compare_clusters` result row: each mean against the other
    /// cluster, the tests as `a` against `b`.
    fn compare_row(&mut self, base: &str, row: &Value, ga: &str, gb: &str) {
        if is_num(row, "mean_a") {
            self.statistic(format!("{base}/mean_a"), ga, gb);
        }
        if is_num(row, "mean_b") {
            self.statistic(format!("{base}/mean_b"), gb, ga);
        }
        for k in ["p_val_t", "p_val_ks", "cohens_d"] {
            if is_num(row, k) {
                self.statistic(format!("{base}/{k}"), ga, gb);
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
    builder.cluster_counts(
        "/cluster_index/clusters",
        "size",
        &payload["cluster_index"]["clusters"],
    );
    builder.cluster_entries("/clusters", &payload["clusters"]);
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
    builder.cluster_entries("/clusters", &payload["clusters"]);
    payload[KEY] = builder.finish();
}

/// Attach `io.phirewall/release` to a `merge_small_groups` payload.
pub fn attach_merge(state: &ServerState, run_id: &RunId, labels: &[i64], payload: &mut Value) {
    let Some(mut builder) = ReleaseBuilder::new(state, run_id, labels) else {
        return;
    };
    builder.cluster_counts("/cluster_sizes", "size", &payload["cluster_sizes"]);
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
    builder.cluster_entries("/signals/clusters", &payload["signals"]["clusters"]);
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
    let (ga, gb) = (group_id(a), group_id(b));
    for (i, row) in items(&payload["results"]) {
        builder.compare_row(&format!("/results/{i}"), row, &ga, &gb);
    }
    payload[KEY] = builder.finish();
}
