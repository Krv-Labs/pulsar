//! `merge_small_groups`: merge the clusters under `min_patients` patients into
//! their most-connected neighbour, by the run's full edge weights, and publish
//! the result as a new assignment the other cluster tools read.
//!
//! The merge itself is [`pulsar_analysis::clustering::merge`]; this module
//! counts patients, publishes through the dossier's path (the new assignment
//! is stored with method `merged:<original method>`, then built on with
//! `method='assignment'`) and reports what was merged.

use std::collections::BTreeMap;

use pulsar_analysis::clustering::merge::{label_sizes, merge_small_by, Linkage, Merge};
use pulsar_analysis::disclosure::{Patients, RowCount, Unit};
use pulsar_store::{ClusterAssignment, ClusterAssignmentId};
use rmcp::model::CallToolResult;
use serde_json::{json, Value};

use super::clustering::{self, DossierDetail, MethodParam};
use super::ToolSpec;
use crate::envelope::{ok_result, ToolError};
use crate::phirewall;
use crate::release;
use crate::schema::{self as s, input_schema};
use crate::state::ServerState;

pub const NAME: &str = "merge_small_groups";

/// The fewest patients a group may have. Matches Phirewall's release floor;
/// it is Pulsar-local so a change to Phirewall's row floor cannot drop it.
const MIN_PATIENTS: usize = 8;

#[derive(Default, serde::Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct Params {
    /// Stored clustering to merge (from generate_cluster_dossier or an earlier merge). Omit for the session's current clustering.
    pub cluster_assignment_id: Option<String>,
    /// Smallest number of distinct patients a group may keep (>= 8, the default; can only be raised).
    pub min_patients: Option<usize>,
}

pub fn spec() -> ToolSpec {
    let open = s::open_object();
    ToolSpec {
        name: NAME,
        title: "Merge Small Groups",
        description: "Merge clusters under min_patients patients (default 8, via the patient token column) into their strongest-linked neighbor by full edge weights (isolated clusters join the largest), chaining until every group meets the floor. Publishes a new cluster_assignment_id (method 'merged:<original>') and returns merges [{from, into, linkage, weight_share}] (consecutive isolated merges into one group collapse to {into, linkage:'isolated', from:[...], count}) plus rounded sizes. generate_cluster_dossier with method='assignment' and this cluster_assignment_id builds the full evidence on the merged groups.",
        input_schema: input_schema::<Params>(),
        output_schema: s::as_object(s::object(
            &[
                ("status", s::string()),
                ("cluster_assignment_id", s::string()),
                ("merged_from", s::string()),
                ("run_id", s::string()),
                ("min_patients", s::integer()),
                ("n_clusters", s::integer()),
                ("merges", s::array(open.clone())),
                ("cluster_sizes", s::array(open.clone())),
                (phirewall::KEY, open.clone()),
                (release::KEY, open),
                ("recommended_next_tools", s::array(s::string())),
            ],
            &[
                "status",
                "cluster_assignment_id",
                "n_clusters",
                "cluster_sizes",
            ],
            true,
        )),
        read_only: false,
        idempotent: false,
    }
}

fn merge_json(m: &Merge) -> Value {
    json!({
        "from": m.from,
        "into": m.into,
        "group": m.group,
        "linkage": m.linkage,
        "weight_share": (m.weight_share * 100.0).round() / 100.0,
    })
}

/// The merges in order, with each run of consecutive `isolated` merges (no
/// edge to any cluster; each joined the largest) collapsed into one entry.
fn merges_json(merges: &[Merge]) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    let mut run: Vec<&Merge> = Vec::new();
    let flush = |run: &mut Vec<&Merge>, out: &mut Vec<Value>| {
        match run.as_slice() {
            [] => {}
            [one] => out.push(merge_json(one)),
            [first, ..] => out.push(json!({
                "into": first.into,
                "linkage": "isolated",
                "from": run.iter().map(|m| m.from).collect::<Vec<_>>(),
                "count": run.len(),
            })),
        }
        run.clear();
    };
    for m in merges {
        if m.linkage == Linkage::Isolated && run.first().is_none_or(|f| f.into == m.into) {
            run.push(m);
        } else {
            flush(&mut run, &mut out);
            if m.linkage == Linkage::Isolated {
                run.push(m);
            } else {
                out.push(merge_json(m));
            }
        }
    }
    flush(&mut run, &mut out);
    out
}

fn render_text(
    n_merged: usize,
    n_clusters: usize,
    k: usize,
    new_id: &ClusterAssignmentId,
    id: &ClusterAssignmentId,
) -> String {
    if n_merged == 0 {
        format!(
            "## Merge small groups\n\nEvery cluster already has at least {k} patients; nothing was merged. `cluster_assignment_id`: `{new_id}`.\n"
        )
    } else {
        format!(
            "## Merge small groups\n\n{n_merged} cluster(s) merged into {n_clusters} groups of at least {k} patients each. `cluster_assignment_id`: `{new_id}` (from `{id}`). Run `generate_cluster_dossier` with `method='assignment'` and this id for the full evidence.\n"
        )
    }
}

fn failed(code: &str, reason: impl Into<String>, action: impl Into<String>) -> ToolError {
    ToolError::new(NAME, code, reason, action)
}

pub async fn call(state: &ServerState, p: Params) -> Result<CallToolResult, ToolError> {
    let k = p.min_patients.unwrap_or(MIN_PATIENTS);
    if k < MIN_PATIENTS {
        // A privacy floor: a group this small is never released.
        return Err(ToolError::invalid_argument(
            NAME,
            format!("min_patients must be >= {MIN_PATIENTS}, got '{k}'"),
            format!(
                "Pass min_patients >= {MIN_PATIENTS} or omit it; the floor can only be raised."
            ),
        ));
    }
    let id = match p.cluster_assignment_id.as_deref().filter(|i| !i.is_empty()) {
        Some(id) => ClusterAssignmentId::parse(id).map_err(|e| ToolError::from_store(NAME, e))?,
        None => state
            .session()
            .clusters
            .as_ref()
            .map(|c| c.cluster_assignment_id.clone())
            .ok_or_else(|| {
                failed(
                    "NO_CLUSTERING",
                    "There is no clustering in this session to merge.",
                    "Run generate_cluster_dossier first, or pass the cluster_assignment_id of a stored clustering.",
                )
            })?,
    };
    let (record, labels, view, live) = clustering::resolve_assignment_run(state, NAME, &id)?;
    let run_id = view.run_id().clone();
    let patients = Patients::of(&live.table, &live.identifier_columns);
    if patients.unit() == Unit::Rows {
        return Err(failed(
            "NO_PATIENT_IDENTIFIER",
            "This table has no patient token column, so the number of distinct patients in a cluster cannot be counted; a cluster's size in rows would overstate it when a patient has several rows.",
            "Merge on a dataset registered through Phirewall, whose twin carries a patient token column (PP-...). Ask the user to register the data there; a Pulsar-only upload cannot be merged by patients.",
        ));
    }
    if patients.total() < k {
        return Err(failed(
            "TOO_FEW_PATIENTS",
            format!("The whole dataset has fewer than {k} patients, so no grouping of it can reach {k}."),
            "Analyse it as one group, or use a larger dataset.",
        ));
    }
    let affinity = state
        .graph_cache_for_live(&live, &view.record)
        .affinity
        .clone();

    let merged = {
        let labels = labels.clone();
        tokio::task::spawn_blocking(move || {
            let mut rows: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
            for (i, &l) in labels.iter().enumerate() {
                rows.entry(l).or_default().push(i);
            }
            let support_of = |members: &[i64]| {
                patients.count(members.iter().flat_map(|l| rows[l].iter().copied()))
            };
            merge_small_by(&labels, &support_of, &affinity.edges, k)
        })
        .await
        .map_err(|e| ToolError::worker_stopped(NAME, "MERGE_FAILED", &e))?
    };
    if merged.unresolved {
        return Err(failed(
            "TOO_FEW_PATIENTS",
            format!("Merging everything still leaves a group under {k} patients."),
            "Analyse the data as one group.",
        ));
    }

    // Nothing under the floor: the stored assignment already is the answer.
    // Its labels are the ids the caller gets back, so sizes must use them,
    // not a renumbered copy.
    let published = if merged.merges.is_empty() {
        &labels
    } else {
        &merged.labels
    };
    let new_id = if merged.merges.is_empty() {
        id.clone()
    } else {
        let method = if record.method.starts_with("merged:") {
            record.method.clone()
        } else {
            format!("merged:{}", record.method)
        };
        state
            .store
            .save_cluster_assignment(ClusterAssignment {
                run_id: Some(run_id.clone()),
                dataset_id: record.dataset_id.clone(),
                method,
                interpretation_edge_weight_threshold: record.interpretation_edge_weight_threshold,
                threshold_source: record.threshold_source.clone(),
                construction_threshold: record.construction_threshold,
                labels: &merged.labels,
            })
            .map_err(|e| ToolError::from_store(NAME, e))?
            .cluster_assignment_id
    };

    // Publish through the dossier's path: it builds the evidence on the new
    // labels, makes them the session's clustering and returns the summary.
    let (dossier, _) = clustering::run(
        state,
        clustering::Params {
            method: MethodParam::Assignment,
            cluster_assignment_id: Some(new_id.to_string()),
            detail: DossierDetail::Summary,
            ..Default::default()
        },
    )
    .await?;

    let merges = merges_json(&merged.merges);
    let cluster_sizes: Vec<Value> = label_sizes(published)
        .into_iter()
        .map(|(c, n)| json!({"cluster_id": c, "size": RowCount::of(n, MIN_PATIENTS)}))
        .collect();

    let mut summary = dossier[phirewall::KEY].clone();
    if !merged.merges.is_empty() {
        let full_merges: Vec<Value> = merged.merges.iter().map(merge_json).collect();
        if let Some(Value::Object(grouping)) = summary.get_mut("grouping") {
            grouping.insert("merged_from".into(), json!(id));
            grouping.insert("merges".into(), json!(full_merges));
        }
    }
    let n_clusters = merged.groups.len();
    let text = render_text(merged.merges.len(), n_clusters, k, &new_id, &id);
    let mut payload = json!({
        "status": "ok",
        "cluster_assignment_id": new_id,
        "merged_from": id,
        "run_id": run_id,
        "min_patients": k,
        "n_clusters": n_clusters,
        "merges": merges,
        "cluster_sizes": cluster_sizes,
        phirewall::KEY: summary,
        "recommended_next_tools": [
            "generate_cluster_dossier",
            "get_cluster_profile",
            "compare_clusters",
        ],
    });
    // Nothing merged: there is no parent id and no merge list to report.
    if merged.merges.is_empty() {
        if let Some(object) = payload.as_object_mut() {
            object.remove("merged_from");
            object.remove("merges");
        }
    }
    release::attach_merge(state, &run_id, published, &mut payload);
    Ok(ok_result(payload, Some(text)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(from: i64, into: i64, linkage: Linkage) -> Merge {
        Merge {
            from,
            into,
            group: into,
            linkage,
            weight_share: 0.5,
        }
    }

    #[test]
    fn a_run_of_isolated_merges_is_one_entry() {
        let merges = [
            m(1, 0, Linkage::EdgeWeight),
            m(2, 0, Linkage::Isolated),
            m(3, 0, Linkage::Isolated),
            m(4, 0, Linkage::Isolated),
            m(5, 9, Linkage::EdgeWeight),
            m(6, 0, Linkage::Isolated),
        ];
        let out = merges_json(&merges);
        assert_eq!(out.len(), 4, "{out:?}");
        assert_eq!(out[0]["from"], 1);
        assert_eq!(
            out[1],
            json!({"into": 0, "linkage": "isolated", "from": [2, 3, 4], "count": 3})
        );
        assert_eq!(out[2]["from"], 5);
        // A lone isolated merge keeps the plain shape.
        assert_eq!(out[3]["from"], 6);
        assert_eq!(out[3]["linkage"], "isolated");

        let new_id = ClusterAssignmentId::parse("ca_111111111111").unwrap();
        let old_id = ClusterAssignmentId::parse("ca_000000000000").unwrap();
        let text = render_text(merges.len(), 2, 8, &new_id, &old_id);
        assert!(
            text.contains("6 cluster(s) merged into 2 groups"),
            "markdown must count real merges, not collapsed entries: {text}"
        );
    }
}
