//! `export_dataset_bundle` (`pulsar/exports.py`): the fitted graph, its
//! node/edge/group tables and the source data as a Parquet bundle under
//! `{output_dir}/{slug}/`:
//!
//! ```text
//! tabular/raw.parquet     node_id + the raw columns
//! tabular/clean.parquet   node_id + the preprocessed matrix (optional)
//! graph/nodes.parquet     node_id, group_id, val, archetype, ex, ey, ez, is_live + raw passthrough
//! graph/edges.parquet     snapshot edges (above edges_threshold)
//! graph/cosmic.parquet    every weighted edge (above 0)
//! graph/groups.parquet    group_id, name, desc, member_ids
//! export_manifest.json    versions, thresholds and counts (optional)
//! ```
//!
//! The bundle is written to a staging directory beside the destination and
//! promoted file by file, so files a user keeps under the slug survive a
//! re-export and a failed export changes nothing.
//!
//! Known divergences from Python:
//! - `config_hash` is the SHA-256 of the resolved Rust config as key-sorted
//!   compact JSON; Python hashed its own config dataclass, so the values
//!   differ.
//! - `layout="spectral"` returns eigenvectors of the graph Laplacian
//!   (`nx.spectral_layout`). Eigenvectors are unique only up to sign, and a
//!   repeated eigenvalue (one zero eigenvalue per connected component) makes
//!   them an arbitrary basis of the eigenspace, so coordinates can differ
//!   from Python's while being an equally valid layout. Graphs of 500 nodes
//!   and more use LOBPCG instead of ARPACK.
//! - Parquet files carry no pandas metadata.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, Float32Array, Float64Array, Int64Array, LargeStringArray, ListBuilder,
    StringArray, UInt32Array, UInt32Builder,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use nalgebra::{DMatrix, SymmetricEigen};
use parquet::arrow::ArrowWriter;
use pulsar_data::{Column, Table};
use pulsar_pipeline::{Fitted, Graph};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

/// Every file the bundle may write, relative to its root.
pub const BUNDLE_FILES: [&str; 7] = [
    "tabular/raw.parquet",
    "tabular/clean.parquet",
    "graph/nodes.parquet",
    "graph/edges.parquet",
    "graph/cosmic.parquet",
    "graph/groups.parquet",
    "export_manifest.json",
];

/// Canonical node fields; raw columns with these names are renamed.
const NODE_RESERVED_FIELDS: [&str; 8] = [
    "node_id",
    "group_id",
    "val",
    "archetype",
    "ex",
    "ey",
    "ez",
    "is_live",
];

/// `nx.spectral_layout` solves densely below this many nodes.
const SPECTRAL_DENSE_LIMIT: usize = 500;

#[derive(Debug, thiserror::Error)]
pub enum BundleError {
    #[error("{0}")]
    Invalid(String),
    #[error("{path}: {message}")]
    Write { path: PathBuf, message: String },
}

fn write_err(path: &Path, e: impl std::fmt::Display) -> BundleError {
    BundleError::Write {
        path: path.to_path_buf(),
        message: e.to_string(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// The first three dimensions of the first projection embedding.
    Projection,
    /// Laplacian eigenvectors of the constructed graph.
    Spectral,
    Zeros,
}

/// Everything besides the model and data that the bundle depends on.
pub struct BundleOptions<'a> {
    /// One label per node.
    pub labels: &'a [i64],
    pub cluster_names: &'a BTreeMap<i64, String>,
    pub cluster_descriptions: &'a BTreeMap<i64, String>,
    /// Snapshot cut for `graph/edges.parquet`; default the construction threshold.
    pub edges_threshold: Option<f64>,
    pub layout: Layout,
    pub include_clean: bool,
    pub write_manifest: bool,
    pub pulsar_version: &'a str,
}

// ---- tables ----------------------------------------------------------------------------

struct Frame {
    fields: Vec<Field>,
    arrays: Vec<ArrayRef>,
}

impl Frame {
    fn new() -> Self {
        Self {
            fields: vec![],
            arrays: vec![],
        }
    }

    fn push(&mut self, name: &str, nullable: bool, array: ArrayRef) {
        self.fields
            .push(Field::new(name, array.data_type().clone(), nullable));
        self.arrays.push(array);
    }

    /// A raw table column as pandas `to_parquet` / `from_pandas` types it.
    fn push_column(&mut self, name: &str, col: &Column) {
        let array: ArrayRef = match col {
            Column::Float(v) => Arc::new(Float64Array::from(
                v.iter()
                    .map(|x| (!x.is_nan()).then_some(*x))
                    .collect::<Vec<_>>(),
            )),
            Column::Int(v) => Arc::new(Int64Array::from(v.clone())),
            Column::Bool(v) => Arc::new(BooleanArray::from(v.clone())),
            Column::Text(v) => Arc::new(LargeStringArray::from(
                v.iter().map(|s| s.as_deref()).collect::<Vec<_>>(),
            )),
        };
        self.push(name, true, array);
    }

    fn write(self, path: &Path) -> Result<usize, BundleError> {
        let schema = Arc::new(Schema::new(self.fields));
        let batch =
            RecordBatch::try_new(schema.clone(), self.arrays).map_err(|e| write_err(path, e))?;
        let rows = batch.num_rows();
        let file = fs::File::create(path).map_err(|e| write_err(path, e))?;
        let mut writer =
            ArrowWriter::try_new(file, schema, None).map_err(|e| write_err(path, e))?;
        writer.write(&batch).map_err(|e| write_err(path, e))?;
        writer.close().map_err(|e| write_err(path, e))?;
        Ok(rows)
    }
}

/// `(from, to)` column-name pairs, in column order.
type NamePairs = Vec<(String, String)>;

/// `(output name, raw name)` for the node passthrough columns, and the
/// `(raw name, output name)` renames of raw columns that collide with
/// canonical node fields.
fn passthrough_columns(raw: &[String]) -> (NamePairs, NamePairs) {
    let mut used: BTreeSet<String> = NODE_RESERVED_FIELDS.iter().map(|s| s.to_string()).collect();
    let mut passthrough = vec![];
    let mut renamed = vec![];
    for col in raw {
        let mut out = col.clone();
        if used.contains(&out) {
            let base = format!("raw_{col}");
            out = base.clone();
            let mut suffix = 1;
            while used.contains(&out) || raw.contains(&out) {
                out = format!("{base}_{suffix}");
                suffix += 1;
            }
            renamed.push((col.clone(), out.clone()));
        }
        used.insert(out.clone());
        passthrough.push((out, col.clone()));
    }
    (passthrough, renamed)
}

fn node_ids(n: usize) -> ArrayRef {
    Arc::new(UInt32Array::from((0..n as u32).collect::<Vec<_>>()))
}

fn edge_frame(edges: &[(usize, usize, f64)]) -> Frame {
    let mut f = Frame::new();
    f.push(
        "source_id",
        false,
        Arc::new(UInt32Array::from(
            edges.iter().map(|e| e.0 as u32).collect::<Vec<_>>(),
        )),
    );
    f.push(
        "target_id",
        false,
        Arc::new(UInt32Array::from(
            edges.iter().map(|e| e.1 as u32).collect::<Vec<_>>(),
        )),
    );
    f.push(
        "weight",
        false,
        Arc::new(Float32Array::from(
            edges.iter().map(|e| e.2 as f32).collect::<Vec<_>>(),
        )),
    );
    f
}

/// Per-node sum of snapshot weights in float32, sources first then targets
/// (`np.add.at` twice).
fn weighted_degree(n: usize, edges: &[(usize, usize, f64)]) -> Vec<f32> {
    let mut val = vec![0f32; n];
    for &(i, _, w) in edges {
        val[i] += w as f32;
    }
    for &(_, j, w) in edges {
        val[j] += w as f32;
    }
    val
}

fn name_of(names: &BTreeMap<i64, String>, gid: i64) -> String {
    names
        .get(&gid)
        .cloned()
        .unwrap_or_else(|| format!("Cluster {gid}"))
}

// ---- layout ------------------------------------------------------------------------------

/// `nx.spectral_layout(G, dim=3, weight="weight")`: eigenvectors 1..=3 of the
/// weighted Laplacian `D - A`, each column centred, then all scaled so the
/// largest absolute coordinate is 1.
fn spectral_layout(graph: &Graph) -> Vec<[f64; 3]> {
    let n = graph.n;
    match n {
        0 => return vec![],
        1 => return vec![[0.0; 3]],
        // nx places two nodes at the origin and at (2, 2, 2) before rescaling.
        2 => {
            return vec![[-1.0; 3], [1.0; 3]];
        }
        _ => {}
    }
    let dim = 3.min(n - 1);
    let columns: Vec<Vec<f64>> = if n < SPECTRAL_DENSE_LIMIT {
        let mut l = DMatrix::<f64>::zeros(n, n);
        for &(i, j, w) in &graph.edges {
            if i == j {
                continue;
            }
            l[(i, j)] -= w;
            l[(j, i)] -= w;
            l[(i, i)] += w;
            l[(j, j)] += w;
        }
        let eig = SymmetricEigen::new(l);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));
        order[1..=dim]
            .iter()
            .map(|&k| eig.eigenvectors.column(k).iter().copied().collect())
            .collect()
    } else {
        let mut degree = vec![0.0; n];
        for &(i, j, w) in &graph.edges {
            degree[i] += w;
            degree[j] += w;
        }
        let apply = |x: &DMatrix<f64>| {
            let mut y = DMatrix::<f64>::zeros(x.nrows(), x.ncols());
            for c in 0..x.ncols() {
                for i in 0..n {
                    y[(i, c)] = degree[i] * x[(i, c)];
                }
                for &(i, j, w) in &graph.edges {
                    y[(i, c)] -= w * x[(j, c)];
                    y[(j, c)] -= w * x[(i, c)];
                }
            }
            y
        };
        match pulsar_analysis::clustering::lobpcg::lobpcg_smallest(
            &apply,
            n,
            dim + 1,
            42,
            1e-6,
            500,
        ) {
            Ok(pairs) => (1..=dim)
                .map(|k| pairs.vectors.column(k).iter().copied().collect())
                .collect(),
            Err(_) => vec![vec![0.0; n]; dim],
        }
    };
    let mut pos = vec![[0.0; 3]; n];
    let mut lim: f64 = 0.0;
    for (c, col) in columns.iter().enumerate() {
        let mean = col.iter().sum::<f64>() / n as f64;
        for (i, v) in col.iter().enumerate() {
            pos[i][c] = v - mean;
            lim = lim.max(pos[i][c].abs());
        }
    }
    if lim > 0.0 {
        for p in &mut pos {
            for v in p.iter_mut() {
                *v /= lim;
            }
        }
    }
    pos
}

fn layout_coordinates(fitted: &Fitted, layout: Layout) -> Vec<[f32; 3]> {
    let n = fitted.n;
    match layout {
        Layout::Zeros => vec![[0.0; 3]; n],
        Layout::Projection => match fitted.embeddings.first() {
            None => vec![[0.0; 3]; n],
            Some(e) => (0..e.nrows())
                .map(|i| {
                    let mut p = [0f32; 3];
                    for (c, v) in p.iter_mut().enumerate().take(e.ncols().min(3)) {
                        *v = e[[i, c]] as f32;
                    }
                    p
                })
                .collect(),
        },
        Layout::Spectral => spectral_layout(&fitted.graph)
            .into_iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect(),
    }
}

// ---- manifest ----------------------------------------------------------------------------

fn sorted_keys(v: Value) -> Value {
    match v {
        Value::Object(m) => {
            let sorted: BTreeMap<String, Value> =
                m.into_iter().map(|(k, v)| (k, sorted_keys(v))).collect();
            Value::Object(sorted.into_iter().collect())
        }
        Value::Array(a) => Value::Array(a.into_iter().map(sorted_keys).collect()),
        other => other,
    }
}

fn config_hash(fitted: &Fitted) -> String {
    let canonical = sorted_keys(serde_json::to_value(&fitted.config).unwrap_or(Value::Null));
    let text = serde_json::to_string(&canonical).unwrap_or_default();
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// ---- export ------------------------------------------------------------------------------

/// Write the bundle under `{output_dir}/{slug}/`; returns the manifest
/// (empty when `write_manifest` is false).
pub fn export_dataset_bundle(
    fitted: &Fitted,
    raw: &Table,
    output_dir: &Path,
    slug: &str,
    opts: &BundleOptions<'_>,
) -> Result<Value, BundleError> {
    let n = fitted.n;
    if raw.n_rows() != n {
        return Err(BundleError::Invalid(format!(
            "raw data has {} rows but graph has {n} nodes",
            raw.n_rows()
        )));
    }
    if opts.labels.len() != n {
        return Err(BundleError::Invalid(format!(
            "cluster labels have {} entries but graph has {n} nodes",
            opts.labels.len()
        )));
    }
    let base = output_dir.join(slug);
    let parent = base.parent().unwrap_or(output_dir).to_path_buf();
    fs::create_dir_all(&parent).map_err(|e| write_err(&parent, e))?;
    let leaf = base
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "bundle".into());
    let staging = parent.join(format!(
        ".{leaf}_staging_{}_{}",
        std::process::id(),
        pulsar_store_nonce()
    ));
    let result = write_bundle(fitted, raw, &staging, opts).and_then(|manifest| {
        promote(&staging, &base)?;
        Ok(manifest)
    });
    let _ = fs::remove_dir_all(&staging);
    result
}

fn pulsar_store_nonce() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos())
}

fn write_bundle(
    fitted: &Fitted,
    raw: &Table,
    staging: &Path,
    opts: &BundleOptions<'_>,
) -> Result<Value, BundleError> {
    let n = fitted.n;
    let tabular = staging.join("tabular");
    let graph = staging.join("graph");
    for d in [&tabular, &graph] {
        fs::create_dir_all(d).map_err(|e| write_err(d, e))?;
    }

    // tabular/raw.parquet: node_id first; a raw `node_id` keeps its renamed name.
    let (passthrough, renamed) = passthrough_columns(&raw.names);
    let raw_node_id = renamed
        .iter()
        .find(|(c, _)| c == "node_id")
        .map(|(_, o)| o.clone());
    let mut f = Frame::new();
    f.push("node_id", true, node_ids(n));
    for (name, col) in raw.names.iter().zip(&raw.columns) {
        let out = if name == "node_id" {
            raw_node_id.as_deref().unwrap_or(name)
        } else {
            name
        };
        f.push_column(out, col);
    }
    f.write(&tabular.join("raw.parquet"))?;

    if opts.include_clean {
        let pre = fitted.preprocessed.as_ref().ok_or_else(|| {
            BundleError::Invalid(
                "this run has no single preprocessed table (multi-representation fit)".into(),
            )
        })?;
        let names = pre.feature_names();
        let (_, clean_renames) = passthrough_columns(names);
        let mut f = Frame::new();
        f.push("node_id", true, node_ids(n));
        for (k, name) in names.iter().enumerate() {
            let out = if name == "node_id" {
                clean_renames
                    .iter()
                    .find(|(c, _)| c == "node_id")
                    .map_or(name.as_str(), |(_, o)| o.as_str())
            } else {
                name.as_str()
            };
            let values: Vec<Option<f64>> = pre
                .matrix
                .column(k)
                .iter()
                .map(|x| (!x.is_nan()).then_some(*x))
                .collect();
            f.push(out, true, Arc::new(Float64Array::from(values)));
        }
        f.write(&tabular.join("clean.parquet"))?;
    }

    let cosmic = fitted.weighted_edges(Some(0.0));
    let cosmic_count = edge_frame(&cosmic).write(&graph.join("cosmic.parquet"))?;
    let snapshot = fitted.weighted_edges(opts.edges_threshold);
    let edges_count = edge_frame(&snapshot).write(&graph.join("edges.parquet"))?;

    let coords = layout_coordinates(fitted, opts.layout);
    let mut f = Frame::new();
    f.push("node_id", false, node_ids(n));
    f.push(
        "group_id",
        false,
        Arc::new(UInt32Array::from(
            opts.labels.iter().map(|&l| l as u32).collect::<Vec<_>>(),
        )),
    );
    f.push(
        "val",
        false,
        Arc::new(Float32Array::from(weighted_degree(n, &snapshot))),
    );
    f.push(
        "archetype",
        false,
        Arc::new(StringArray::from(
            opts.labels
                .iter()
                .map(|&l| name_of(opts.cluster_names, l as u32 as i64))
                .collect::<Vec<_>>(),
        )),
    );
    for (c, name) in ["ex", "ey", "ez"].iter().enumerate() {
        f.push(
            name,
            false,
            Arc::new(Float32Array::from(
                coords.iter().map(|p| p[c]).collect::<Vec<_>>(),
            )),
        );
    }
    f.push(
        "is_live",
        false,
        Arc::new(BooleanArray::from(vec![true; n])),
    );
    for (out, source) in &passthrough {
        if let Some(col) = raw.column(source) {
            f.push_column(out, col);
        }
    }
    f.write(&graph.join("nodes.parquet"))?;

    let gids: BTreeSet<u32> = opts.labels.iter().map(|&l| l as u32).collect();
    let mut members = ListBuilder::new(UInt32Builder::new()).with_field(Field::new(
        "element",
        DataType::UInt32,
        true,
    ));
    let (mut names, mut descs) = (vec![], vec![]);
    for &gid in &gids {
        for (i, &l) in opts.labels.iter().enumerate() {
            if l as u32 == gid {
                members.values().append_value(i as u32);
            }
        }
        members.append(true);
        names.push(name_of(opts.cluster_names, gid as i64));
        descs.push(
            opts.cluster_descriptions
                .get(&(gid as i64))
                .cloned()
                .unwrap_or_default(),
        );
    }
    let mut f = Frame::new();
    f.push(
        "group_id",
        false,
        Arc::new(UInt32Array::from(gids.iter().copied().collect::<Vec<_>>())),
    );
    f.push("name", false, Arc::new(StringArray::from(names)));
    f.push("desc", false, Arc::new(StringArray::from(descs)));
    f.push("member_ids", false, Arc::new(members.finish()));
    let groups_count = f.write(&graph.join("groups.parquet"))?;

    if !opts.write_manifest {
        return Ok(Value::Object(Map::new()));
    }
    let renames: Map<String, Value> = renamed.into_iter().map(|(k, v)| (k, json!(v))).collect();
    let manifest = json!({
        "pulsar_version": opts.pulsar_version,
        "config_hash": config_hash(fitted),
        "thresholds": {
            "resolved_construction_threshold": fitted.construction_threshold,
            "edges_threshold": opts.edges_threshold.unwrap_or(fitted.construction_threshold),
        },
        "phil_config": {"enabled": opts.include_clean},
        "counts": {
            "nodes": n,
            "edges": edges_count,
            "cosmic_edges": cosmic_count,
            "groups": groups_count,
        },
        "node_passthrough_column_renames": renames,
    });
    let path = staging.join("export_manifest.json");
    let text = serde_json::to_string_pretty(&manifest).map_err(|e| write_err(&path, e))?;
    fs::write(&path, text).map_err(|e| write_err(&path, e))?;
    Ok(manifest)
}

/// Move the staged bundle into place. A new slug is renamed in one step; an
/// existing one has only the bundle's own files replaced (or removed when
/// this export did not write them).
fn promote(staging: &Path, base: &Path) -> Result<(), BundleError> {
    if !base.exists() {
        return fs::rename(staging, base).map_err(|e| write_err(base, e));
    }
    for rel in BUNDLE_FILES {
        let (staged, target) = (staging.join(rel), base.join(rel));
        if staged.exists() {
            if let Some(dir) = target.parent() {
                fs::create_dir_all(dir).map_err(|e| write_err(dir, e))?;
            }
            fs::rename(&staged, &target).map_err(|e| write_err(&target, e))?;
        } else if target.exists() {
            fs::remove_file(&target).map_err(|e| write_err(&target, e))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_renames_avoid_every_existing_name() {
        let raw: Vec<String> = ["x", "node_id", "val", "raw_val"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let (pass, renamed) = passthrough_columns(&raw);
        assert_eq!(
            pass,
            vec![
                ("x".into(), "x".into()),
                ("raw_node_id".into(), "node_id".into()),
                ("raw_val_1".into(), "val".into()),
                ("raw_val".into(), "raw_val".into()),
            ]
        );
        assert_eq!(
            renamed,
            vec![
                ("node_id".to_string(), "raw_node_id".to_string()),
                ("val".into(), "raw_val_1".into())
            ]
        );
    }

    #[test]
    fn weighted_degree_adds_sources_then_targets() {
        assert_eq!(
            weighted_degree(3, &[(0, 1, 0.5), (1, 2, 0.25)]),
            vec![0.5, 0.75, 0.25]
        );
    }

    #[test]
    fn spectral_layout_of_a_path_is_its_fiedler_vector() {
        let g = Graph {
            n: 3,
            edges: vec![(0, 1, 1.0), (1, 2, 1.0)],
        };
        let pos = spectral_layout(&g);
        // Eigenvectors 1 and 2 of P3: (1, 0, -1)/sqrt(2) and (1, -2, 1)/sqrt(6),
        // up to sign, scaled together so the largest |coordinate| is 1.
        assert!(pos[1][0].abs() < 1e-12);
        assert!((pos[0][0] + pos[2][0]).abs() < 1e-12);
        assert!((pos[0][0].abs() - 3f64.sqrt() / 2.0).abs() < 1e-12);
        assert!((pos[1][1].abs() - 1.0).abs() < 1e-12);
        assert_eq!(
            pos[0][2], 0.0,
            "a 3-node graph has only two non-trivial eigenvectors"
        );
    }
}
