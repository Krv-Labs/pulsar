//! Arrow IPC sidecars for the unbounded parts of a record: edge lists
//! (O(edges)) and label vectors (O(rows)). JSON metadata stays small and
//! greppable; the arrays stay typed and zero-copy readable.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, Int64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;
use pulsar_pipeline::Graph;

use crate::atomic::{stage_atomic, write_atomic, StagedFile};
use crate::error::{io, StoreError};

const META_N: &str = "pulsar.n";

fn corrupt<'a>(path: &'a Path, what: &'static str) -> impl Fn(String) -> StoreError + 'a {
    move |message| StoreError::CorruptSidecar {
        path: path.to_path_buf(),
        what,
        message,
    }
}

fn encode(
    schema: Schema,
    columns: Vec<ArrayRef>,
    path: &Path,
    what: &'static str,
) -> Result<Vec<u8>, StoreError> {
    let err = corrupt(path, what);
    let schema = Arc::new(schema);
    let batch = RecordBatch::try_new(schema.clone(), columns).map_err(|e| err(e.to_string()))?;
    let mut writer = FileWriter::try_new(Vec::new(), &schema).map_err(|e| err(e.to_string()))?;
    writer.write(&batch).map_err(|e| err(e.to_string()))?;
    writer.finish().map_err(|e| err(e.to_string()))?;
    writer.into_inner().map_err(|e| err(e.to_string()))
}

fn open(path: &Path, what: &'static str) -> Result<FileReader<File>, StoreError> {
    let file = File::open(path).map_err(io("open", path))?;
    FileReader::try_new(file, None).map_err(|e| corrupt(path, what)(e.to_string()))
}

fn column<'a, T: Array + 'static>(
    batch: &'a RecordBatch,
    index: usize,
    path: &Path,
    what: &'static str,
) -> Result<&'a T, StoreError> {
    batch
        .columns()
        .get(index)
        .and_then(|c| c.as_any().downcast_ref::<T>())
        .ok_or_else(|| corrupt(path, what)(format!("column {index} has the wrong type")))
}

/// Write an edge list as `(i: u64, j: u64, weight: f64)` with `n` in the
/// schema metadata. Atomic like every other write.
pub(crate) fn write_edges(path: &Path, graph: &Graph) -> Result<(), StoreError> {
    let (mut i, mut j, mut w) = (
        Vec::with_capacity(graph.edges.len()),
        Vec::with_capacity(graph.edges.len()),
        Vec::with_capacity(graph.edges.len()),
    );
    for &(a, b, weight) in &graph.edges {
        i.push(a as u64);
        j.push(b as u64);
        w.push(weight);
    }
    let schema = Schema::new_with_metadata(
        vec![
            Field::new("i", DataType::UInt64, false),
            Field::new("j", DataType::UInt64, false),
            Field::new("weight", DataType::Float64, false),
        ],
        HashMap::from([(META_N.to_owned(), graph.n.to_string())]),
    );
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from(i)),
        Arc::new(UInt64Array::from(j)),
        Arc::new(Float64Array::from(w)),
    ];
    let bytes = encode(schema, columns, path, "edge list")?;
    write_atomic(path, &bytes, true)
}

pub(crate) fn read_edges(path: &Path) -> Result<Graph, StoreError> {
    const WHAT: &str = "edge list";
    let reader = open(path, WHAT)?;
    let n = reader
        .schema()
        .metadata()
        .get(META_N)
        .and_then(|v| v.parse::<usize>().ok())
        .ok_or_else(|| corrupt(path, WHAT)(format!("missing `{META_N}` metadata")))?;
    let mut edges = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|e| corrupt(path, WHAT)(e.to_string()))?;
        let i = column::<UInt64Array>(&batch, 0, path, WHAT)?;
        let j = column::<UInt64Array>(&batch, 1, path, WHAT)?;
        let w = column::<Float64Array>(&batch, 2, path, WHAT)?;
        for row in 0..batch.num_rows() {
            edges.push((i.value(row) as usize, j.value(row) as usize, w.value(row)));
        }
    }
    if edges.iter().any(|&(i, j, _)| i >= n || j >= n) {
        return Err(corrupt(path, WHAT)(format!(
            "edge endpoint out of range for n = {n}"
        )));
    }
    Ok(Graph { n, edges })
}

/// Stage a label vector as one `label: i64` column; the caller commits it.
pub(crate) fn stage_labels(path: &Path, labels: &[i64]) -> Result<StagedFile, StoreError> {
    let schema = Schema::new(vec![Field::new("label", DataType::Int64, false)]);
    let columns: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(labels.to_vec()))];
    let bytes = encode(schema, columns, path, "label vector")?;
    stage_atomic(path, &bytes, true)
}

pub(crate) fn read_labels(path: &Path) -> Result<Vec<i64>, StoreError> {
    const WHAT: &str = "label vector";
    let reader = open(path, WHAT)?;
    let mut labels = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|e| corrupt(path, WHAT)(e.to_string()))?;
        let col = column::<Int64Array>(&batch, 0, path, WHAT)?;
        labels.extend(col.values().iter().copied());
    }
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::TempRoot;

    #[test]
    fn edge_lists_and_labels_round_trip() {
        let root = TempRoot::new("sidecar");
        let graph = Graph {
            n: 5,
            edges: vec![(0, 1, 0.5), (1, 4, 1.0), (2, 3, 0.125)],
        };
        let path = root.path().join("graph.arrow");
        write_edges(&path, &graph).unwrap();
        assert_eq!(read_edges(&path).unwrap(), graph);

        let empty = Graph {
            n: 3,
            edges: vec![],
        };
        write_edges(&path, &empty).unwrap();
        assert_eq!(read_edges(&path).unwrap(), empty);

        let labels = vec![0, 0, 1, -1, 2];
        let lpath = root.path().join("labels.arrow");
        stage_labels(&lpath, &labels).unwrap().commit().unwrap();
        assert_eq!(read_labels(&lpath).unwrap(), labels);

        std::fs::write(&lpath, b"not arrow").unwrap();
        assert_eq!(read_labels(&lpath).unwrap_err().code(), "SIDECAR_CORRUPT");
    }
}
