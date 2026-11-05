//! `DataFrame.to_csv(index=False)`: the text pandas writes for a table, so a
//! labeled export reads back exactly like the source.
//!
//! Floats use the shortest repr (`astype(str)`), missing cells are empty,
//! booleans are `True` / `False`, and fields are quoted only when they hold a
//! comma, a quote or a line break (Python `csv.QUOTE_MINIMAL`), with quotes
//! doubled. Lines end in `\n`.

use std::fs;
use std::io::Write;
use std::path::Path;

use crate::error::DataError;
use crate::preprocess::py_float_repr;
use crate::table::{Column, Table};

fn cell(col: &Column, i: usize) -> String {
    match col {
        Column::Float(v) if v[i].is_nan() => String::new(),
        Column::Float(v) => py_float_repr(v[i]),
        Column::Int(v) => v[i].to_string(),
        Column::Bool(v) => match v[i] {
            Some(true) => "True".into(),
            Some(false) => "False".into(),
            None => String::new(),
        },
        Column::Text(v) => v[i].clone().unwrap_or_default(),
    }
}

fn quote(field: &str) -> String {
    if field.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_owned()
    }
}

fn line(fields: &[String]) -> String {
    // Python's csv writer quotes a lone empty field so the row is not blank.
    if let [only] = fields {
        if only.is_empty() {
            return "\"\"\n".into();
        }
    }
    let quoted: Vec<String> = fields.iter().map(|f| quote(f)).collect();
    quoted.join(",") + "\n"
}

/// The CSV text pandas writes for `table` (header row, no index).
pub fn to_csv_string(table: &Table) -> String {
    let mut out = line(&table.names);
    for i in 0..table.n_rows() {
        let fields: Vec<String> = table.columns.iter().map(|c| cell(c, i)).collect();
        out.push_str(&line(&fields));
    }
    out
}

/// Write `table` as CSV to `path`, atomically: a sibling temporary file is
/// written and then renamed over the destination.
pub fn write_csv(table: &Table, path: impl AsRef<Path>) -> Result<(), DataError> {
    let path = path.as_ref();
    let io = |source: std::io::Error| DataError::Io {
        path: path.to_path_buf(),
        source,
    };
    let dir = path
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(dir).map_err(io)?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let tmp = dir.join(format!(".{name}.{}.tmp", std::process::id()));
    let result = (|| {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(to_csv_string(table).as_bytes())?;
        f.sync_all()?;
        fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result.map_err(io)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cells_and_quoting_follow_pandas() {
        let t = Table::new(
            vec!["a,b".into(), "x".into(), "f".into()],
            vec![
                Column::Float(vec![0.1, f64::NAN, 1e16]),
                Column::Text(vec![
                    Some("say \"hi\"".into()),
                    None,
                    Some("two\nlines".into()),
                ]),
                Column::Bool(vec![Some(true), None, Some(false)]),
            ],
        )
        .unwrap();
        assert_eq!(
            to_csv_string(&t),
            "\"a,b\",x,f\n0.1,\"say \"\"hi\"\"\",True\n,,\n1e+16,\"two\nlines\",False\n"
        );
        let single = Table::new(vec!["s".into()], vec![Column::Text(vec![None])]).unwrap();
        assert_eq!(to_csv_string(&single), "s\n\"\"\n");
    }
}
