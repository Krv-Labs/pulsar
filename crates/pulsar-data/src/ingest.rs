//! CSV and Parquet readers that type columns the way pandas did.

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray};
use arrow::compute::{cast, concat};
use arrow::datatypes::{DataType, Float64Type, Int64Type};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::error::DataError;
use crate::table::{Column, Table};

/// Field values `pandas.read_csv` reads as missing by default
/// (`pandas._libs.parsers.STR_NA_VALUES`). Matching is exact: no trimming,
/// no case folding. The golden fixtures were produced with this set.
pub const NULL_TOKENS: [&str; 19] = [
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
    "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null",
];

/// Whether a raw CSV field is a missing value.
pub fn is_null_token(field: &str) -> bool {
    NULL_TOKENS.contains(&field)
}

/// Read a `.parquet` file as Parquet and anything else as CSV.
pub fn read_table(path: impl AsRef<Path>) -> Result<Table, DataError> {
    let path = path.as_ref();
    let is_parquet = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("parquet"));
    if is_parquet {
        read_parquet(path)
    } else {
        read_csv_path(path)
    }
}

/// Read a CSV file; see [`read_csv`].
pub fn read_csv_path(path: impl AsRef<Path>) -> Result<Table, DataError> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|source| DataError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    read_csv(file, &path.display().to_string())
}

/// Read CSV text with a header row. Columns are typed like `pandas.read_csv`
/// defaults: int64 when every value is an integer and none is missing,
/// float64 when every value is numeric, bool for `True`/`False` spellings,
/// otherwise str. Missing values are the [`NULL_TOKENS`].
pub fn read_csv<R: Read>(reader: R, source_name: &str) -> Result<Table, DataError> {
    let csv_err = |e: csv::Error| DataError::Csv {
        source_name: source_name.to_string(),
        message: e.to_string(),
    };
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(reader);
    let headers = rdr.headers().map_err(csv_err)?.clone();
    let names = header_names(headers.iter());
    if names.is_empty() {
        return Err(DataError::NoColumns);
    }
    let mut raw: Vec<Vec<Option<String>>> = vec![Vec::new(); names.len()];
    for record in rdr.records() {
        let record = record.map_err(csv_err)?;
        for (col, field) in raw.iter_mut().zip(record.iter()) {
            col.push(if is_null_token(field) {
                None
            } else {
                Some(field.to_string())
            });
        }
    }
    let columns = raw.into_iter().map(infer_column).collect();
    Table::new(names, columns)
}

/// pandas header handling: blank names become `Unnamed: <i>`, repeated
/// names get `.1`, `.2`, … suffixes.
fn header_names<'a>(headers: impl Iterator<Item = &'a str>) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for (i, h) in headers.enumerate() {
        let base = if h.is_empty() {
            format!("Unnamed: {i}")
        } else {
            h.to_string()
        };
        let mut name = base.clone();
        let mut k = 1;
        while names.contains(&name) {
            name = format!("{base}.{k}");
            k += 1;
        }
        names.push(name);
    }
    names
}

/// Type one column of raw fields (`None` = missing) like pandas would.
pub fn infer_column(tokens: Vec<Option<String>>) -> Column {
    let n = tokens.len();
    let any_null = tokens.iter().any(Option::is_none);
    let present = || tokens.iter().flatten().map(String::as_str);
    if present().next().is_none() {
        return Column::Float(vec![f64::NAN; n]);
    }
    if !any_null {
        let ints: Option<Vec<i64>> = present().map(parse_int).collect();
        if let Some(ints) = ints {
            return Column::Int(ints);
        }
    }
    if present().all(|s| parse_float(s).is_some()) {
        return Column::Float(
            tokens
                .iter()
                .map(|t| t.as_deref().and_then(parse_float).unwrap_or(f64::NAN))
                .collect(),
        );
    }
    if present().all(|s| parse_bool(s).is_some()) {
        return Column::Bool(
            tokens
                .iter()
                .map(|t| t.as_deref().and_then(parse_bool))
                .collect(),
        );
    }
    Column::Text(tokens)
}

fn parse_int(s: &str) -> Option<i64> {
    s.trim().parse().ok()
}

fn parse_float(s: &str) -> Option<f64> {
    // NaN spellings that are not null tokens stay strings, as in pandas.
    s.trim().parse::<f64>().ok().filter(|x| !x.is_nan())
}

fn parse_bool(s: &str) -> Option<bool> {
    match s {
        "True" | "TRUE" | "true" => Some(true),
        "False" | "FALSE" | "false" => Some(false),
        _ => None,
    }
}

/// Read a Parquet file. Integer columns without nulls become int64, other
/// numeric columns float64 (nulls → NaN), string and dictionary-of-string
/// columns str, booleans bool, and temporal columns their string form.
pub fn read_parquet(path: impl AsRef<Path>) -> Result<Table, DataError> {
    let path = path.as_ref();
    let pq_err = |e: parquet::errors::ParquetError| DataError::Parquet {
        path: path.to_path_buf(),
        message: e.to_string(),
    };
    let file = File::open(path).map_err(|source| DataError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(pq_err)?;
    let schema = builder.schema().clone();
    let reader = builder.build().map_err(pq_err)?;
    let batches: Vec<RecordBatch> =
        reader
            .collect::<Result<_, _>>()
            .map_err(|e| DataError::Parquet {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?;
    if schema.fields().is_empty() {
        return Err(DataError::NoColumns);
    }
    let mut names = Vec::with_capacity(schema.fields().len());
    let mut columns = Vec::with_capacity(schema.fields().len());
    for (i, field) in schema.fields().iter().enumerate() {
        let parts: Vec<ArrayRef> = batches.iter().map(|b| b.column(i).clone()).collect();
        let array: ArrayRef = if parts.is_empty() {
            arrow::array::new_empty_array(field.data_type())
        } else {
            let refs: Vec<&dyn Array> = parts.iter().map(|a| a.as_ref()).collect();
            concat(&refs).map_err(|e| DataError::Parquet {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?
        };
        names.push(field.name().clone());
        columns.push(column_from_arrow(field.name(), &array)?);
    }
    Table::new(names, columns)
}

fn column_from_arrow(name: &str, array: &ArrayRef) -> Result<Column, DataError> {
    let unsupported = || DataError::UnsupportedColumnType {
        column: name.to_string(),
        dtype: array.data_type().to_string(),
    };
    let dtype = array.data_type();
    let is_string =
        |t: &DataType| matches!(t, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View);
    if dtype == &DataType::Boolean {
        let a = array.as_boolean();
        return Ok(Column::Bool(
            (0..a.len())
                .map(|i| (!a.is_null(i)).then(|| a.value(i)))
                .collect(),
        ));
    }
    if dtype == &DataType::Null {
        return Ok(Column::Float(vec![f64::NAN; array.len()]));
    }
    let dict_of_string = matches!(dtype, DataType::Dictionary(_, v) if is_string(v));
    if is_string(dtype) || dict_of_string || dtype.is_temporal() {
        let a: ArrayRef = cast(array, &DataType::Utf8).map_err(|_| unsupported())?;
        let a = a.as_string::<i32>();
        return Ok(Column::Text(
            (0..a.len())
                .map(|i| (!a.is_null(i)).then(|| a.value(i).to_string()))
                .collect(),
        ));
    }
    if dtype.is_integer() && array.null_count() == 0 {
        if let Ok(a) = cast(array, &DataType::Int64) {
            let a = a.as_primitive::<Int64Type>();
            return Ok(Column::Int(a.values().to_vec()));
        }
    }
    if dtype.is_numeric() {
        let a: ArrayRef = cast(array, &DataType::Float64).map_err(|_| unsupported())?;
        let a = a.as_primitive::<Float64Type>();
        return Ok(Column::Float(
            (0..a.len())
                .map(|i| if a.is_null(i) { f64::NAN } else { a.value(i) })
                .collect(),
        ));
    }
    Err(unsupported())
}

/// Write a table as Parquet (used by tests and, later, exports).
pub fn write_parquet(table: &Table, path: impl AsRef<Path>) -> Result<(), DataError> {
    use arrow::array::{BooleanArray, Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{Field, Schema};
    use parquet::arrow::ArrowWriter;

    let path = path.as_ref();
    let mut fields = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();
    for (name, col) in table.names.iter().zip(&table.columns) {
        let (dt, arr): (DataType, ArrayRef) = match col {
            Column::Float(v) => (DataType::Float64, Arc::new(Float64Array::from(v.clone()))),
            Column::Int(v) => (DataType::Int64, Arc::new(Int64Array::from(v.clone()))),
            Column::Bool(v) => (DataType::Boolean, Arc::new(BooleanArray::from(v.clone()))),
            Column::Text(v) => (
                DataType::Utf8,
                Arc::new(StringArray::from(
                    v.iter().map(|s| s.as_deref()).collect::<Vec<_>>(),
                )),
            ),
        };
        fields.push(Field::new(name, dt, true));
        arrays.push(arr);
    }
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays).map_err(|e| DataError::Parquet {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;
    let io_err = |source: std::io::Error| DataError::Io {
        path: path.to_path_buf(),
        source,
    };
    let pq_err = |e: parquet::errors::ParquetError| DataError::Parquet {
        path: path.to_path_buf(),
        message: e.to_string(),
    };
    let file = File::create(path).map_err(io_err)?;
    let mut writer = ArrowWriter::try_new(file, schema, None).map_err(pq_err)?;
    writer.write(&batch).map_err(pq_err)?;
    writer.close().map_err(pq_err)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn types_columns_like_pandas() {
        let text = "id,x,flag,name,blank\n1,1.5,True,a,\n2,NA,false,,\n3,2,TRUE,b,NA\n";
        let t = read_csv(text.as_bytes(), "inline").unwrap();
        assert_eq!(t.names, ["id", "x", "flag", "name", "blank"]);
        assert_eq!(t.columns[0], Column::Int(vec![1, 2, 3]));
        match &t.columns[1] {
            Column::Float(v) => {
                assert_eq!(v[0], 1.5);
                assert!(v[1].is_nan());
                assert_eq!(v[2], 2.0);
            }
            other => panic!("x should be float, got {other:?}"),
        }
        assert_eq!(
            t.columns[2],
            Column::Bool(vec![Some(true), Some(false), Some(true)])
        );
        assert_eq!(
            t.columns[3],
            Column::Text(vec![Some("a".into()), None, Some("b".into())])
        );
        match &t.columns[4] {
            Column::Float(v) => assert!(v.iter().all(|x| x.is_nan())),
            other => panic!("blank should be all-NaN float, got {other:?}"),
        }
    }

    #[test]
    fn integers_with_gaps_become_floats() {
        let t = read_csv("a\n1\nNA\n3\n".as_bytes(), "inline").unwrap();
        assert_eq!(t.columns[0].dtype(), "float64");
        assert_eq!(t.columns[0].null_count(), 1);
        // Blank lines are skipped, as pandas does with skip_blank_lines=True.
        let t = read_csv("a\n1\n\n3\n".as_bytes(), "inline").unwrap();
        assert_eq!(t.columns[0], Column::Int(vec![1, 3]));
    }

    #[test]
    fn header_mangling_follows_pandas() {
        let t = read_csv("a,,a,a\n1,2,3,4\n".as_bytes(), "inline").unwrap();
        assert_eq!(t.names, ["a", "Unnamed: 1", "a.1", "a.2"]);
    }

    #[test]
    fn ragged_rows_are_errors() {
        let err = read_csv("a,b\n1,2\n3\n".as_bytes(), "inline").unwrap_err();
        assert_eq!(err.code(), "CSV_MALFORMED");
    }

    #[test]
    fn parquet_round_trip() {
        let table = Table::new(
            vec!["f".into(), "i".into(), "b".into(), "s".into()],
            vec![
                Column::Float(vec![1.0, f64::NAN]),
                Column::Int(vec![7, 8]),
                Column::Bool(vec![Some(true), None]),
                Column::Text(vec![None, Some("x".into())]),
            ],
        )
        .unwrap();
        let dir = std::env::temp_dir().join(format!("pulsar-data-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.parquet");
        write_parquet(&table, &path).unwrap();
        let back = read_table(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(back.names, table.names);
        assert_eq!(back.columns[1], table.columns[1]);
        assert_eq!(back.columns[2], table.columns[2]);
        assert_eq!(back.columns[3], table.columns[3]);
        match &back.columns[0] {
            Column::Float(v) => {
                assert_eq!(v[0], 1.0);
                assert!(v[1].is_nan());
            }
            other => panic!("{other:?}"),
        }
    }
}
