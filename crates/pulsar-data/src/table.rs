//! An in-memory column table with pandas-like typing.

use crate::error::DataError;

/// One column. Typing follows `pandas.read_csv`: a numeric column with any
/// missing value is `Float` (NaN marks the gap); `Int` columns never have
/// gaps.
#[derive(Debug, Clone, PartialEq)]
pub enum Column {
    /// 64-bit floats; `NaN` is a missing value.
    Float(Vec<f64>),
    /// 64-bit integers, never missing.
    Int(Vec<i64>),
    /// Booleans, `None` is a missing value.
    Bool(Vec<Option<bool>>),
    /// Strings, `None` is a missing value.
    Text(Vec<Option<String>>),
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Self::Float(v) => v.len(),
            Self::Int(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::Text(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// pandas dtype name: `float64`, `int64`, `bool` or `str`.
    pub fn dtype(&self) -> &'static str {
        match self {
            Self::Float(_) => "float64",
            Self::Int(_) => "int64",
            Self::Bool(_) => "bool",
            Self::Text(_) => "str",
        }
    }

    pub fn is_null(&self, i: usize) -> bool {
        match self {
            Self::Float(v) => v[i].is_nan(),
            Self::Int(_) => false,
            Self::Bool(v) => v[i].is_none(),
            Self::Text(v) => v[i].is_none(),
        }
    }

    pub fn null_count(&self) -> usize {
        (0..self.len()).filter(|&i| self.is_null(i)).count()
    }
}

/// Named columns of equal length.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Table {
    pub names: Vec<String>,
    pub columns: Vec<Column>,
}

impl Table {
    /// Build a table, checking that every column has the same length.
    pub fn new(names: Vec<String>, columns: Vec<Column>) -> Result<Self, DataError> {
        if names.is_empty() || columns.is_empty() {
            return Err(DataError::NoColumns);
        }
        let expected = columns[0].len();
        for (name, col) in names.iter().zip(&columns) {
            if col.len() != expected {
                return Err(DataError::ColumnLength {
                    column: name.clone(),
                    got: col.len(),
                    expected,
                });
            }
        }
        Ok(Self { names, columns })
    }

    pub fn n_rows(&self) -> usize {
        self.columns.first().map_or(0, Column::len)
    }

    /// A new table with the given rows, in the given order.
    ///
    /// Panics if an index is out of range; callers pass indices below
    /// [`Table::n_rows`].
    pub fn take_rows(&self, rows: &[usize]) -> Table {
        fn pick<T: Clone>(v: &[T], rows: &[usize]) -> Vec<T> {
            rows.iter().map(|&i| v[i].clone()).collect()
        }
        let columns = self
            .columns
            .iter()
            .map(|c| match c {
                Column::Float(v) => Column::Float(pick(v, rows)),
                Column::Int(v) => Column::Int(pick(v, rows)),
                Column::Bool(v) => Column::Bool(pick(v, rows)),
                Column::Text(v) => Column::Text(pick(v, rows)),
            })
            .collect();
        Table {
            names: self.names.clone(),
            columns,
        }
    }

    pub fn n_cols(&self) -> usize {
        self.columns.len()
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    pub fn column(&self, name: &str) -> Option<&Column> {
        self.index_of(name).map(|i| &self.columns[i])
    }

    /// `(name, dtype)` pairs in column order; the input-schema part of the
    /// projection fingerprint.
    pub fn schema(&self) -> Vec<(String, &'static str)> {
        self.names
            .iter()
            .zip(&self.columns)
            .map(|(n, c)| (n.clone(), c.dtype()))
            .collect()
    }
}
