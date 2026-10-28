//! Structured ingest and preprocessing errors.

use std::path::PathBuf;

use thiserror::Error;

/// A failure while reading or preprocessing a table.
///
/// Each variant has a stable [`DataError::code`] so MCP and CLI callers can
/// branch on it without parsing the message.
#[derive(Debug, Error)]
pub enum DataError {
    #[error("cannot read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("{source_name}: malformed CSV: {message}")]
    Csv {
        source_name: String,
        message: String,
    },

    #[error("{path}: cannot read Parquet: {message}")]
    Parquet { path: PathBuf, message: String },

    #[error(
        "column {column:?} has unsupported type {dtype}; cast it to numeric, string or boolean"
    )]
    UnsupportedColumnType { column: String, dtype: String },

    #[error("table has no columns")]
    NoColumns,

    #[error("column {column:?} has {got} rows; expected {expected}")]
    ColumnLength {
        column: String,
        got: usize,
        expected: usize,
    },

    #[error(
        "column {column:?} is configured for numeric imputation ({method}) but contains \
         non-numeric values. Sample: {sample:?}. Clean the data or use a string imputation method."
    )]
    CoercionFailed {
        column: String,
        method: String,
        sample: Vec<String>,
    },

    #[error("column {column:?} is all-missing; cannot impute/encode without observed values")]
    AllMissing { column: String },

    #[error(
        "column {column:?} is a string column; method {method:?} requires numeric data. \
         Use fill_mode, sample_categorical or fill_value for string columns."
    )]
    NumericMethodOnText { column: String, method: String },

    #[error("fill_value for numeric column {column:?} must be numeric, got {value:?}")]
    FillValueType { column: String, value: String },

    #[error(
        "column {column:?} has {n_categories} categories, exceeding max_categories={max}. \
         Reduce cardinality, drop the column, or increase max_categories."
    )]
    TooManyCategories {
        column: String,
        n_categories: usize,
        max: u64,
    },

    #[error("NaN values remain after imputation in {} column(s): {}. Add imputation rules in preprocessing.impute, or drop them via drop_columns.", .columns.len(), fmt_nan(.columns))]
    NanRemaining { columns: Vec<(String, usize)> },

    #[error("non-numeric columns remain after preprocessing: {}. Encode them via preprocessing.encode, or drop them via drop_columns.", .columns.join(", "))]
    NonNumericRemaining { columns: Vec<String> },

    #[error("column {column:?} holds a non-finite value ({value}) at row {row}. Infinite values are not missing and are not imputed; they would turn the whole scaled column into NaN. Fix or null the cell, or drop the column via drop_columns.")]
    NonFiniteValue {
        column: String,
        row: usize,
        value: f64,
    },

    #[error("feature layout does not match the reference dataset. Missing columns: {missing:?}. Extra columns: {extra:?}.")]
    LayoutMismatch {
        missing: Vec<String>,
        extra: Vec<String>,
    },

    #[error("row count {got} does not match the reference dataset ({expected} rows)")]
    RowCountMismatch { got: usize, expected: usize },

    #[error(transparent)]
    Core(#[from] pulsar_core::PulsarError),
}

fn fmt_nan(columns: &[(String, usize)]) -> String {
    columns
        .iter()
        .map(|(c, n)| format!("{c:?} ({n} rows)"))
        .collect::<Vec<_>>()
        .join(", ")
}

impl DataError {
    /// Stable machine-readable code.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Io { .. } => "IO",
            Self::Csv { .. } => "CSV_MALFORMED",
            Self::Parquet { .. } => "PARQUET_MALFORMED",
            Self::UnsupportedColumnType { .. } => "UNSUPPORTED_COLUMN_TYPE",
            Self::NoColumns => "NO_COLUMNS",
            Self::ColumnLength { .. } => "COLUMN_LENGTH",
            Self::CoercionFailed { .. } => "COERCION_FAILED",
            Self::AllMissing { .. } => "ALL_MISSING",
            Self::NumericMethodOnText { .. } => "NUMERIC_METHOD_ON_TEXT",
            Self::FillValueType { .. } => "FILL_VALUE_TYPE",
            Self::TooManyCategories { .. } => "TOO_MANY_CATEGORIES",
            Self::NanRemaining { .. } => "NAN_REMAINING",
            Self::NonNumericRemaining { .. } => "NON_NUMERIC_REMAINING",
            Self::NonFiniteValue { .. } => "NON_FINITE_VALUE",
            Self::LayoutMismatch { .. } => "LAYOUT_MISMATCH",
            Self::RowCountMismatch { .. } => "ROW_COUNT_MISMATCH",
            Self::Core(e) => e.code(),
        }
    }

    /// The column the error is about, when there is one.
    pub fn column(&self) -> Option<&str> {
        match self {
            Self::UnsupportedColumnType { column, .. }
            | Self::ColumnLength { column, .. }
            | Self::CoercionFailed { column, .. }
            | Self::AllMissing { column }
            | Self::NumericMethodOnText { column, .. }
            | Self::FillValueType { column, .. }
            | Self::TooManyCategories { column, .. } => Some(column),
            _ => None,
        }
    }
}
