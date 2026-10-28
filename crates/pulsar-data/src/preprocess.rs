//! The v0.5 `preprocessing` block, executed in Rust.
//!
//! Sequence (identical to Python `preprocess_dataframe` on the #57 branch):
//! group indicators over soon-dropped columns → drop → coerce → all-missing
//! guard → missing indicators → impute → one-hot encode → NaN check →
//! numeric check → layout check.

use std::collections::HashSet;

use indexmap::IndexMap;
use ndarray::Array2;
use pulsar_config::{FillValue, Impute, ImputeMethod, Preprocessing, MISSING_SUFFIX};
use pulsar_core::impute_column_inplace;

use crate::error::DataError;
use crate::pyrandom::{cumulative, PyRandom};
use crate::table::{Column, Table};

/// Shared one-hot vocabularies for `fit_multi`: column → category labels in
/// output order.
pub type Vocab = IndexMap<String, Vec<String>>;

/// The output schema of a preprocessing pass; `fit_multi` requires every
/// dataset to produce the same one.
#[derive(Debug, Clone, PartialEq)]
pub struct Layout {
    pub feature_names: Vec<String>,
    pub vocab: Vocab,
    pub n_rows: usize,
}

/// An all-numeric, NaN-free feature matrix plus its layout.
#[derive(Debug, Clone)]
pub struct Preprocessed {
    /// `(n_rows, n_features)`, row-major.
    pub matrix: Array2<f64>,
    pub layout: Layout,
}

impl Preprocessed {
    pub fn feature_names(&self) -> &[String] {
        &self.layout.feature_names
    }
}

/// A column while preprocessing runs.
enum Work {
    /// Numeric; NaN is missing. `int_like` columns came in as integers and
    /// are labelled without a decimal point when one-hot encoded.
    Num {
        values: Vec<f64>,
        int_like: bool,
    },
    Text(Vec<Option<String>>),
}

impl Work {
    fn null_mask(&self) -> Vec<bool> {
        match self {
            Work::Num { values, .. } => values.iter().map(|x| x.is_nan()).collect(),
            Work::Text(v) => v.iter().map(Option::is_none).collect(),
        }
    }

    fn all_null(&self) -> bool {
        self.null_mask().iter().all(|&m| m)
    }
}

struct Frame {
    names: Vec<String>,
    cols: Vec<Work>,
}

impl Frame {
    fn index(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }
}

/// The union of each encoded column's categories across `tables`, sorted,
/// so one-hot blocks line up across datasets.
pub fn shared_vocab(tables: &[&Table], pre: &Preprocessing) -> Vocab {
    let mut vocab = Vocab::new();
    for col in pre.encode.keys() {
        let mut numeric: Vec<f64> = Vec::new();
        let mut int_like = true;
        let mut text: Vec<String> = Vec::new();
        let mut all_numeric = true;
        let mut seen_any = false;
        for table in tables {
            let Some(c) = table.column(col) else { continue };
            seen_any = true;
            match c {
                Column::Float(v) => {
                    int_like = false;
                    numeric.extend(v.iter().copied().filter(|x| !x.is_nan()));
                }
                Column::Int(v) => numeric.extend(v.iter().map(|&x| x as f64)),
                Column::Bool(v) => {
                    all_numeric = false;
                    text.extend(v.iter().flatten().map(|b| bool_label(*b)));
                }
                Column::Text(v) => {
                    all_numeric = false;
                    text.extend(v.iter().flatten().cloned());
                }
            }
        }
        if !seen_any {
            continue;
        }
        let labels: Vec<String> = if all_numeric {
            let mut u = numeric;
            u.sort_by(|a, b| a.total_cmp(b));
            u.dedup();
            u.into_iter().map(|x| category_label(x, int_like)).collect()
        } else {
            text.extend(numeric.into_iter().map(|x| category_label(x, int_like)));
            text.sort();
            text.dedup();
            text
        };
        vocab.insert(col.clone(), labels);
    }
    vocab
}

/// Apply a resolved `preprocessing` block to a table.
///
/// `vocab` fixes one-hot categories (shared across `fit_multi` datasets);
/// `expected` is the layout every later dataset must reproduce.
pub fn preprocess(
    table: &Table,
    pre: &Preprocessing,
    vocab: Option<&Vocab>,
    expected: Option<&Layout>,
) -> Result<Preprocessed, DataError> {
    let n_rows = table.n_rows();
    let drop: HashSet<&str> = pre.drop_columns.iter().map(String::as_str).collect();
    let grouped: HashSet<&str> = pre
        .missing_indicator_groups
        .values()
        .flatten()
        .map(String::as_str)
        .collect();

    // 1. Group indicators must see columns that are about to be dropped.
    let mut pre_drop_masks: IndexMap<&str, Vec<bool>> = IndexMap::new();
    for (group, cols) in &pre.missing_indicator_groups {
        let dropped_present: Vec<&Column> = cols
            .iter()
            .filter(|c| drop.contains(c.as_str()))
            .filter_map(|c| table.column(c))
            .collect();
        if !dropped_present.is_empty() {
            let mask = (0..n_rows)
                .map(|i| dropped_present.iter().any(|c| c.is_null(i)))
                .collect();
            pre_drop_masks.insert(group.as_str(), mask);
        }
    }

    // 2. Drop.
    let mut frame = Frame {
        names: Vec::new(),
        cols: Vec::new(),
    };
    for (name, col) in table.names.iter().zip(&table.columns) {
        if drop.contains(name.as_str()) {
            continue;
        }
        frame.names.push(name.clone());
        frame.cols.push(to_work(name, col, pre));
    }

    // 3. Coerce string columns that are targets of numeric imputation.
    for (col, spec) in &pre.impute {
        let Some(idx) = frame.index(col) else {
            continue;
        };
        if let Work::Text(values) = &frame.cols[idx] {
            if needs_numeric_coercion(spec) {
                frame.cols[idx] = coerce_numeric(col, spec, values)?;
            }
        }
    }

    // 4. All-missing guard for impute/encode targets.
    for col in pre.impute.keys().chain(pre.encode.keys()) {
        let Some(idx) = frame.index(col) else {
            continue;
        };
        let fill_value = pre
            .impute
            .get(col)
            .is_some_and(|s| s.method == ImputeMethod::FillValue);
        if !fill_value && frame.cols[idx].all_null() {
            return Err(DataError::AllMissing {
                column: col.clone(),
            });
        }
    }

    // 5. Missing indicators: groups first, then per-column flags.
    for (group, cols) in &pre.missing_indicator_groups {
        let kept: Vec<usize> = cols.iter().filter_map(|c| frame.index(c)).collect();
        let mut mask = pre_drop_masks.get(group.as_str()).cloned();
        if !kept.is_empty() {
            let masks: Vec<Vec<bool>> = kept.iter().map(|&i| frame.cols[i].null_mask()).collect();
            let kept_mask: Vec<bool> = (0..n_rows).map(|r| masks.iter().any(|m| m[r])).collect();
            mask = Some(match mask {
                Some(pre) => pre.iter().zip(&kept_mask).map(|(a, b)| *a || *b).collect(),
                None => kept_mask,
            });
        }
        if let Some(mask) = mask {
            frame.names.push(format!("{group}{MISSING_SUFFIX}"));
            frame.cols.push(flag_column(&mask));
        }
    }
    for (col, spec) in &pre.impute {
        let Some(idx) = frame.index(col) else {
            continue;
        };
        if spec.indicator && !grouped.contains(col.as_str()) {
            let mask = frame.cols[idx].null_mask();
            frame.names.push(format!("{col}{MISSING_SUFFIX}"));
            frame.cols.push(flag_column(&mask));
        }
    }

    // 5b. Infinite values are not missing: imputation would average them into
    // NaN and one-hot encoding would turn them into an `_inf` category, so
    // reject them here, while the row index is still the original row.
    for (name, col) in frame.names.iter().zip(&frame.cols) {
        if let Work::Num { values, .. } = col {
            if let Some((row, &value)) = values.iter().enumerate().find(|(_, x)| x.is_infinite()) {
                return Err(DataError::NonFiniteValue {
                    column: name.clone(),
                    row,
                    value,
                });
            }
        }
    }

    // 6. Impute.
    for (col, spec) in &pre.impute {
        let Some(idx) = frame.index(col) else {
            continue;
        };
        match &mut frame.cols[idx] {
            Work::Num { values, .. } => impute_numeric(col, spec, values)?,
            Work::Text(values) => impute_text(col, spec, values)?,
        }
    }

    // 7. One-hot encode: the column is removed and its block appended.
    let mut built_vocab = Vocab::new();
    for (col, spec) in &pre.encode {
        let Some(idx) = frame.index(col) else {
            continue;
        };
        frame.names.remove(idx);
        let work = frame.cols.remove(idx);
        let labels: Vec<Option<String>> = match &work {
            Work::Text(v) => v.clone(),
            Work::Num { values, int_like } => values
                .iter()
                .map(|&x| (!x.is_nan()).then(|| category_label(x, *int_like)))
                .collect(),
        };
        let categories: Vec<String> = match vocab.and_then(|v| v.get(col)) {
            Some(fixed) => fixed.clone(),
            None => match &work {
                Work::Num { values, int_like } => {
                    let mut u: Vec<f64> = values.iter().copied().filter(|x| !x.is_nan()).collect();
                    u.sort_by(|a, b| a.total_cmp(b));
                    u.dedup();
                    u.into_iter()
                        .map(|x| category_label(x, *int_like))
                        .collect()
                }
                Work::Text(v) => {
                    let mut u: Vec<String> = v.iter().flatten().cloned().collect();
                    u.sort();
                    u.dedup();
                    u
                }
            },
        };
        if let Some(max) = spec.max_categories {
            if categories.len() as u64 > max {
                return Err(DataError::TooManyCategories {
                    column: col.clone(),
                    n_categories: categories.len(),
                    max,
                });
            }
        }
        for cat in &categories {
            let values = labels
                .iter()
                .map(|l| {
                    if l.as_deref() == Some(cat.as_str()) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            frame.names.push(format!("{col}_{cat}"));
            frame.cols.push(Work::Num {
                values,
                int_like: false,
            });
        }
        if let Some(v) = vocab {
            if let Some(fixed) = v.get(col) {
                built_vocab.insert(col.clone(), fixed.clone());
            }
        }
    }

    // 8. No NaN may remain.
    let nan_cols: Vec<(String, usize)> = frame
        .names
        .iter()
        .zip(&frame.cols)
        .filter_map(|(n, c)| match c {
            Work::Num { values, .. } => {
                let k = values.iter().filter(|x| x.is_nan()).count();
                (k > 0).then(|| (n.clone(), k))
            }
            Work::Text(_) => None,
        })
        .collect();
    if !nan_cols.is_empty() {
        return Err(DataError::NanRemaining { columns: nan_cols });
    }

    // 9. Everything must be numeric.
    let text_cols: Vec<String> = frame
        .names
        .iter()
        .zip(&frame.cols)
        .filter(|(_, c)| matches!(c, Work::Text(_)))
        .map(|(n, _)| n.clone())
        .collect();
    if !text_cols.is_empty() {
        return Err(DataError::NonNumericRemaining { columns: text_cols });
    }

    // 10–11. Layout.
    let layout = Layout {
        feature_names: frame.names.clone(),
        vocab: built_vocab,
        n_rows,
    };
    if let Some(expected) = expected {
        if layout.feature_names != expected.feature_names {
            let want: HashSet<&String> = expected.feature_names.iter().collect();
            let have: HashSet<&String> = layout.feature_names.iter().collect();
            let mut missing: Vec<String> = want.difference(&have).map(|s| (*s).clone()).collect();
            let mut extra: Vec<String> = have.difference(&want).map(|s| (*s).clone()).collect();
            missing.sort();
            extra.sort();
            return Err(DataError::LayoutMismatch { missing, extra });
        }
        if layout.n_rows != expected.n_rows {
            return Err(DataError::RowCountMismatch {
                got: layout.n_rows,
                expected: expected.n_rows,
            });
        }
    }

    let n_features = frame.cols.len();
    let mut matrix = Array2::<f64>::zeros((n_rows, n_features));
    for (j, col) in frame.cols.iter().enumerate() {
        if let Work::Num { values, .. } = col {
            for (i, &x) in values.iter().enumerate() {
                matrix[[i, j]] = x;
            }
        }
    }
    Ok(Preprocessed { matrix, layout })
}

fn to_work(name: &str, col: &Column, pre: &Preprocessing) -> Work {
    match col {
        Column::Float(v) => Work::Num {
            values: v.clone(),
            int_like: false,
        },
        Column::Int(v) => Work::Num {
            values: v.iter().map(|&x| x as f64).collect(),
            int_like: true,
        },
        Column::Text(v) => Work::Text(v.clone()),
        Column::Bool(v) => {
            if bool_as_text(name, pre) {
                Work::Text(v.iter().map(|b| b.map(bool_label)).collect())
            } else {
                Work::Num {
                    values: v
                        .iter()
                        .map(|b| b.map_or(f64::NAN, |b| if b { 1.0 } else { 0.0 }))
                        .collect(),
                    int_like: false,
                }
            }
        }
    }
}

/// A boolean column stays categorical when it is one-hot encoded or imputed
/// with a string method; otherwise it is numeric 0/1.
fn bool_as_text(name: &str, pre: &Preprocessing) -> bool {
    if pre.encode.contains_key(name) {
        return true;
    }
    matches!(
        pre.impute.get(name),
        Some(Impute {
            method: ImputeMethod::FillMode | ImputeMethod::SampleCategorical,
            ..
        }) | Some(Impute {
            method: ImputeMethod::FillValue,
            value: Some(FillValue::Text(_)),
            ..
        })
    )
}

fn bool_label(b: bool) -> String {
    if b { "True" } else { "False" }.to_string()
}

fn flag_column(mask: &[bool]) -> Work {
    Work::Num {
        values: mask.iter().map(|&m| if m { 1.0 } else { 0.0 }).collect(),
        int_like: false,
    }
}

fn needs_numeric_coercion(spec: &Impute) -> bool {
    match spec.method {
        ImputeMethod::FillMean | ImputeMethod::FillMedian | ImputeMethod::SampleNormal => true,
        ImputeMethod::FillValue => matches!(
            spec.value,
            Some(FillValue::Integer(_)) | Some(FillValue::Float(_))
        ),
        _ => false,
    }
}

/// `pd.to_numeric(errors="coerce")`, failing when fewer than half of the
/// observed values parse.
fn coerce_numeric(col: &str, spec: &Impute, values: &[Option<String>]) -> Result<Work, DataError> {
    let mut out = Vec::with_capacity(values.len());
    let mut observed = 0usize;
    let mut parsed = 0usize;
    let mut bad: Vec<String> = Vec::new();
    for v in values {
        match v {
            None => out.push(f64::NAN),
            Some(s) => {
                observed += 1;
                match s.trim().parse::<f64>().ok().filter(|x| !x.is_nan()) {
                    Some(x) => {
                        parsed += 1;
                        out.push(x);
                    }
                    None => {
                        if bad.len() < 5 {
                            bad.push(s.clone());
                        }
                        out.push(f64::NAN);
                    }
                }
            }
        }
    }
    if observed > 0 && (parsed as f64) < observed as f64 * 0.5 {
        return Err(DataError::CoercionFailed {
            column: col.to_string(),
            method: method_name(spec.method).to_string(),
            sample: bad,
        });
    }
    Ok(Work::Num {
        values: out,
        int_like: false,
    })
}

fn method_name(m: ImputeMethod) -> &'static str {
    match m {
        ImputeMethod::SampleNormal => "sample_normal",
        ImputeMethod::SampleCategorical => "sample_categorical",
        ImputeMethod::FillMean => "fill_mean",
        ImputeMethod::FillMedian => "fill_median",
        ImputeMethod::FillMode => "fill_mode",
        ImputeMethod::FillValue => "fill_value",
    }
}

fn impute_numeric(col: &str, spec: &Impute, values: &mut [f64]) -> Result<(), DataError> {
    if spec.method == ImputeMethod::FillValue {
        let fill = match &spec.value {
            Some(FillValue::Integer(i)) => *i as f64,
            Some(FillValue::Float(f)) => *f,
            other => {
                return Err(DataError::FillValueType {
                    column: col.to_string(),
                    value: match other {
                        Some(FillValue::Text(t)) => t.clone(),
                        _ => "<missing>".to_string(),
                    },
                })
            }
        };
        for v in values.iter_mut() {
            if v.is_nan() {
                *v = fill;
            }
        }
        return Ok(());
    }
    impute_column_inplace(values, method_name(spec.method), spec.seed).map_err(|e| match e {
        pulsar_core::PulsarError::EmptyInput { .. } => DataError::AllMissing {
            column: col.to_string(),
        },
        other => DataError::Core(other),
    })
}

fn impute_text(col: &str, spec: &Impute, values: &mut [Option<String>]) -> Result<(), DataError> {
    let missing: Vec<usize> = values
        .iter()
        .enumerate()
        .filter_map(|(i, v)| v.is_none().then_some(i))
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    match spec.method {
        ImputeMethod::FillValue => {
            let fill = match &spec.value {
                Some(FillValue::Text(t)) => t.clone(),
                Some(FillValue::Integer(i)) => i.to_string(),
                Some(FillValue::Float(f)) => py_float_repr(*f),
                None => {
                    return Err(DataError::FillValueType {
                        column: col.to_string(),
                        value: "<missing>".to_string(),
                    })
                }
            };
            for &i in &missing {
                values[i] = Some(fill.clone());
            }
        }
        ImputeMethod::FillMode => {
            // pandas `mode()` is sorted, so ties resolve to the smallest value.
            let counts = value_counts(values);
            let best = counts
                .iter()
                .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
                .map(|(v, _)| v.clone())
                .ok_or_else(|| DataError::AllMissing {
                    column: col.to_string(),
                })?;
            for &i in &missing {
                values[i] = Some(best.clone());
            }
        }
        ImputeMethod::SampleCategorical => {
            // random.Random(seed).choices(categories, weights=counts, k=n_missing)
            // with categories in pandas `value_counts()` order.
            let counts = value_counts(values);
            if counts.is_empty() {
                return Err(DataError::AllMissing {
                    column: col.to_string(),
                });
            }
            let weights: Vec<f64> = counts.iter().map(|(_, c)| *c as f64).collect();
            let cum = cumulative(&weights);
            let mut rng = PyRandom::new(spec.seed);
            for &i in &missing {
                let k = rng.choice_index(&cum);
                values[i] = Some(counts[k].0.clone());
            }
        }
        other => {
            return Err(DataError::NumericMethodOnText {
                column: col.to_string(),
                method: method_name(other).to_string(),
            })
        }
    }
    Ok(())
}

/// `Series.value_counts()`: categories by descending count, ties in order of
/// first appearance.
fn value_counts(values: &[Option<String>]) -> Vec<(String, usize)> {
    let mut counts: IndexMap<&str, usize> = IndexMap::new();
    for v in values.iter().flatten() {
        *counts.entry(v.as_str()).or_insert(0) += 1;
    }
    let mut out: Vec<(String, usize)> = counts
        .into_iter()
        .map(|(k, c)| (k.to_string(), c))
        .collect();
    out.sort_by_key(|a| std::cmp::Reverse(a.1));
    out
}

/// The label pandas gives a numeric one-hot category: `3` for integer
/// columns, `repr(float)` otherwise.
pub fn category_label(x: f64, int_like: bool) -> String {
    if int_like && x.fract() == 0.0 && x.abs() < 9.2e18 {
        format!("{}", x as i64)
    } else {
        py_float_repr(x)
    }
}

/// Python `repr(float)`: shortest round-trip digits, exponent form outside
/// `1e-4 <= |x| < 1e16`.
pub fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    let sci = format!("{x:e}");
    let (mantissa, exp) = sci.split_once('e').expect("{:e} always has an exponent");
    let exp: i32 = exp.parse().expect("exponent is an integer");
    if (-4..16).contains(&exp) {
        let plain = format!("{x}");
        if plain.contains('.') {
            plain
        } else {
            format!("{plain}.0")
        }
    } else {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exp.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pulsar_config::from_yaml_str;

    fn table(csv: &str) -> Table {
        crate::read_csv(csv.as_bytes(), "inline").unwrap()
    }

    fn run(csv: &str, yaml: &str) -> Result<Preprocessed, DataError> {
        let cfg = from_yaml_str(yaml).unwrap();
        preprocess(&table(csv), &cfg.preprocessing, None, None)
    }

    #[test]
    fn float_repr_matches_python() {
        for (x, want) in [
            (1.0, "1.0"),
            (2.5, "2.5"),
            (1e20, "1e+20"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (-3.25, "-3.25"),
            (0.1, "0.1"),
        ] {
            assert_eq!(py_float_repr(x), want, "{x}");
        }
        assert_eq!(category_label(3.0, true), "3");
        assert_eq!(category_label(3.0, false), "3.0");
    }

    #[test]
    fn known_answer_impute_flag_and_encode() {
        let out = run(
            "a,b,c\n1,x,10\n,y,20\n3,,30\n",
            "preprocessing:\n  impute:\n    a: {method: fill_mean}\n    b: {method: fill_mode}\n  encode:\n    b: {method: one_hot}\n",
        )
        .unwrap();
        assert_eq!(
            out.feature_names(),
            ["a", "c", "a_was_missing", "b_was_missing", "b_x", "b_y"]
        );
        let m = out.matrix;
        assert_eq!(m.row(1).to_vec(), [2.0, 20.0, 1.0, 0.0, 0.0, 1.0]);
        // fill_mode tie (x and y once each) resolves to the smallest: x.
        assert_eq!(m.row(2).to_vec(), [3.0, 30.0, 0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn indicator_opt_out_and_groups() {
        let out = run(
            "a,b,c\n1,2,\n,,3\n",
            "preprocessing:\n  drop_columns: [c]\n  impute:\n    a: {method: fill_value, value: 0, indicator: false}\n    b: {method: fill_value, value: -1}\n  missing_indicator_groups:\n    lab: [b, c]\n",
        )
        .unwrap();
        // Group flag first (and it sees the dropped column c), b is grouped
        // so it gets no per-column flag, a opted out.
        assert_eq!(out.feature_names(), ["a", "b", "lab_was_missing"]);
        assert_eq!(out.matrix.row(0).to_vec(), [1.0, 2.0, 1.0]);
        assert_eq!(out.matrix.row(1).to_vec(), [0.0, -1.0, 1.0]);
    }

    #[test]
    fn integer_and_bool_categories_are_labelled_like_pandas() {
        let out = run(
            "i,b,f\n1,True,1.5\n2,False,2\n2,True,1.5\n",
            "preprocessing:\n  encode:\n    i: {method: one_hot}\n    b: {method: one_hot}\n    f: {method: one_hot}\n",
        )
        .unwrap();
        assert_eq!(
            out.feature_names(),
            ["i_1", "i_2", "b_False", "b_True", "f_1.5", "f_2.0"]
        );
    }

    #[test]
    fn unencoded_bool_is_numeric() {
        let out = run("b\nTrue\nfalse\n", "run: {name: t}\n").unwrap();
        assert_eq!(out.matrix.column(0).to_vec(), [1.0, 0.0]);
    }

    #[test]
    fn structured_errors() {
        let e = run(
            "a\nx\n",
            "preprocessing:\n  impute:\n    a: {method: fill_mean}\n",
        )
        .unwrap_err();
        assert_eq!(e.code(), "COERCION_FAILED");
        let e = run(
            "a\n\n\n",
            "preprocessing:\n  impute:\n    a: {method: fill_mean}\n",
        )
        .unwrap_err();
        assert_eq!(e.code(), "ALL_MISSING");
        let e = run("a\n1\nNA\n", "run: {name: t}\n").unwrap_err();
        assert_eq!(e.code(), "NAN_REMAINING");
        let e = run("a\n1\ninf\n", "run: {name: t}\n").unwrap_err();
        assert_eq!(e.code(), "NON_FINITE_VALUE");
        assert!(e.to_string().contains("row 1"), "{e}");
        let e = run(
            "a\n1\ninf\n",
            "preprocessing:\n  encode:\n    a: {method: one_hot}\n",
        )
        .unwrap_err();
        assert_eq!(e.code(), "NON_FINITE_VALUE");
        let e = run(
            "a\n1\ninf\n\n",
            "preprocessing:\n  impute:\n    a: {method: fill_mean}\n",
        )
        .unwrap_err();
        assert_eq!(e.code(), "NON_FINITE_VALUE");
        assert!(e.to_string().contains("row 1"), "{e}");
        let e = run("a\nx\ny\n", "run: {name: t}\n").unwrap_err();
        assert_eq!(e.code(), "NON_NUMERIC_REMAINING");
        let e = run(
            "a\nx\ny\nz\n",
            "preprocessing:\n  encode:\n    a: {method: one_hot, max_categories: 2}\n",
        )
        .unwrap_err();
        assert_eq!(e.code(), "TOO_MANY_CATEGORIES");
        let e = run(
            "a\nx\n\n",
            "preprocessing:\n  impute:\n    a: {method: sample_normal}\n",
        )
        .unwrap_err();
        assert_eq!(e.code(), "COERCION_FAILED");
    }

    #[test]
    fn shared_vocab_aligns_layouts() {
        let cfg = from_yaml_str("preprocessing:\n  encode:\n    s: {method: one_hot}\n").unwrap();
        let t1 = table("s\nb\na\n");
        let t2 = table("s\nc\na\n");
        let vocab = shared_vocab(&[&t1, &t2], &cfg.preprocessing);
        assert_eq!(vocab["s"], ["a", "b", "c"]);
        let p1 = preprocess(&t1, &cfg.preprocessing, Some(&vocab), None).unwrap();
        let p2 = preprocess(&t2, &cfg.preprocessing, Some(&vocab), Some(&p1.layout)).unwrap();
        assert_eq!(p1.feature_names(), ["s_a", "s_b", "s_c"]);
        assert_eq!(p2.matrix.row(0).to_vec(), [0.0, 0.0, 1.0]);
        // Without the vocab the layouts differ, and the check says how.
        let p3 = preprocess(&t2, &cfg.preprocessing, None, Some(&p1.layout)).unwrap_err();
        match p3 {
            DataError::LayoutMismatch { missing, extra } => {
                assert_eq!(missing, ["s_b"]);
                assert_eq!(extra, Vec::<String>::new());
            }
            other => panic!("{other:?}"),
        }
    }
}
