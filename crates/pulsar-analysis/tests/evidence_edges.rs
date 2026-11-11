//! Edge cases and known answers for `measure_preprocessing_evidence`.
//! Reference parity lives in `pulsar-parity/tests/evidence.rs`.

use pulsar_analysis::{measure_preprocessing_evidence, ColumnKind, EvidenceError, EvidenceOptions};
use pulsar_data::{Column, Table};

fn table(cols: Vec<(&str, Column)>) -> Table {
    let (names, columns) = cols.into_iter().map(|(n, c)| (n.to_string(), c)).unzip();
    Table::new(names, columns).unwrap()
}

fn floats(v: &[f64]) -> Column {
    Column::Float(v.to_vec())
}

fn measure(t: &Table) -> pulsar_analysis::PreprocessingEvidence {
    measure_preprocessing_evidence(t, &EvidenceOptions::default()).unwrap()
}

#[test]
fn zero_rows_is_a_named_error() {
    let t = table(vec![("a", floats(&[]))]);
    let err = measure_preprocessing_evidence(&t, &EvidenceOptions::default()).unwrap_err();
    assert!(matches!(err, EvidenceError::NoRows));
    assert_eq!(err.to_string(), "dataset has no rows");
    assert_eq!(err.code(), "EMPTY_DATASET");
}

#[test]
fn excluding_every_column_is_a_named_error() {
    let t = table(vec![("a", floats(&[1.0, 2.0]))]);
    let opts = EvidenceOptions {
        exclude_columns: vec!["a".into()],
        ..Default::default()
    };
    let err = measure_preprocessing_evidence(&t, &opts).unwrap_err();
    assert_eq!(err.to_string(), "dataset has no columns");
}

#[test]
fn zero_rows_and_columns_after_exclusion() {
    let t = table(vec![("a", floats(&[]))]);
    let opts = EvidenceOptions {
        exclude_columns: vec!["a".into()],
        ..Default::default()
    };
    let err = measure_preprocessing_evidence(&t, &opts).unwrap_err();
    assert_eq!(err.to_string(), "dataset has no rows and no columns");
}

#[test]
fn duplicate_column_names_are_rejected() {
    let t = Table {
        names: vec!["a".into(), "b".into(), "a".into()],
        columns: vec![floats(&[1.0]), floats(&[2.0]), floats(&[3.0])],
    };
    let err = measure_preprocessing_evidence(&t, &EvidenceOptions::default()).unwrap_err();
    assert_eq!(
        err.to_string(),
        "dataset has duplicate column names: [\"a\"]"
    );
    assert_eq!(err.code(), "DUPLICATE_COLUMNS");
}

#[test]
fn single_row_measures_columns_and_skips_relations() {
    let t = table(vec![("a", floats(&[3.0])), ("b", floats(&[4.0]))]);
    let ev = measure(&t);
    assert_eq!(ev.n_rows, 1);
    assert!(ev.columns.iter().all(|c| c.is_constant && c.n_unique == 1));
    assert!(ev.relations.is_empty());
    assert_eq!(
        ev.cost.steps_skipped[0].reason,
        "fewer than 2 rows in dataset"
    );
}

#[test]
fn single_numeric_column_skips_relations() {
    let t = table(vec![
        ("a", floats(&[1.0, 2.0, 3.0])),
        (
            "s",
            Column::Text(vec![Some("x".into()), None, Some("y".into())]),
        ),
    ]);
    let ev = measure(&t);
    assert_eq!(
        ev.cost.steps_skipped[0].reason,
        "fewer than 2 numeric or boolean columns"
    );
    assert_eq!(ev.cost.numeric_columns_considered, 1);
}

#[test]
fn too_many_numeric_columns_skips_relations_with_the_limit_in_the_reason() {
    let t = table(vec![
        ("a", floats(&[1.0, 2.0, 3.0])),
        ("b", floats(&[3.0, 1.0, 2.0])),
    ]);
    let opts = EvidenceOptions {
        max_relation_columns: 1,
        ..Default::default()
    };
    let ev = measure_preprocessing_evidence(&t, &opts).unwrap();
    assert_eq!(
        ev.cost.steps_skipped[0].reason,
        "numeric/boolean column count (2) exceeds max_relation_columns (1)"
    );
    assert_eq!(ev.cost.numeric_columns_considered, 0);
}

#[test]
fn all_missing_column_is_constant_with_no_statistics() {
    let t = table(vec![
        ("gone", floats(&[f64::NAN, f64::NAN, f64::NAN])),
        ("a", floats(&[1.0, 2.0, 3.0])),
    ]);
    let c = &measure(&t).columns[0];
    assert_eq!(
        (c.n_missing, c.n_unique, c.is_constant, c.kind),
        (3, 0, true, ColumnKind::Numeric)
    );
    assert_eq!(
        (c.observed_min, c.n_zero, c.nonnegative, c.max_abs_z),
        (None, None, None, None)
    );
}

#[test]
fn known_answers_for_one_numeric_column() {
    // values 0, 2, 4, 6, NaN: mean 3, population std sqrt(5).
    let t = table(vec![("x", floats(&[0.0, 2.0, 4.0, 6.0, f64::NAN]))]);
    let c = &measure(&t).columns[0];
    assert_eq!((c.n_rows, c.n_missing, c.n_unique), (5, 1, 4));
    assert_eq!(c.missing_fraction, 0.2);
    assert_eq!(c.unique_ratio, 1.0);
    assert_eq!((c.observed_min, c.observed_max), (Some(0.0), Some(6.0)));
    assert_eq!(c.n_zero, Some(1));
    assert_eq!(c.integer_valued, Some(true));
    assert_eq!(c.nonnegative, Some(true));
    let z = c.max_abs_z.unwrap();
    assert!((z - 3.0 / 5.0_f64.sqrt()).abs() < 1e-12, "{z}");
}

#[test]
fn zero_one_floats_are_boolean_and_report_minority_share() {
    let t = table(vec![("flag", floats(&[0.0, 1.0, 1.0, 1.0]))]);
    let c = &measure(&t).columns[0];
    assert_eq!(c.kind, ColumnKind::Boolean);
    assert_eq!((c.n_unique, c.minority_fraction), (2, Some(0.25)));
    assert_eq!(c.integer_valued, None);
}

#[test]
fn infinite_values_are_not_finite_statistics() {
    let t = table(vec![("x", floats(&[1.0, f64::INFINITY, 3.0]))]);
    let c = &measure(&t).columns[0];
    assert_eq!((c.observed_min, c.observed_max), (Some(1.0), Some(3.0)));
    assert_eq!(c.integer_valued, Some(false), "inf is not integer-valued");
}

#[test]
fn shared_missing_mask_forms_a_group_and_partner_zero_a_dependency() {
    let n = f64::NAN;
    let t = table(vec![
        ("flag", floats(&[0.0, 1.0, 0.0, 1.0, 1.0])),
        ("a", floats(&[n, 1.0, n, 2.0, 3.0])),
        ("b", floats(&[n, 5.0, n, 6.0, 7.0])),
    ]);
    let ev = measure(&t);
    assert_eq!(ev.missing_groups.len(), 1);
    assert_eq!(ev.missing_groups[0].columns, ["a", "b"]);
    assert_eq!(ev.missing_groups[0].n_missing, 2);
    assert_eq!(ev.missing_groups[0].zero_free_columns, ["a", "b"]);
    assert_eq!(ev.columns[1].missing_group.as_deref(), Some("mg1"));
    let deps: Vec<_> = ev
        .mask_dependencies
        .iter()
        .map(|d| (d.column.as_str(), d.partner.as_str(), d.n_other_partners))
        .collect();
    assert_eq!(deps, [("a", "flag", 0), ("b", "flag", 0)]);
    assert_eq!(
        ev.columns[1]
            .mask_dependency
            .as_ref()
            .unwrap()
            .relation
            .as_str(),
        "missing_when_partner_zero"
    );
}

#[test]
fn duplicate_and_exact_sum_columns_are_found() {
    let a: Vec<f64> = (0..30).map(|i| f64::from(i * i % 17)).collect();
    let b: Vec<f64> = (0..30).map(|i| f64::from(i * 7 % 11)).collect();
    let c: Vec<f64> = (0..30).map(|i| f64::from((i * 13 + 5) % 19)).collect();
    let sum: Vec<f64> = (0..30).map(|i| a[i] + b[i] + c[i]).collect();
    let t = table(vec![
        ("a", floats(&a)),
        ("a_copy", floats(&a)),
        ("b", floats(&b)),
        ("c", floats(&c)),
        ("total", floats(&sum)),
    ]);
    let ev = measure(&t);
    let dup = ev
        .relations
        .iter()
        .find(|r| r.kind.as_str() == "duplicate")
        .unwrap();
    assert_eq!(
        (dup.target.as_str(), dup.sources.as_slice()),
        ("a_copy", ["a".to_string()].as_slice())
    );
    let total = ev
        .relations
        .iter()
        .find(|r| r.kind.as_str() == "sum_of")
        .unwrap();
    assert_eq!(total.target, "total");
    assert_eq!(total.sources, ["a", "b", "c"]);
    assert_eq!(total.rows_used, 30);
    assert!(total.value <= 1e-9);
}

#[test]
fn relations_are_measured_on_a_deterministic_sample_of_large_tables() {
    let n = 400;
    let a: Vec<f64> = (0..n).map(|i| ((i * 37) % 101) as f64).collect();
    let b: Vec<f64> = a.iter().map(|v| v * 2.0 + 1.0).collect();
    let t = table(vec![("a", floats(&a)), ("b", floats(&b))]);
    let opts = EvidenceOptions {
        sample_rows: 50,
        seed: 3,
        ..Default::default()
    };
    let first = measure_preprocessing_evidence(&t, &opts).unwrap();
    let second = measure_preprocessing_evidence(&t, &opts).unwrap();
    assert_eq!(first.cost.rows_sampled, 50);
    assert_eq!(first.relations, second.relations, "same seed, same sample");
    assert_eq!(first.relations[0].rows_used, 50);
    assert!(
        (first.relations[0].value - 1.0).abs() < 1e-9,
        "perfect line"
    );
}

#[test]
fn text_columns_report_unique_counts_and_numeric_text_tokens() {
    let s = |v: &str| Some(v.to_string());
    let t = table(vec![(
        "money",
        Column::Text(vec![
            s("1,200"),
            s("300"),
            s("tbd"),
            s("?"),
            s("tbd"),
            s("45"),
            None,
        ]),
    )]);
    let c = &measure(&t).columns[0];
    assert_eq!(c.kind, ColumnKind::NumericText);
    assert_eq!(c.n_non_numeric, Some(3));
    assert_eq!(
        c.non_numeric_values.as_deref(),
        Some([("tbd".to_string(), 2), ("?".to_string(), 1)].as_slice())
    );
    assert_eq!(c.n_missing, 1);
}
