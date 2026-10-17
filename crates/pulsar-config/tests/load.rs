//! Loading: defaults, #57 preprocessing fields, and exact error codes/paths.

use pulsar_config::{
    from_value, from_yaml_str, to_yaml, ConfigErrors, ErrorCode, FillValue, ImputeMethod,
    ProjectionMethod, Threshold,
};
use serde_json::json;

/// `(path, code)` pairs, in report order.
fn codes(err: &ConfigErrors) -> Vec<(&str, ErrorCode)> {
    err.iter().map(|e| (e.path.as_str(), e.code)).collect()
}

fn load_err(yaml: &str) -> ConfigErrors {
    from_yaml_str(yaml).expect_err("config should be rejected")
}

#[test]
fn empty_document_resolves_to_defaults() {
    let cfg = from_yaml_str("").unwrap();
    assert_eq!(cfg.version, 1);
    assert_eq!(cfg.data, "");
    assert_eq!(cfg.projection.method, ProjectionMethod::Jl);
    assert_eq!(cfg.projection.dimensions, vec![2]);
    assert_eq!(cfg.projection.seeds, vec![42]);
    assert!(cfg.projection.center);
    assert_eq!(cfg.ball_mapper.epsilons, vec![0.5]);
    assert_eq!(cfg.cosmic_graph.construction_threshold, Threshold::Auto);
    assert_eq!(cfg.cosmic_graph.minhash_d, 256);
    assert_eq!(cfg.n_reps, 4);
}

#[test]
fn null_sections_and_keys_mean_default() {
    let cfg = from_yaml_str(
        "preprocessing:\nsweep:\n  projection:\n    seed: null\ncosmic_graph: {minhash_d: ~}\n",
    )
    .unwrap();
    assert_eq!(cfg.projection.seeds, vec![42]);
    assert_eq!(cfg.cosmic_graph.minhash_d, 256);
}

#[test]
fn legacy_top_level_data_and_pca_block() {
    let cfg = from_yaml_str(
        "data: legacy.csv\nsweep:\n  pca:\n    dimensions: {values: [3, 5]}\n    seed: 7\n",
    )
    .unwrap();
    assert_eq!(cfg.data, "legacy.csv");
    assert_eq!(cfg.projection.method, ProjectionMethod::Jl);
    assert_eq!(cfg.projection.dimensions, vec![3, 5]);
    assert_eq!(cfg.projection.seeds, vec![7]);

    // run.data wins over top-level data; projection wins over pca.
    let cfg = from_yaml_str(
        "data: a.csv\nrun: {data: b.csv}\nsweep:\n  projection: {method: pca, dimensions: [4]}\n  pca: {dimensions: [9]}\n",
    )
    .unwrap();
    assert_eq!(cfg.data, "b.csv");
    assert_eq!(cfg.projection.method, ProjectionMethod::Pca);
    assert_eq!(cfg.projection.dimensions, vec![4]);
}

#[test]
fn sweep_spec_forms() {
    let cfg = from_yaml_str(
        "sweep:\n  ball_mapper:\n    epsilon: {range: {min: 0.0, max: 1.0, steps: 5}}\n",
    );
    // 0.0 is not a valid radius.
    assert_eq!(
        codes(&cfg.unwrap_err()),
        vec![("/sweep/ball_mapper/epsilon", ErrorCode::OutOfRange)]
    );
    let cfg = from_yaml_str(
        "sweep:\n  ball_mapper:\n    epsilon: {distribution: {type: uniform, min: 0.2, max: 0.9}}\n",
    )
    .unwrap();
    assert_eq!(cfg.ball_mapper.epsilons, vec![0.2, 0.9]);
    let cfg = from_yaml_str("sweep: {ball_mapper: {epsilon: 0.7}}").unwrap();
    assert_eq!(cfg.ball_mapper.epsilons, vec![0.7]);
    let cfg =
        from_yaml_str("sweep: {ball_mapper: {epsilon: {range: {min: 0.1, max: 1.0}}}}").unwrap();
    assert_eq!(cfg.ball_mapper.epsilons.len(), 10, "steps defaults to 10");
}

#[test]
fn preprocessing_fields_from_pr57() {
    let cfg = from_yaml_str(
        r#"
preprocessing:
  drop_columns: [id, 2010]
  missing_indicator_groups:
    inpatient: [n_ip_claims, ip_paid]
  impute:
    n_ip_claims: {method: fill_value, value: 0}
    ip_paid: {method: fill_value, value: 0.0, indicator: false}
    esrd: {method: fill_value, value: "N"}
    age: {method: sample_normal}
  encode:
    esrd: {method: one_hot, max_categories: 3}
"#,
    )
    .unwrap();
    let pre = &cfg.preprocessing;
    assert_eq!(pre.drop_columns, vec!["id", "2010"]);
    assert_eq!(
        pre.missing_indicator_groups["inpatient"],
        vec!["n_ip_claims", "ip_paid"]
    );
    assert_eq!(pre.impute["n_ip_claims"].value, Some(FillValue::Integer(0)));
    assert_eq!(pre.impute["ip_paid"].value, Some(FillValue::Float(0.0)));
    assert!(!pre.impute["ip_paid"].indicator);
    assert_eq!(pre.impute["esrd"].value, Some(FillValue::Text("N".into())));
    assert_eq!(pre.impute["age"].method, ImputeMethod::SampleNormal);
    assert_eq!(pre.impute["age"].seed, 42);
    assert!(pre.impute["age"].indicator);
    // Column order is preserved.
    let order: Vec<&str> = pre.impute.keys().map(String::as_str).collect();
    assert_eq!(order, ["n_ip_claims", "ip_paid", "esrd", "age"]);

    // Emitting and reloading is lossless, including the fill values' types.
    assert_eq!(from_yaml_str(&to_yaml(&cfg)).unwrap(), cfg);
}

#[test]
fn fill_value_rules() {
    let err = load_err(
        "preprocessing:\n  impute:\n    a: {method: fill_value}\n    b: {method: fill_mean, value: 1}\n",
    );
    assert_eq!(
        codes(&err),
        vec![
            ("/preprocessing/impute/a/value", ErrorCode::MissingField),
            ("/preprocessing/impute/b/value", ErrorCode::InvalidValue),
        ]
    );
    let err = load_err("preprocessing: {impute: {a: {method: fill_value, value: true}}}");
    assert_eq!(
        codes(&err),
        vec![("/preprocessing/impute/a/value", ErrorCode::InvalidType)]
    );
    let err = load_err("preprocessing: {impute: {a: {method: fill_value, value: .nan}}}");
    assert_eq!(
        codes(&err),
        vec![("/preprocessing/impute/a/value", ErrorCode::InvalidValue)]
    );
}

#[test]
fn indicator_group_rules() {
    let err = load_err(
        r#"
preprocessing:
  missing_indicator_groups:
    g1: [a, b]
    g2: [b]
"#,
    );
    assert_eq!(
        codes(&err),
        vec![(
            "/preprocessing/missing_indicator_groups/g2/0",
            ErrorCode::DuplicateColumn
        )]
    );

    // g_was_missing would collide with column g's own marker.
    let err = load_err(
        "preprocessing:\n  missing_indicator_groups: {g: [x]}\n  impute: {g: {method: fill_mean}}\n",
    );
    assert_eq!(
        codes(&err),
        vec![(
            "/preprocessing/missing_indicator_groups/g",
            ErrorCode::NameCollision
        )]
    );
    // ...but not when g opts out of its marker; then it is only a configured
    // column, which still collides with nothing named g_was_missing.
    from_yaml_str(
        "preprocessing:\n  missing_indicator_groups: {g: [x]}\n  impute: {g: {method: fill_mean, indicator: false}}\n",
    )
    .unwrap();
    // A group flag equal to a configured column name collides.
    let err = load_err(
        "preprocessing:\n  missing_indicator_groups: {g: [x]}\n  drop_columns: [g_was_missing]\n",
    );
    assert_eq!(
        codes(&err),
        vec![(
            "/preprocessing/missing_indicator_groups/g",
            ErrorCode::NameCollision
        )]
    );
}

#[test]
fn unknown_and_legacy_keys_are_all_reported() {
    let err = load_err(
        r#"
runn: {name: x}
sweep:
  ball_mapper: {epsilon: [0.5], eps: 1}
cosmic_graph:
  threshold: 0.2
  sparsfy: true
preprocessing:
  impute:
    a: {method: fill_mean, sed: 1}
"#,
    );
    assert_eq!(
        codes(&err),
        vec![
            ("/runn", ErrorCode::UnknownKey),
            ("/sweep/ball_mapper/eps", ErrorCode::UnknownKey),
            ("/cosmic_graph/threshold", ErrorCode::LegacyKey),
            ("/cosmic_graph/sparsfy", ErrorCode::UnknownKey),
            ("/preprocessing/impute/a/sed", ErrorCode::UnknownKey),
        ]
    );
    let sparsfy = err
        .iter()
        .find(|e| e.path == "/cosmic_graph/sparsfy")
        .unwrap();
    assert!(sparsfy
        .allowed
        .as_ref()
        .unwrap()
        .contains(&"sparsify".to_string()));
    let legacy = err.iter().find(|e| e.code == ErrorCode::LegacyKey).unwrap();
    assert!(legacy.message.contains("construction_threshold"));
}

#[test]
fn key_errors_do_not_hide_value_errors() {
    let err = load_err(
        "cosmic_graph:\n  threshold: 0.2\n  minhash_d: 0\nsweep:\n  projection: {dimensions: [0]}\n",
    );
    assert_eq!(
        codes(&err),
        vec![
            ("/cosmic_graph/threshold", ErrorCode::LegacyKey),
            ("/sweep/projection/dimensions", ErrorCode::OutOfRange),
            ("/cosmic_graph/minhash_d", ErrorCode::OutOfRange),
        ]
    );
}

#[test]
fn missing_required_method() {
    let err = load_err("preprocessing: {impute: {a: {seed: 3}}, encode: {b: {}}}");
    assert_eq!(
        codes(&err),
        vec![
            ("/preprocessing/impute/a/method", ErrorCode::MissingField),
            ("/preprocessing/encode/b/method", ErrorCode::MissingField),
        ]
    );
}

#[test]
fn type_errors_carry_paths_and_allowed_hints() {
    let err = load_err("preprocessing: {impute: {a: {method: fill_avg}}}");
    assert_eq!(
        codes(&err),
        vec![("/preprocessing/impute/a/method", ErrorCode::InvalidType)]
    );
    assert!(
        err.0[0].message.contains("fill_mean"),
        "{}",
        err.0[0].message
    );

    let err = load_err("sweep: {projection: {seed: \"42\"}}");
    assert_eq!(
        codes(&err),
        vec![("/sweep/projection/seed", ErrorCode::InvalidType)]
    );
    assert!(err.0[0].message.contains("{values: [...]}"));

    let err = load_err("sweep: {ball_mapper: {epsilon: {values: [0.5], extra: 1}}}");
    assert_eq!(
        codes(&err),
        vec![("/sweep/ball_mapper/epsilon", ErrorCode::InvalidType)]
    );

    let err = load_err("cosmic_graph: {construction_threshold: high}");
    assert_eq!(
        codes(&err),
        vec![(
            "/cosmic_graph/construction_threshold",
            ErrorCode::InvalidType
        )]
    );

    let err = load_err("cosmic_graph: {construction: approximate}");
    assert_eq!(
        codes(&err),
        vec![("/cosmic_graph/construction", ErrorCode::InvalidType)]
    );
}

#[test]
fn range_and_grid_errors_are_collected() {
    let err = load_err(
        r#"
version: 2
sweep:
  projection: {dimensions: [0, 2], seed: {values: []}}
  ball_mapper: {epsilon: {range: {min: 0.1, max: 1.0, steps: 0}}}
cosmic_graph:
  construction_threshold: 1.5
  minhash_d: 0
  sparsify_pcg_tol: 0
output: {n_reps: 0}
"#,
    );
    assert_eq!(
        codes(&err),
        vec![
            ("/version", ErrorCode::UnsupportedVersion),
            ("/sweep/projection/dimensions", ErrorCode::OutOfRange),
            ("/sweep/projection/seed", ErrorCode::EmptyGrid),
            ("/sweep/ball_mapper/epsilon", ErrorCode::EmptyGrid),
            (
                "/cosmic_graph/construction_threshold",
                ErrorCode::OutOfRange
            ),
            ("/cosmic_graph/sparsify_pcg_tol", ErrorCode::OutOfRange),
            ("/cosmic_graph/minhash_d", ErrorCode::OutOfRange),
            ("/output/n_reps", ErrorCode::OutOfRange),
        ]
    );
}

#[test]
fn integer_params_are_not_coerced() {
    let err = load_err("sweep: {projection: {dimensions: [2.5]}}");
    assert_eq!(
        codes(&err),
        vec![("/sweep/projection/dimensions", ErrorCode::InvalidValue)]
    );
    let err = load_err("sweep: {projection: {seed: [-1]}}");
    assert_eq!(
        codes(&err),
        vec![("/sweep/projection/seed", ErrorCode::OutOfRange)]
    );
}

#[test]
fn yaml_syntax_and_root_type() {
    let err = load_err("sweep: [unclosed");
    assert_eq!(codes(&err), vec![("", ErrorCode::YamlSyntax)]);
    let err = load_err("- just\n- a list\n");
    assert_eq!(codes(&err), vec![("", ErrorCode::InvalidType)]);
}

#[test]
fn from_value_accepts_mcp_style_json() {
    let cfg = from_value(json!({
        "run": {"name": "mcp", "data": null},
        "sweep": {"ball_mapper": {"epsilon": {"values": [0.25, 0.5]}}},
        "cosmic_graph": {"construction_threshold": 0.1, "sparsify_sketch_dim": null}
    }))
    .unwrap();
    assert_eq!(cfg.run_name, "mcp");
    assert_eq!(cfg.data, "");
    assert_eq!(
        cfg.cosmic_graph.construction_threshold,
        Threshold::Value(0.1)
    );
    assert_eq!(cfg.cosmic_graph.sparsify_sketch_dim, None);
}

#[test]
fn errors_serialize_as_structured_json() {
    let err = load_err("cosmic_graph: {construction: approximate, sparsfy: 1}");
    let v = serde_json::to_value(&err).unwrap();
    assert_eq!(v[0]["code"], "UNKNOWN_KEY");
    assert_eq!(v[0]["path"], "/cosmic_graph/sparsfy");
    assert!(v[0]["allowed"].is_array());
}
