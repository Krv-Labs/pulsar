//! YAML/JSON → [`PulsarConfig`], in three passes:
//!
//! 1. **Normalize**: YAML → JSON value. Null-valued keys are dropped (they
//!    mean "use the default"), scalar map keys are stringified, and
//!    non-finite numbers and YAML tags are rejected.
//! 2. **Shape**: walk the value against the generated JSON Schema to report
//!    every unknown, legacy, or missing key at once, then deserialize into
//!    [`ConfigFile`] (type errors carry their JSON pointer).
//! 3. **Resolve**: apply defaults, expand sweep parameters, and check value
//!    ranges and cross-field rules, collecting every error.

use std::path::Path;

use indexmap::IndexMap;
use serde_json::{Map, Value};

use crate::error::{ConfigError, ConfigErrors, ErrorCode, Pointer};
use crate::expand::{expand_integers, expand_positive_floats};
use crate::file::{
    ColumnName, ConfigFile, CosmicGraphFile, ImputeFile, ParamSpec, PreprocessingFile, SweepFile,
    ThresholdFile,
};
use crate::model::{
    BallMapper, Construction, CosmicGraph, Encode, FillValue, Impute, ImputeMethod, Preprocessing,
    Projection, ProjectionMethod, PulsarConfig, Threshold, CONFIG_VERSION,
};
use crate::schema::json_schema;

/// Suffix of generated missingness-marker columns.
pub const MISSING_SUFFIX: &str = "_was_missing";

/// Parse and validate a YAML (or JSON) config document.
pub fn from_yaml_str(text: &str) -> Result<PulsarConfig, ConfigErrors> {
    let yaml: serde_yaml_ng::Value = serde_yaml_ng::from_str(text).map_err(|e| {
        ConfigErrors::single(ConfigError::new(
            "",
            ErrorCode::YamlSyntax,
            format!("Config is not valid YAML: {e}"),
        ))
    })?;
    let mut errors = Vec::new();
    let value = normalize(yaml, &Pointer::root(), &mut errors);
    if !errors.is_empty() {
        return Err(ConfigErrors(errors));
    }
    from_value(value)
}

/// Read, parse and validate a config file.
pub fn from_path(path: impl AsRef<Path>) -> Result<PulsarConfig, ConfigErrors> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|e| {
        ConfigErrors::single(ConfigError::new(
            "",
            ErrorCode::InvalidValue,
            format!("Cannot read config {}: {e}", path.display()),
        ))
    })?;
    from_yaml_str(&text)
}

/// Validate an already-parsed JSON value (e.g. a config sent over MCP).
/// Null-valued keys mean "use the default", as in YAML.
pub fn from_value(value: Value) -> Result<PulsarConfig, ConfigErrors> {
    let mut value = match strip_nulls(value) {
        Value::Null => Value::Object(Map::new()),
        v @ Value::Object(_) => v,
        other => {
            return Err(ConfigErrors::single(ConfigError::new(
                "",
                ErrorCode::InvalidType,
                format!("Config must be a mapping, got {}.", type_name(&other)),
            )))
        }
    };

    let schema = json_schema();
    // Key problems are reported and the offending keys dropped, so the
    // remaining passes still surface every value error in the same report.
    let mut errors = Vec::new();
    check_keys(&mut value, &schema, &schema, &Pointer::root(), &mut errors);

    let file: ConfigFile = match serde_path_to_error::deserialize(value) {
        Ok(file) => file,
        Err(e) => {
            let err = type_error(e);
            // serde re-reports a missing key the walk already reported.
            let duplicate = errors.iter().any(|prev| {
                prev.code == ErrorCode::MissingField && prev.path.starts_with(&err.path)
            });
            if !duplicate {
                errors.push(err);
            }
            return Err(ConfigErrors(errors));
        }
    };
    match resolve(file) {
        Ok(cfg) if errors.is_empty() => Ok(cfg),
        Ok(_) => Err(ConfigErrors(errors)),
        Err(more) => {
            errors.extend(more.0);
            Err(ConfigErrors(errors))
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 1: normalize
// ---------------------------------------------------------------------------

fn normalize(yaml: serde_yaml_ng::Value, at: &Pointer, errors: &mut Vec<ConfigError>) -> Value {
    use serde_yaml_ng::Value as Y;
    match yaml {
        Y::Null => Value::Null,
        Y::Bool(b) => Value::Bool(b),
        Y::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::from(i)
            } else if let Some(u) = n.as_u64() {
                Value::from(u)
            } else {
                let f = n.as_f64().unwrap_or(f64::NAN);
                serde_json::Number::from_f64(f)
                    .map(Value::Number)
                    .unwrap_or_else(|| {
                        errors.push(ConfigError::new(
                            at.clone(),
                            ErrorCode::InvalidValue,
                            format!("Numbers must be finite, got {f}."),
                        ));
                        Value::Null
                    })
            }
        }
        Y::String(s) => Value::String(s),
        Y::Sequence(seq) => Value::Array(
            seq.into_iter()
                .enumerate()
                .map(|(i, v)| normalize(v, &at.index(i), errors))
                .collect(),
        ),
        Y::Mapping(map) => {
            let mut out = Map::new();
            for (k, v) in map {
                let key = match k {
                    Y::String(s) => s,
                    Y::Number(n) => n.to_string(),
                    Y::Bool(b) => b.to_string(),
                    other => {
                        errors.push(ConfigError::new(
                            at.clone(),
                            ErrorCode::InvalidType,
                            format!("Mapping keys must be scalars, got {other:?}."),
                        ));
                        continue;
                    }
                };
                let child = at.key(&key);
                out.insert(key, normalize(v, &child, errors));
            }
            Value::Object(out)
        }
        Y::Tagged(t) => {
            errors.push(ConfigError::new(
                at.clone(),
                ErrorCode::InvalidType,
                format!("YAML tags are not supported (found !{}).", t.tag),
            ));
            Value::Null
        }
    }
}

fn strip_nulls(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .filter(|(_, v)| !v.is_null())
                .map(|(k, v)| (k, strip_nulls(v)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.into_iter().map(strip_nulls).collect()),
        other => other,
    }
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "a list",
        Value::Object(_) => "a mapping",
    }
}

// ---------------------------------------------------------------------------
// Pass 2: shape
// ---------------------------------------------------------------------------

/// Keys older Pulsar versions accepted, with the replacement to suggest.
fn legacy_key_hint(parent: &str, key: &str) -> Option<&'static str> {
    match (parent, key) {
        ("/cosmic_graph", "threshold") => Some(
            "Unsupported legacy key cosmic_graph.threshold. \
             Use cosmic_graph.construction_threshold instead.",
        ),
        _ => None,
    }
}

fn resolve_ref<'a>(schema: &'a Value, root: &'a Value) -> &'a Value {
    let mut current = schema;
    while let Some(target) = current.get("$ref").and_then(Value::as_str) {
        match target.strip_prefix('#').and_then(|p| root.pointer(p)) {
            Some(next) => current = next,
            None => break,
        }
    }
    current
}

/// Report unknown, legacy and missing keys by walking `value` against the
/// generated schema, removing unknown keys as they are reported. Union
/// types (sweep specs, thresholds) are left to serde.
fn check_keys(
    value: &mut Value,
    schema: &Value,
    root: &Value,
    at: &Pointer,
    errors: &mut Vec<ConfigError>,
) {
    let schema = resolve_ref(schema, root);
    if let Some(branches) = schema
        .get("anyOf")
        .or_else(|| schema.get("oneOf"))
        .and_then(Value::as_array)
    {
        let non_null: Vec<&Value> = branches
            .iter()
            .filter(|b| b.get("type").and_then(Value::as_str) != Some("null"))
            .collect();
        if let [only] = non_null.as_slice() {
            check_keys(value, only, root, at, errors);
        }
        return;
    }

    match value {
        Value::Object(obj) => {
            if let Some(props) = schema.get("properties").and_then(Value::as_object) {
                for req in schema
                    .get("required")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                {
                    if !obj.contains_key(req) {
                        errors.push(ConfigError::new(
                            at.key(req),
                            ErrorCode::MissingField,
                            format!("Missing required key {req:?}."),
                        ));
                    }
                }
                let closed = schema.get("additionalProperties") == Some(&Value::Bool(false));
                let mut unknown = Vec::new();
                for (key, child) in obj.iter_mut() {
                    match props.get(key) {
                        Some(child_schema) => {
                            check_keys(child, child_schema, root, &at.key(key), errors)
                        }
                        None if closed => {
                            errors.push(unknown_key(at, key, props));
                            unknown.push(key.clone());
                        }
                        None => {}
                    }
                }
                for key in unknown {
                    obj.shift_remove(&key);
                }
            } else if let Some(ap) = schema.get("additionalProperties").filter(|s| s.is_object()) {
                for (key, child) in obj.iter_mut() {
                    check_keys(child, ap, root, &at.key(key), errors);
                }
            }
        }
        Value::Array(items) => {
            if let Some(item_schema) = schema.get("items").filter(|s| s.is_object()) {
                for (i, item) in items.iter_mut().enumerate() {
                    check_keys(item, item_schema, root, &at.index(i), errors);
                }
            }
        }
        _ => {}
    }
}

fn unknown_key(at: &Pointer, key: &str, props: &Map<String, Value>) -> ConfigError {
    let path = at.key(key);
    match legacy_key_hint(at.as_str(), key) {
        Some(hint) => ConfigError::new(path, ErrorCode::LegacyKey, hint),
        None => {
            let parent = if at.as_str().is_empty() {
                "the top level".to_string()
            } else {
                at.as_str().to_string()
            };
            ConfigError::new(
                path,
                ErrorCode::UnknownKey,
                format!("Unknown key {key:?} in {parent}."),
            )
            .with_allowed(props.keys().cloned())
        }
    }
}

fn type_error(err: serde_path_to_error::Error<serde_json::Error>) -> ConfigError {
    use serde_path_to_error::Segment;
    let mut at = Pointer::root();
    for seg in err.path().iter() {
        at = match seg {
            Segment::Seq { index } => at.index(*index),
            Segment::Map { key } => at.key(key),
            Segment::Enum { variant } => at.key(variant),
            Segment::Unknown => at,
        };
    }
    let inner = err.into_inner().to_string();
    let last = at.as_str().rsplit('/').next().unwrap_or("");
    let parent_is_columns = at.as_str().contains("/drop_columns/")
        || at.as_str().contains("/missing_indicator_groups/");
    let message = if inner.contains("did not match any variant") {
        match last {
            "dimensions" | "seed" | "epsilon" => "Expected a number, a list of numbers, \
                 {values: [...]}, {range: {min, max, steps}}, or \
                 {distribution: {type: uniform, min, max}}."
                .to_string(),
            "construction_threshold" => {
                "Expected \"auto\" or a number between 0.0 and 1.0.".to_string()
            }
            "value" => "Expected a finite number or a string.".to_string(),
            _ if parent_is_columns => "Expected a column name (string or integer).".to_string(),
            _ => inner,
        }
    } else {
        inner
    };
    ConfigError::new(at, ErrorCode::InvalidType, message)
}

// ---------------------------------------------------------------------------
// Pass 3: resolve
// ---------------------------------------------------------------------------

fn resolve(file: ConfigFile) -> Result<PulsarConfig, ConfigErrors> {
    let root = Pointer::root();
    let mut errors = Vec::new();

    let version = file.version.unwrap_or(CONFIG_VERSION);
    if version != CONFIG_VERSION {
        errors.push(
            ConfigError::new(
                root.key("version"),
                ErrorCode::UnsupportedVersion,
                format!("Config version {version} is not supported by this build."),
            )
            .with_allowed([CONFIG_VERSION.to_string()]),
        );
    }

    let preprocessing = resolve_preprocessing(file.preprocessing, &root, &mut errors);
    let (projection, ball_mapper) = resolve_sweep(&file.sweep, &root, &mut errors);
    let cosmic_graph = resolve_cosmic_graph(&file.cosmic_graph, &root, &mut errors);

    let n_reps = file.output.n_reps.unwrap_or(4);
    let n_reps = match u32::try_from(n_reps) {
        Ok(n) if n >= 1 => n,
        _ => {
            errors.push(ConfigError::new(
                root.key("output").key("n_reps"),
                ErrorCode::OutOfRange,
                format!(
                    "output.n_reps must be between 1 and {}, got {n_reps}.",
                    u32::MAX
                ),
            ));
            0
        }
    };

    if !errors.is_empty() {
        return Err(ConfigErrors(errors));
    }
    Ok(PulsarConfig {
        version,
        run_name: file.run.name.unwrap_or_default(),
        data: file.run.data.or(file.data).unwrap_or_default(),
        preprocessing,
        projection,
        ball_mapper,
        cosmic_graph,
        n_reps,
    })
}

fn resolve_preprocessing(
    pre: PreprocessingFile,
    root: &Pointer,
    errors: &mut Vec<ConfigError>,
) -> Preprocessing {
    let at = root.key("preprocessing");
    let drop_columns: Vec<String> = pre
        .drop_columns
        .into_iter()
        .map(ColumnName::into_string)
        .collect();

    let impute: IndexMap<String, Impute> = pre
        .impute
        .into_iter()
        .map(|(col, spec)| {
            let spec_at = at.key("impute").key(&col);
            let resolved = resolve_impute(spec, &spec_at, errors);
            (col, resolved)
        })
        .collect();

    let encode: IndexMap<String, Encode> = pre
        .encode
        .into_iter()
        .map(|(col, spec)| {
            if spec.max_categories == Some(0) {
                errors.push(ConfigError::new(
                    at.key("encode").key(&col).key("max_categories"),
                    ErrorCode::OutOfRange,
                    "max_categories must be >= 1.",
                ));
            }
            (
                col,
                Encode {
                    method: spec.method,
                    max_categories: spec.max_categories,
                },
            )
        })
        .collect();

    let groups = resolve_indicator_groups(pre.missing_indicator_groups, &at, errors);
    let resolved = Preprocessing {
        drop_columns,
        impute,
        encode,
        missing_indicator_groups: groups,
    };
    check_group_flag_collisions(&resolved, &at, errors);
    resolved
}

fn resolve_impute(spec: ImputeFile, at: &Pointer, errors: &mut Vec<ConfigError>) -> Impute {
    let value = match (spec.method, spec.value) {
        (ImputeMethod::FillValue, None) => {
            errors.push(ConfigError::new(
                at.key("value"),
                ErrorCode::MissingField,
                "Method 'fill_value' requires 'value' (a finite number or a string).",
            ));
            None
        }
        (ImputeMethod::FillValue, Some(v)) => Some(v),
        (method, Some(_)) => {
            errors.push(ConfigError::new(
                at.key("value"),
                ErrorCode::InvalidValue,
                format!(
                    "'value' is only valid with method 'fill_value', got method {:?}.",
                    serde_json::to_value(method)
                        .ok()
                        .and_then(|v| v.as_str().map(str::to_owned))
                        .unwrap_or_default()
                ),
            ));
            None
        }
        (_, None) => None,
    };
    if let Some(FillValue::Float(f)) = value {
        // Normalization already rejects non-finite YAML numbers; this guards
        // values built in code.
        if !f.is_finite() {
            errors.push(ConfigError::new(
                at.key("value"),
                ErrorCode::InvalidValue,
                "'value' must be finite.",
            ));
        }
    }
    Impute {
        method: spec.method,
        seed: spec.seed.unwrap_or(42),
        value,
        indicator: spec.indicator.unwrap_or(true),
    }
}

fn resolve_indicator_groups(
    groups: IndexMap<String, Vec<ColumnName>>,
    at: &Pointer,
    errors: &mut Vec<ConfigError>,
) -> IndexMap<String, Vec<String>> {
    let at = at.key("missing_indicator_groups");
    let mut owner: IndexMap<String, String> = IndexMap::new();
    let mut out = IndexMap::new();
    for (group, cols) in groups {
        let group_at = at.key(&group);
        if group.trim().is_empty() {
            errors.push(ConfigError::new(
                group_at.clone(),
                ErrorCode::InvalidValue,
                "Group names must be non-empty.",
            ));
        }
        let mut names = Vec::with_capacity(cols.len());
        for (i, col) in cols.into_iter().enumerate() {
            let col = col.into_string();
            if col.trim().is_empty() {
                errors.push(ConfigError::new(
                    group_at.index(i),
                    ErrorCode::InvalidValue,
                    "Column names must be non-empty.",
                ));
            } else if let Some(prev) = owner.get(&col) {
                errors.push(ConfigError::new(
                    group_at.index(i),
                    ErrorCode::DuplicateColumn,
                    format!(
                        "Column {col:?} appears in more than one missing_indicator_group \
                         ({prev:?} and {group:?})."
                    ),
                ));
            } else {
                owner.insert(col.clone(), group.clone());
            }
            names.push(col);
        }
        out.insert(group, names);
    }
    out
}

/// A group's `<group>_was_missing` marker must not collide with a per-column
/// marker, a configured column, or another group's marker.
fn check_group_flag_collisions(pre: &Preprocessing, at: &Pointer, errors: &mut Vec<ConfigError>) {
    let grouped: std::collections::HashSet<&str> = pre
        .missing_indicator_groups
        .values()
        .flatten()
        .map(String::as_str)
        .collect();
    let dropped: std::collections::HashSet<&str> =
        pre.drop_columns.iter().map(String::as_str).collect();
    let per_column_flags: std::collections::HashSet<String> = pre
        .impute
        .iter()
        .filter(|(col, spec)| {
            spec.indicator && !grouped.contains(col.as_str()) && !dropped.contains(col.as_str())
        })
        .map(|(col, _)| format!("{col}{MISSING_SUFFIX}"))
        .collect();
    let configured: std::collections::HashSet<&str> = pre
        .impute
        .keys()
        .chain(pre.encode.keys())
        .map(String::as_str)
        .chain(dropped.iter().copied())
        .chain(grouped.iter().copied())
        .collect();

    let mut seen = std::collections::HashSet::new();
    for group in pre.missing_indicator_groups.keys() {
        let flag = format!("{group}{MISSING_SUFFIX}");
        let reason = if per_column_flags.contains(&flag) {
            Some("collides with a per-column indicator")
        } else if configured.contains(flag.as_str()) {
            Some("collides with a configured column")
        } else if !seen.insert(flag.clone()) {
            Some("duplicates another group's indicator")
        } else {
            None
        };
        if let Some(reason) = reason {
            errors.push(ConfigError::new(
                at.key("missing_indicator_groups").key(group),
                ErrorCode::NameCollision,
                format!("Group {group:?} indicator {flag:?} {reason}."),
            ));
        }
    }
}

fn resolve_sweep(
    sweep: &SweepFile,
    root: &Pointer,
    errors: &mut Vec<ConfigError>,
) -> (Projection, BallMapper) {
    let at = root.key("sweep");

    // `projection` wins; the legacy `pca` block is read as a JL grid only
    // when `projection` is absent (Python v0.5 behaviour).
    let (proj_at, method, dims, seeds, center) = match (&sweep.projection, &sweep.pca) {
        (Some(p), _) => (
            at.key("projection"),
            p.method.unwrap_or(ProjectionMethod::Jl),
            p.dimensions.as_ref(),
            p.seed.as_ref(),
            p.center.unwrap_or(true),
        ),
        (None, Some(legacy)) => (
            at.key("pca"),
            ProjectionMethod::Jl,
            legacy.dimensions.as_ref(),
            legacy.seed.as_ref(),
            true,
        ),
        (None, None) => (at.key("projection"), ProjectionMethod::Jl, None, None, true),
    };

    let default_dims = ParamSpec::List(vec![2.into()]);
    let default_seeds = ParamSpec::List(vec![42.into()]);
    let dimensions = expand_integers(
        dims.unwrap_or(&default_dims),
        &proj_at.key("dimensions"),
        1,
        "dimensions",
    )
    .map(|v| v.into_iter().map(|d| d as usize).collect())
    .unwrap_or_else(|e| {
        errors.push(e);
        Vec::new()
    });
    let seeds = expand_integers(
        seeds.unwrap_or(&default_seeds),
        &proj_at.key("seed"),
        0,
        "seeds",
    )
    .unwrap_or_else(|e| {
        errors.push(e);
        Vec::new()
    });

    let default_eps = ParamSpec::List(vec![serde_json::Number::from_f64(0.5).expect("finite")]);
    let epsilons = expand_positive_floats(
        sweep.ball_mapper.epsilon.as_ref().unwrap_or(&default_eps),
        &at.key("ball_mapper").key("epsilon"),
        "epsilon",
    )
    .unwrap_or_else(|e| {
        errors.push(e);
        Vec::new()
    });

    (
        Projection {
            method,
            dimensions,
            seeds,
            center,
        },
        BallMapper { epsilons },
    )
}

fn resolve_cosmic_graph(
    cg: &CosmicGraphFile,
    root: &Pointer,
    errors: &mut Vec<ConfigError>,
) -> CosmicGraph {
    let at = root.key("cosmic_graph");

    let construction_threshold = match cg.construction_threshold {
        None | Some(ThresholdFile::Auto(_)) => Threshold::Auto,
        Some(ThresholdFile::Value(t)) => {
            if !(t.is_finite() && (0.0..=1.0).contains(&t)) {
                errors.push(ConfigError::new(
                    at.key("construction_threshold"),
                    ErrorCode::OutOfRange,
                    format!("Threshold must be finite and between 0.0 and 1.0, got {t}."),
                ));
            }
            Threshold::Value(t)
        }
    };

    let mut positive_f64 = |key: &str, v: Option<f64>, default: f64| -> f64 {
        let v = v.unwrap_or(default);
        if !(v.is_finite() && v > 0.0) {
            errors.push(ConfigError::new(
                at.key(key),
                ErrorCode::OutOfRange,
                format!("cosmic_graph.{key} must be finite and > 0, got {v}."),
            ));
        }
        v
    };
    let sparsify_epsilon = positive_f64("sparsify_epsilon", cg.sparsify_epsilon, 1.0);
    let sparsify_pcg_tol = positive_f64("sparsify_pcg_tol", cg.sparsify_pcg_tol, 1e-6);

    let mut at_least_one = |key: &str, v: Option<u64>| {
        if v == Some(0) {
            errors.push(ConfigError::new(
                at.key(key),
                ErrorCode::OutOfRange,
                format!("cosmic_graph.{key} must be >= 1."),
            ));
        }
        v
    };
    let sparsify_sketch_dim = at_least_one("sparsify_sketch_dim", cg.sparsify_sketch_dim);
    let sparsify_sample_count = at_least_one("sparsify_sample_count", cg.sparsify_sample_count);
    let sparsify_max_iter = at_least_one("sparsify_max_iter", cg.sparsify_max_iter).unwrap_or(1000);
    let minhash_d = at_least_one("minhash_d", cg.minhash_d).unwrap_or(256);

    CosmicGraph {
        construction_threshold,
        neighborhood: cg
            .neighborhood
            .clone()
            .unwrap_or_else(|| "node".to_string()),
        sparsify: cg.sparsify.unwrap_or(false),
        sparsify_epsilon,
        sparsify_seed: cg.sparsify_seed.unwrap_or(42),
        sparsify_sketch_dim,
        sparsify_sample_count,
        sparsify_pcg_tol,
        sparsify_max_iter,
        construction: cg.construction.unwrap_or(Construction::Minhash),
        minhash_d,
        minhash_seed: cg.minhash_seed.unwrap_or(42),
    }
}
