//! Constrained overrides of a sweep config (`apply_overrides` in
//! `pulsar/mcp/config_tools.py`): what `refine_config` applies between sweeps.
//!
//! Overrides are a nested map. It is flattened to dotted keys, so
//! `{"sweep": {"ball_mapper": {"epsilon": {"range": {"max": 1.0}}}}}` sets the
//! one leaf `sweep.ball_mapper.epsilon.range.max`. Short keys
//! (`projection_dimensions`, `epsilon_values`, ...) set a whole parameter and
//! keep the legacy `sweep.pca` mirror in step. `remove_keys` lists dotted
//! paths to delete.
//!
//! Known behaviors kept from Python so the two agree: a short key whose value
//! is an object (`{"epsilon_range": {...}}`) flattens to an unknown dotted root
//! and is rejected; use the nested form above. The tool layer, not this
//! module, validates and normalizes the resulting config.

use serde_json::{json, Map, Value};

use crate::plan::{py_repr_list, py_str};

const VALID_KEYS: [&str; 17] = [
    "dataset_path",
    "drop_columns",
    "encode",
    "epsilon_range",
    "epsilon_values",
    "impute",
    "missing_indicator_groups",
    "n_imputations",
    "n_reps",
    "pca_dims",
    "pca_seeds",
    "projection_dimensions",
    "projection_method",
    "projection_seeds",
    "raw",
    "run_name",
    "construction_threshold",
];
const DOTTED_ROOTS: [&str; 5] = ["cosmic_graph", "output", "preprocessing", "run", "sweep"];

/// Dotted spellings of the short keys.
const PREFIX_MAP: [(&str, &str); 16] = [
    ("sweep.projection_dimensions", "projection_dimensions"),
    ("sweep.projection_seeds", "projection_seeds"),
    ("sweep.pca_dims", "pca_dims"),
    ("sweep.pca_seeds", "pca_seeds"),
    ("sweep.projection_method", "projection_method"),
    ("sweep.epsilon_values", "epsilon_values"),
    ("sweep.epsilon_range", "epsilon_range"),
    (
        "cosmic_graph.construction_threshold",
        "construction_threshold",
    ),
    ("output.n_reps", "n_reps"),
    ("preprocessing.n_imputations", "n_imputations"),
    ("preprocessing.drop_columns", "drop_columns"),
    ("preprocessing.impute", "impute"),
    ("preprocessing.encode", "encode"),
    (
        "preprocessing.missing_indicator_groups",
        "missing_indicator_groups",
    ),
    ("run.run_name", "run_name"),
    ("run.dataset_path", "dataset_path"),
];

/// Canonical path each short key writes (in Python's dict order).
const CANONICAL: [(&str, &str); 16] = [
    ("projection_method", "sweep.projection.method"),
    (
        "projection_dimensions",
        "sweep.projection.dimensions.values",
    ),
    ("projection_seeds", "sweep.projection.seed.values"),
    ("pca_dims", "sweep.projection.dimensions.values"),
    ("pca_seeds", "sweep.projection.seed.values"),
    ("epsilon_values", "sweep.ball_mapper.epsilon.values"),
    ("epsilon_range", "sweep.ball_mapper.epsilon.range"),
    (
        "construction_threshold",
        "cosmic_graph.construction_threshold",
    ),
    ("n_reps", "output.n_reps"),
    ("n_imputations", "preprocessing.n_imputations"),
    ("drop_columns", "preprocessing.drop_columns"),
    ("impute", "preprocessing.impute"),
    ("encode", "preprocessing.encode"),
    (
        "missing_indicator_groups",
        "preprocessing.missing_indicator_groups",
    ),
    ("run_name", "run.name"),
    ("dataset_path", "run.data"),
];

#[derive(Debug, Clone, PartialEq)]
pub enum OverrideError {
    /// An override key or `remove_keys` root the engine does not know.
    UnknownKey(String),
    /// A recognised key with a value of the wrong shape.
    Invalid(String),
}

impl std::fmt::Display for OverrideError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownKey(m) | Self::Invalid(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for OverrideError {}

/// Validated overrides, ready to apply to a parsed config.
#[derive(Debug)]
pub struct Checked {
    remove_keys: Vec<String>,
    dotted: Vec<(String, Value)>,
    flat: Vec<(String, Value)>,
    /// Short key -> the spelling the caller used.
    original: Vec<(String, String)>,
}

/// What an application changed, in the order it happened.
#[derive(Debug, Default, PartialEq)]
pub struct Applied {
    pub applied: Vec<String>,
    pub diff: Vec<Value>,
}

/// Parse config YAML into its top-level mapping.
pub fn parse_yaml_mapping(config_yaml: &str) -> Result<Map<String, Value>, String> {
    match serde_yaml_ng::from_str::<Value>(config_yaml) {
        Ok(Value::Object(map)) => Ok(map),
        Ok(_) => Err("config_yaml must be a valid YAML mapping".into()),
        Err(e) => Err(format!("config_yaml is not valid YAML: {e}")),
    }
}

fn flatten(nested: &Map<String, Value>, prefix: &str, out: &mut Vec<(String, Value)>) {
    for (k, v) in nested {
        let key = format!("{prefix}{k}");
        match v {
            Value::Object(inner) => flatten(inner, &format!("{key}."), out),
            _ => out.push((key, v.clone())),
        }
    }
}

fn sorted_repr(mut items: Vec<String>) -> String {
    items.sort();
    items.dedup();
    py_repr_list(&items)
}

/// Check the override keys before any config is parsed.
pub fn check_overrides(overrides: &Map<String, Value>) -> Result<Checked, OverrideError> {
    let remove_keys: Vec<String> = match overrides.get("remove_keys") {
        None => vec![],
        Some(Value::Array(items)) => items.iter().map(py_str_of).collect(),
        Some(_) => {
            return Err(OverrideError::UnknownKey(
                "remove_keys must be a list of dotted config paths, e.g. ['preprocessing.encode.species'].".into(),
            ))
        }
    };
    let root = |p: &str| p.split('.').next().unwrap_or("").to_owned();
    let unknown_roots: Vec<String> = remove_keys
        .iter()
        .map(|p| root(p))
        .filter(|r| !DOTTED_ROOTS.contains(&r.as_str()))
        .collect();
    if !unknown_roots.is_empty() {
        return Err(OverrideError::UnknownKey(format!(
            "Unknown remove_keys root(s): {}. Valid roots: {}.",
            sorted_repr(unknown_roots),
            py_repr_list(&DOTTED_ROOTS)
        )));
    }

    let without_remove: Map<String, Value> = overrides
        .iter()
        .filter(|(k, _)| k.as_str() != "remove_keys")
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let mut flat_pairs = Vec::new();
    flatten(&without_remove, "", &mut flat_pairs);

    let (mut dotted, mut flat, mut original) = (vec![], vec![], vec![]);
    for (key, value) in flat_pairs {
        let normalized = PREFIX_MAP
            .iter()
            .find(|(dotted_key, _)| *dotted_key == key)
            .map_or(key.clone(), |(_, short)| (*short).to_owned());
        original.push((normalized.clone(), key));
        if normalized.contains('.') {
            dotted.push((normalized, value));
        } else {
            flat.push((normalized, value));
        }
    }

    let unknown_dotted: Vec<String> = dotted
        .iter()
        .map(|(k, _)| root(k))
        .filter(|r| !DOTTED_ROOTS.contains(&r.as_str()))
        .collect();
    if !unknown_dotted.is_empty() {
        return Err(OverrideError::UnknownKey(format!(
            "Unknown dotted override root(s): {}. Valid roots: {}",
            sorted_repr(unknown_dotted),
            py_repr_list(&DOTTED_ROOTS)
        )));
    }
    let unknown_keys: Vec<String> = flat
        .iter()
        .map(|(k, _)| k.clone())
        .filter(|k| !VALID_KEYS.contains(&k.as_str()))
        .collect();
    if !unknown_keys.is_empty() {
        let mut valid = VALID_KEYS.to_vec();
        valid.sort_unstable();
        return Err(OverrideError::UnknownKey(format!(
            "Unknown override key(s): {}. Valid keys: {}, or use dotted YAML paths under {}.",
            sorted_repr(unknown_keys),
            py_repr_list(&valid),
            py_repr_list(&DOTTED_ROOTS)
        )));
    }
    Ok(Checked {
        remove_keys,
        dotted,
        flat,
        original,
    })
}

fn py_str_of(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => py_str(other),
    }
}

/// Walk `path` into `raw`, creating objects on the way; set the leaf and
/// return what it held.
fn set_path(raw: &mut Map<String, Value>, path: &str, value: Value) -> Value {
    let parts: Vec<&str> = path.split('.').collect();
    let mut cursor = raw;
    for part in &parts[..parts.len() - 1] {
        let slot = cursor
            .entry((*part).to_owned())
            .or_insert_with(|| json!({}));
        if !slot.is_object() {
            *slot = json!({});
        }
        cursor = slot.as_object_mut().expect("just ensured an object");
    }
    cursor
        .insert(parts[parts.len() - 1].to_owned(), value)
        .unwrap_or(Value::Null)
}

/// Delete the leaf at `path`; `None` when any step is missing.
fn remove_path(raw: &mut Map<String, Value>, path: &str) -> Option<Value> {
    let parts: Vec<&str> = path.split('.').collect();
    let mut cursor = raw;
    for part in &parts[..parts.len() - 1] {
        cursor = cursor.get_mut(*part)?.as_object_mut()?;
    }
    cursor.shift_remove(parts[parts.len() - 1])
}

fn get_path(raw: &Map<String, Value>, path: &[&str]) -> Value {
    let mut cursor = raw;
    for part in &path[..path.len() - 1] {
        match cursor.get(*part).and_then(Value::as_object) {
            Some(next) => cursor = next,
            None => return Value::Null,
        }
    }
    cursor
        .get(path[path.len() - 1])
        .cloned()
        .unwrap_or(Value::Null)
}

fn object_at<'a>(raw: &'a mut Map<String, Value>, path: &[&str]) -> &'a mut Map<String, Value> {
    let mut cursor = raw;
    for part in path {
        let slot = cursor
            .entry((*part).to_owned())
            .or_insert_with(|| json!({}));
        if !slot.is_object() {
            *slot = json!({});
        }
        cursor = slot.as_object_mut().expect("just ensured an object");
    }
    cursor
}

fn as_list(key: &str, value: &Value) -> Result<Vec<Value>, OverrideError> {
    value.as_array().cloned().ok_or_else(|| {
        OverrideError::Invalid(format!(
            "Override '{key}' must be a list, got {}.",
            py_str(value)
        ))
    })
}

fn as_object(key: &str, value: &Value) -> Result<Map<String, Value>, OverrideError> {
    value.as_object().cloned().ok_or_else(|| {
        OverrideError::Invalid(format!(
            "Override '{key}' must be an object, got {}.",
            py_str(value)
        ))
    })
}

fn as_int(key: &str, value: &Value) -> Result<i64, OverrideError> {
    let parsed = match value {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f.trunc() as i64)),
        Value::String(s) => s.trim().parse().ok(),
        Value::Bool(b) => Some(i64::from(*b)),
        _ => None,
    };
    parsed.ok_or_else(|| {
        OverrideError::Invalid(format!(
            "Override '{key}' must be an integer, got {}.",
            py_str(value)
        ))
    })
}

impl Checked {
    /// Apply to the parsed YAML mapping `raw`.
    pub fn apply(&self, raw: &mut Map<String, Value>) -> Result<Applied, OverrideError> {
        let mut out = Applied::default();
        let original = |short: &str| {
            self.original
                .iter()
                .find(|(s, _)| s == short)
                .map(|(_, o)| o.as_str())
        };
        let track = |out: &mut Applied, path: &str, old: Value, new: Value| {
            out.applied.push(path.to_owned());
            out.diff.push(json!({"path": path, "old": old, "new": new}));
            if let Some((short, _)) = CANONICAL.iter().find(|(_, canonical)| *canonical == path) {
                if let Some(orig) = original(short) {
                    if orig != path && !out.applied.iter().any(|a| a == orig) {
                        out.applied.push(orig.to_owned());
                    }
                }
            }
        };

        for (path, value) in &self.dotted {
            let old = set_path(raw, path, value.clone());
            if let Some(rest) = path.strip_prefix("sweep.pca.") {
                set_path(raw, &format!("sweep.projection.{rest}"), value.clone());
            }
            track(&mut out, path, old, value.clone());
        }

        for (key, value) in &self.flat {
            let key = key.as_str();
            match key {
                "run_name" => {
                    let old = get_path(raw, &["run", "name"]);
                    object_at(raw, &["run"]).insert("name".into(), value.clone());
                    track(&mut out, "run.name", old, value.clone());
                }
                "dataset_path" => {
                    let old = get_path(raw, &["run", "data"]);
                    object_at(raw, &["run"]).insert("data".into(), value.clone());
                    track(&mut out, "run.data", old, value.clone());
                }
                "projection_method" => {
                    let old = get_path(raw, &["sweep", "projection", "method"]);
                    object_at(raw, &["sweep", "projection"]).insert("method".into(), value.clone());
                    track(&mut out, "sweep.projection.method", old, value.clone());
                }
                "projection_dimensions" | "pca_dims" => {
                    let list = Value::Array(as_list(key, value)?);
                    let side = if key == "pca_dims" {
                        "pca"
                    } else {
                        "projection"
                    };
                    let old = get_path(raw, &["sweep", side, "dimensions"]);
                    for target in ["projection", "pca"] {
                        object_at(raw, &["sweep", target])
                            .insert("dimensions".into(), json!({"values": list}));
                    }
                    track(&mut out, "sweep.projection.dimensions.values", old, list);
                }
                "projection_seeds" | "pca_seeds" => {
                    let list = Value::Array(as_list(key, value)?);
                    let side = if key == "pca_seeds" {
                        "pca"
                    } else {
                        "projection"
                    };
                    let old = get_path(raw, &["sweep", side, "seed"]);
                    for target in ["projection", "pca"] {
                        object_at(raw, &["sweep", target])
                            .insert("seed".into(), json!({"values": list}));
                    }
                    track(&mut out, "sweep.projection.seed.values", old, list);
                }
                "epsilon_values" => {
                    let list = Value::Array(as_list(key, value)?);
                    let old = get_path(raw, &["sweep", "ball_mapper", "epsilon"]);
                    object_at(raw, &["sweep", "ball_mapper"])
                        .insert("epsilon".into(), json!({"values": list}));
                    track(&mut out, "sweep.ball_mapper.epsilon.values", old, list);
                }
                "epsilon_range" => {
                    let range = Value::Object(as_object(key, value)?);
                    let old = get_path(raw, &["sweep", "ball_mapper", "epsilon"]);
                    object_at(raw, &["sweep", "ball_mapper"])
                        .insert("epsilon".into(), json!({"range": range}));
                    track(&mut out, "sweep.ball_mapper.epsilon.range", old, range);
                }
                "construction_threshold" => {
                    let old = get_path(raw, &["cosmic_graph", "construction_threshold"]);
                    object_at(raw, &["cosmic_graph"])
                        .insert("construction_threshold".into(), value.clone());
                    track(
                        &mut out,
                        "cosmic_graph.construction_threshold",
                        old,
                        value.clone(),
                    );
                }
                "n_reps" => {
                    let old = get_path(raw, &["output", "n_reps"]);
                    object_at(raw, &["output"]).insert("n_reps".into(), value.clone());
                    track(&mut out, "output.n_reps", old, value.clone());
                }
                "n_imputations" => {
                    let n = json!(as_int(key, value)?);
                    let old = get_path(raw, &["preprocessing", "n_imputations"]);
                    object_at(raw, &["preprocessing"]).insert("n_imputations".into(), n.clone());
                    track(&mut out, "preprocessing.n_imputations", old, n);
                }
                "drop_columns" => {
                    let list = Value::Array(as_list(key, value)?);
                    let old = get_path(raw, &["preprocessing", "drop_columns"]);
                    object_at(raw, &["preprocessing"]).insert("drop_columns".into(), list.clone());
                    track(&mut out, "preprocessing.drop_columns", old, list);
                }
                "impute" | "encode" | "missing_indicator_groups" => {
                    let update = as_object(key, value)?;
                    let section = object_at(raw, &["preprocessing"]);
                    let old = section
                        .get(key)
                        .and_then(Value::as_object)
                        .map_or(json!({}), |m| Value::Object(m.clone()));
                    let slot = section.entry(key.to_owned()).or_insert_with(|| json!({}));
                    if !slot.is_object() {
                        *slot = json!({});
                    }
                    slot.as_object_mut()
                        .expect("just ensured an object")
                        .extend(update.clone());
                    track(
                        &mut out,
                        &format!("preprocessing.{key}"),
                        old,
                        Value::Object(update),
                    );
                }
                "raw" => {
                    let update = as_object(key, value)?;
                    deep_merge(raw, &update);
                    track(
                        &mut out,
                        "raw",
                        json!("(deep merge)"),
                        Value::Object(update),
                    );
                }
                other => unreachable!("checked key {other}"),
            }
        }

        for path in &self.remove_keys {
            if let Some(old) = remove_path(raw, path) {
                out.applied.push(format!("-{path}"));
                out.diff
                    .push(json!({"path": path, "old": old, "new": null, "removed": true}));
            }
        }
        Ok(out)
    }
}

fn deep_merge(target: &mut Map<String, Value>, update: &Map<String, Value>) {
    for (k, v) in update {
        match (target.get_mut(k), v) {
            (Some(Value::Object(t)), Value::Object(u)) => deep_merge(t, u),
            _ => {
                target.insert(k.clone(), v.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    fn base() -> Map<String, Value> {
        obj(json!({
            "run": {"name": "t", "data": "d.csv"},
            "sweep": {
                "projection": {"method": "jl", "dimensions": {"values": [3, 5]}, "seed": {"values": [42]}},
                "ball_mapper": {"epsilon": {"range": {"min": 0.5, "max": 2.0, "steps": 10}}}
            },
            "preprocessing": {"encode": {"species": {"method": "one_hot"}}},
        }))
    }

    fn run(overrides: Value, raw: &mut Map<String, Value>) -> Applied {
        check_overrides(&obj(overrides))
            .unwrap()
            .apply(raw)
            .unwrap()
    }

    #[test]
    fn nested_overrides_set_one_leaf_each_and_keep_the_rest() {
        let mut raw = base();
        let a = run(
            json!({"sweep": {"ball_mapper": {"epsilon": {"range": {"max": 3.0}}}}}),
            &mut raw,
        );
        assert_eq!(a.applied, ["sweep.ball_mapper.epsilon.range.max"]);
        assert_eq!(
            a.diff,
            [json!({"path": "sweep.ball_mapper.epsilon.range.max", "old": 2.0, "new": 3.0})]
        );
        assert_eq!(raw["sweep"]["ball_mapper"]["epsilon"]["range"]["min"], 0.5);
        assert_eq!(raw["sweep"]["ball_mapper"]["epsilon"]["range"]["max"], 3.0);
    }

    #[test]
    fn short_keys_set_the_whole_parameter_and_the_legacy_mirror() {
        let mut raw = base();
        let a = run(json!({"projection_dimensions": [2, 4, 8]}), &mut raw);
        // The caller's spelling follows the canonical path.
        assert_eq!(
            a.applied,
            [
                "sweep.projection.dimensions.values",
                "projection_dimensions"
            ]
        );
        assert_eq!(a.diff[0]["old"], json!({"values": [3, 5]}));
        assert_eq!(
            raw["sweep"]["pca"]["dimensions"]["values"],
            json!([2, 4, 8])
        );
        assert_eq!(
            raw["sweep"]["projection"]["dimensions"]["values"],
            json!([2, 4, 8])
        );
    }

    #[test]
    fn dotted_pca_paths_mirror_into_projection() {
        let mut raw = base();
        run(
            json!({"sweep": {"pca": {"seed": {"values": [1, 2]}}}}),
            &mut raw,
        );
        assert_eq!(raw["sweep"]["projection"]["seed"]["values"], json!([1, 2]));
    }

    #[test]
    fn remove_keys_is_idempotent_and_reports_only_what_it_removed() {
        let mut raw = base();
        let a = run(
            json!({"remove_keys": ["preprocessing.encode.species", "preprocessing.encode.species", "run.nope.x"]}),
            &mut raw,
        );
        assert_eq!(a.applied, ["-preprocessing.encode.species"]);
        assert_eq!(
            a.diff,
            [
                json!({"path": "preprocessing.encode.species", "old": {"method": "one_hot"}, "new": null, "removed": true})
            ]
        );
        assert!(raw["preprocessing"]["encode"]
            .as_object()
            .unwrap()
            .is_empty());
    }

    #[test]
    fn unknown_keys_and_roots_are_listed_with_the_valid_ones() {
        let e = check_overrides(&obj(json!({"bogus": 1}))).unwrap_err();
        let msg = e.to_string();
        assert!(
            msg.starts_with(
                "Unknown override key(s): ['bogus']. Valid keys: ['construction_threshold',"
            ),
            "{msg}"
        );
        assert!(msg.ends_with("or use dotted YAML paths under ['cosmic_graph', 'output', 'preprocessing', 'run', 'sweep']."), "{msg}");
        let e = check_overrides(&obj(json!({"nope": {"x": 1}}))).unwrap_err();
        assert_eq!(
            e.to_string(),
            "Unknown dotted override root(s): ['nope']. Valid roots: ['cosmic_graph', 'output', 'preprocessing', 'run', 'sweep']"
        );
        let e = check_overrides(&obj(json!({"remove_keys": ["zzz.a", "aaa"]}))).unwrap_err();
        assert_eq!(
            e.to_string(),
            "Unknown remove_keys root(s): ['aaa', 'zzz']. Valid roots: ['cosmic_graph', 'output', 'preprocessing', 'run', 'sweep']."
        );
        assert!(check_overrides(&obj(json!({"remove_keys": "x"}))).is_err());
    }

    #[test]
    fn an_object_under_a_short_key_flattens_to_an_unknown_root() {
        // Python behavior: use the nested dotted form for ranges instead.
        let e = check_overrides(&obj(json!({"epsilon_range": {"min": 1, "max": 2}}))).unwrap_err();
        assert!(e.to_string().contains("['epsilon_range']"), "{e}");
    }

    #[test]
    fn wrongly_shaped_values_are_named() {
        let mut raw = base();
        let checked = check_overrides(&obj(json!({"projection_dimensions": 5}))).unwrap();
        let e = checked.apply(&mut raw).unwrap_err();
        assert_eq!(
            e.to_string(),
            "Override 'projection_dimensions' must be a list, got 5."
        );
        let mut raw = base();
        let a = run(json!({"n_imputations": "3"}), &mut raw);
        assert_eq!(a.diff[0]["new"], 3);
    }
}
