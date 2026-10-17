//! Projection-cache fingerprint: embeddings are invariant to epsilon changes,
//! so a sweep retry with the same data, preprocessing and projection grid can
//! reuse them. SHA-256 over a canonical JSON payload.

use std::collections::BTreeMap;
use std::time::UNIX_EPOCH;

use pulsar_config::PulsarConfig;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// Fingerprint of everything that determines the projection embeddings:
/// data path and mtime, the preprocessing block, the projection grid, the
/// row count and (when given) the input schema `(name, dtype)` pairs.
pub fn projection_fingerprint(
    cfg: &PulsarConfig,
    n_rows: usize,
    input_schema: Option<&[(String, &'static str)]>,
) -> String {
    let mtime = std::fs::metadata(&cfg.data)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0.0, |d| d.as_secs_f64());
    let mut drop_columns = cfg.preprocessing.drop_columns.clone();
    drop_columns.sort();
    let pre = &cfg.preprocessing;
    let impute: BTreeMap<&String, Value> = pre
        .impute
        .iter()
        .map(|(c, s)| (c, serde_json::to_value(s).unwrap_or(Value::Null)))
        .collect();
    let encode: BTreeMap<&String, Value> = pre
        .encode
        .iter()
        .map(|(c, s)| (c, serde_json::to_value(s).unwrap_or(Value::Null)))
        .collect();
    let groups: BTreeMap<&String, &Vec<String>> = pre.missing_indicator_groups.iter().collect();
    let schema: Value = match input_schema {
        Some(s) => s
            .iter()
            .map(|(name, dtype)| json!({"dtype": dtype, "name": name}))
            .collect(),
        None => Value::Null,
    };
    let mut payload: BTreeMap<&str, Value> = BTreeMap::new();
    payload.insert("center", json!(cfg.projection.center));
    payload.insert("data_path", json!(cfg.data));
    payload.insert("dimensions", json!(cfg.projection.dimensions));
    payload.insert("drop_columns", json!(drop_columns));
    payload.insert("encode", json!(encode));
    payload.insert("impute", json!(impute));
    payload.insert("input_schema", schema);
    payload.insert("method", json!(cfg.projection.method));
    payload.insert("missing_indicator_groups", json!(groups));
    payload.insert("mtime", json!(mtime));
    payload.insert("n_rows", json!(n_rows));
    payload.insert("seeds", json!(cfg.projection.seeds));
    let text = serde_json::to_string(&payload).expect("payload is plain JSON");
    let digest = Sha256::digest(text.as_bytes());
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changes_with_projection_but_not_epsilon() {
        let a = pulsar_config::from_yaml_str(
            "sweep: {projection: {dimensions: [2]}, ball_mapper: {epsilon: [0.5]}}",
        )
        .unwrap();
        let b = pulsar_config::from_yaml_str(
            "sweep: {projection: {dimensions: [2]}, ball_mapper: {epsilon: [0.9]}}",
        )
        .unwrap();
        let c = pulsar_config::from_yaml_str(
            "sweep: {projection: {dimensions: [3]}, ball_mapper: {epsilon: [0.5]}}",
        )
        .unwrap();
        let fa = projection_fingerprint(&a, 10, None);
        assert_eq!(fa.len(), 64);
        assert_eq!(fa, projection_fingerprint(&b, 10, None));
        assert_ne!(fa, projection_fingerprint(&c, 10, None));
        assert_ne!(fa, projection_fingerprint(&a, 11, None));
    }
}
