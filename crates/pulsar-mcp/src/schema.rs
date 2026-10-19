//! JSON Schema builders for tool output contracts (`toolmeta.py`) and the
//! input schema derived from a parameter struct.

use std::sync::Arc;

use rmcp::model::JsonObject;
use schemars::JsonSchema;
use serde_json::{json, Map, Value};

pub fn string() -> Value {
    json!({"type": "string"})
}

pub fn integer() -> Value {
    json!({"type": "integer"})
}

pub fn number() -> Value {
    json!({"type": "number"})
}

pub fn boolean() -> Value {
    json!({"type": "boolean"})
}

/// Any JSON value.
pub fn any() -> Value {
    json!({})
}

/// An object with unspecified properties.
pub fn open_object() -> Value {
    json!({"type": "object", "additionalProperties": true})
}

pub fn array(items: Value) -> Value {
    json!({"type": "array", "items": items})
}

pub fn nullable(schema: Value) -> Value {
    json!({"anyOf": [schema, {"type": "null"}]})
}

pub fn enumeration(values: &[&str]) -> Value {
    json!({"type": "string", "enum": values})
}

/// An object with the given properties. `additional` allows undeclared keys.
pub fn object(properties: &[(&str, Value)], required: &[&str], additional: bool) -> Value {
    let props: Map<String, Value> = properties
        .iter()
        .map(|(k, v)| ((*k).to_owned(), v.clone()))
        .collect();
    json!({
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": additional,
    })
}

pub fn as_object(schema: Value) -> Arc<JsonObject> {
    match schema {
        Value::Object(map) => Arc::new(map),
        other => Arc::new(Map::from_iter([("value".to_owned(), other)])),
    }
}

/// The input schema of a parameter struct: schemars output with the
/// document-level `$schema` and `title` keys removed (they cost catalog
/// characters and say nothing an agent needs).
pub fn input_schema<T: JsonSchema>() -> Arc<JsonObject> {
    let schema = schemars::schema_for!(T);
    let mut value = serde_json::to_value(schema).unwrap_or_else(|_| json!({"type": "object"}));
    if let Some(map) = value.as_object_mut() {
        map.remove("$schema");
        map.remove("title");
        map.entry("type").or_insert_with(|| json!("object"));
    }
    as_object(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Deserialize, JsonSchema, Default)]
    #[serde(default)]
    struct Params {
        /// A handle.
        run_id: String,
        /// A limit.
        limit: usize,
    }

    #[test]
    fn input_schema_keeps_descriptions_and_drops_document_keys() {
        let schema = input_schema::<Params>();
        assert_eq!(schema["type"], "object");
        assert!(schema.get("$schema").is_none());
        assert!(schema.get("title").is_none());
        assert_eq!(schema["properties"]["run_id"]["description"], "A handle.");
        assert_eq!(schema["properties"]["limit"]["type"], "integer");
    }

    #[test]
    fn builders_produce_the_python_shapes() {
        let s = object(
            &[("a", string()), ("b", nullable(integer()))],
            &["a"],
            false,
        );
        assert_eq!(s["required"], json!(["a"]));
        assert_eq!(s["properties"]["b"]["anyOf"][1]["type"], "null");
        assert_eq!(enumeration(&["x", "y"])["enum"], json!(["x", "y"]));
    }
}
