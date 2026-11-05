//! Compact, agent-sized views of a characterization and of column probes
//! (`pulsar/mcp/characterization.py`).
//!
//! The analysis layer keeps every column profile; the map view must not dump
//! O(width) schema, so it shows a ranked preview and counts what it omits.

use serde_json::{json, Map, Value};

use super::{ColumnProfile, DatasetProfile};
use crate::minhash::massive_dataset_advisory;
use crate::plan::py_str;
use crate::pyround::round_to;

const DEFAULT_PREVIEW_LIMIT: usize = 20;
const COLUMN_NAME_PREVIEW_LIMIT: usize = 20;
const VALUE_PREVIEW_LIMIT: usize = 80;

fn is_high_cardinality(cp: &ColumnProfile, n_samples: usize) -> bool {
    if cp.is_numeric {
        return false;
    }
    cp.n_unique > 50 || (n_samples > 0 && cp.n_unique as f64 / n_samples as f64 > 0.9)
}

fn interest_score(cp: &ColumnProfile, n_samples: usize) -> u32 {
    let mut score = 0;
    if cp.missing_pct >= 100.0 {
        score += 8;
    } else if cp.n_missing > 0 {
        score += 4;
    }
    if !cp.is_numeric {
        score += 2;
    }
    if is_high_cardinality(cp, n_samples) {
        score += 3;
    }
    score
}

fn compact_column_profile(cp: &ColumnProfile) -> Value {
    json!({
        "name": cp.name,
        "dtype": cp.dtype,
        "is_numeric": cp.is_numeric,
        "n_unique": cp.n_unique,
        "n_missing": cp.n_missing,
        "missing_pct": cp.missing_pct,
    })
}

/// The columns most worth a second look: blank, categorical or
/// high-cardinality, most notable first.
fn interesting_preview(profiles: &[ColumnProfile], n_samples: usize, limit: usize) -> Vec<Value> {
    let mut ranked: Vec<(u32, &ColumnProfile)> = profiles
        .iter()
        .map(|cp| (interest_score(cp, n_samples), cp))
        .filter(|(score, _)| *score > 0)
        .collect();
    ranked.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| b.1.missing_pct.total_cmp(&a.1.missing_pct))
            .then_with(|| a.1.name.cmp(&b.1.name))
    });
    ranked
        .into_iter()
        .take(limit)
        .map(|(_, cp)| compact_column_profile(cp))
        .collect()
}

fn schema_summary(profiles: &[ColumnProfile], n_samples: usize, preview_count: usize) -> Value {
    let numeric = profiles.iter().filter(|c| c.is_numeric).count();
    json!({
        "numeric_columns": numeric,
        "categorical_columns": profiles.len() - numeric,
        "columns_with_missing": profiles.iter().filter(|c| c.n_missing > 0).count(),
        "high_cardinality_columns": profiles.iter().filter(|c| is_high_cardinality(c, n_samples)).count(),
        "all_missing_columns": profiles.iter().filter(|c| c.missing_pct >= 100.0).count(),
        "previewed_column_profiles": preview_count,
    })
}

fn overall_missingness_pct(profiles: &[ColumnProfile], n_samples: usize) -> f64 {
    if profiles.is_empty() || n_samples == 0 {
        return 0.0;
    }
    let missing: usize = profiles.iter().map(|c| c.n_missing).sum();
    round_to(
        missing as f64 / (n_samples * profiles.len()) as f64 * 100.0,
        2,
    )
}

fn column_name_preview(profiles: &[ColumnProfile]) -> Value {
    let block = |numeric: bool| {
        let names: Vec<&String> = profiles
            .iter()
            .filter(|c| c.is_numeric == numeric)
            .map(|c| &c.name)
            .collect();
        json!({
            "columns": names.iter().take(COLUMN_NAME_PREVIEW_LIMIT).collect::<Vec<_>>(),
            "omitted": names.len().saturating_sub(COLUMN_NAME_PREVIEW_LIMIT),
        })
    };
    json!({"numeric": block(true), "categorical": block(false)})
}

/// The `characterize_dataset` payload, with the default 20-column preview.
pub fn compact_characterization_payload(geo: &DatasetProfile) -> Value {
    compact_with_limit(geo, DEFAULT_PREVIEW_LIMIT)
}

pub(crate) fn compact_with_limit(geo: &DatasetProfile, preview_limit: usize) -> Value {
    let profiles = &geo.column_profiles;
    let preview = interesting_preview(profiles, geo.n_samples, preview_limit);
    let advisory = massive_dataset_advisory(geo.n_samples, crate::minhash::DEFAULT_D);

    let mut p = Map::new();
    p.insert("status".into(), json!("ok"));
    p.insert("n_samples".into(), json!(geo.n_samples));
    p.insert("n_features".into(), json!(geo.n_features));
    p.insert("n_columns_total".into(), json!(geo.n_columns_total));
    p.insert("missingness_pct".into(), json!(geo.missingness_pct));
    p.insert("numeric_missingness_pct".into(), json!(geo.missingness_pct));
    p.insert(
        "overall_missingness_pct".into(),
        json!(overall_missingness_pct(profiles, geo.n_samples)),
    );
    p.insert(
        "raw_numeric_geometry".into(),
        json!({
            "knn_k5_mean": geo.knn_k5_mean,
            "knn_k10_mean": geo.knn_k10_mean,
            "knn_k20_mean": geo.knn_k20_mean,
            "pca_cumulative_variance": geo.pca_cumulative_variance,
        }),
    );
    let preview_count = preview.len();
    p.insert(
        "schema_summary".into(),
        schema_summary(profiles, geo.n_samples, preview_count),
    );
    p.insert("column_name_preview".into(), column_name_preview(profiles));
    let omitted = profiles.len().saturating_sub(preview.len());
    p.insert("column_profile_preview".into(), Value::Array(preview));
    p.insert("omitted_column_profiles".into(), json!(omitted));
    p.insert("recommended_next_tool".into(), json!("create_config"));
    p.insert(
        "agent_guidance".into(),
        json!("Use create_config(dataset_id) for processed-space calibration. Use probe_columns(dataset_id, columns) only for specific columns listed in column_profile_preview or named by the user."),
    );
    // Only present on massive datasets.
    if let Some(a) = advisory {
        p.insert("minhash_advisory".into(), a);
    }
    Value::Object(p)
}

fn fmt_pct(v: &Value) -> String {
    format!("{:.2}%", v.as_f64().unwrap_or(0.0))
}

fn fmt_float(v: &Value) -> String {
    v.as_f64()
        .map_or_else(|| "n/a".to_string(), |x| format!("{x:.4}"))
}

fn text(v: &Value) -> String {
    py_str(v)
}

fn int(v: &Value) -> i64 {
    v.as_i64().unwrap_or(0)
}

/// A readable Markdown view of the characterization payload.
pub fn characterization_payload_to_markdown(payload: &Value) -> String {
    let (schema, geometry, names) = (
        &payload["schema_summary"],
        &payload["raw_numeric_geometry"],
        &payload["column_name_preview"],
    );
    let mut lines = vec![
        "# Dataset Characterization".to_string(),
        String::new(),
        format!("- Rows: {}", text(&payload["n_samples"])),
        format!("- Numeric features: {}", text(&payload["n_features"])),
        format!("- Total columns: {}", text(&payload["n_columns_total"])),
        format!(
            "- Numeric missingness: {}",
            fmt_pct(&payload["numeric_missingness_pct"])
        ),
        format!(
            "- Overall missingness: {}",
            fmt_pct(&payload["overall_missingness_pct"])
        ),
        String::new(),
        "## Schema".into(),
        String::new(),
        format!("- Numeric columns: {}", int(&schema["numeric_columns"])),
        format!(
            "- Categorical columns: {}",
            int(&schema["categorical_columns"])
        ),
        format!(
            "- Columns with missing values: {}",
            int(&schema["columns_with_missing"])
        ),
        format!(
            "- High-cardinality columns: {}",
            int(&schema["high_cardinality_columns"])
        ),
        format!(
            "- All-missing columns: {}",
            int(&schema["all_missing_columns"])
        ),
        String::new(),
        "## Geometry".into(),
        String::new(),
        format!("- kNN mean k=5: {}", fmt_float(&geometry["knn_k5_mean"])),
        format!("- kNN mean k=10: {}", fmt_float(&geometry["knn_k10_mean"])),
        format!("- kNN mean k=20: {}", fmt_float(&geometry["knn_k20_mean"])),
    ];
    let variance: Vec<String> = geometry["pca_cumulative_variance"]
        .as_array()
        .map(|a| {
            a.iter()
                .take(6)
                .map(|pair| format!("{}d={:.3}", int(&pair[0]), pair[1].as_f64().unwrap_or(0.0)))
                .collect()
        })
        .unwrap_or_default();
    if !variance.is_empty() {
        lines.push(format!(
            "- PCA cumulative variance: {}",
            variance.join(", ")
        ));
    }

    lines.extend([
        String::new(),
        "## Column Name Preview".into(),
        String::new(),
    ]);
    for (kind, label) in [("numeric", "Numeric"), ("categorical", "Categorical")] {
        let block = &names[kind];
        let cols: Vec<String> = block["columns"]
            .as_array()
            .map(|a| a.iter().map(|n| format!("`{}`", text(n))).collect())
            .unwrap_or_default();
        if cols.is_empty() {
            lines.push(format!("- {label}: none"));
        } else {
            lines.push(format!("- {label}: {}", cols.join(", ")));
        }
        let omitted = int(&block["omitted"]);
        if omitted > 0 {
            lines.push(format!("  - {omitted} more {kind} columns omitted."));
        }
    }

    lines.extend([String::new(), "## Columns To Inspect".into(), String::new()]);
    let preview = payload["column_profile_preview"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    if preview.is_empty() {
        lines.push(
            "- No high-missingness, categorical, or high-cardinality columns flagged.".into(),
        );
    }
    for cp in &preview {
        lines.push(format!(
            "- `{}`: {}, unique={}, missing={}",
            text(&cp["name"]),
            text(&cp["dtype"]),
            text(&cp["n_unique"]),
            fmt_pct(&cp["missing_pct"])
        ));
    }
    let omitted_profiles = int(&payload["omitted_column_profiles"]);
    if omitted_profiles > 0 {
        lines.push(format!(
            "- Column profiles omitted from preview: {omitted_profiles}"
        ));
    }

    if let Some(advisory) = payload.get("minhash_advisory").filter(|a| a.is_object()) {
        let (cur, sug) = (&advisory["current_profile"], &advisory["suggested_profile"]);
        lines.extend([
            String::new(),
            "## Cosmic Graph Construction (MinHash)".into(),
            String::new(),
            format!("- {}", text(&advisory["message"])),
            format!(
                "- Current `minhash_d={}`: 95% CI \u{b1}{}, signature {}.",
                text(&advisory["current"]),
                text(&cur["ci95_half_width_worst"]),
                text(&cur["signature_memory_human"])
            ),
            format!(
                "- Suggested `minhash_d={}`: 95% CI \u{b1}{}, signature {}.",
                text(&advisory["suggested"]),
                text(&sug["ci95_half_width_worst"]),
                text(&sug["signature_memory_human"])
            ),
            "- Set via `refine_config(..., cosmic_graph.minhash_d=<value>)`.".into(),
        ]);
    }

    let next = payload
        .get("recommended_next_tool")
        .map_or_else(|| "create_config".to_string(), text);
    lines.extend([
        String::new(),
        "## Next Tools".into(),
        String::new(),
        format!("- `{next}` for processed-space calibration."),
        "- `probe_columns` for specific columns from the previews above.".into(),
    ]);
    lines.join("\n").trim().to_string()
}

fn truncate_value(value: &str) -> (String, bool) {
    if value.chars().count() <= VALUE_PREVIEW_LIMIT {
        return (value.to_string(), false);
    }
    (
        format!(
            "{}...",
            value.chars().take(VALUE_PREVIEW_LIMIT).collect::<String>()
        ),
        true,
    )
}

fn compact_detailed_profile(p: &ColumnProfile) -> Value {
    let mut truncated_samples = 0;
    let samples: Vec<String> = p
        .sample_values
        .iter()
        .map(|v| {
            let (t, was) = truncate_value(v);
            truncated_samples += usize::from(was);
            t
        })
        .collect();
    let mut truncated_top = 0;
    let top = p.top_values.as_ref().map(|tv| {
        tv.iter()
            .map(|(v, n)| {
                let (t, was) = truncate_value(v);
                truncated_top += usize::from(was);
                json!([t, n])
            })
            .collect::<Vec<_>>()
    });
    json!({
        "name": p.name,
        "dtype": p.dtype,
        "is_numeric": p.is_numeric,
        "n_unique": p.n_unique,
        "n_missing": p.n_missing,
        "missing_pct": p.missing_pct,
        "sample_values": samples,
        "mean": p.mean,
        "std": p.std,
        "min_val": p.min_val,
        "max_val": p.max_val,
        "top_values": top,
        "truncation": {
            "sample_values_truncated": truncated_samples,
            "top_values_truncated": truncated_top,
        },
    })
}

/// The `probe_columns` payload.
pub fn probe_columns_payload(
    profiles: &[ColumnProfile],
    requested: &[String],
    missing: &[String],
) -> Value {
    json!({
        "status": "ok",
        "columns_requested": requested.len(),
        "columns_returned": profiles.len(),
        "missing_columns": missing,
        "column_profiles": profiles.iter().map(compact_detailed_profile).collect::<Vec<_>>(),
    })
}

/// A readable Markdown view of the probe payload.
pub fn probe_columns_payload_to_markdown(payload: &Value) -> String {
    let mut lines = vec![
        "# Column Probe".to_string(),
        String::new(),
        format!(
            "- Columns requested: {}",
            int(&payload["columns_requested"])
        ),
        format!("- Columns returned: {}", int(&payload["columns_returned"])),
    ];
    let missing: Vec<String> = payload["missing_columns"]
        .as_array()
        .map(|a| a.iter().map(|n| format!("`{}`", text(n))).collect())
        .unwrap_or_default();
    if !missing.is_empty() {
        lines.push(format!("- Missing columns: {}", missing.join(", ")));
    }
    for profile in payload["column_profiles"].as_array().into_iter().flatten() {
        lines.extend([
            String::new(),
            format!("## `{}`", text(&profile["name"])),
            String::new(),
            format!("- Type: {}", text(&profile["dtype"])),
            format!("- Numeric: {}", text(&profile["is_numeric"])),
            format!("- Unique values: {}", text(&profile["n_unique"])),
            format!(
                "- Missing: {} ({})",
                text(&profile["n_missing"]),
                fmt_pct(&profile["missing_pct"])
            ),
        ]);
        if profile["is_numeric"] == json!(true) {
            lines.extend([
                format!("- Mean: {}", fmt_float(&profile["mean"])),
                format!("- Std: {}", fmt_float(&profile["std"])),
                format!(
                    "- Min / max: {} / {}",
                    fmt_float(&profile["min_val"]),
                    fmt_float(&profile["max_val"])
                ),
            ]);
        }
        if let Some(top) = profile["top_values"].as_array().filter(|t| !t.is_empty()) {
            lines.push("- Top values:".into());
            for pair in top {
                lines.push(format!("  - `{}`: {}", text(&pair[0]), text(&pair[1])));
            }
        }
        if let Some(samples) = profile["sample_values"]
            .as_array()
            .filter(|s| !s.is_empty())
        {
            lines.push(format!(
                "- Sample values: {}",
                samples
                    .iter()
                    .map(|v| format!("`{}`", text(v)))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if let Some(t) = profile["truncation"]
            .as_object()
            .filter(|t| t.values().any(|v| v.as_i64().unwrap_or(0) != 0))
        {
            let parts: Vec<String> = t.iter().map(|(k, v)| format!("{k}={}", text(v))).collect();
            lines.push(format!("- Truncation: {}", parts.join(", ")));
        }
    }
    lines.join("\n").trim().to_string()
}
