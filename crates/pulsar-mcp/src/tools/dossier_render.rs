//! The Markdown views of a fitted dossier and of `get_cluster_detail`:
//! cluster sections, the cluster index and the dossier header.

use serde_json::Value;

fn fmt_num(v: &Value, dp: usize) -> String {
    v.as_f64()
        .map_or_else(|| "?".into(), |x| format!("{x:.dp$}"))
}

fn feature_line(row: &Value) -> String {
    let tier = row["signal_tier"].as_str().unwrap_or("?");
    if row.get("value").is_some_and(|v| !v.is_null()) {
        format!(
            "{}={} ({tier}, lift {})",
            row["column"].as_str().unwrap_or("?"),
            row["value"]
                .as_str()
                .map_or_else(|| row["value"].to_string(), str::to_owned),
            fmt_num(&row["lift"], 2)
        )
    } else {
        let z = if row["z_score"].is_number() {
            format!(", z {}", fmt_num(&row["z_score"], 2))
        } else {
            String::new()
        };
        format!(
            "{}: {} ({tier}{z})",
            row["column"].as_str().unwrap_or("?"),
            row["direction"].as_str().unwrap_or("mixed"),
        )
    }
}

/// `- Neighbours: ...`, when the section lists any.
fn neighbours_line(c: &Value) -> Option<String> {
    let n = c["topological_neighbors"]
        .as_array()
        .filter(|n| !n.is_empty())?;
    let list: Vec<String> = n
        .iter()
        .map(|x| {
            format!(
                "{} ({})",
                x["cluster_id"],
                fmt_num(&x["bridge_strength"], 2)
            )
        })
        .collect();
    Some(format!("- Neighbours: {}", list.join(", ")))
}

/// `- Shortened: ...`, when the section was capped.
fn shortened_line(c: &Value) -> Option<String> {
    let t = c["sections_trimmed"].as_object()?;
    let cut: Vec<String> = t
        .iter()
        .map(|(k, v)| format!("{k} {}/{}", v["shown"], v["total"]))
        .collect();
    Some(format!("- Shortened: {}", cut.join(", ")))
}

/// Markdown of one cluster section.
pub fn section_markdown(c: &Value) -> String {
    let mut lines = vec![
        format!(
            "### Cluster {}: {}",
            c["cluster_id"],
            c["semantic_name"].as_str().unwrap_or("")
        ),
        format!("- Size: {} ({}%)", c["size"], fmt_num(&c["size_pct"], 1)),
    ];
    lines.extend(neighbours_line(c));
    for (key, title) in [
        ("feature_preview", "Feature preview"),
        ("numeric_features", "Numeric features"),
        ("categorical_features", "Categorical features"),
    ] {
        if let Some(rows) = c[key].as_array().filter(|r| !r.is_empty()) {
            lines.push(format!("- {title}:"));
            lines.extend(rows.iter().map(|r| format!("  - {}", feature_line(r))));
        }
    }
    if let Some(rows) = c["central_rows"].as_array().filter(|r| !r.is_empty()) {
        lines.push(format!(
            "- Central rows: {} (structured content)",
            rows.len()
        ));
    }
    lines.extend(shortened_line(c));
    lines.join("\n")
}

/// Markdown of a cluster index.
pub fn index_markdown(index: &Value) -> String {
    let mut lines = vec![
        format!("## Cluster Index ({} clusters)", index["total"]),
        String::new(),
    ];
    for c in index["clusters"].as_array().into_iter().flatten() {
        let label = c["label"]
            .as_str()
            .map(|l| format!(": {l}"))
            .unwrap_or_default();
        lines.push(format!("- {} (n={}){label}", c["cluster_id"], c["size"]));
    }
    for g in index["grouped"].as_array().into_iter().flatten() {
        lines.push(format!(
            "- {} clusters of size {}: {}",
            g["count"],
            g["size"],
            g["cluster_ids"].as_str().unwrap_or("")
        ));
    }
    lines.join("\n")
}

/// The dossier header in Markdown, from the payload.
pub fn dossier_header(p: &Value) -> String {
    [
        "# Cluster Dossier".to_owned(),
        String::new(),
        format!(
            "- cluster_assignment_id: `{}` (the session's live labels)",
            p["cluster_assignment_id"].as_str().unwrap_or("")
        ),
        format!("- run_id: `{}`", p["run_id"].as_str().unwrap_or("")),
        format!(
            "- Method: {}; clusters: {}; interpretation threshold {} (construction {})",
            p["method_applied"].as_str().unwrap_or("?"),
            p["cluster_count"],
            fmt_num(&p["interpretation_edge_weight_threshold_applied"], 4),
            fmt_num(&p["construction_threshold"], 4),
        ),
        format!(
            "- Readiness: {}",
            p["cluster_result"]["interpretation_readiness"]["status"]
                .as_str()
                .unwrap_or("unknown")
        ),
    ]
    .join("\n")
}

/// The Markdown view of a fitted dossier `p` whose truncation reads `line`:
/// `original` (the dossier's own view) with the index appended, or else the
/// header, the index and every section `p` keeps.
pub fn dossier_markdown(p: &Value, line: &str, original: Option<&str>) -> String {
    let index = index_markdown(&p["cluster_index"]);
    if let Some(original) = original {
        return format!("{original}\n\n{index}\n\n{line}");
    }
    let body: Vec<String> = p["clusters"]
        .as_array()
        .into_iter()
        .flatten()
        .map(section_markdown)
        .collect();
    format!(
        "{}\n\n{line}\n\n{index}\n\n## Clusters\n\n{}",
        dossier_header(p),
        body.join("\n\n")
    )
}
