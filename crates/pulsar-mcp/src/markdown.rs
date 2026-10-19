//! Markdown text views. The structured content is the contract; these are
//! the readable renderings (`payloads.py`, `diagnostics.py`).

use serde_json::Value;

fn metric(value: &Value) -> String {
    match value {
        Value::Null => "n/a".into(),
        Value::Number(n) => match n.as_f64() {
            Some(f) if !n.is_i64() && !n.is_u64() => format!("{f:.4}"),
            _ => n.to_string(),
        },
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn pct(value: &Value) -> String {
    match value.as_f64() {
        Some(f) => format!("{:.1}%", f * 100.0),
        None => "n/a".into(),
    }
}

fn graph_health_status(payload: &Value) -> String {
    let health = payload["graph_health"].as_str().unwrap_or("unknown");
    let metrics = &payload["key_metrics"];
    let giant = metrics["giant_fraction"].as_f64().unwrap_or(0.0);
    let components = metrics["component_count"].as_u64().unwrap_or(0);
    let singletons = metrics["singleton_fraction"].as_f64().unwrap_or(0.0);
    if health == "connected" && components > 1 {
        if giant >= 0.85 {
            return "usable_giant_component_with_tail".into();
        }
        return "usable_multicomponent".into();
    }
    if health == "fragmented" && singletons >= 0.25 {
        return "fragmented_singleton_heavy".into();
    }
    health.to_owned()
}

/// `sweep_payload_to_markdown` over the summary payload.
pub fn sweep(payload: &Value) -> String {
    let gate = &payload["finalization_gate"];
    let mut lines = vec![
        "# Sweep Run Complete".to_owned(),
        String::new(),
        format!("- Run ID: `{}`", metric(&payload["run_id"])),
        format!("- Dataset ID: `{}`", metric(&payload["dataset_id"])),
        format!("- Graph health status: {}", graph_health_status(payload)),
        format!(
            "- Constructed graph connected: {}",
            payload["constructed_graph_connected"]
        ),
        format!(
            "- Construction threshold (edge-weight cut): {}",
            metric(&payload["construction_threshold"])
        ),
        format!("- Analysis status: {}", metric(&payload["analysis_status"])),
        format!(
            "- Next required check: `{}`",
            metric(&payload["next_required_check"])
        ),
        format!(
            "- Finalization gate: {}",
            gate["status"].as_str().unwrap_or("unknown")
        ),
    ];
    if let Some(advisory) = payload["config_advisory"].as_str() {
        lines.push(format!("- Config advisory: {advisory}"));
    }
    lines.extend([
        "".into(),
        "## Key Metrics".into(),
        "".into(),
        "| Metric | Value |".into(),
        "|---|---:|".into(),
    ]);
    if let Some(metrics) = payload["key_metrics"].as_object() {
        for (key, value) in metrics {
            lines.push(format!("| {key} | {} |", metric(value)));
        }
    }
    let diff = &payload["diff_summary"];
    let params = diff["parameter_changes"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let metric_changes = diff["metric_changes"]
        .as_object()
        .cloned()
        .unwrap_or_default();
    if !params.is_empty() || !metric_changes.is_empty() {
        lines.extend(["".into(), "## Diff From Previous Sweep".into(), "".into()]);
        if !params.is_empty() {
            let items: Vec<String> = params.iter().map(|p| format!("`{}`", metric(p))).collect();
            lines.push(format!("- Parameter changes: {}", items.join(", ")));
        }
        if !metric_changes.is_empty() {
            lines.extend([
                "".into(),
                "| Metric | Previous | Current |".into(),
                "|---|---:|---:|".into(),
            ]);
            for (key, change) in &metric_changes {
                lines.push(format!(
                    "| {key} | {} | {} |",
                    metric(&change["previous"]),
                    metric(&change["current"])
                ));
            }
        }
    }
    if let Some(stability) = payload["threshold_stability_summary"].as_object() {
        lines.extend([
            "".into(),
            "## Threshold Stability".into(),
            "".into(),
            format!(
                "- Selected threshold: {}",
                metric(stability.get("selected_threshold").unwrap_or(&Value::Null))
            ),
        ]);
        if let Some(w) = stability.get("warning").and_then(Value::as_str) {
            lines.push(format!("- Warning: {w}"));
        }
    }
    if matches!(gate["status"].as_str(), Some("blocked") | Some("caution")) {
        lines.extend([
            "".into(),
            "## Gate Advisory".into(),
            "".into(),
            format!("- Status: `{}`", metric(&gate["status"])),
            format!("- Code: `{}`", metric(&gate["code"])),
            format!("- Message: {}", metric(&gate["message"])),
        ]);
        if let Some(action) = gate["recommended_action"].as_object() {
            let tool = action
                .get("tool")
                .and_then(Value::as_str)
                .unwrap_or("generate_cluster_dossier");
            let method = action
                .get("method")
                .and_then(Value::as_str)
                .unwrap_or("components");
            let mut command = format!("{tool}(method=\"{method}\"");
            if let Some(t) = action.get("interpretation_edge_weight_threshold") {
                if !t.is_null() {
                    command.push_str(&format!(
                        ", interpretation_edge_weight_threshold={}",
                        metric(t)
                    ));
                }
            }
            command.push(')');
            lines.push(format!("- Recommended action: `{command}`"));
            if let Some(reason) = action.get("reason").and_then(Value::as_str) {
                lines.push(format!("- Reason: {reason}"));
            }
        }
    }
    lines.join("\n")
}

/// `_diagnosis_to_markdown`.
pub fn diagnosis(payload: &Value) -> String {
    let surface = &payload["graph_surface"];
    let scale = &payload["scale"];
    let morph = &payload["component_morphology"];
    let weights = &payload["weight_distribution"];
    let support = &payload["sweep_support"];
    let risks: Vec<&str> = payload["risk_factors"]
        .as_array()
        .map(|a| a.iter().filter_map(|r| r["code"].as_str()).collect())
        .unwrap_or_default();
    let patterns: Vec<&str> = payload["observed_patterns"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let sizes: Vec<String> = morph["top_component_sizes"]
        .as_array()
        .map(|a| a.iter().map(metric).collect())
        .unwrap_or_default();
    let mut lines = vec![
        "# Cosmic Graph Diagnosis".to_owned(),
        String::new(),
        format!("- Source: {}", metric(&payload["source"])),
        format!("- Run ID: `{}`", metric(&payload["run_id"])),
        format!(
            "- Surface: {} @ construction threshold {}",
            metric(&surface["kind"]),
            metric(&surface["construction_threshold"])
        ),
        format!(
            "- Nodes / edges: {} / {}",
            metric(&scale["n_nodes"]),
            metric(&scale["n_edges"])
        ),
        format!("- Density: {}", metric(&scale["density"])),
        format!("- Components: {}", metric(&morph["component_count"])),
        format!("- Top component sizes: {}", sizes.join(", ")),
        format!(
            "- Largest component: {}",
            pct(&morph["largest_component_fraction"])
        ),
        format!(
            "- Singleton fraction: {}",
            pct(&morph["singleton_fraction"])
        ),
        format!(
            "- Nontrivial components: {} (floor={})",
            metric(&morph["nontrivial_component_count"]),
            metric(&morph["nontrivial_component_floor"])
        ),
        format!(
            "- Grid support: {} ({} ball maps)",
            metric(&support["grid_adequacy_status"]),
            metric(&support["n_ball_maps"])
        ),
        format!(
            "- Weight p25/p50/p95: {} / {} / {}",
            metric(&weights["p25"]),
            metric(&weights["p50"]),
            metric(&weights["p95"])
        ),
        format!("- Observed patterns: {}", patterns.join(", ")),
        format!(
            "- Risk factors: {}",
            if risks.is_empty() {
                "none".to_owned()
            } else {
                risks.join(", ")
            }
        ),
    ];
    if let Some(mh) = payload["minhash_profile"].as_object() {
        lines.push(format!(
            "- MinHash construction (d={}): 95% CI \u{b1}{}, signature {}",
            metric(mh.get("d").unwrap_or(&Value::Null)),
            metric(mh.get("ci95_half_width_worst").unwrap_or(&Value::Null)),
            metric(mh.get("signature_memory_human").unwrap_or(&Value::Null))
        ));
    }
    lines.join("\n")
}

/// Compact rendering of the threshold curve packet.
pub fn threshold_curve(payload: &Value) -> String {
    let mut lines = vec![
        "# Threshold Stability Curve".to_owned(),
        String::new(),
        format!("- Run ID: `{}`", metric(&payload["run_id"])),
        format!(
            "- Resolved construction threshold: {}",
            metric(&payload["resolved_construction_threshold"])
        ),
        format!(
            "- H0 longest plateau threshold: {} (matches current: {})",
            metric(&payload["h0_longest_plateau_threshold"]),
            payload["matches_current_threshold"]
        ),
        format!(
            "- Policy: {}",
            metric(&payload["threshold_candidate_policy"])
        ),
        format!("- Readout: {}", metric(&payload["h0_plateau_readout"])),
    ];
    if let Some(candidates) = payload["threshold_candidates"].as_array() {
        lines.extend([
            "".into(),
            "## Candidate Lenses".into(),
            "".into(),
            "| Threshold | Kind | Tier | Components | Singletons |".into(),
            "|---:|---|---|---:|---:|".into(),
        ]);
        for c in candidates {
            let morph = &c["morphology"];
            lines.push(format!(
                "| {} | {} | {} | {} | {} |",
                metric(&c["threshold"]),
                metric(&c["candidate_kind"]),
                metric(&c["interpretability_tier"]),
                metric(&morph["component_count"]),
                pct(&morph["singleton_fraction"])
            ));
        }
    }
    if let Some(bps) = payload["structural_breakpoints"].as_array() {
        if !bps.is_empty() {
            lines.extend(["".into(), "## Structural Breakpoints".into(), "".into()]);
            for b in bps {
                lines.push(format!(
                    "- {} at {}: {} \u{2192} {} components, absorbed {}",
                    metric(&b["event"]),
                    metric(&b["threshold"]),
                    metric(&b["component_count_before"]),
                    metric(&b["component_count_after"]),
                    pct(&b["absorbed_mass_fraction"])
                ));
            }
        }
    }
    lines.join("\n")
}

/// `_history_markdown`.
pub fn history(payload: &Value) -> String {
    let rows = payload["runs"].as_array().cloned().unwrap_or_default();
    let detail = payload["detail"].as_str().unwrap_or("table");
    let table = {
        let mut t = vec![
            "| Run | Run ID | Projection Dims | Epsilon Range | Nodes | Edges | Components | Giant Fraction |".to_owned(),
            "|---|---|---|---|---|---|---|---|".to_owned(),
        ];
        for row in &rows {
            t.push(format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} |",
                metric(&row["run_index"]),
                metric(&row["run_id"]),
                row["projection_dimensions"],
                metric(&row["epsilon"]),
                metric(&row["nodes"]),
                metric(&row["edges"]),
                metric(&row["components"]),
                pct(&row["giant_fraction"])
            ));
        }
        t.join("\n")
    };
    if rows.is_empty() {
        return format!("No sweeps persisted yet.\n\n{table}");
    }
    if detail == "table" {
        return table;
    }
    let summary = &payload["summary"];
    let mut lines = vec![
        "# Sweep History Summary".to_owned(),
        String::new(),
        format!("- Runs: {}", rows.len()),
    ];
    if let Some(obs) = summary["observations"].as_array() {
        lines.extend(["".into(), "## Observations".into()]);
        lines.extend(obs.iter().map(|o| format!("- {}", metric(o))));
    }
    if let Some(r) = summary["rationale"].as_str() {
        if !r.is_empty() {
            lines.extend(["".into(), "## Rationale".into(), format!("- {r}")]);
        }
    }
    let summary_md = lines.join("\n");
    if detail == "summary" {
        return summary_md;
    }
    format!("# Sweep History\n\n{table}\n\n{summary_md}")
}

/// `compare_sweeps` table.
pub fn compare(payload: &Value) -> String {
    let a = &payload["run_a"];
    let b = &payload["run_b"];
    let (ma, mb) = (&a["metrics"], &b["metrics"]);
    [
        "### Sweep Comparison".to_owned(),
        String::new(),
        format!("- Run A: {}", metric(&a["run_id"])),
        format!("- Run B: {}", metric(&b["run_id"])),
        format!("- Dataset A: {}", metric(&a["dataset_id"])),
        format!("- Dataset B: {}", metric(&b["dataset_id"])),
        String::new(),
        "| Field | Run A | Run B |".to_owned(),
        "|---|---|---|".to_owned(),
        format!(
            "| projection_dimensions | {} | {} |",
            a["projection_dimensions"], b["projection_dimensions"]
        ),
        format!(
            "| epsilon | {} | {} |",
            metric(&a["epsilon"]),
            metric(&b["epsilon"])
        ),
        format!(
            "| threshold | {} | {} |",
            metric(&a["threshold"]),
            metric(&b["threshold"])
        ),
        format!(
            "| nodes | {} | {} |",
            metric(&ma["n_nodes"]),
            metric(&mb["n_nodes"])
        ),
        format!(
            "| edges | {} | {} |",
            metric(&ma["n_edges"]),
            metric(&mb["n_edges"])
        ),
        format!(
            "| components | {} | {} |",
            metric(&ma["component_count"]),
            metric(&mb["component_count"])
        ),
        format!(
            "| giant_fraction | {} | {} |",
            pct(&ma["giant_fraction"]),
            pct(&mb["giant_fraction"])
        ),
        format!(
            "| weight_p95 | {} | {} |",
            metric(&ma["weight_p95"]),
            metric(&mb["weight_p95"])
        ),
    ]
    .join("\n")
}
