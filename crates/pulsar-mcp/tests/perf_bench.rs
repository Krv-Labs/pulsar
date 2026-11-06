//! Perf harness (phase 1: measure only). Run with
//! `PERF_SIZES=5000,20000 PERF_OUT=/path/out.jsonl cargo test --release -p pulsar-mcp --test perf_bench -- --ignored --nocapture`.

#![allow(
    unused_assignments,
    clippy::approx_constant,
    clippy::needless_range_loop
)]

mod common;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use common::{args, call, serve_state, TempRoot};
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, ClientCapabilities, ClientConfig, GetTaskParams,
    Implementation,
};
use serde_json::{json, Value};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn norm(&mut self) -> f64 {
        let (a, b) = (self.next().max(1e-12), self.next());
        (-2.0 * a.ln()).sqrt() * (6.283185307 * b).cos()
    }
    fn pick(&mut self, n: usize) -> usize {
        ((self.next() * n as f64) as usize).min(n - 1)
    }
}

/// Mixed-type clinical-like table. Returns (path, n_cols, numeric feature names).
fn make_csv(dir: &Path, rows: usize, ncols: usize) -> (PathBuf, Vec<String>) {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ rows as u64);
    let k = 6usize; // latent groups
    let n_num = ncols / 2;
    let n_cat = ncols / 6;
    let n_code = ncols / 8;
    let n_flag = ncols / 10;
    let mut header: Vec<String> = vec!["patient_id".into(), "visit".into(), "visit_date".into()];
    let mut num_names = vec![];
    for j in 0..n_num {
        let n = format!("lab_{j:03}");
        num_names.push(n.clone());
        header.push(n);
    }
    for j in 0..n_cat {
        header.push(format!("cat_{j:02}"));
    }
    for j in 0..n_code {
        header.push(format!("code_{j:02}"));
    }
    for j in 0..n_flag {
        header.push(format!("flag_{j:02}"));
    }
    while header.len() < ncols - 1 {
        header.push(format!("extra_{:02}", header.len()));
    }
    header.push("outcome".into());
    let ncols_real = header.len();
    let mut out = header.join(",");
    out.push('\n');
    let loadings: Vec<Vec<f64>> = (0..k)
        .map(|_| (0..n_num).map(|_| rng.norm() * 1.5).collect())
        .collect();
    let mut row = 0usize;
    let mut pid = 0usize;
    while row < rows {
        pid += 1;
        let g = rng.pick(k);
        let visits = 1 + rng.pick(3);
        let base: Vec<f64> = (0..n_num).map(|j| loadings[g][j] + rng.norm()).collect();
        for v in 0..visits {
            if row >= rows {
                break;
            }
            row += 1;
            let mut f: Vec<String> = Vec::with_capacity(ncols_real);
            f.push(token(pid));
            f.push((v + 1).to_string());
            f.push(format!(
                "2023-{:02}-{:02}",
                1 + (pid + v) % 12,
                1 + (pid * 3 + v) % 28
            ));
            for j in 0..n_num {
                // a share of columns carry blanks
                if j % 3 == 0 && rng.next() < 0.1 {
                    f.push(String::new());
                } else if j % 7 == 0 {
                    f.push(format!(
                        "{}",
                        (base[j] * 4.0 + 30.0 + rng.norm() * 3.0).round()
                    ));
                } else {
                    f.push(format!("{:.4}", base[j] + 0.4 * rng.norm()));
                }
            }
            for j in 0..n_cat {
                let card = 3 + j % 5;
                if rng.next() < 0.05 {
                    f.push(String::new());
                } else {
                    let lvl = (g + j) % card;
                    let lvl = if rng.next() < 0.3 {
                        rng.pick(card)
                    } else {
                        lvl
                    };
                    f.push(format!("lvl{lvl}"));
                }
            }
            for j in 0..n_code {
                if j % 2 == 0 {
                    // ICD-like strings, ~120 levels
                    let c = if rng.next() < 0.5 {
                        (g * 17 + j) % 120
                    } else {
                        rng.pick(120)
                    };
                    f.push(format!(
                        "{}{:02}.{}",
                        (b'A' + (c % 20) as u8) as char,
                        c,
                        c % 9
                    ));
                } else {
                    f.push(format!("{}", 1 + (g + rng.pick(3)) % 9)); // integer codes
                }
            }
            for j in 0..n_flag {
                let yes = rng.next() < 0.3 + 0.1 * ((g + j) % 3) as f64;
                f.push(if j % 2 == 0 {
                    (yes as u8).to_string()
                } else {
                    (if yes { "Y" } else { "N" }).to_string()
                });
            }
            while f.len() < ncols_real - 1 {
                f.push(if rng.next() < 0.2 {
                    String::new()
                } else {
                    format!("{:.3}", rng.norm())
                });
            }
            f.push(
                ["recovered", "readmitted", "died"][(g + (rng.next() < 0.2) as usize) % 3]
                    .to_string(),
            );
            out.push_str(&f.join(","));
            out.push('\n');
        }
    }
    let path = dir.join(format!("synthetic_{rows}.csv"));
    std::fs::write(&path, out).unwrap();
    (path, num_names)
}

fn token(i: usize) -> String {
    const ALPHABET: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";
    const CHECK: &[u8; 37] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ*~$=U";
    let mut data = *b"D7000000";
    let mut n = i;
    for slot in data[2..].iter_mut().rev() {
        *slot = ALPHABET[n % 32];
        n /= 32;
    }
    let check = data.iter().fold(0u32, |acc, c| {
        (acc * 32 + ALPHABET.iter().position(|a| a == c).unwrap() as u32) % 37
    });
    let d = std::str::from_utf8(&data).unwrap();
    format!("{}-{}-{}", &d[..2], &d[2..], CHECK[check as usize] as char)
}

fn rss_mb() -> f64 {
    let o = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .unwrap();
    String::from_utf8_lossy(&o.stdout)
        .trim()
        .parse::<f64>()
        .unwrap_or(0.0)
        / 1024.0
}

struct Bench {
    size: usize,
    out: std::fs::File,
    timeout: Duration,
}

type Client = rmcp::service::RunningService<rmcp::service::RoleClient, ()>;

impl Bench {
    /// Time `reps` calls; the first is the cold call. Returns the first result's payload.
    async fn tool(
        &mut self,
        client: &Client,
        label: &str,
        name: &str,
        reps: usize,
        mk: impl Fn(usize) -> Value,
    ) -> Value {
        let mut times = vec![];
        let mut first = Value::Null;
        let (mut chars, mut sc_chars, mut is_err) = (0, 0, false);
        let mut code = String::new();
        for i in 0..reps {
            let a = serde_json::from_value(mk(i)).unwrap();
            let t = Instant::now();
            let r = tokio::time::timeout(self.timeout, call(client, name, a)).await;
            let dt = t.elapsed().as_secs_f64();
            let Ok(r) = r else {
                times.push(dt);
                code = "TIMEOUT".into();
                is_err = true;
                break;
            };
            times.push(dt);
            let payload = r.structured_content.clone().unwrap_or(Value::Null);
            if i == 0 {
                chars = r
                    .content
                    .iter()
                    .filter_map(|c| c.as_text().map(|t| t.text.chars().count()))
                    .sum::<usize>();
                sc_chars = payload.to_string().chars().count();
                is_err = r.is_error == Some(true);
                if is_err {
                    code = payload.to_string().chars().take(300).collect();
                }
                first = payload;
            }
            if is_err {
                break;
            }
        }
        self.record(label, name, &times, chars, sc_chars, &code);
        first
    }

    fn record(
        &mut self,
        label: &str,
        name: &str,
        times: &[f64],
        chars: usize,
        sc: usize,
        err: &str,
    ) {
        let mut s = times[1..].to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let warm = |p: f64| -> Option<f64> {
            if s.is_empty() {
                None
            } else {
                Some(s[((s.len() - 1) as f64 * p).round() as usize])
            }
        };
        let rec = json!({"size": self.size, "label": label, "tool": name, "cold_s": times[0],
            "warm_p50_s": warm(0.5), "warm_p95_s": warm(0.95), "n": times.len(),
            "text_chars": chars, "structured_chars": sc, "err": err, "rss_mb": rss_mb()});
        println!("{rec}");
        writeln!(self.out, "{rec}").unwrap();
    }
}

fn ids(v: &Value, key: &str) -> Vec<i64> {
    v["clusters"]
        .as_array()
        .map(|a| a.iter().filter_map(|c| c[key].as_i64()).collect())
        .unwrap_or_default()
}

async fn run_size(rows: usize, ncols: usize, reps: usize) {
    let root = TempRoot::new("perf");
    let t = Instant::now();
    let (csv, nums) = make_csv(root.path(), rows, ncols);
    eprintln!(
        "generated {rows}x{ncols} in {:?}, {} bytes",
        t.elapsed(),
        std::fs::metadata(&csv).unwrap().len()
    );
    let out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(std::env::var("PERF_OUT").unwrap_or("/tmp/perf.jsonl".into()))
        .unwrap();
    let secs: u64 = std::env::var("PERF_TIMEOUT_S")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(900);
    let mut b = Bench {
        size: rows,
        out,
        timeout: Duration::from_secs(secs),
    };
    let only: Option<String> = std::env::var("PERF_UNTIL").ok();

    let store = pulsar_store::Store::open_at(root.path()).unwrap();
    let state =
        std::sync::Arc::new(pulsar_mcp::ServerState::new(store, common::test_options()).unwrap());
    let client = serve_state(state.clone(), ()).await;
    let r = reps;

    b.tool(&client, "-", "get_workflow_guide", r, |_| json!({}))
        .await;
    let ing = b
        .tool(
            &client,
            "-",
            "ingest_dataset",
            r,
            |_| json!({"path": csv.display().to_string()}),
        )
        .await;
    let ds = ing["dataset_id"].clone();
    b.tool(
        &client,
        "-",
        "characterize_dataset",
        r,
        |_| json!({"dataset_id": ds, "response_format": "json"}),
    )
    .await;
    let probe_cols: Vec<String> = nums
        .iter()
        .take(8)
        .cloned()
        .chain(["cat_00".into(), "code_00".into(), "outcome".into()])
        .collect();
    b.tool(
        &client,
        "-",
        "probe_columns",
        r,
        |_| json!({"dataset_id": ds, "columns": probe_cols, "response_format": "json"}),
    )
    .await;

    let draft = b.tool(&client, "-", "draft_preprocessing_plan", r, |_| json!({"dataset_id": ds, "response_format": "json", "entity_column": "patient_id", "time_column": "visit"})).await;
    let draft_full = b
        .tool(
            &client,
            "detail=full",
            "draft_preprocessing_plan",
            1,
            |_| json!({"dataset_id": ds, "response_format": "json", "detail": "full"}),
        )
        .await;
    let _ = draft_full;
    let _cfg0 = "run:\n  name: t\npreprocessing:\n  drop_columns: [patient_id]\nsweep:\n  projection:\n    method: jl\n    dimensions: [2]\n    seed: [42]\n  ball_mapper:\n    epsilon: [0.5]\n";
    let plan_id = draft["plan_id"].clone();
    b.tool(&client, "-", "edit_preprocessing_plan", r, |i| json!({"plan_id": plan_id, "edits": [{"column": "lab_003", "impute_method": if i % 2 == 0 {"fill_mean"} else {"fill_median"}}], "response_format": "json"})).await;

    let created = b
        .tool(
            &client,
            "-",
            "create_config",
            r,
            |_| json!({"dataset_id": ds}),
        )
        .await;
    let cfg = created["config_yaml"].as_str().unwrap_or("").to_owned();
    let light = std::env::var("PERF_LIGHT").is_ok();
    let sweep_cfg: Option<String> = light.then(|| {
        let steps = std::env::var("PERF_STEPS").unwrap_or("4".into());
        let one = cfg
            .replace("[42, 7, 13]", "[42]")
            .replace("steps: 24", &format!("steps: {steps}"));
        // Optional coarse grid: one epsilon at `max * PERF_EPS_SCALE` (large balls, few centres).
        let one = match std::env::var("PERF_EPS_SCALE")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
        {
            Some(scale) => {
                let key = "max: ";
                let i = one.find(key).unwrap() + key.len();
                let end = i + one[i..].find('\n').unwrap();
                let max: f64 = one[i..end].trim().parse().unwrap();
                let lo = one.find("min: ").unwrap();
                let lo_end = lo + one[lo..].find('\n').unwrap();
                format!(
                    "{}min: {:.4}{}max: {:.4}{}",
                    &one[..lo],
                    max * scale,
                    &one[lo_end..i - key.len()],
                    max * scale,
                    &one[end..]
                )
                .replace(&format!("steps: {steps}"), "steps: 1")
            }
            None => one,
        };
        // First projection dimension only.
        let key = "dimensions:\n      values: [";
        match one.find(key) {
            Some(i) => {
                let start = i + key.len();
                let end = start + one[start..].find(']').unwrap();
                let first = one[start..end].split(',').next().unwrap().trim().to_owned();
                format!("{}{}{}", &one[..start], first, &one[end..])
            }
            None => one,
        }
    });
    let _ = std::fs::write(format!("/tmp/perf_cfg_{rows}.yaml"), &cfg);
    let sweep_args = |detail: &str| {
        let mut v = json!({"dataset_id": ds, "response_format": "json", "detail": detail});
        if let Some(c) = &sweep_cfg {
            v["config_yaml"] = json!(c);
        }
        v
    };
    let sr = if light { 1 } else { r };
    b.tool(
        &client,
        "-",
        "validate_preprocessing_config",
        r,
        |_| json!({"dataset_id": ds, "config_yaml": cfg}),
    )
    .await;
    b.tool(
        &client,
        "-",
        "validate_config",
        r,
        |_| json!({"dataset_id": ds, "config_yaml": cfg}),
    )
    .await;
    b.tool(
        &client,
        "-",
        "refine_config",
        r,
        |_| json!({"config_yaml": cfg, "overrides": {}, "response_format": "json"}),
    )
    .await;
    if only.as_deref() == Some("config") {
        return;
    }

    // Cold sync sweep, then warm (projection cache hit).
    let sweep = b
        .tool(&client, "sync", "run_topological_sweep", sr, |_| {
            sweep_args("summary")
        })
        .await;
    let run_id = sweep["run_id"].clone();
    if !light {
        b.tool(
            &client,
            "detail=full",
            "run_topological_sweep",
            if light { 1 } else { 2 },
            |_| sweep_args("full"),
        )
        .await;
    }
    b.tool(&client, "-", "get_job", r, |_| json!({"run_id": run_id}))
        .await;
    b.tool(
        &client,
        "-",
        "wait_job",
        r,
        |_| json!({"run_id": run_id, "timeout_s": 5.0}),
    )
    .await;
    b.tool(
        &client,
        "summary",
        "diagnose_cosmic_graph",
        r,
        |_| json!({"run_id": run_id, "response_format": "json"}),
    )
    .await;
    b.tool(
        &client,
        "full",
        "diagnose_cosmic_graph",
        r,
        |_| json!({"run_id": run_id, "detail": "full", "response_format": "json"}),
    )
    .await;
    b.tool(
        &client,
        "summary",
        "get_threshold_stability_curve",
        r,
        |_| json!({"run_id": run_id, "response_format": "json"}),
    )
    .await;
    b.tool(
        &client,
        "full",
        "get_threshold_stability_curve",
        r,
        |_| json!({"run_id": run_id, "detail": "full", "response_format": "json"}),
    )
    .await;
    for d in ["summary", "nodes", "edges", "full"] {
        b.tool(
            &client,
            d,
            "get_topological_skeleton",
            r,
            |_| json!({"run_id": run_id, "detail": d, "max_edges": 500, "max_nodes": 500}),
        )
        .await;
    }
    b.tool(
        &client,
        "table",
        "get_sweep_history",
        r,
        |_| json!({"detail": "table", "response_format": "json"}),
    )
    .await;
    b.tool(
        &client,
        "full",
        "get_sweep_history",
        r,
        |_| json!({"detail": "full", "response_format": "json"}),
    )
    .await;

    b.tool(
        &client,
        "estimate",
        "create_graph_artifact",
        r,
        |_| json!({"kind": "spectral_sparsifier", "run_id": run_id, "estimate_only": true}),
    )
    .await;
    b.tool(
        &client,
        "build",
        "create_graph_artifact",
        1,
        |_| json!({"kind": "spectral_sparsifier", "run_id": run_id}),
    )
    .await;

    // Dossier and cluster tools.
    let dossier = b
        .tool(
            &client,
            "summary",
            "generate_cluster_dossier",
            r,
            |_| json!({"response_format": "json"}),
        )
        .await;
    let ca = dossier["cluster_assignment_id"].clone();
    let cids = {
        let st = pulsar_store::Store::open_at(root.path()).unwrap();
        let id = pulsar_store::ClusterAssignmentId::parse(ca.as_str().unwrap()).unwrap();
        let mut v = st.cluster_labels(&id).unwrap();
        v.sort();
        v.dedup();
        v
    };
    let _ = ids(&dossier, "cluster_id");
    if let Ok(list) = std::env::var("PERF_PROFILE") {
        for item in list.split(',') {
            let (tool, a): (&str, Value) = match item {
                "skeleton" => (
                    "get_topological_skeleton",
                    json!({"run_id": run_id, "detail": "summary"}),
                ),
                "skeleton_edges" => (
                    "get_topological_skeleton",
                    json!({"run_id": run_id, "detail": "edges", "max_edges": 500}),
                ),
                "curve" => (
                    "get_threshold_stability_curve",
                    json!({"run_id": run_id, "detail": "full", "response_format": "json"}),
                ),
                "curve_summary" => (
                    "get_threshold_stability_curve",
                    json!({"run_id": run_id, "response_format": "json"}),
                ),
                "report" => ("export_html_report", json!({"dataset_id": ds})),
                "diagnose" => (
                    "diagnose_cosmic_graph",
                    json!({"run_id": run_id, "response_format": "json"}),
                ),
                "dossier" => (
                    "generate_cluster_dossier",
                    json!({"response_format": "json"}),
                ),
                "merge" => ("merge_small_groups", json!({"cluster_assignment_id": ca})),
                "bundle" => (
                    "export_dataset_bundle",
                    json!({"output_dir": root.path().join("pb").display().to_string()}),
                ),
                "detail" => (
                    "get_cluster_detail",
                    json!({"cluster_assignment_id": ca, "response_format": "json"}),
                ),
                "sweep" => ("run_topological_sweep", sweep_args("summary")),
                _ => continue,
            };
            eprintln!("PROFILE {item}");
            b.tool(&client, item, tool, 6, |_| a.clone()).await;
            eprintln!("PROFILE-END {item}");
        }
        return;
    }
    eprintln!("clusters: {cids:?}");
    b.tool(
        &client,
        "standard",
        "generate_cluster_dossier",
        2,
        |_| json!({"detail": "standard", "response_format": "json"}),
    )
    .await;
    b.tool(
        &client,
        "full",
        "generate_cluster_dossier",
        2,
        |_| json!({"detail": "full", "response_format": "json"}),
    )
    .await;
    let ca2 = ca.clone();
    b.tool(
        &client,
        "-",
        "get_cluster_detail",
        r,
        |_| json!({"cluster_assignment_id": ca2, "response_format": "json"}),
    )
    .await;
    b.tool(
        &client,
        "-",
        "merge_small_groups",
        r,
        |_| json!({"cluster_assignment_id": ca}),
    )
    .await;
    let c0 = cids[0];
    let c1 = cids[1];
    b.tool(
        &client,
        "summary",
        "get_cluster_profile",
        r,
        |_| json!({"cluster_id": c0, "response_format": "json"}),
    )
    .await;
    b.tool(
        &client,
        "full",
        "get_cluster_profile",
        r,
        |_| json!({"cluster_id": c0, "detail": "full", "response_format": "json"}),
    )
    .await;
    let feats: Vec<String> = nums.iter().take(6).cloned().collect();
    b.tool(
        &client,
        "-",
        "get_feature_signal",
        r,
        |_| json!({"feature_names": feats, "response_format": "json"}),
    )
    .await;
    b.tool(&client, "-", "get_cluster_signal_matrix", r, |_| json!({}))
        .await;
    b.tool(
        &client,
        "-",
        "compare_clusters",
        r,
        |_| json!({"cluster_a": c0, "cluster_b": c1}),
    )
    .await;

    // Exports.
    let names: serde_json::Map<String, Value> = cids
        .iter()
        .map(|c| (c.to_string(), json!(format!("group {c}"))))
        .collect();
    let outp = root.path().join("labeled.csv");
    b.tool(&client, "-", "export_labeled_data", r, |_| json!({"cluster_assignment_id": ca2, "cluster_names": names, "output_path": outp.display().to_string()})).await;
    let bdir = root.path().join("bundle");
    b.tool(
        &client,
        "-",
        "export_dataset_bundle",
        r,
        |i| json!({"output_dir": bdir.display().to_string(), "slug": format!("b{i}")}),
    )
    .await;
    b.tool(
        &client,
        "-",
        "export_html_report",
        r,
        |_| json!({"dataset_id": ds}),
    )
    .await;

    // Longitudinal.
    let ent_feats: Vec<String> = nums.iter().take(12).cloned().collect();
    b.tool(&client, "-", "sample_longitudinal_entities", r, |_| json!({"dataset_id": ds, "entity_column": "patient_id", "max_entities": 500, "min_observations": 2})).await;
    let panel_yaml = "run:\n  name: panel\nsweep:\n  projection:\n    method: jl\n    dimensions:\n      values: [4]\n    seed:\n      values: [42]\n  ball_mapper:\n    epsilon:\n      values: [0.5, 1.0]\ncosmic_graph:\n  construction: exact\n";
    let lg = b.tool(&client, "-", "build_longitudinal_graph", r, |_| json!({"dataset_id": ds, "entity_column": "patient_id", "time_column": "visit", "feature_columns": ent_feats, "config_yaml": panel_yaml, "response_format": "json"})).await;
    let lid = lg["longitudinal_id"].clone();
    b.tool(
        &client,
        "-",
        "diagnose_longitudinal_graph",
        r,
        |_| json!({"longitudinal_id": lid, "response_format": "json"}),
    )
    .await;
    b.tool(
        &client,
        "-",
        "get_trajectory_archetypes",
        r,
        |_| json!({"longitudinal_id": lid, "response_format": "json"}),
    )
    .await;
    let ent = "x";
    b.tool(
        &client,
        "-",
        "get_cross_time_neighbors",
        r,
        |_| json!({"longitudinal_id": lid, "entity_id": ent, "observation_id": 0, "response_format": "json"}),
    )
    .await;
    for m in ["complexity", "transition", "sequence", "levenshtein", "dtw"] {
        b.tool(
            &client,
            m,
            "classify_trajectories",
            r,
            |_| json!({"longitudinal_id": lid, "method": m, "response_format": "json"}),
        )
        .await;
    }
    let other = b
        .tool(&client, "-", "run_topological_sweep", 1, |_| {
            sweep_args("summary")
        })
        .await;
    b.tool(
        &client,
        "-",
        "compare_sweeps",
        r,
        |_| json!({"run_a": run_id, "run_b": other["run_id"]}),
    )
    .await;
    b.tool(&client, "-", "get_runtime_context", r, |_| json!({}))
        .await;
    client.cancel().await.unwrap();

    if std::env::var("PERF_SKIP_TASK").is_ok() {
        return;
    }
    // Task and async sweeps on a fresh server (cold projection cache).
    let root2 = TempRoot::new("perf-task");
    let store = pulsar_store::Store::open_at(root2.path()).unwrap();
    let state2 =
        std::sync::Arc::new(pulsar_mcp::ServerState::new(store, common::test_options()).unwrap());
    let cfgc = ClientConfig::new(
        ClientCapabilities::builder().enable_tasks().build(),
        Implementation::new("perf", "0"),
    );
    let tclient = serve_state(state2.clone(), cfgc).await;
    let ing = call(
        &tclient,
        "ingest_dataset",
        args(&[("path", json!(csv.display().to_string()))]),
    )
    .await;
    let ds2 = ing.structured_content.unwrap()["dataset_id"].clone();
    call(
        &tclient,
        "create_config",
        args(&[("dataset_id", ds2.clone())]),
    )
    .await;
    let mut p = CallToolRequestParams::new("run_topological_sweep");
    p.arguments = Some(args(&[
        ("dataset_id", ds2.clone()),
        ("response_format", json!("json")),
    ]));
    let t = Instant::now();
    let resp = tclient.call_tool_once(p).await.unwrap();
    if let CallToolResponse::Task(created) = resp {
        let id = created.task.task_id.clone();
        let t_created = t.elapsed().as_secs_f64();
        let mut size = 0;
        loop {
            let got = tclient
                .get_task(GetTaskParams::new(id.clone()))
                .await
                .unwrap();
            if got.task.status().is_terminal() {
                if let rmcp::model::TaskPayload::Completed { result } = got.task.payload {
                    size = Value::Object(result).to_string().chars().count();
                }
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        b.record(
            "task(cold, create+poll)",
            "run_topological_sweep",
            &[t.elapsed().as_secs_f64()],
            0,
            size,
            &format!("handle_in={t_created:.3}s"),
        );
    }
    tclient.cancel().await.unwrap();

    let root3 = TempRoot::new("perf-async");
    let store = pulsar_store::Store::open_at(root3.path()).unwrap();
    let state3 =
        std::sync::Arc::new(pulsar_mcp::ServerState::new(store, common::test_options()).unwrap());
    let aclient = serve_state(state3, ()).await;
    let ing = call(
        &aclient,
        "ingest_dataset",
        args(&[("path", json!(csv.display().to_string()))]),
    )
    .await;
    let ds3 = ing.structured_content.unwrap()["dataset_id"].clone();
    call(
        &aclient,
        "create_config",
        args(&[("dataset_id", ds3.clone())]),
    )
    .await;
    let t = Instant::now();
    let sub = call(
        &aclient,
        "run_topological_sweep",
        args(&[("dataset_id", ds3), ("execution", json!("async"))]),
    )
    .await;
    let rid = sub.structured_content.unwrap()["run_id"].clone();
    let submit = t.elapsed().as_secs_f64();
    let w = call(
        &aclient,
        "wait_job",
        args(&[("run_id", rid), ("timeout_s", json!(secs as f64))]),
    )
    .await;
    b.record(
        "async(cold, submit+wait)",
        "run_topological_sweep",
        &[t.elapsed().as_secs_f64()],
        0,
        w.structured_content.unwrap().to_string().len(),
        &format!("submit={submit:.3}s"),
    );
    aclient.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn perf_bench() {
    let sizes = std::env::var("PERF_SIZES").unwrap_or("5000".into());
    let reps: usize = std::env::var("PERF_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    for s in sizes.split(',') {
        let rows: usize = s.parse().unwrap();
        let ncols = match rows {
            r if r <= 5000 => 40,
            r if r <= 20000 => 60,
            _ => 80,
        };
        run_size(rows, ncols, reps).await;
    }
}

/// Stage timers for Ball Mapper on a wide embedding (the linear path, d > 16).
/// `PERF_BM_N`, `PERF_BM_D` override the cloud shape.
#[test]
#[ignore]
fn perf_ballmapper_stages() {
    use ndarray::Array2;
    let n: usize = std::env::var("PERF_BM_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20000);
    let d: usize = std::env::var("PERF_BM_D")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(213);
    let mut rng = Rng(7);
    let k_groups = 6;
    let centres: Vec<Vec<f64>> = (0..k_groups)
        .map(|_| (0..d).map(|_| rng.norm() * 1.5).collect())
        .collect();
    let x = Array2::from_shape_fn((n, d), |(i, j)| centres[i % k_groups][j] + rng.norm());
    let l2 =
        |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum() };
    let l2_abandon = |a: &[f64], b: &[f64], lim: f64| -> bool {
        let mut s = 0.0;
        for (ca, cb) in a.chunks(16).zip(b.chunks(16)) {
            for (x, y) in ca.iter().zip(cb) {
                s += (x - y) * (x - y);
            }
            if s > lim {
                return false;
            }
        }
        s <= lim
    };
    // eps from sampled pair distances: a low and a median quantile.
    let mut sample: Vec<f64> = (0..2000)
        .map(|i| {
            l2(
                x.row(i).to_slice().unwrap(),
                x.row((i * 7 + 13) % n).to_slice().unwrap(),
            )
            .sqrt()
        })
        .collect();
    sample.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for q in [0.70, 0.80, 0.90] {
        let eps = q * (2.0 * d as f64).sqrt();
        let eps_sq = eps * eps;
        let t = Instant::now();
        let mut centres_idx: Vec<usize> = vec![];
        'o: for i in 0..n {
            let pi = x.row(i).to_slice().unwrap();
            for &c in &centres_idx {
                if l2(pi, x.row(c).to_slice().unwrap()) <= eps_sq {
                    continue 'o;
                }
            }
            centres_idx.push(i);
        }
        let t_centres = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let mut centres_abandon: Vec<usize> = vec![];
        'o2: for i in 0..n {
            let pi = x.row(i).to_slice().unwrap();
            for &c in &centres_abandon {
                if l2_abandon(pi, x.row(c).to_slice().unwrap(), eps_sq) {
                    continue 'o2;
                }
            }
            centres_abandon.push(i);
        }
        let t_centres_abandon = t.elapsed().as_secs_f64();
        assert_eq!(centres_idx, centres_abandon);
        let t = Instant::now();
        let nodes: Vec<Vec<usize>> = centres_idx
            .iter()
            .map(|&c| {
                let pc = x.row(c).to_slice().unwrap();
                (0..n)
                    .filter(|&i| l2(x.row(i).to_slice().unwrap(), pc) <= eps_sq)
                    .collect()
            })
            .collect();
        let t_member = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let nodes2: Vec<Vec<usize>> = {
            let threads = 8;
            let chunk = centres_idx.len().div_ceil(threads).max(1);
            let (x, l2_abandon) = (&x, &l2_abandon);
            std::thread::scope(|sc| {
                let handles: Vec<_> = centres_idx
                    .chunks(chunk)
                    .map(|cs| {
                        sc.spawn(move || {
                            cs.iter()
                                .map(|&c| {
                                    let pc = x.row(c).to_slice().unwrap();
                                    (0..n)
                                        .filter(|&i| {
                                            l2_abandon(x.row(i).to_slice().unwrap(), pc, eps_sq)
                                        })
                                        .collect::<Vec<usize>>()
                                })
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .flat_map(|h| h.join().unwrap())
                    .collect()
            })
        };
        let t_member_par_abandon = t.elapsed().as_secs_f64();
        assert_eq!(nodes, nodes2);
        let t = Instant::now();
        let mut edges = 0usize;
        for a in 0..nodes.len() {
            let set_a: std::collections::HashSet<usize> = nodes[a].iter().copied().collect();
            for b in (a + 1)..nodes.len() {
                if nodes[b].iter().any(|v| set_a.contains(v)) {
                    edges += 1;
                }
            }
        }
        let t_edges = t.elapsed().as_secs_f64();
        let mean_ball = nodes.iter().map(Vec::len).sum::<usize>() as f64 / nodes.len() as f64;
        let line = json!({"n": n, "d": d, "q": q, "eps": eps, "k": centres_idx.len(), "mean_ball": mean_ball, "edges": edges,
            "centres_s": t_centres, "centres_abandon_s": t_centres_abandon, "member_serial_s": t_member,
            "member_par_abandon_s": t_member_par_abandon, "edges_s": t_edges});
        println!("{line}");
    }
}
