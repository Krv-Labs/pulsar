//! Parity for `CosmicTrajectory` (`pulsar/representations/trajectory.py`):
//! the Rust port must reproduce what
//! `generate_trajectory.py` (Python generator, see `tests/fixtures/golden/README.md`) froze.
//!
//! Three builds (aligned + MinHash + JL, ragged + exact + PCA with a
//! similarity threshold, positional + unscaled): observations, balls and
//! their members, CO_SIMILAR edges, meta, memberships, cross-time edges and
//! component labels at three thresholds; plus the input-check messages.

use ndarray::Array2;
use pulsar_data::Key;
use pulsar_parity::{assert_close, golden_dir, Manifest};
use pulsar_pipeline::{CosmicTrajectory, EntityIds};
use serde_json::{json, Value};

fn fixture() -> Value {
    let path = golden_dir().join("temporal/trajectory.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn key(v: &Value) -> Key {
    match v {
        Value::String(s) => Key::Text(s.clone()),
        Value::Bool(b) => Key::Bool(*b),
        v if v.is_i64() => Key::Int(v.as_i64().unwrap()),
        v => Key::Float(v.as_f64().unwrap()),
    }
}

fn keys(v: &Value) -> Vec<Key> {
    v.as_array().unwrap().iter().map(key).collect()
}

fn snapshot(m: &Value) -> Array2<f64> {
    let s: Vec<usize> = serde_json::from_value(m["shape"].clone()).unwrap();
    let d: Vec<f64> = serde_json::from_value(m["data"].clone()).unwrap();
    Array2::from_shape_vec((s[0], s[1]), d).unwrap()
}

fn build(input: &Value) -> CosmicTrajectory {
    let snapshots: Vec<Array2<f64>> = input["snapshots"]
        .as_array()
        .unwrap()
        .iter()
        .map(snapshot)
        .collect();
    let config = pulsar_config::from_yaml_str(input["config_yaml"].as_str().unwrap()).unwrap();
    let ids = match input["ids_kind"].as_str().unwrap() {
        "shared" => EntityIds::Shared(keys(&input["ids"])),
        "per_snapshot" => {
            EntityIds::PerSnapshot(input["ids"].as_array().unwrap().iter().map(keys).collect())
        }
        _ => EntityIds::Positional,
    };
    let stamps = (!input["timestamps"].is_null()).then(|| keys(&input["timestamps"]));
    CosmicTrajectory::from_snapshots(
        &snapshots,
        &config,
        &ids,
        stamps.as_deref(),
        input["similarity_threshold"].as_f64().unwrap(),
        input["scale"].as_bool().unwrap(),
    )
    .unwrap()
}

#[test]
fn builds_reproduce_the_python_trajectory() {
    let fx = fixture();
    let rtol = Manifest::load().rtol("trajectory");
    for case in fx["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let ct = build(&case["input"]);

        let obs: Vec<Value> = ct.obs.iter().map(|o| json!(o)).collect();
        assert_eq!(Value::Array(obs), case["obs"], "{name}: obs");
        assert_eq!(
            ct.n_entities() as u64,
            case["n_entities"].as_u64().unwrap(),
            "{name}"
        );
        assert_eq!(
            ct.n_times() as u64,
            case["n_times"].as_u64().unwrap(),
            "{name}"
        );

        let balls: Vec<Value> = ct
            .balls
            .iter()
            .map(|b| json!({"eps": b.eps, "dim": b.dim, "size": b.members.len()}))
            .collect();
        assert_eq!(Value::Array(balls), case["balls"], "{name}: balls");
        let members: Vec<Value> = ct.balls.iter().map(|b| json!(b.members)).collect();
        assert_eq!(Value::Array(members), case["members"], "{name}: members");
        assert_eq!(
            json!(ct.memberships()),
            case["memberships"],
            "{name}: memberships"
        );
        assert_eq!(ct.meta, case["meta"], "{name}: meta");

        let mut edges = ct.similarity.edges.clone();
        edges.sort_by_key(|&(i, j, _)| (i, j));
        let want = case["edges"].as_array().unwrap();
        assert_eq!(edges.len(), want.len(), "{name}: edge count");
        let ends = |e: &Value| {
            (
                e[0].as_u64().unwrap() as usize,
                e[1].as_u64().unwrap() as usize,
            )
        };
        assert!(
            edges.iter().zip(want).all(|(g, w)| (g.0, g.1) == ends(w)),
            "{name}: edge endpoints"
        );
        let got_w: Vec<f64> = edges.iter().map(|e| e.2).collect();
        let want_w: Vec<f64> = want.iter().map(|e| e[2].as_f64().unwrap()).collect();
        assert_close(&format!("{name} weights"), &got_w, &want_w, rtol);

        for (th, labels) in case["cluster_labels"].as_object().unwrap() {
            let t: f64 = th.parse().unwrap();
            assert_eq!(
                json!(ct.cluster_labels(t)),
                *labels,
                "{name}: labels @ {th}"
            );
            let mut cross: Vec<[usize; 2]> = ct
                .cross_time(t)
                .edges
                .iter()
                .map(|&(i, j, _)| [i, j])
                .collect();
            cross.sort_unstable();
            assert_eq!(json!(cross), case["cross_time"][th], "{name}: cross @ {th}");
        }
    }
}

#[test]
fn input_checks_report_the_python_messages() {
    let fx = fixture();
    let errors = &fx["errors"];
    let config =
        pulsar_config::from_yaml_str(fx["cases"][2]["input"]["config_yaml"].as_str().unwrap())
            .unwrap();
    let z = |r: usize, c: usize| Array2::<f64>::zeros((r, c));
    let run = |snaps: &[Array2<f64>], ids: EntityIds, stamps: Option<&[Key]>, th: f64| {
        CosmicTrajectory::from_snapshots(snaps, &config, &ids, stamps, th, true)
            .unwrap_err()
            .to_string()
    };
    let want = |k: &str| errors[k].as_str().unwrap().to_owned();
    assert_eq!(run(&[], EntityIds::Positional, None, 0.0), want("empty"));
    assert_eq!(
        run(&[z(2, 3), z(2, 4)], EntityIds::Positional, None, 0.0),
        want("widths")
    );
    assert_eq!(
        run(&[z(0, 3)], EntityIds::Positional, None, 0.0),
        want("no_rows")
    );
    assert_eq!(
        run(
            &[z(2, 3), z(3, 3)],
            EntityIds::Shared(vec![Key::Int(1), Key::Int(2)]),
            None,
            0.0
        ),
        want("id_mismatch")
    );
    assert_eq!(
        run(
            &[z(2, 3), z(3, 3)],
            EntityIds::PerSnapshot(vec![vec![Key::Int(1), Key::Int(2)]]),
            None,
            0.0
        ),
        want("slices")
    );
    assert_eq!(
        run(
            &[z(2, 3), z(3, 3)],
            EntityIds::Positional,
            Some(&[Key::Int(1)]),
            0.0
        ),
        want("timestamps")
    );
    assert_eq!(
        run(&[z(2, 3)], EntityIds::Positional, None, -0.25),
        want("negative_threshold")
    );
}
