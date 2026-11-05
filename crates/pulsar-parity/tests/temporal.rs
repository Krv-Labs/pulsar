//! Parity for `TemporalCosmicGraph` (`pulsar/representations/temporal.py`):
//! the Rust port must reproduce what
//! `generate_temporal.py` (Python generator, see `tests/fixtures/golden/README.md`) froze.
//!
//! `from_snapshots` on a drifting 30-entity, 4-step panel (tensor and every
//! aggregation), aggregations of hand-made tensors with 1, 2 and 5 steps, and
//! the exact input-check messages.

use ndarray::{ArrayD, IxDyn};
use pulsar_core::TemporalCosmicGraph;
use pulsar_parity::{assert_close, golden_dir, Manifest};
use serde_json::Value;

fn fixture() -> Value {
    let path = golden_dir().join("temporal/temporal.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn shape(m: &Value) -> Vec<usize> {
    m["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

fn data(m: &Value) -> Vec<f64> {
    m["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn dense(m: &Value) -> ArrayD<f64> {
    ArrayD::from_shape_vec(IxDyn(&shape(m)), data(m)).unwrap()
}

fn check(label: &str, got: &ndarray::Array2<f64>, want: &Value, rtol: f64) {
    assert_eq!(got.shape(), shape(want).as_slice(), "{label}: shape");
    assert_close(label, got.as_slice().unwrap(), &data(want), rtol);
}

fn check_aggregations(label: &str, tcg: &TemporalCosmicGraph, want: &Value, rtol: f64) {
    let a = &want["aggregations"];
    check(
        &format!("{label} persistence"),
        &tcg.persistence_graph(None),
        &a["persistence"],
        rtol,
    );
    check(
        &format!("{label} persistence 0.3"),
        &tcg.persistence_graph(Some(0.3)),
        &a["persistence_0.3"],
        rtol,
    );
    check(
        &format!("{label} mean"),
        &tcg.mean_graph(),
        &a["mean"],
        rtol,
    );
    check(
        &format!("{label} recency"),
        &tcg.recency_graph(0.9).unwrap(),
        &a["recency"],
        rtol,
    );
    check(
        &format!("{label} recency 0.5"),
        &tcg.recency_graph(0.5).unwrap(),
        &a["recency_0.5"],
        rtol,
    );
    check(
        &format!("{label} volatility"),
        &tcg.volatility_graph(),
        &a["volatility"],
        rtol,
    );
    check(
        &format!("{label} trend"),
        &tcg.trend_graph(),
        &a["trend"],
        rtol,
    );
    check(
        &format!("{label} change point"),
        &tcg.change_point_graph(),
        &a["change_point"],
        rtol,
    );
}

#[test]
fn from_snapshots_reproduces_the_python_tensor_and_aggregations() {
    let fx = fixture();
    let rtol = Manifest::load().rtol("temporal");
    let p = &fx["pipeline"];
    let snapshots: Vec<ndarray::Array2<f64>> = p["snapshots"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| dense(m).into_dimensionality().unwrap())
        .collect();
    let ints = |v: &Value| -> Vec<u64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap())
            .collect()
    };
    let dims: Vec<usize> = ints(&p["dimensions"])
        .into_iter()
        .map(|d| d as usize)
        .collect();
    let tcg = TemporalCosmicGraph::from_snapshots(
        &snapshots,
        &dims,
        &ints(&p["seeds"]),
        &data(&serde_json::json!({"data": p["epsilons"]})),
        0.0,
    )
    .unwrap();
    assert_eq!(tcg.shape(), (30, 30, 4));
    assert_eq!(tcg.nbytes(), 30 * 30 * 4 * 8);
    let tensor = &p["tensor"];
    assert_eq!(tcg.tensor().shape(), shape(tensor).as_slice());
    assert_close(
        "tensor",
        tcg.tensor().as_slice().unwrap(),
        &data(tensor),
        rtol,
    );
    check_aggregations("pipeline", &tcg, p, rtol);
    check(
        "slice(1, 3) mean",
        &tcg.slice(1, Some(3)).mean_graph(),
        &p["slice_1_3_mean"],
        rtol,
    );
}

#[test]
fn aggregations_of_hand_made_tensors_match_numpy() {
    let fx = fixture();
    let rtol = Manifest::load().rtol("temporal");
    for (k, case) in fx["tensors"].as_array().unwrap().iter().enumerate() {
        let tcg =
            TemporalCosmicGraph::new(dense(&case["tensor"]), case["threshold"].as_f64().unwrap())
                .unwrap();
        let label = format!("tensor {k} (T={})", tcg.T());
        check_aggregations(&label, &tcg, case, rtol);
        check(
            &format!("{label} default threshold"),
            &tcg.persistence_graph(None),
            &case["persistence_default_threshold"],
            rtol,
        );
    }
}

#[test]
fn input_checks_report_the_python_messages() {
    let fx = fixture();
    let errors = &fx["errors"];
    let msg = |e: pulsar_core::PulsarError| e.to_string();
    let want = |name: &str| format!("Invalid parameter: {}", errors[name].as_str().unwrap());
    let zeros = |s: &[usize]| ArrayD::<f64>::zeros(IxDyn(s));

    assert_eq!(
        msg(TemporalCosmicGraph::new(zeros(&[2, 2]), 0.0).unwrap_err()),
        want("not_3d")
    );
    assert_eq!(
        msg(TemporalCosmicGraph::new(zeros(&[2, 3, 1]), 0.0).unwrap_err()),
        want("not_square")
    );
    let grid = |snaps: &[ndarray::Array2<f64>]| {
        TemporalCosmicGraph::from_snapshots(snaps, &[2], &[42], &[0.5], 0.0)
    };
    assert_eq!(msg(grid(&[]).unwrap_err()), want("empty_snapshots"));
    assert_eq!(
        msg(grid(&[
            ndarray::Array2::zeros((4, 3)),
            ndarray::Array2::zeros((5, 3))
        ])
        .unwrap_err()),
        want("inconsistent_nodes")
    );
    let tcg = TemporalCosmicGraph::new(zeros(&[2, 2, 2]), 0.0).unwrap();
    assert_eq!(msg(tcg.recency_graph(1.0).unwrap_err()), want("decay_one"));
    assert_eq!(msg(tcg.recency_graph(0.0).unwrap_err()), want("decay_zero"));
}
