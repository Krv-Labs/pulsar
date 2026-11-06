use ndarray::{Array2, ArrayView2};
use pulsar_core::ballmapper::fit_inner;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashSet;
use std::hint::black_box;
use std::time::Instant;

// The original serial implementation, kept here as an independent output oracle.
fn original_fit(points: ArrayView2<f64>, eps: f64) -> (Vec<Vec<usize>>, Vec<(usize, usize)>) {
    let eps_sq = eps * eps;
    let mut centers = Vec::new();
    'outer: for i in 0..points.nrows() {
        let pi = points.row(i);
        let pi = pi.as_slice().unwrap();
        for &c in &centers {
            let pc = points.row(c);
            let pc = pc.as_slice().unwrap();
            let distance: f64 = pi
                .iter()
                .zip(pc.iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum();
            if distance <= eps_sq {
                continue 'outer;
            }
        }
        centers.push(i);
    }

    let nodes: Vec<Vec<usize>> = centers
        .iter()
        .map(|&c| {
            let pc = points.row(c);
            let pc = pc.as_slice().unwrap();
            (0..points.nrows())
                .filter(|&i| {
                    let pi = points.row(i);
                    let pi = pi.as_slice().unwrap();
                    let distance: f64 = pi
                        .iter()
                        .zip(pc.iter())
                        .map(|(x, y)| (x - y) * (x - y))
                        .sum();
                    distance <= eps_sq
                })
                .collect()
        })
        .collect();
    let mut edges = Vec::new();
    for a in 0..nodes.len() {
        let set_a: HashSet<usize> = nodes[a].iter().copied().collect();
        for (b, node_b) in nodes.iter().enumerate().skip(a + 1) {
            if node_b.iter().any(|x| set_a.contains(x)) {
                edges.push((a, b));
            }
        }
    }
    (nodes, edges)
}

#[test]
fn wide_fit_matches_original_on_240_random_datasets() {
    let mut rng = StdRng::seed_from_u64(0x82a9_12c4);
    for case in 0..240 {
        let n = rng.gen_range(1..=40);
        let d = rng.gen_range(17..=72);
        let groups = rng.gen_range(1..=n.min(10));
        let prototypes: Vec<f64> = (0..groups * d).map(|_| rng.gen_range(-2.0..2.0)).collect();
        let mut data = Vec::with_capacity(n * d);
        for i in 0..n {
            let group = rng.gen_range(0..groups);
            for j in 0..d {
                let noise = if i % 5 == 0 {
                    0.0
                } else {
                    rng.gen_range(-0.2..0.2)
                };
                data.push(prototypes[group * d + j] + noise);
            }
        }
        let points = Array2::from_shape_vec((n, d), data).unwrap();
        let eps = match case % 5 {
            0 => 0.0,
            1 => 0.2 * (d as f64).sqrt(),
            2 => 0.6 * (d as f64).sqrt(),
            3 => 2.0 * (d as f64).sqrt(),
            _ => rng.gen_range(0.0..3.0) * (d as f64).sqrt(),
        };
        assert_eq!(
            fit_inner(points.view(), eps),
            original_fit(points.view(), eps),
            "case {case}, n {n}, d {d}, eps {eps}"
        );
    }
}

#[test]
fn wide_fit_keeps_inclusive_boundary_and_edge_order() {
    let points = Array2::from_shape_fn(
        (3, 17),
        |(i, j)| {
            if j == 0 {
                [0.0, 2.0, 1.0][i]
            } else {
                0.0
            }
        },
    );
    assert_eq!(
        fit_inner(points.view(), 1.0),
        (vec![vec![0, 2], vec![1, 2]], vec![(0, 1)])
    );
}

#[test]
#[ignore = "release-mode performance measurement"]
fn profile_wide_fit_20k() {
    for d in [64, 128, 256] {
        let mut rng = StdRng::seed_from_u64(0x5544_3311 + d as u64);
        let prototypes: Vec<f64> = (0..96 * d).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let points = Array2::from_shape_fn((20_000, d), |(i, j)| {
            prototypes[(i % 96) * d + j] + rng.gen_range(-0.05..0.05)
        });
        let eps = 0.2 * (d as f64).sqrt();
        for sample in 1..=2 {
            let start = Instant::now();
            let result = fit_inner(black_box(points.view()), black_box(eps));
            println!(
                "fit_inner n=20000 d={d} sample={sample} elapsed_ms={} balls={} edges={}",
                start.elapsed().as_millis(),
                result.0.len(),
                result.1.len()
            );
            black_box(result);
        }
    }
}
