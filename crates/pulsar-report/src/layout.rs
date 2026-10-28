//! `nx.spring_layout(G, seed=42, k=...)` for the report's topology figure:
//! numpy `RandomState(seed).rand(n, 2)` start positions, then networkx's
//! Fruchterman-Reingold ("force" method) with linear cooling, then
//! `rescale_layout` to `[-1, 1]`. The force accumulation reproduces numpy's
//! `einsum` exactly (sequential fused multiply-add): the dynamics amplify
//! last-bit differences into visibly different figures.
//!
//! networkx 3.7 switches to an L-BFGS "energy" method at 500 nodes; this
//! port keeps the force method at every size (the figure is a sampled sketch
//! and its coordinates are arbitrary either way).

use pulsar_data::pyrandom::PyRandom;

const ITERATIONS: usize = 50;
const THRESHOLD: f64 = 1e-4;

/// 2-D positions of `n` nodes; `weight(i, j)` is the adjacency weight (0 when
/// there is no edge).
pub fn spring_layout(n: usize, weights: &[Vec<(usize, f64)>], k: f64, seed: u32) -> Vec<[f64; 2]> {
    match n {
        0 => return vec![],
        1 => return vec![[0.0, 0.0]],
        _ => {}
    }
    let mut rng = PyRandom::numpy_legacy(seed);
    let mut pos: Vec<[f64; 2]> = (0..n).map(|_| [rng.random(), rng.random()]).collect();
    // Dense adjacency rows (the figure is capped at 1000 nodes).
    let mut a = vec![vec![0.0; n]; n];
    for (i, row) in weights.iter().enumerate() {
        for &(j, w) in row {
            a[i][j] = w;
        }
    }
    let span = |c: usize| {
        let (lo, hi) = pos
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
                (lo.min(p[c]), hi.max(p[c]))
            });
        hi - lo
    };
    let mut t = span(0).max(span(1)) * 0.1;
    let dt = t / (ITERATIONS + 1) as f64;
    for _ in 0..ITERATIONS {
        let mut delta_pos = vec![[0.0; 2]; n];
        for i in 0..n {
            let mut disp = [0.0; 2];
            for j in 0..n {
                let d = [pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]];
                let distance = (d[0] * d[0] + d[1] * d[1]).sqrt().max(0.01);
                let f = k * k / (distance * distance) - a[i][j] * distance / k;
                // numpy's einsum("ijk,ij->ik") accumulates with fused
                // multiply-add, and the layout is chaotic enough that the
                // rounding shows: match it exactly.
                disp[0] = d[0].mul_add(f, disp[0]);
                disp[1] = d[1].mul_add(f, disp[1]);
            }
            let length = (disp[0] * disp[0] + disp[1] * disp[1]).sqrt().max(0.01);
            delta_pos[i] = [disp[0] * (t / length), disp[1] * (t / length)];
        }
        let mut norm = 0.0;
        for (p, d) in pos.iter_mut().zip(&delta_pos) {
            p[0] += d[0];
            p[1] += d[1];
            norm += d[0] * d[0] + d[1] * d[1];
        }
        t -= dt;
        if norm.sqrt() / (n as f64) < THRESHOLD {
            break;
        }
    }
    // rescale_layout(scale=1): centre each axis, divide by the largest |coordinate|.
    let mut lim: f64 = 0.0;
    for c in 0..2 {
        let mean = pos.iter().map(|p| p[c]).sum::<f64>() / n as f64;
        for p in &mut pos {
            p[c] -= mean;
        }
    }
    for p in &pos {
        lim = lim.max(p[0].abs()).max(p[1].abs());
    }
    if lim > 0.0 {
        for p in &mut pos {
            p[0] /= lim;
            p[1] /= lim;
        }
    }
    pos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numpy_random_state_42_matches() {
        let mut rng = PyRandom::numpy_legacy(42);
        let got: Vec<f64> = (0..4).map(|_| rng.random()).collect();
        assert_eq!(
            got,
            vec![
                0.3745401188473625,
                0.9507143064099162,
                0.7319939418114051,
                0.5986584841970366
            ]
        );
    }

    #[test]
    fn two_connected_nodes_end_on_the_unit_box() {
        let w = vec![vec![(1, 1.0)], vec![(0, 1.0)]];
        let pos = spring_layout(2, &w, 1.0, 42);
        let max = pos
            .iter()
            .flat_map(|p| p.iter())
            .fold(0.0f64, |m, v| m.max(v.abs()));
        assert!((max - 1.0).abs() < 1e-12);
        assert!((pos[0][0] + pos[1][0]).abs() < 1e-12, "centred");
    }
}
