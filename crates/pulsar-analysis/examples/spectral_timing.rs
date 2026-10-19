//! Timing probe for the #64 acceptance criterion: spectral clustering on a
//! ~6k-node graph must finish well inside what v0.5 (ARPACK) needed.
//!
//! `cargo run --release -p pulsar-analysis --example spectral_timing [n] [k]`

use std::time::Instant;

use pulsar_analysis::clustering::cluster_spectral_with;
use pulsar_pipeline::Graph;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(6098);
    let k: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(5);
    // k planted communities of equal size, p_in = 0.01 (~60 neighbours per
    // node at n = 6k), p_out = 0.0005, random weights in (0.2, 1].
    let mut rng = StdRng::seed_from_u64(1);
    let mut edges = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            let same = i * k / n == j * k / n;
            let p = if same { 0.01 } else { 0.0005 };
            if rng.gen::<f64>() < p {
                edges.push((i, j, 0.2 + 0.8 * rng.gen::<f64>()));
            }
        }
    }
    let graph = Graph { n, edges };
    println!("n = {n}, k = {k}, edges = {}", graph.n_edges());
    let start = Instant::now();
    let labels = cluster_spectral_with(&graph, k, 42, true).expect("spectral clustering");
    let elapsed = start.elapsed();
    let mut sizes = vec![0usize; k];
    for &l in &labels {
        sizes[l] += 1;
    }
    let truth: Vec<usize> = (0..n).map(|i| i * k / n).collect();
    let agree = (0..n)
        .flat_map(|i| (i + 1..n.min(i + 50)).map(move |j| (i, j)))
        .filter(|&(i, j)| (labels[i] == labels[j]) == (truth[i] == truth[j]))
        .count();
    let pairs = (0..n).map(|i| n.min(i + 50) - i - 1).sum::<usize>();
    println!(
        "LOBPCG + k-means: {:.2?}; cluster sizes {sizes:?}; local pair agreement {:.4}",
        elapsed,
        agree as f64 / pairs as f64
    );
}
