//! Approximate cosmic-graph construction via MinHash signatures + LSH banding.
//!
//! ## What this replaces
//!
//! The exact path (`pseudolaplacian.rs` → `cosmic.rs`) materializes every
//! co-occurring member pair of every ball — cost **Σ_c |B_c|²** (sum of squared ball
//! sizes), which explodes for coarse `epsilon` and across `fit_multi`'s many ball
//! maps. The cosmic edge weight is exactly the **Jaccard similarity of the two
//! points' ball-sets**:
//!
//! ```text
//! W[i,j] = |balls(i) ∩ balls(j)| / |balls(i) ∪ balls(j)|
//! ```
//!
//! MinHash is an unbiased estimator of this Jaccard, computed in **O(d·M)** time
//! (M = total memberships) with no pair materialization. LSH banding then surfaces
//! the high-similarity pairs in sub-quadratic time so only candidates get a weight.
//!
//! ## Statistical guarantee
//!
//! `Ŵ = (1/d)·Σ_r 1[M[r,i] = M[r,j]]` is the mean of `d` Bernoulli(J) trials, so it
//! is unbiased with `Var = J(1−J)/d` — error depends only on the signature depth `d`,
//! **independent of n or the number of balls**. Hoeffding: `P(|Ŵ−J|≥ε) ≤ 2e^{−2dε²}`.
//!
//! ## Determinism
//!
//! Fully seeded. Candidate pairs are deduped and the final edge list is sorted by
//! `(i, j)`, so a given `(balls, n, d, seed)` always yields the identical graph.
//!
//! References: Broder (1997); Broder, Charikar, Frieze, Mitzenmacher (1998/2000);
//! Indyk & Motwani (1998); Leskovec, Rajaraman, Ullman, *Mining of Massive Datasets*
//! ch. 3 (the `b`/`r` S-curve, `τ ≈ (1/b)^{1/r}`); Hoeffding (1963). One-Permutation
//! Hashing (Li, Owen, Zhang 2012; Shrivastava 2017) is a documented faster follow-up
//! that produces the same `d×n` signature interface.

use ahash::AHashMap;
use rayon::prelude::*;

/// LSH S-curve inflection target. Set safely below any plausible construction
/// threshold (which the interpretation layer selects afterward): at/below the sketch
/// noise floor, where Jaccard estimates are statistically indistinguishable from
/// zero. LSH is a high-recall *candidate filter*, not the thresholder — false
/// positives are cheap (verified exactly, then dropped by the real threshold), false
/// negatives below this floor are edges nobody would keep.
const TAU_FLOOR: f64 = 0.05;

/// Minimum per-bucket size cap for skew mitigation. Buckets larger than the
/// data-derived p99 (but never smaller than this floor) are subsampled to bound the
/// O(bucket²) pair blow-up from heavy points that collide with everything.
const MIN_BUCKET_CAP: usize = 256;

/// SplitMix64 finalizer — a strong integer mixer. We deliberately avoid the
/// `(a·c + b) mod p` family (only 2-universal → biased min-wise collision
/// probabilities); a good mixer makes the MinHash estimator's `P(collision) = J`
/// assumption hold in practice.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Per-row permutation seed derived from the base seed. The row index is spread
/// across the full word (multiply by the golden-ratio constant) *before* combining
/// with the base seed, so nearby base seeds and row indices don't alias onto shared
/// permutations — a naive `base_seed ^ (row+1)` collides for small seeds/rows and
/// destroys row independence (correlated, biased estimates).
#[inline]
fn row_seed(base_seed: u64, row: usize) -> u64 {
    splitmix64(
        base_seed.wrapping_add(
            (row as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(1),
        ),
    )
}

/// Hash of ball index `c` under permutation `row` (high 32 bits of the mixer for
/// quality). u32 codomain halves signature memory; K balls < 2³² in practice.
#[inline]
fn hash_ball(row_seed: u64, c: usize) -> u32 {
    (splitmix64(row_seed ^ (c as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 32) as u32
}

/// Choose `(r, b)` with `d = b·r` placing the S-curve inflection `τ ≈ (1/b)^{1/r}`
/// closest to `tau`. Ties break toward more bands (larger `b`) for higher recall.
/// Fully derived from `d` and `tau` — no user-facing knob.
fn choose_bands(d: usize, tau: f64) -> (usize, usize) {
    let mut best_r = 1usize;
    let mut best_b = d;
    let mut best_score = f64::INFINITY;
    for r in 1..=d {
        if !d.is_multiple_of(r) {
            continue;
        }
        let b = d / r;
        let tau_est = (1.0 / b as f64).powf(1.0 / r as f64);
        let score = (tau_est - tau).abs();
        // Strictly better, or an effective tie with more bands (higher recall).
        if score < best_score - 1e-12 || ((score - best_score).abs() <= 1e-12 && b > best_b) {
            best_score = score;
            best_r = r;
            best_b = b;
        }
    }
    (best_r, best_b)
}

/// p99 of a slice of bucket sizes (used to derive the skew cap). Returns 0 for empty.
fn percentile_99(sizes: &[usize]) -> usize {
    if sizes.is_empty() {
        return 0;
    }
    let mut sorted = sizes.to_vec();
    sorted.sort_unstable();
    let idx = ((sorted.len() as f64 * 0.99).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[idx]
}

/// Row-major `d×n` MinHash signature accumulator. `sig[r*n + p]` is point `p`'s
/// value under permutation `r`; `u32::MAX` marks "point in no ball".
///
/// Balls are hashed by a *global* id so distinct balls (across batches/datasets) get
/// distinct permutation values — `ball_offset` tracks the running id so streaming
/// accumulation (`fit_multi`) is identical to a single-shot build. The signature is
/// constant `d×n` memory regardless of how many ball maps stream through, which is
/// why streaming is the memory-efficient path.
pub struct MinHashSignatures {
    n: usize,
    d: usize,
    seed: u64,
    sig: Vec<u32>,
    ball_offset: usize,
}

impl MinHashSignatures {
    pub fn new(n: usize, d: usize, seed: u64) -> Self {
        MinHashSignatures {
            n,
            d,
            seed,
            sig: vec![u32::MAX; d * n],
            ball_offset: 0,
        }
    }

    /// Fold a batch of balls into the running signature via element-wise `min` (an
    /// associative, order-independent reduction). Rows are independent → filled in
    /// parallel with contention-free writes.
    pub fn accumulate(&mut self, balls: &[&[usize]]) {
        let n = self.n;
        let seed = self.seed;
        let base = self.ball_offset;
        self.sig.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
            let rs = row_seed(seed, r);
            for (c, members) in balls.iter().enumerate() {
                let h = hash_ball(rs, base + c);
                for &p in *members {
                    if p < n && h < row[p] {
                        row[p] = h;
                    }
                }
            }
        });
        self.ball_offset += balls.len();
    }

    /// Number of points (signature column count).
    pub fn n(&self) -> usize {
        self.n
    }

    /// Run LSH banding + candidate weight estimation over the accumulated signature.
    pub fn edges(&self) -> Vec<(usize, usize, f64)> {
        lsh_and_estimate(&self.sig, self.n, self.d)
    }
}

/// Generate candidate pairs for one band by bucketing the length-`r` sub-signatures.
/// Points sharing a bucket are candidates. Oversized buckets (heavy points colliding
/// with everything) are subsampled to `cap` to bound the O(bucket²) blow-up.
/// Returns `(pairs, n_capped)` where `n_capped` counts buckets that were subsampled.
fn band_candidates(
    sig: &[u32],
    n: usize,
    band_start: usize,
    r: usize,
    cap: usize,
) -> (Vec<(u32, u32)>, usize) {
    // Group points by their band signature. Insertion order = ascending point index
    // → deterministic.
    let mut buckets: AHashMap<u64, Vec<u32>> = AHashMap::new();
    for p in 0..n {
        // Skip points in no ball: an all-MAX signature would spuriously collide with
        // every other isolated point and estimate Jaccard 1.
        let mut h = splitmix64(0xA5A5_5A5A ^ band_start as u64);
        let mut all_max = true;
        for t in 0..r {
            let v = sig[(band_start + t) * n + p];
            if v != u32::MAX {
                all_max = false;
            }
            h = splitmix64(h ^ v as u64);
        }
        if all_max {
            continue;
        }
        buckets.entry(h).or_default().push(p as u32);
    }

    let mut pairs = Vec::new();
    let mut n_capped = 0usize;
    for members in buckets.values() {
        let m = members.len();
        if m < 2 {
            continue;
        }
        if m <= cap {
            for a in 0..m {
                for b in (a + 1)..m {
                    pairs.push((members[a], members[b]));
                }
            }
        } else {
            // Deterministic stride subsample to `cap` representatives.
            n_capped += 1;
            let stride = m / cap;
            let reps: Vec<u32> = (0..m)
                .step_by(stride.max(1))
                .take(cap)
                .map(|k| members[k])
                .collect();
            for a in 0..reps.len() {
                for b in (a + 1)..reps.len() {
                    pairs.push((reps[a], reps[b]));
                }
            }
        }
    }
    (pairs, n_capped)
}

/// Estimate the Jaccard weight of a candidate pair from the signature columns:
/// fraction of the `d` rows on which the two points' min-hashes agree.
#[inline]
fn estimate_weight(sig: &[u32], n: usize, d: usize, i: usize, j: usize) -> f64 {
    let mut matches = 0usize;
    for r in 0..d {
        if sig[r * n + i] == sig[r * n + j] {
            matches += 1;
        }
    }
    matches as f64 / d as f64
}

/// Approximate cosmic edges from ball memberships via MinHash + LSH.
///
/// `balls[c]` is the list of point indices in ball `c` (balls are globally enumerated
/// across all ball maps). Returns weighted edges `(i, j, ŵ)` with `i < j`,
/// `ŵ > TAU_FLOOR`, sorted by `(i, j)`. Pure (no Python types) for unit testing.
///
/// Caller must ensure `d >= 1`. `n < 2` yields no edges.
pub fn cosmic_edges_minhash(
    balls: &[&[usize]],
    n: usize,
    d: usize,
    seed: u64,
) -> Vec<(usize, usize, f64)> {
    if n < 2 || d == 0 || balls.is_empty() {
        return Vec::new();
    }
    let mut sigs = MinHashSignatures::new(n, d, seed);
    sigs.accumulate(balls);
    sigs.edges()
}

/// LSH banding + candidate weight estimation over a built `d×n` signature.
/// Returns weighted edges `(i, j, ŵ)` with `i < j`, `ŵ > TAU_FLOOR`, sorted.
fn lsh_and_estimate(sig: &[u32], n: usize, d: usize) -> Vec<(usize, usize, f64)> {
    if n < 2 || d == 0 {
        return Vec::new();
    }
    let (r, b) = choose_bands(d, TAU_FLOOR);

    // Derive the per-bucket skew cap from the realized bucket-size distribution.
    let mut all_sizes: Vec<usize> = Vec::new();
    for band in 0..b {
        let band_start = band * r;
        let mut counts: AHashMap<u64, usize> = AHashMap::new();
        for p in 0..n {
            let mut h = splitmix64(0xA5A5_5A5A ^ band_start as u64);
            let mut all_max = true;
            for t in 0..r {
                let v = sig[(band_start + t) * n + p];
                if v != u32::MAX {
                    all_max = false;
                }
                h = splitmix64(h ^ v as u64);
            }
            if !all_max {
                *counts.entry(h).or_default() += 1;
            }
        }
        all_sizes.extend(counts.values().copied());
    }
    let cap = percentile_99(&all_sizes).max(MIN_BUCKET_CAP);

    // Candidate generation per band, in parallel.
    let band_results: Vec<(Vec<(u32, u32)>, usize)> = (0..b)
        .into_par_iter()
        .map(|band| band_candidates(sig, n, band * r, r, cap))
        .collect();

    let total_capped: usize = band_results.iter().map(|(_, c)| *c).sum();
    if total_capped > 0 {
        eprintln!(
            "pulsar.minhash: subsampled {total_capped} oversized LSH bucket(s) (cap={cap}) \
             to bound candidate generation; some low-weight edges among heavy points may be omitted."
        );
    }

    // Flatten, normalize to i<j, dedup across bands.
    let mut candidates: Vec<(u32, u32)> = band_results
        .into_iter()
        .flat_map(|(pairs, _)| pairs)
        .map(|(x, y)| if x < y { (x, y) } else { (y, x) })
        .collect();
    candidates.par_sort_unstable();
    candidates.dedup();

    // Estimate weights on candidates in parallel; keep edges above the noise floor.
    let mut edges: Vec<(usize, usize, f64)> = candidates
        .par_iter()
        .filter_map(|&(i, j)| {
            let w = estimate_weight(sig, n, d, i as usize, j as usize);
            if w > TAU_FLOOR {
                Some((i as usize, j as usize, w))
            } else {
                None
            }
        })
        .collect();
    edges.par_sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact Jaccard of two points' ball-sets, brute force.
    fn exact_jaccard(balls: &[&[usize]], i: usize, j: usize) -> f64 {
        let bi: Vec<usize> = balls
            .iter()
            .enumerate()
            .filter(|(_, m)| m.contains(&i))
            .map(|(c, _)| c)
            .collect();
        let bj: Vec<usize> = balls
            .iter()
            .enumerate()
            .filter(|(_, m)| m.contains(&j))
            .map(|(c, _)| c)
            .collect();
        let inter = bi.iter().filter(|c| bj.contains(c)).count();
        let union = bi.len() + bj.len() - inter;
        if union == 0 {
            0.0
        } else {
            inter as f64 / union as f64
        }
    }

    #[test]
    fn choose_bands_factorizes_d() {
        let (r, b) = choose_bands(256, TAU_FLOOR);
        assert_eq!(r * b, 256);
        assert!(b >= 1 && r >= 1);
    }

    #[test]
    fn identical_ballsets_estimate_to_one() {
        // Points 0 and 1 share every ball → Jaccard 1.
        let balls: Vec<&[usize]> = vec![&[0, 1, 2], &[0, 1, 3], &[0, 1]];
        let edges = cosmic_edges_minhash(&balls, 4, 512, 7);
        let w01 = edges
            .iter()
            .find(|&&(i, j, _)| i == 0 && j == 1)
            .map(|&(_, _, w)| w);
        assert!(w01.is_some(), "expected an edge between identical points");
        assert!((w01.unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn estimates_are_close_to_exact_jaccard() {
        // A handful of overlapping balls; high d → tight estimate.
        let balls: Vec<&[usize]> = vec![
            &[0, 1, 2, 3],
            &[1, 2, 3, 4],
            &[2, 3, 4, 5],
            &[0, 1],
            &[3, 4, 5],
            &[0, 2, 4],
        ];
        let n = 6;
        let d = 4096; // large d for a tight check
        let edges = cosmic_edges_minhash(&balls, n, d, 42);
        for &(i, j, w) in &edges {
            let exact = exact_jaccard(&balls, i, j);
            assert!(
                (w - exact).abs() < 0.05,
                "pair ({i},{j}): est {w} vs exact {exact}"
            );
        }
    }

    #[test]
    fn isolated_points_produce_no_edges() {
        // Point 3 is in no ball → must never appear in an edge.
        let balls: Vec<&[usize]> = vec![&[0, 1], &[0, 2], &[1, 2]];
        let edges = cosmic_edges_minhash(&balls, 4, 256, 1);
        assert!(edges.iter().all(|&(i, j, _)| i != 3 && j != 3));
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let balls: Vec<&[usize]> = vec![&[0, 1, 2], &[1, 2, 3], &[2, 3, 0]];
        let a = cosmic_edges_minhash(&balls, 4, 256, 99);
        let b = cosmic_edges_minhash(&balls, 4, 256, 99);
        assert_eq!(a, b);
    }

    #[test]
    fn handles_degenerate_inputs() {
        assert!(cosmic_edges_minhash(&[], 5, 256, 0).is_empty());
        let balls: Vec<&[usize]> = vec![&[0, 1]];
        assert!(cosmic_edges_minhash(&balls, 1, 256, 0).is_empty()); // n < 2
        assert!(cosmic_edges_minhash(&balls, 2, 0, 0).is_empty()); // d == 0
    }
}
