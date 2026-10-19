//! MinHash signature-depth accuracy and memory profile
//! (`pulsar/mcp/minhash_advisor.py`). Edge weights from the MinHash
//! construction are means of `d` Bernoulli(J) trials, so their error
//! depends only on `d`.

use serde::Serialize;

use crate::pyround::round_to;

pub const DEFAULT_D: usize = 256;
/// 95% confidence.
pub const DEFAULT_DELTA: f64 = 0.05;
const BYTES_PER_SIGNATURE_ENTRY: usize = 4;

/// Hoeffding worst-case error `ε` with `P(|Ŵ − W| ≥ ε) ≤ delta`.
pub fn epsilon_at_confidence(d: usize, delta: f64) -> f64 {
    ((2.0 / delta).ln() / (2.0 * d as f64)).sqrt()
}

/// Minimum depth guaranteeing Hoeffding error `≤ eps` at `1 − delta`.
pub fn depth_for_epsilon(eps: f64, delta: f64) -> usize {
    ((2.0 / delta).ln() / (2.0 * eps * eps)).ceil() as usize
}

/// CLT standard error of the Jaccard estimate at true similarity `w`.
pub fn standard_error(w: f64, d: usize) -> f64 {
    let w = w.clamp(0.0, 1.0);
    (w * (1.0 - w) / d as f64).sqrt()
}

pub fn ci_half_width(d: usize, w: f64, z: f64) -> f64 {
    z * standard_error(w, d)
}

/// Bytes of the `d × n` u32 signature matrix.
pub fn signature_memory_bytes(d: usize, n: usize) -> usize {
    d * n * BYTES_PER_SIGNATURE_ENTRY
}

pub fn fmt_bytes(b: usize) -> String {
    let gb = b as f64 / 1e9;
    if gb >= 1.0 {
        format!("{gb:.2} GB")
    } else {
        format!("{:.0} MB", b as f64 / 1e6)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DepthProfile {
    pub d: usize,
    pub hoeffding_epsilon: f64,
    pub ci95_half_width_worst: f64,
    pub se_at_0_15: f64,
    pub signature_memory_bytes: usize,
    pub signature_memory_human: String,
    pub confidence: f64,
}

/// Accuracy and memory profile of depth `d` at `n` rows.
pub fn depth_profile(d: usize, n: usize, delta: f64) -> DepthProfile {
    DepthProfile {
        d,
        hoeffding_epsilon: round_to(epsilon_at_confidence(d, delta), 4),
        ci95_half_width_worst: round_to(ci_half_width(d, 0.5, 1.96), 4),
        se_at_0_15: round_to(standard_error(0.15, d), 4),
        signature_memory_bytes: signature_memory_bytes(d, n),
        signature_memory_human: fmt_bytes(signature_memory_bytes(d, n)),
        confidence: round_to(1.0 - delta, 4),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_matches_the_python_numbers() {
        let p = depth_profile(256, 6098, DEFAULT_DELTA);
        assert_eq!(p.hoeffding_epsilon, 0.0849);
        assert_eq!(p.ci95_half_width_worst, 0.0612);
        assert_eq!(p.se_at_0_15, 0.0223);
        assert_eq!(p.signature_memory_bytes, 6_244_352);
        assert_eq!(p.signature_memory_human, "6 MB");
        assert_eq!(p.confidence, 0.95);
        assert_eq!(fmt_bytes(2_500_000_000), "2.50 GB");
        assert_eq!(depth_for_epsilon(0.05, 0.05), 738);
    }
}
