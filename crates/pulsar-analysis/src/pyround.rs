//! Python-compatible decimal rounding.
//!
//! CPython's `round(x, n)` is correctly rounded on the exact binary value
//! with ties to even; so is Rust's `{:.n}` formatting. Going through the
//! formatted decimal therefore reproduces Python bit for bit, where
//! `(x * 10^n).round() / 10^n` would not (it rounds the already-rounded
//! product, half away from zero).

/// `round(x, digits)` as Python computes it.
pub fn round_to(x: f64, digits: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.digits$}").parse().unwrap_or(x)
}

/// `round(x)` as Python computes it (ties to even), as an integer.
pub fn round_int(x: f64) -> i64 {
    round_to(x, 0) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_python_round() {
        // Values checked against CPython 3.12.
        assert_eq!(round_to(0.6636363636363637, 4), 0.6636);
        assert_eq!(round_to(2.675, 2), 2.67); // 2.675 is below the tie in binary
        assert_eq!(round_to(0.125, 2), 0.12); // exact tie → even
        assert_eq!(round_to(0.375, 2), 0.38);
        assert_eq!(round_to(1.0 / 3.0, 4), 0.3333);
        assert_eq!(round_to(0.0, 4), 0.0);
        assert_eq!(round_int(18.547236990991408), 19);
        assert_eq!(round_int(1.72), 2);
        assert_eq!(round_int(2.5), 2);
        assert_eq!(round_int(3.5), 4);
    }
}
