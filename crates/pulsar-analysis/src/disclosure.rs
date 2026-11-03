//! How a summary discloses names and counts: column names and levels cut to
//! a display label, row counts rounded or withheld, row fractions to 2 dp.
//! These are the `io.phirewall/summary` rules; every summary that names a
//! column or counts rows uses them.

use serde::Serialize;

use crate::pyround::round_to;

/// Longest column name or level emitted, in characters, before the `…`.
pub const MAX_LABEL_CHARS: usize = 80;

/// `s` cut to [`MAX_LABEL_CHARS`] characters plus `…`, and whether it was.
pub fn label(s: &str) -> (String, bool) {
    match s.char_indices().nth(MAX_LABEL_CHARS) {
        Some((end, _)) => (format!("{}…", &s[..end]), true),
        None => (s.to_owned(), false),
    }
}

/// A disclosed row count: 0, or rounded to the nearest 5, or withheld
/// (`"under <floor>"`) when nonzero and under `floor`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum RowCount {
    Rounded(usize),
    Withheld(String),
}

impl RowCount {
    pub fn of(rows: usize, floor: usize) -> Self {
        if rows == 0 || rows >= floor {
            Self::Rounded(round_count(rows))
        } else {
            Self::Withheld(format!("under {floor}"))
        }
    }
}

/// `count` rounded to the nearest 5, halves up: Phirewall's `round_count`.
pub fn round_count(count: usize) -> usize {
    (count + 2) / 5 * 5
}

/// A share of rows to 2 decimals; a nonzero share is never shown as 0.
pub fn row_fraction(x: f64) -> f64 {
    let r = round_to(x, 2);
    if x > 0.0 && r == 0.0 {
        0.01
    } else {
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_round_halves_up_and_small_ones_are_withheld() {
        assert_eq!(RowCount::of(0, 8), RowCount::Rounded(0));
        assert_eq!(RowCount::of(7, 8), RowCount::Withheld("under 8".into()));
        assert_eq!(RowCount::of(8, 8), RowCount::Rounded(10));
        assert_eq!(RowCount::of(342, 8), RowCount::Rounded(340));
        assert_eq!(RowCount::of(343, 8), RowCount::Rounded(345));
    }

    #[test]
    fn a_nonzero_fraction_is_never_zero() {
        assert_eq!(row_fraction(0.0), 0.0);
        assert_eq!(row_fraction(0.003), 0.01);
        assert_eq!(row_fraction(0.0058), 0.01);
        assert_eq!(row_fraction(0.4249), 0.42);
    }
}
