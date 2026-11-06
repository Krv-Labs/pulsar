//! How a summary discloses names and counts: column names and levels cut to
//! a display label, row counts rounded or withheld, row fractions to 2 dp.
//! These are the `io.phirewall/summary` rules; every summary that names a
//! column or counts rows uses them.

use std::collections::{BTreeMap, HashSet};

use pulsar_data::{Column, Table};
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

/// Who each row of a table is, for patient floors: the distinct tokens of
/// its identifier columns, or each row itself when it has none.
///
/// An identifier column is one whose every non-blank cell is a Phirewall
/// patient token (`PP-RRRRRR-C`, Crockford base32 with its check symbol), or
/// one the caller names (a plan's identifier columns, a panel's entity
/// column). Like Phirewall, a group's patients are the fewest distinct
/// tokens across those columns.
#[derive(Debug, Clone)]
pub struct Patients {
    /// Per identifier column, each row's token (`None` for a blank).
    columns: Vec<Vec<Option<String>>>,
    rows: usize,
}

/// What patient counts are counted in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    Patients,
    Rows,
}

impl Unit {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Patients => "patients",
            Self::Rows => "rows",
        }
    }
}

impl Patients {
    /// Identity from `table`'s token columns plus the columns `named`.
    pub fn of(table: &Table, named: &[String]) -> Self {
        let columns = table
            .names
            .iter()
            .zip(&table.columns)
            .filter(|(name, col)| named.contains(name) || is_token_column(col))
            .map(|(_, col)| cells(col))
            .collect();
        Self {
            columns,
            rows: table.n_rows(),
        }
    }

    pub fn unit(&self) -> Unit {
        if self.columns.is_empty() {
            Unit::Rows
        } else {
            Unit::Patients
        }
    }

    /// Distinct patients among `rows` (row indices).
    pub fn count(&self, rows: impl IntoIterator<Item = usize> + Clone) -> usize {
        if self.columns.is_empty() {
            return rows.into_iter().filter(|&i| i < self.rows).count();
        }
        self.columns
            .iter()
            .map(|col| {
                rows.clone()
                    .into_iter()
                    .filter_map(|i| col.get(i).and_then(Option::as_deref))
                    .collect::<HashSet<&str>>()
                    .len()
            })
            .min()
            .unwrap_or(0)
    }

    /// Distinct patients in the whole table.
    pub fn total(&self) -> usize {
        self.count(0..self.rows)
    }

    /// The levels of a categorical (text or boolean) column as
    /// `(level, rows, patients)`, most rows first, ties by level; `None` for a
    /// numeric column.
    pub fn levels(&self, col: &Column) -> Option<Vec<(String, usize, usize)>> {
        let cells = match col {
            Column::Text(_) | Column::Bool(_) => cells(col),
            _ => return None,
        };
        let mut rows: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, c) in cells.into_iter().enumerate() {
            if let Some(c) = c.filter(|c| !c.trim().is_empty()) {
                rows.entry(c).or_default().push(i);
            }
        }
        let mut out: Vec<(String, usize, usize)> = rows
            .into_iter()
            .map(|(l, idx)| {
                let p = self.count(idx.iter().copied());
                (l, idx.len(), p)
            })
            .collect();
        out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        Some(out)
    }

    /// Distinct patients per label (`labels[i]` is row `i`'s group).
    pub fn per_label(&self, labels: &[i64]) -> BTreeMap<i64, (usize, usize)> {
        let mut rows: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        for (i, &l) in labels.iter().enumerate() {
            rows.entry(l).or_default().push(i);
        }
        rows.into_iter()
            .map(|(l, idx)| (l, (idx.len(), self.count(idx.iter().copied()))))
            .collect()
    }
}

fn cells(col: &Column) -> Vec<Option<String>> {
    match col {
        Column::Text(v) => v.clone(),
        Column::Int(v) => v.iter().map(|x| Some(x.to_string())).collect(),
        Column::Float(v) => v
            .iter()
            .map(|x| (!x.is_nan()).then(|| x.to_string()))
            .collect(),
        Column::Bool(v) => v.iter().map(|b| b.map(|b| b.to_string())).collect(),
    }
}

/// Whether every non-blank cell of `col` is a Phirewall patient token.
pub fn is_token_column(col: &Column) -> bool {
    let Column::Text(v) = col else { return false };
    let mut any = false;
    for cell in v.iter().flatten() {
        let cell = cell.trim();
        if cell.is_empty() {
            continue;
        }
        if !looks_like_token(cell) {
            return false;
        }
        any = true;
    }
    any
}

/// A Phirewall patient token: `PP-RRRRRR-C`, eight Crockford base32 data
/// symbols and a mod-37 check symbol (`phirewall-core/src/token.rs`).
pub fn looks_like_token(s: &str) -> bool {
    const ALPHABET: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";
    const CHECK: &[u8; 37] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ*~$=U";
    let b = s.as_bytes();
    if b.len() != 11 || b[2] != b'-' || b[9] != b'-' {
        return false;
    }
    let value = |c: u8| {
        let c = match c.to_ascii_uppercase() {
            b'I' | b'L' => b'1',
            b'O' => b'0',
            c => c,
        };
        ALPHABET.iter().position(|&a| a == c).map(|i| i as u32)
    };
    let mut n = 0u32;
    for &c in b[..2].iter().chain(&b[3..9]) {
        match value(c) {
            Some(v) => n = (n * 32 + v) % 37,
            None => return false,
        }
    }
    b[10].to_ascii_uppercase() == CHECK[n as usize]
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

    fn token(prefix_and_random: &str) -> String {
        // The check symbol, computed the way Phirewall does.
        const ALPHABET: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";
        const CHECK: &[u8; 37] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ*~$=U";
        let n = prefix_and_random.bytes().fold(0u32, |n, c| {
            (n * 32 + ALPHABET.iter().position(|&a| a == c).unwrap() as u32) % 37
        });
        format!(
            "{}-{}-{}",
            &prefix_and_random[..2],
            &prefix_and_random[2..],
            CHECK[n as usize] as char
        )
    }

    #[test]
    fn tokens_are_recognized_by_shape_and_check_symbol() {
        let t = token("D7K3F9Q2");
        assert!(looks_like_token(&t), "{t}");
        let mut wrong = t.clone().into_bytes();
        wrong[10] = if wrong[10] == b'0' { b'1' } else { b'0' };
        assert!(!looks_like_token(std::str::from_utf8(&wrong).unwrap()));
        assert!(!looks_like_token("D7-K3F9Q2"));
        assert!(!looks_like_token("patient-001"));
    }

    #[test]
    fn patients_are_the_fewest_distinct_tokens() {
        let (a, b) = (token("D7K3F9Q2"), token("D7AAAAAA"));
        let table = Table::new(
            vec!["pid".into(), "x".into()],
            vec![
                Column::Text(vec![Some(a.clone()), Some(a.clone()), Some(b.clone())]),
                Column::Float(vec![1.0, 2.0, 3.0]),
            ],
        )
        .unwrap();
        let p = Patients::of(&table, &[]);
        assert_eq!(p.unit(), Unit::Patients);
        assert_eq!(p.total(), 2);
        assert_eq!(p.per_label(&[0, 0, 1])[&0], (2, 1));
        let rows = Patients::of(&table, &[]).count([0usize, 1]);
        assert_eq!(rows, 1);
        let none = Table::new(vec!["x".into()], vec![Column::Float(vec![1.0, 2.0])]).unwrap();
        assert_eq!(Patients::of(&none, &[]).unit(), Unit::Rows);
        assert_eq!(Patients::of(&none, &[]).total(), 2);
    }
}
