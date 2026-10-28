//! Classify a text column as `numeric_text`, `datetime_like` or `text`.
//!
//! Mirrors `_detect_object_kind`: sample up to 200 observed values, then a
//! column is numeric text when at least half parse as finite numbers (after
//! stripping thousands commas) and datetime-like when at least half are
//! non-numeric dates containing a digit-separator-digit run.
//!
//! The date recogniser accepts a conservative subset of what
//! `pd.to_datetime(format="mixed")` parses: `YYYY-MM-DD`, `YYYY/MM/DD`,
//! `YYYY.MM.DD`, `YYYY-MM`, `MM/DD/YYYY` or `DD/MM/YYYY` (also with `-`
//! and `.`), each optionally followed by `T` or a space and a time, and bare
//! times (`HH:MM[:SS[.f]]`) with an optional `Z` / `±HH[:MM]` zone. Month
//! names and 2-digit years are not recognised, so such columns stay `text`.

use super::ColumnKind;

const TYPE_SAMPLE_SIZE: usize = 200;

/// Remove commas that group thousands: `(?<=\d),(?=\d{3}(?!\d))`.
pub(crate) fn strip_thousands(s: &str) -> String {
    if !s.contains(',') {
        return s.to_string();
    }
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    for (i, &c) in chars.iter().enumerate() {
        if c == ','
            && i > 0
            && chars[i - 1].is_ascii_digit()
            && chars
                .get(i + 1..i + 4)
                .is_some_and(|w| w.iter().all(char::is_ascii_digit))
            && chars.get(i + 4).is_none_or(|d| !d.is_ascii_digit())
        {
            continue;
        }
        out.push(c);
    }
    out
}

/// `pd.to_numeric(errors="coerce")` for one token; `None` when it is not a
/// number or is NaN.
pub(crate) fn to_numeric(token: &str) -> Option<f64> {
    token.trim().parse::<f64>().ok().filter(|x| !x.is_nan())
}

/// `\d[-/:.T]\d` anywhere in `s`.
fn has_date_sep(s: &str) -> bool {
    let chars: Vec<char> = s.chars().collect();
    chars.windows(3).any(|w| {
        w[0].is_ascii_digit()
            && matches!(w[1], '-' | '/' | ':' | '.' | 'T')
            && w[2].is_ascii_digit()
    })
}

/// Evenly spaced sample, as `np.linspace(0, n - 1, 200, dtype=int)`.
fn sample_indices(n: usize) -> Vec<usize> {
    if n <= TYPE_SAMPLE_SIZE {
        return (0..n).collect();
    }
    let step = (n - 1) as f64 / (TYPE_SAMPLE_SIZE - 1) as f64;
    let mut idx: Vec<usize> = (0..TYPE_SAMPLE_SIZE)
        .map(|i| (i as f64 * step) as usize)
        .collect();
    *idx.last_mut().expect("sample is non-empty") = n - 1;
    idx
}

/// Kind of a text column given its observed (non-missing) values in order.
pub(super) fn detect_text_kind(observed: &[&str]) -> ColumnKind {
    if observed.is_empty() {
        return ColumnKind::Text;
    }
    let sample: Vec<&str> = sample_indices(observed.len())
        .into_iter()
        .map(|i| observed[i].trim())
        .collect();

    let plain: Vec<bool> = sample
        .iter()
        .map(|s| !s.is_empty() && to_numeric(&strip_thousands(s)).is_some_and(f64::is_finite))
        .collect();
    let frac = |flags: &[bool]| flags.iter().filter(|&&b| b).count() as f64 / flags.len() as f64;

    if frac(&plain) >= 0.5 {
        return ColumnKind::NumericText;
    }
    let sep: Vec<bool> = sample.iter().map(|s| has_date_sep(s)).collect();
    if !sep.iter().zip(&plain).any(|(&s, &p)| s && !p) {
        return ColumnKind::Text;
    }
    let is_date: Vec<bool> = sample
        .iter()
        .zip(plain.iter().zip(&sep))
        .map(|(s, (&p, &d))| d && !p && !s.is_empty() && parses_as_datetime(s))
        .collect();
    if frac(&is_date) >= 0.5 {
        ColumnKind::DatetimeLike
    } else {
        ColumnKind::Text
    }
}

fn all_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

fn days_in_month(year: u32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400) => {
            29
        }
        2 => 28,
        _ => 0,
    }
}

fn valid_ymd(y: &str, m: &str, d: &str) -> bool {
    let (Ok(y), Ok(m), Ok(d)) = (y.parse::<u32>(), m.parse::<u32>(), d.parse::<u32>()) else {
        return false;
    };
    (1..=12).contains(&m) && d >= 1 && d <= days_in_month(y, m)
}

fn date_ok(d: &str) -> bool {
    let Some(sep) = ['-', '/', '.'].into_iter().find(|&s| d.contains(s)) else {
        return false;
    };
    let parts: Vec<&str> = d.split(sep).collect();
    if !parts.iter().all(|p| all_digits(p) && p.len() <= 4) {
        return false;
    }
    match parts.as_slice() {
        [y, m] if y.len() == 4 && sep != '.' => {
            m.parse::<u32>().is_ok_and(|m| (1..=12).contains(&m))
        }
        [y, m, d] if y.len() == 4 => valid_ymd(y, m, d),
        [a, b, y] if y.len() == 4 && a.len() <= 2 && b.len() <= 2 => {
            valid_ymd(y, a, b) || valid_ymd(y, b, a)
        }
        _ => false,
    }
}

fn time_ok(t: &str) -> bool {
    let t = t.trim();
    let body = if let Some(rest) = t.strip_suffix('Z') {
        rest
    } else if let Some(i) = t.rfind(['+', '-']).filter(|&i| i > 0) {
        let zone = &t[i + 1..];
        let digits: String = zone.chars().filter(|c| *c != ':').collect();
        if !(all_digits(&digits) && matches!(digits.len(), 2 | 4)) {
            return false;
        }
        &t[..i]
    } else {
        t
    };
    let parts: Vec<&str> = body.trim().split(':').collect();
    if !(2..=3).contains(&parts.len()) {
        return false;
    }
    let in_range = |s: &str, max: u32| {
        all_digits(s) && s.len() <= 2 && s.parse::<u32>().is_ok_and(|v| v <= max)
    };
    if !in_range(parts[0], 23) || !in_range(parts[1], 59) {
        return false;
    }
    match parts.get(2) {
        None => true,
        Some(sec) => {
            let (whole, frac) = sec.split_once('.').unwrap_or((sec, "0"));
            in_range(whole, 59) && all_digits(frac)
        }
    }
}

fn parses_as_datetime(s: &str) -> bool {
    let s = s.trim();
    if s.contains(':') && !s.contains(['-', '/']) && !s.contains(' ') && !s.contains('T') {
        return time_ok(s);
    }
    match s.split_once(['T', ' ']) {
        Some((date, time)) => date_ok(date) && time_ok(time),
        None => date_ok(s),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thousands_commas_only_group_digits() {
        assert_eq!(strip_thousands("1,234"), "1234");
        assert_eq!(strip_thousands("1,234,567"), "1234567");
        assert_eq!(strip_thousands("1,2345"), "1,2345");
        assert_eq!(strip_thousands("a,123"), "a,123");
        assert_eq!(strip_thousands("12,34"), "12,34");
    }

    #[test]
    fn numeric_text_needs_half_plain_numbers() {
        assert_eq!(
            detect_text_kind(&["1", "2", "x", "4"]),
            ColumnKind::NumericText
        );
        assert_eq!(detect_text_kind(&["1", "x", "y", "z"]), ColumnKind::Text);
        assert_eq!(
            detect_text_kind(&["1,000", "2,500"]),
            ColumnKind::NumericText
        );
        assert_eq!(detect_text_kind(&["inf", "nan-ish"]), ColumnKind::Text);
    }

    #[test]
    fn dates_are_recognised_by_format_and_range() {
        for ok in [
            "2020-01-05",
            "2020/01/05",
            "2020-01-05T10:20:30Z",
            "2020-01-05 10:20",
            "2020-01-05 10:20:30.123+02:00",
            "01/05/2020",
            "31/12/2020",
            "2020-02",
            "12:30",
            "12:30:45",
        ] {
            assert!(parses_as_datetime(ok), "{ok}");
        }
        for bad in [
            "2020-13-01",
            "2021-02-29",
            "1.2.3",
            "hello",
            "24:00",
            "2020-01-05T",
            "99/99/2020",
        ] {
            assert!(!parses_as_datetime(bad), "{bad}");
        }
        assert!(parses_as_datetime("2020-02-29"));
    }

    #[test]
    fn datetime_like_excludes_plain_numbers() {
        assert_eq!(
            detect_text_kind(&["2020-01-05", "2020-02-11", "n/a-ish"]),
            ColumnKind::DatetimeLike
        );
        // "3.14" has a digit.digit run but is a plain number.
        assert_eq!(detect_text_kind(&["3.14", "2.71"]), ColumnKind::NumericText);
        assert_eq!(detect_text_kind(&["a-1b", "c-2d"]), ColumnKind::Text);
        assert_eq!(detect_text_kind(&[]), ColumnKind::Text);
    }

    #[test]
    fn sample_matches_numpy_linspace() {
        let idx = sample_indices(1000);
        assert_eq!(idx.len(), 200);
        assert_eq!((idx[0], idx[1], idx[199]), (0, 5, 999));
    }
}
