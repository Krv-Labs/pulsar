//! Per-column measurements (`_profile_numeric` / `_profile_object`).

use std::collections::HashMap;

use pulsar_data::Column;

use super::kind::{detect_text_kind, strip_thousands, to_numeric};
use super::{ColumnEvidence, ColumnKind};

pub(super) struct Profile {
    pub evidence: ColumnEvidence,
    /// Missing mask, only when `0 < n_missing < n_rows`.
    pub partial_mask: Option<Vec<bool>>,
    /// Float view of numeric and boolean columns (NaN = missing).
    pub values: Option<Vec<f64>>,
}

fn blank(name: &str, kind: ColumnKind, n_rows: usize, n_missing: usize) -> ColumnEvidence {
    ColumnEvidence {
        name: name.to_string(),
        kind,
        n_rows,
        n_missing,
        missing_fraction: n_missing as f64 / n_rows as f64,
        n_unique: 0,
        unique_ratio: 0.0,
        is_constant: true,
        n_zero: None,
        observed_min: None,
        observed_max: None,
        integer_valued: None,
        nonnegative: None,
        minority_fraction: None,
        max_abs_z: None,
        missing_group: None,
        mask_dependency: None,
        non_numeric_values: None,
        n_non_numeric: None,
    }
}

pub(super) fn profile_column(name: &str, col: &Column, n_rows: usize) -> Profile {
    match col {
        Column::Float(v) => profile_numeric(name, v.clone(), n_rows, false, false),
        Column::Int(v) => profile_numeric(
            name,
            v.iter().map(|&x| x as f64).collect(),
            n_rows,
            false,
            true,
        ),
        Column::Bool(v) => profile_numeric(
            name,
            v.iter()
                .map(|b| b.map_or(f64::NAN, |b| f64::from(u8::from(b))))
                .collect(),
            n_rows,
            true,
            false,
        ),
        Column::Text(v) => profile_text(name, v, n_rows),
    }
}

/// Canonical bit pattern of a value for exact-equality counting (-0.0 == 0.0).
fn key(x: f64) -> u64 {
    (x + 0.0).to_bits()
}

fn profile_numeric(
    name: &str,
    float_arr: Vec<f64>,
    n_rows: usize,
    is_bool: bool,
    is_int: bool,
) -> Profile {
    let isna: Vec<bool> = float_arr.iter().map(|x| x.is_nan()).collect();
    let n_missing = isna.iter().filter(|&&b| b).count();
    let n_observed = n_rows - n_missing;

    if n_observed == 0 {
        let kind = if is_bool {
            ColumnKind::Boolean
        } else {
            ColumnKind::Numeric
        };
        return Profile {
            evidence: blank(name, kind, n_rows, n_missing),
            partial_mask: None,
            values: Some(float_arr),
        };
    }

    let obs: Vec<f64> = float_arr.iter().copied().filter(|x| !x.is_nan()).collect();
    let all_finite = obs.iter().all(|x| x.is_finite());
    let finite: Vec<f64> = if all_finite {
        Vec::new()
    } else {
        obs.iter().copied().filter(|x| x.is_finite()).collect()
    };
    let obs_finite: &[f64] = if all_finite { &obs } else { &finite };
    let obs_min = obs_finite.iter().copied().reduce(f64::min);
    let obs_max = obs_finite.iter().copied().reduce(f64::max);
    let n_zero = obs.iter().filter(|&&x| x == 0.0).count();

    let is_01 = is_bool
        || (all_finite
            && matches!((obs_min, obs_max), (Some(lo), Some(hi)) if lo >= 0.0 && hi <= 1.0)
            && obs.iter().filter(|&&x| x == 0.0 || x == 1.0).count() == n_observed);
    let kind = if is_01 {
        ColumnKind::Boolean
    } else {
        ColumnKind::Numeric
    };

    let (n_unique, minority_fraction, integer_valued);
    if is_01 {
        let n_one = n_observed - n_zero;
        n_unique = usize::from(n_zero > 0) + usize::from(n_one > 0);
        minority_fraction = (n_unique == 2).then(|| n_zero.min(n_one) as f64 / n_observed as f64);
        integer_valued = None;
    } else {
        if all_finite && obs_min.is_some() && obs_min == obs_max {
            n_unique = 1;
            minority_fraction = None;
        } else {
            let mut seen = HashMap::<u64, ()>::new();
            let mut first = f64::NAN;
            for &x in &obs {
                if seen.insert(key(x), ()).is_none() && seen.len() == 1 {
                    first = x;
                }
            }
            n_unique = seen.len();
            minority_fraction = (n_unique == 2).then(|| {
                let c_first = obs.iter().filter(|&&x| x == first).count();
                c_first.min(n_observed - c_first) as f64 / n_observed as f64
            });
        }
        integer_valued = Some(is_int || (all_finite && obs.iter().all(|x| x.floor() == *x)));
    }

    let unique_ratio = n_unique as f64 / n_observed.max(1) as f64;
    let is_constant = n_unique <= 1;
    let nonnegative = obs_min.map(|lo| lo >= 0.0);

    let max_abs_z = match (obs_min, obs_max) {
        (Some(lo), Some(hi)) if !is_constant && obs_finite.len() >= 2 => {
            let n = obs_finite.len() as f64;
            let mean = obs_finite.iter().sum::<f64>() / n;
            let var = obs_finite.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            let std = var.sqrt();
            (std > 0.0 && std.is_finite()).then(|| (hi - mean).abs().max((lo - mean).abs()) / std)
        }
        _ => None,
    };

    let partial_mask = (0 < n_missing && n_missing < n_rows).then_some(isna);
    let mut ev = blank(name, kind, n_rows, n_missing);
    ev.n_unique = n_unique;
    ev.unique_ratio = unique_ratio;
    ev.is_constant = is_constant;
    ev.n_zero = Some(n_zero);
    ev.observed_min = obs_min;
    ev.observed_max = obs_max;
    ev.integer_valued = integer_valued;
    ev.nonnegative = nonnegative;
    ev.minority_fraction = minority_fraction;
    ev.max_abs_z = max_abs_z;
    Profile {
        evidence: ev,
        partial_mask,
        values: Some(float_arr),
    }
}

/// Up to five most frequent tokens that do not parse as numbers, and the
/// count of such tokens (`_non_numeric_tokens`).
fn non_numeric_tokens(observed: &[&str]) -> (Vec<(String, usize)>, usize) {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    let mut n_bad = 0;
    for s in observed.iter().map(|s| s.trim()).filter(|s| !s.is_empty()) {
        if to_numeric(&strip_thousands(s)).is_none() {
            n_bad += 1;
            *counts.entry(s).or_default() += 1;
        }
    }
    let mut items: Vec<(String, usize)> = counts
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    items.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    items.truncate(5);
    (items, n_bad)
}

fn profile_text(name: &str, values: &[Option<String>], n_rows: usize) -> Profile {
    let isna: Vec<bool> = values.iter().map(Option::is_none).collect();
    let n_missing = isna.iter().filter(|&&b| b).count();
    let n_observed = n_rows - n_missing;
    let observed: Vec<&str> = values.iter().flatten().map(String::as_str).collect();

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for s in &observed {
        *counts.entry(s).or_default() += 1;
    }
    let n_unique = counts.len();
    let kind = detect_text_kind(&observed);
    let minority_fraction = (n_unique == 2)
        .then(|| *counts.values().min().expect("two values") as f64 / n_observed as f64);
    let (non_numeric_values, n_non_numeric) = if kind == ColumnKind::NumericText {
        let (v, n) = non_numeric_tokens(&observed);
        (Some(v), Some(n))
    } else {
        (None, None)
    };

    let mut ev = blank(name, kind, n_rows, n_missing);
    ev.n_unique = n_unique;
    ev.unique_ratio = n_unique as f64 / n_observed.max(1) as f64;
    ev.is_constant = n_unique <= 1;
    ev.minority_fraction = minority_fraction;
    ev.non_numeric_values = non_numeric_values;
    ev.n_non_numeric = n_non_numeric;
    Profile {
        evidence: ev,
        partial_mask: (0 < n_missing && n_missing < n_rows).then_some(isna),
        values: None,
    }
}
