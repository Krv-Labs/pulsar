//! Sweep-parameter expansion (`values`, `range`, `distribution`).

use serde_json::Number;

use crate::error::{ConfigError, ErrorCode, Pointer};
use crate::file::ParamSpec;

/// Default `steps` for `{range: ...}` when omitted.
pub const DEFAULT_RANGE_STEPS: u64 = 10;

/// `numpy.linspace(start, stop, num)` (endpoint included), reproduced
/// bit-for-bit: `start + i * step`, with the last element pinned to `stop`.
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    match num {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let div = (num - 1) as f64;
            let delta = stop - start;
            let step = delta / div;
            let mut out: Vec<f64> = if step == 0.0 {
                (0..num).map(|i| (i as f64 / div) * delta + start).collect()
            } else {
                (0..num).map(|i| i as f64 * step + start).collect()
            };
            out[num - 1] = stop;
            out
        }
    }
}

/// Expand a spec into its flat list of values.
pub(crate) fn expand(spec: &ParamSpec, at: &Pointer) -> Result<Vec<Number>, ConfigError> {
    let values = match spec {
        ParamSpec::List(v) | ParamSpec::Values { values: v } => v.clone(),
        ParamSpec::Scalar(n) => vec![n.clone()],
        ParamSpec::Range { range } => {
            let steps = range.steps.unwrap_or(DEFAULT_RANGE_STEPS);
            let at = at.key("range").key("steps");
            let steps = usize::try_from(steps)
                .ok()
                .filter(|s| *s <= 1_000_000)
                .ok_or_else(|| {
                    ConfigError::new(
                        at.clone(),
                        ErrorCode::OutOfRange,
                        format!("range.steps must be at most 1000000, got {steps}."),
                    )
                })?;
            linspace(range.min, range.max, steps)
                .into_iter()
                .map(|x| {
                    Number::from_f64(x).ok_or_else(|| {
                        ConfigError::new(
                            at.clone(),
                            ErrorCode::InvalidValue,
                            "range produced a non-finite value.",
                        )
                    })
                })
                .collect::<Result<_, _>>()?
        }
        ParamSpec::Distribution { distribution } => {
            let kind = distribution.kind.as_deref().unwrap_or("uniform");
            if kind != "uniform" {
                return Err(ConfigError::new(
                    at.key("distribution").key("type"),
                    ErrorCode::InvalidValue,
                    format!("Unsupported distribution type {kind:?}."),
                )
                .with_allowed(["uniform"]));
            }
            [distribution.min, distribution.max]
                .into_iter()
                .filter_map(Number::from_f64)
                .collect()
        }
    };
    if values.is_empty() {
        return Err(ConfigError::new(
            at.clone(),
            ErrorCode::EmptyGrid,
            "Sweep parameter expands to no values; give at least one.",
        ));
    }
    Ok(values)
}

fn as_integer(n: &Number) -> Option<i128> {
    if let Some(i) = n.as_i64() {
        return Some(i128::from(i));
    }
    if let Some(u) = n.as_u64() {
        return Some(i128::from(u));
    }
    let f = n.as_f64()?;
    // Integral floats (e.g. from a range) are accepted; 2^63 bounds the cast.
    (f.fract() == 0.0 && f.abs() < 9.2e18).then_some(f as i128)
}

/// Expand and require integers in `[min, u64::MAX]`.
pub(crate) fn expand_integers(
    spec: &ParamSpec,
    at: &Pointer,
    min: u64,
    what: &str,
) -> Result<Vec<u64>, ConfigError> {
    expand(spec, at)?
        .iter()
        .map(|n| match as_integer(n) {
            Some(i) if i >= i128::from(min) && i <= i128::from(u64::MAX) => Ok(i as u64),
            Some(_) => Err(ConfigError::new(
                at.clone(),
                ErrorCode::OutOfRange,
                format!("{what} must be integers >= {min}, got {n}."),
            )),
            None => Err(ConfigError::new(
                at.clone(),
                ErrorCode::InvalidValue,
                format!(
                    "{what} must be integers, got {n}. A range over integers must land on \
                     whole numbers; list them with `values` instead."
                ),
            )),
        })
        .collect()
}

/// Expand and require finite floats strictly greater than zero.
pub(crate) fn expand_positive_floats(
    spec: &ParamSpec,
    at: &Pointer,
    what: &str,
) -> Result<Vec<f64>, ConfigError> {
    expand(spec, at)?
        .iter()
        .map(|n| match n.as_f64() {
            Some(x) if x.is_finite() && x > 0.0 => Ok(x),
            _ => Err(ConfigError::new(
                at.clone(),
                ErrorCode::OutOfRange,
                format!("{what} must be finite and > 0, got {n}."),
            )),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_edge_cases() {
        assert!(linspace(0.0, 1.0, 0).is_empty());
        assert_eq!(linspace(0.3, 2.5, 1), vec![0.3]);
        assert_eq!(linspace(1.0, 1.0, 3), vec![1.0, 1.0, 1.0]);
        assert_eq!(linspace(0.0, 1.0, 5), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        // Endpoint pinned exactly: 3 * step + 0.01 is 0.060000000000000005.
        assert_eq!(
            linspace(0.01, 0.06, 4),
            vec![0.01, 0.026666666666666665, 0.043333333333333335, 0.06]
        );
        let v = linspace(0.1, 1.5, 8);
        assert_eq!(*v.last().unwrap(), 1.5);
        // Reference values from numpy 2.5 `np.linspace`.
        assert_eq!((v[1], v[4], v[6]), (0.3, 0.8999999999999999, 1.3));
        let v = linspace(0.4, 2.5, 40);
        assert_eq!(
            (v[1], v[20], v[38]),
            (0.4538461538461539, 1.476923076923077, 2.4461538461538463)
        );
    }

    #[test]
    fn integer_ranges_must_be_integral() {
        let spec: ParamSpec =
            serde_json::from_str(r#"{"range": {"min": 2, "max": 10, "steps": 5}}"#).unwrap();
        assert_eq!(
            expand_integers(&spec, &Pointer::root(), 1, "dimensions").unwrap(),
            vec![2, 4, 6, 8, 10]
        );
        let spec: ParamSpec =
            serde_json::from_str(r#"{"range": {"min": 2, "max": 10, "steps": 4}}"#).unwrap();
        let err = expand_integers(&spec, &Pointer::root(), 1, "dimensions").unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidValue);
    }
}
