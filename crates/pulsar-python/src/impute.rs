use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use pulsar_core::{fill_constant_inplace, impute_column_inplace, PulsarError};

/// Python-facing wrapper around [`impute_column_inplace`] and [`fill_constant_inplace`].
///
/// Clones the input array, fills NaN values using the chosen method, and
/// returns a new array.  The original array is **not** modified.
///
/// # Parameters (Python)
/// - `values` (`np.ndarray[float64, 1D]`) — column to impute.
/// - `method` (`str`) — one of `"sample_normal"`, `"sample_categorical"`,
///   `"fill_mean"`, `"fill_median"`, `"fill_mode"`, `"fill_value"`.
/// - `seed` (`int`, default `0`) — RNG seed; only used by `"sample_normal"`
///   and `"sample_categorical"`.
/// - `value` (`float | None`, default `None`) — constant replacement value,
///   required when `method == "fill_value"` and disallowed for other methods.
///
/// # Returns
/// A new `np.ndarray[float64, 1D]` with NaN values replaced.
///
/// # Raises
/// `ValueError` — if all values are NaN (except `fill_value`), the method name
/// is unrecognised, or `value` is missing/invalid.
#[pyfunction]
#[pyo3(signature = (values, method, seed=0, value=None))]
pub fn impute_column<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    method: &str,
    seed: u64,
    value: Option<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mut arr = values.as_array().as_standard_layout().into_owned();
    let slice = arr
        .as_slice_mut()
        .expect("standard-layout owned array is contiguous");
    if method == "fill_value" {
        let Some(v) = value else {
            return Err(PulsarError::InvalidParameter {
                msg: "fill_value requires a 'value' parameter".into(),
            }
            .into());
        };
        py.allow_threads(|| fill_constant_inplace(slice, v))?;
    } else {
        if value.is_some() {
            return Err(PulsarError::InvalidParameter {
                msg: format!(
                    "'value' is only valid with method 'fill_value', got method '{method}'"
                ),
            }
            .into());
        }
        py.allow_threads(|| impute_column_inplace(slice, method, seed))?;
    }
    Ok(arr.into_pyarray_bound(py))
}
