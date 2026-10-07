use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use pulsar_core::impute_column_inplace;

/// Python-facing wrapper around [`impute_column_inplace`].
///
/// Clones the input array, fills NaN values using the chosen method, and
/// returns a new array.  The original array is **not** modified.
///
/// # Parameters (Python)
/// - `values` (`np.ndarray[float64, 1D]`) — column to impute.
/// - `method` (`str`) — one of `"sample_normal"`, `"sample_categorical"`,
///   `"fill_mean"`, `"fill_median"`, `"fill_mode"`.
/// - `seed` (`int`, default `0`) — RNG seed; only used by `"sample_normal"`
///   and `"sample_categorical"`.
///
/// # Returns
/// A new `np.ndarray[float64, 1D]` with NaN values replaced.
///
/// # Raises
/// `ValueError` — if all values are NaN or the method name is unrecognised.
#[pyfunction]
#[pyo3(signature = (values, method, seed=0))]
pub fn impute_column<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    method: &str,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mut arr = values.as_array().to_owned();
    impute_column_inplace(arr.as_slice_mut().unwrap(), method, seed)?;
    Ok(arr.into_pyarray_bound(py))
}
