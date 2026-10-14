use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

use pulsar_core::ballmapper as core;
use pulsar_core::PulsarError;

/// Reject empty point clouds before they reach core `assert!`s.
fn validate_points(nrows: usize, ncols: usize) -> Result<(), PulsarError> {
    if nrows == 0 {
        return Err(PulsarError::InvalidParameter {
            msg: "BallMapper input has 0 rows (points)".to_string(),
        });
    }
    if ncols == 0 {
        return Err(PulsarError::InvalidParameter {
            msg: "BallMapper input has 0 columns (features)".to_string(),
        });
    }
    Ok(())
}

/// A fitted Ball Mapper complex.
///
/// Ball Mapper decomposes a point cloud into overlapping balls and
/// represents connectivity as a graph. Designed for large-scale EHR data.
#[pyclass]
pub struct BallMapper {
    pub core: core::BallMapper,
}

#[pymethods]
impl BallMapper {
    /// Create a new Ball Mapper with given radius.
    #[new]
    pub fn new(eps: f64) -> Self {
        BallMapper {
            core: core::BallMapper::new(eps),
        }
    }

    /// Fit the Ball Mapper to a point cloud.
    pub fn fit(&mut self, py: Python<'_>, points: PyReadonlyArray2<f64>) -> PyResult<()> {
        let view = points.as_array();
        validate_points(view.nrows(), view.ncols())?;
        let arr = view.as_standard_layout().into_owned();
        py.allow_threads(|| self.core.fit(arr.view()));
        Ok(())
    }

    #[getter]
    pub fn nodes(&self) -> Vec<Vec<usize>> {
        self.core.nodes.clone()
    }

    #[getter]
    pub fn edges(&self) -> Vec<(usize, usize)> {
        self.core.edges.clone()
    }

    #[getter]
    pub fn eps(&self) -> f64 {
        self.core.eps
    }

    pub fn n_nodes(&self) -> usize {
        self.core.n_nodes()
    }

    pub fn n_edges(&self) -> usize {
        self.core.n_edges()
    }
}

/// Run Ball Mapper for every (embedding, epsilon) pair in parallel.
///
/// This is the main entry point for grid search. Parallelised across all
/// combinations using rayon for maximum throughput on large datasets.
///
/// Complexity per fit: O(n * k) where k = number of balls.
/// No O(n²) memory allocation - scales to large EHR datasets.
#[pyfunction]
pub fn ball_mapper_grid(
    py: Python<'_>,
    embeddings: Vec<PyReadonlyArray2<f64>>,
    epsilons: Vec<f64>,
) -> PyResult<Vec<BallMapper>> {
    let mut owned = Vec::with_capacity(embeddings.len());
    for e in &embeddings {
        let view = e.as_array();
        validate_points(view.nrows(), view.ncols())?;
        owned.push(view.as_standard_layout().into_owned());
    }
    let results = py.allow_threads(|| core::ball_mapper_grid(&owned, &epsilons));
    Ok(results
        .into_iter()
        .map(|core| BallMapper { core })
        .collect())
}
