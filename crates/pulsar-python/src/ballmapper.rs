use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

use pulsar_core::ballmapper as core;

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
    pub fn fit(&mut self, points: PyReadonlyArray2<f64>) -> PyResult<()> {
        self.core.fit(points.as_array());
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
    embeddings: Vec<PyReadonlyArray2<f64>>,
    epsilons: Vec<f64>,
) -> PyResult<Vec<BallMapper>> {
    let owned: Vec<_> = embeddings.iter().map(|e| e.as_array().to_owned()).collect();
    Ok(core::ball_mapper_grid(&owned, &epsilons)
        .into_iter()
        .map(|core| BallMapper { core })
        .collect())
}
