use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use pulsar_core::minhash::MinHashSignatures;
use pulsar_core::{cosmic_edges_minhash, CosmicGraphInner, PulsarError};

use crate::ballmapper::BallMapper;
use crate::pseudolaplacian::SparsePseudoLaplacian;

type WeightedEdge = (usize, usize, f64);

#[pyclass]
pub struct CosmicGraph {
    inner: CosmicGraphInner,
}

#[pymethods]
impl CosmicGraph {
    #[staticmethod]
    pub fn from_pseudo_laplacian<'py>(
        _py: Python<'py>,
        l: PyReadonlyArray2<'py, i64>,
        threshold: f64,
    ) -> PyResult<Self> {
        let arr = l.as_array().to_owned();
        let inner = CosmicGraphInner::from_pseudo_laplacian(&arr, threshold);
        Ok(CosmicGraph { inner })
    }

    /// Build a CosmicGraph from a sparse pseudo-Laplacian (see
    /// [`accumulate_pseudo_laplacians_sparse`]) without materializing an n×n matrix.
    /// Pass `threshold = 0.0` for exact parity with the dense construction path.
    #[staticmethod]
    pub fn from_pseudo_laplacian_sparse(
        spl: PyRef<SparsePseudoLaplacian>,
        threshold: f64,
    ) -> PyResult<Self> {
        let inner = CosmicGraphInner::from_pseudo_laplacian_sparse(
            spl.core.n,
            &spl.core.diag,
            &spl.core.offdiag,
            threshold,
        );
        Ok(CosmicGraph { inner })
    }

    /// Build a CosmicGraph directly from ball memberships via MinHash + LSH, the
    /// approximate construction path. Bypasses the exact pseudo-Laplacian (whose
    /// Σ_c |B_c|² pair materialization is the real bottleneck): edge weights are
    /// unbiased Jaccard estimates of the points' ball-sets with `Var = J(1−J)/d`.
    ///
    /// `d` is the signature depth (accuracy/speed knob; error is size-independent).
    /// `seed` makes the (randomized) construction reproducible. Output is the same
    /// sparse representation as [`CosmicGraph::from_pseudo_laplacian_sparse`], so the
    /// downstream interpretation layer (threshold selection, components) is unchanged.
    #[staticmethod]
    #[pyo3(signature = (ball_maps, n, d=256, seed=42))]
    pub fn from_ball_maps_minhash(
        ball_maps: Vec<PyRef<BallMapper>>,
        n: usize,
        d: usize,
        seed: u64,
    ) -> PyResult<Self> {
        if d == 0 {
            return Err(PulsarError::InvalidParameter {
                msg: "minhash signature depth d must be >= 1".to_string(),
            }
            .into());
        }
        let balls: Vec<&[usize]> = ball_maps
            .iter()
            .flat_map(|bm| bm.core.nodes.iter().map(|v| v.as_slice()))
            .collect();
        let edges = cosmic_edges_minhash(&balls, n, d, seed);
        Ok(CosmicGraph {
            inner: CosmicGraphInner::Sparse { n, edges },
        })
    }

    /// Spielman-Srivastava style spectral sparsifier using JL resistance sketches.
    #[pyo3(signature = (epsilon, seed=42, sketch_dim=None, sample_count=None, pcg_tol=1e-6, max_iter=1000))]
    pub fn spectral_sparsify(
        &self,
        epsilon: f64,
        seed: u64,
        sketch_dim: Option<usize>,
        sample_count: Option<usize>,
        pcg_tol: f64,
        max_iter: usize,
    ) -> PyResult<Self> {
        let inner = self
            .inner
            .spectral_sparsify(epsilon, seed, sketch_dim, sample_count, pcg_tol, max_iter)?;
        Ok(CosmicGraph { inner })
    }

    #[getter]
    pub fn weighted_adj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.inner.weighted_adj().into_pyarray_bound(py))
    }

    #[getter]
    pub fn adj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u8>>> {
        Ok(self.inner.adj().into_pyarray_bound(py))
    }

    #[getter]
    pub fn n(&self) -> usize {
        self.inner.n()
    }

    pub fn weighted_edges(&self) -> Vec<WeightedEdge> {
        self.inner.weighted_edges()
    }

    #[getter]
    pub fn n_edges(&self) -> usize {
        self.inner.weighted_edges().len()
    }
}

/// Streaming MinHash signature accumulator for memory-bounded multi-dataset
/// construction (`fit_multi`). Folds batches of ball maps into a constant-size `d×n`
/// signature via element-wise `min`, then builds the cosmic graph in one shot — the
/// signature never grows with the number of ball maps, so batches can be discarded as
/// they are consumed (matching the exact path's `merge_in_place` streaming).
#[pyclass]
pub struct MinHashAccumulator {
    inner: MinHashSignatures,
}

#[pymethods]
impl MinHashAccumulator {
    #[new]
    #[pyo3(signature = (n, d=256, seed=42))]
    pub fn new(n: usize, d: usize, seed: u64) -> PyResult<Self> {
        if d == 0 {
            return Err(PulsarError::InvalidParameter {
                msg: "minhash signature depth d must be >= 1".to_string(),
            }
            .into());
        }
        Ok(MinHashAccumulator {
            inner: MinHashSignatures::new(n, d, seed),
        })
    }

    /// Fold a batch of ball maps into the running signature.
    pub fn accumulate(&mut self, ball_maps: Vec<PyRef<BallMapper>>) {
        let balls: Vec<&[usize]> = ball_maps
            .iter()
            .flat_map(|bm| bm.core.nodes.iter().map(|v| v.as_slice()))
            .collect();
        self.inner.accumulate(&balls);
    }

    /// Build the cosmic graph (LSH banding + candidate weight estimation) from the
    /// accumulated signature.
    pub fn to_cosmic_graph(&self) -> CosmicGraph {
        CosmicGraph {
            inner: CosmicGraphInner::Sparse {
                n: self.inner.n(),
                edges: self.inner.edges(),
            },
        }
    }
}
