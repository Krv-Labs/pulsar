use pyo3::prelude::*;

mod error;
mod impute;
mod scale;
mod pca;
mod ballmapper;
mod pseudolaplacian;
mod cosmic;

#[pymodule]
fn _pulsar(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(impute::impute_column, m)?)?;
    m.add_class::<scale::StandardScaler>()?;
    m.add_class::<pca::PCA>()?;
    m.add_class::<ballmapper::BallMapper>()?;
    m.add_function(wrap_pyfunction!(ballmapper::ball_mapper_grid, m)?)?;
    m.add_function(wrap_pyfunction!(pseudolaplacian::pseudo_laplacian, m)?)?;
    m.add_class::<cosmic::CosmicGraph>()?;
    Ok(())
}
