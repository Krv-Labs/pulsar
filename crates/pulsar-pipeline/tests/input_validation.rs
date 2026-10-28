//! Non-finite data and invalid Ball Mapper radii are rejected before any
//! ball map is built, so Ball Mapper only ever sees finite points.

use ndarray::Array2;
use pulsar_config::{from_yaml_str, PulsarConfig};
use pulsar_data::{Column, Table};
use pulsar_pipeline::{fit, CosmicTrajectory, EntityIds, FitOptions, PipelineError};

const N_ROWS: usize = 40;

fn table_with(cell: f64) -> Table {
    let a: Vec<f64> = (0..N_ROWS).map(|i| i as f64).collect();
    let mut b: Vec<f64> = (0..N_ROWS).map(|i| (i % 7) as f64).collect();
    b[11] = cell;
    Table::new(
        vec!["a".into(), "b".into()],
        vec![Column::Float(a), Column::Float(b)],
    )
    .unwrap()
}

fn config(yaml: &str) -> PulsarConfig {
    from_yaml_str(yaml).unwrap()
}

fn fit_code(cfg: &PulsarConfig, table: &Table) -> &'static str {
    match fit(cfg, table, FitOptions::default(), None) {
        Ok(_) => panic!("fit accepted invalid input"),
        Err(e) => e.code(),
    }
}

const JL: &str = "sweep: {projection: {dimensions: [2]}, ball_mapper: {epsilon: 0.5}}";
const PCA: &str =
    "sweep: {projection: {method: pca, dimensions: [2]}, ball_mapper: {epsilon: 0.5}}";

#[test]
fn nan_cell_without_imputation_is_rejected_by_preprocessing() {
    for yaml in [JL, PCA] {
        assert_eq!(
            fit_code(&config(yaml), &table_with(f64::NAN)),
            "NAN_REMAINING"
        );
    }
}

#[test]
fn infinite_cell_is_rejected_before_ball_mapper() {
    // Preprocessing names the cell before any projection runs. The projections
    // keep their own finiteness check for programmatic callers (see the
    // pulsar-core unit tests); before it, PCA hung in the SVD on such input.
    for yaml in [JL, PCA] {
        for cell in [f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                fit_code(&config(yaml), &table_with(cell)),
                "NON_FINITE_VALUE",
                "{yaml} {cell}"
            );
        }
    }
}

#[test]
fn finite_table_fits() {
    assert!(fit(&config(JL), &table_with(3.0), FitOptions::default(), None).is_ok());
}

#[test]
fn non_positive_or_non_finite_epsilon_is_rejected() {
    let table = table_with(3.0);
    let snapshots = [Array2::from_shape_fn((N_ROWS, 2), |(i, j)| {
        (i * (j + 1)) as f64
    })];
    for eps in [0.0, -0.5, f64::NAN, f64::INFINITY] {
        let mut cfg = config(JL);
        cfg.ball_mapper.epsilons = vec![0.5, eps];
        assert_eq!(fit_code(&cfg, &table), "INVALID_OPTION", "eps {eps}");

        let trajectory = CosmicTrajectory::from_snapshots(
            &snapshots,
            &cfg,
            &EntityIds::Positional,
            None,
            0.0,
            true,
        );
        assert!(
            matches!(
                trajectory,
                Err(PipelineError::InvalidOption {
                    what: "ball_mapper.epsilons"
                })
            ),
            "eps {eps}"
        );
    }
}
