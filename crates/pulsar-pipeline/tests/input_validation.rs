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
    let grids: Vec<Vec<f64>> = [0.0, -0.5, f64::NAN, f64::INFINITY]
        .into_iter()
        .map(|eps| vec![0.5, eps])
        .chain([vec![]])
        .collect();
    for eps in grids {
        let mut cfg = config(JL);
        cfg.ball_mapper.epsilons = eps.clone();
        assert_eq!(fit_code(&cfg, &table), "INVALID_OPTION", "eps {eps:?}");

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
            "eps {eps:?}"
        );
    }
}

#[test]
fn non_finite_precomputed_embedding_is_rejected() {
    let table = table_with(3.0);
    let good = Array2::from_shape_fn((N_ROWS, 2), |(i, j)| (i * (j + 1)) as f64);
    for cell in [f64::NAN, f64::INFINITY] {
        let mut bad = good.clone();
        bad[[5, 1]] = cell;
        let opts = FitOptions {
            precomputed_embeddings: Some(vec![good.clone(), bad]),
            ..FitOptions::default()
        };
        match fit(&config(JL), &table, opts, None) {
            Ok(_) => panic!("fit accepted a non-finite embedding ({cell})"),
            Err(e) => {
                assert!(matches!(e, PipelineError::Core(_)), "{e:?}");
                assert_eq!(e.code(), "INVALID_PARAMETER");
                assert!(e.to_string().contains("row 5, column 1"), "{e}");
            }
        }
    }
    let opts = FitOptions {
        precomputed_embeddings: Some(vec![good]),
        ..FitOptions::default()
    };
    assert!(fit(&config(JL), &table, opts, None).is_ok());
}

fn trajectory_snapshots() -> Vec<Array2<f64>> {
    (0..2)
        .map(|t| Array2::from_shape_fn((N_ROWS, 2), |(i, j)| (i * (j + 1) + t) as f64))
        .collect()
}

#[test]
fn infinite_snapshot_cell_is_reported_in_the_callers_matrix() {
    let mut snapshots = trajectory_snapshots();
    snapshots[1][[5, 1]] = f64::INFINITY;
    for scale in [true, false] {
        match CosmicTrajectory::from_snapshots(
            &snapshots,
            &config(JL),
            &EntityIds::Positional,
            None,
            0.0,
            scale,
        ) {
            Ok(_) => panic!("from_snapshots accepted an infinite cell (scale={scale})"),
            Err(e) => {
                assert_eq!(e.code(), "INVALID_PARAMETER");
                let msg = e.to_string();
                assert!(msg.contains("snapshot 1: input contains"), "{msg}");
                assert!(msg.contains("(inf) at row 5, column 1"), "{msg}");
            }
        }
    }
}

#[test]
fn nan_similarity_threshold_is_rejected() {
    let snapshots = trajectory_snapshots();
    for threshold in [f64::NAN, f64::INFINITY, -0.25] {
        let result = CosmicTrajectory::from_snapshots(
            &snapshots,
            &config(JL),
            &EntityIds::Positional,
            None,
            threshold,
            true,
        );
        assert!(
            matches!(result, Err(PipelineError::TrajectoryInput(ref m)) if m.starts_with("similarity_threshold must be >= 0")),
            "threshold {threshold}: {:?}",
            result.err()
        );
    }
}
