//! [`PulsarConfig`] → YAML. Lossless: `from_yaml_str(&to_yaml(c)) == c`.

use serde_json::Number;

use crate::expand::linspace;
use crate::file::{
    AutoLiteral, BallMapperFile, ColumnName, ConfigFile, CosmicGraphFile, EncodeFile,
    ImputeEntryFile, ImputeFile, IndicatorSpec, MethodSpec, OutputFile, ParamSpec,
    PreprocessingFile, ProjectionFile, RangeSpec, RunFile, SweepFile, ThresholdFile,
};
use crate::model::{Impute, PulsarConfig, Threshold};

fn impute_to_file(spec: &Impute) -> ImputeFile {
    ImputeFile {
        method: MethodSpec::Single(spec.method),
        seed: Some(ParamSpec::Scalar(spec.seed.into())),
        value: spec.value.clone(),
        indicator: (!spec.indicator).then_some(IndicatorSpec::Single(false)),
    }
}

/// Serialize a resolved config back to the file format, every field explicit.
pub fn to_file(cfg: &PulsarConfig) -> ConfigFile {
    let pre = &cfg.preprocessing;
    ConfigFile {
        version: Some(cfg.version),
        run: RunFile {
            name: Some(cfg.run_name.clone()),
            data: Some(cfg.data.clone()),
        },
        data: None,
        preprocessing: PreprocessingFile {
            drop_columns: pre
                .drop_columns
                .iter()
                .cloned()
                .map(ColumnName::Name)
                .collect(),
            missing_indicator_groups: pre
                .missing_indicator_groups
                .iter()
                .map(|(g, cols)| {
                    (
                        g.clone(),
                        cols.iter().cloned().map(ColumnName::Name).collect(),
                    )
                })
                .collect(),
            n_imputations: (pre.n_imputations > 1).then_some(u64::from(pre.n_imputations)),
            impute: pre
                .impute_columns()
                .into_iter()
                .filter_map(|col| {
                    let cands = pre.candidates_for(col);
                    let first = cands.first()?;
                    let entry = if cands.len() > 1 {
                        ImputeEntryFile::Candidates(cands.iter().map(impute_to_file).collect())
                    } else {
                        ImputeEntryFile::Single(impute_to_file(first))
                    };
                    Some((col.clone(), entry))
                })
                .collect(),
            encode: pre
                .encode
                .iter()
                .map(|(col, spec)| {
                    (
                        col.clone(),
                        EncodeFile {
                            method: spec.method,
                            max_categories: spec.max_categories,
                        },
                    )
                })
                .collect(),
        },
        sweep: SweepFile {
            projection: Some(ProjectionFile {
                method: Some(cfg.projection.method),
                dimensions: Some(ParamSpec::Values {
                    values: cfg
                        .projection
                        .dimensions
                        .iter()
                        .map(|&d| (d as u64).into())
                        .collect(),
                }),
                seed: Some(ParamSpec::Values {
                    values: cfg.projection.seeds.iter().map(|&s| s.into()).collect(),
                }),
                center: Some(cfg.projection.center),
            }),
            pca: None,
            ball_mapper: BallMapperFile {
                epsilon: Some(epsilon_spec(&cfg.ball_mapper.epsilons)),
            },
        },
        cosmic_graph: {
            let cg = &cfg.cosmic_graph;
            CosmicGraphFile {
                construction_threshold: Some(match cg.construction_threshold {
                    Threshold::Auto => ThresholdFile::Auto(AutoLiteral::Auto),
                    Threshold::Value(t) => ThresholdFile::Value(t),
                }),
                neighborhood: Some(cg.neighborhood.clone()),
                sparsify: Some(cg.sparsify),
                sparsify_epsilon: Some(cg.sparsify_epsilon),
                sparsify_seed: Some(cg.sparsify_seed),
                sparsify_sketch_dim: cg.sparsify_sketch_dim,
                sparsify_sample_count: cg.sparsify_sample_count,
                sparsify_pcg_tol: Some(cg.sparsify_pcg_tol),
                sparsify_max_iter: Some(cg.sparsify_max_iter),
                construction: Some(cg.construction),
                minhash_d: Some(cg.minhash_d),
                minhash_seed: Some(cg.minhash_seed),
            }
        },
        output: OutputFile {
            n_reps: Some(u64::from(cfg.n_reps)),
        },
    }
}

/// `range` when the epsilons are exactly a linspace, `values` otherwise.
fn epsilon_spec(eps: &[f64]) -> ParamSpec {
    if let [first, .., last] = eps {
        if linspace(*first, *last, eps.len()) == eps {
            return ParamSpec::Range {
                range: RangeSpec {
                    min: *first,
                    max: *last,
                    steps: Some(eps.len() as u64),
                },
            };
        }
    }
    ParamSpec::Values {
        values: eps.iter().filter_map(|&e| Number::from_f64(e)).collect(),
    }
}

/// Serialize a resolved config to YAML.
pub fn to_yaml(cfg: &PulsarConfig) -> String {
    serde_yaml_ng::to_string(&to_file(cfg)).expect("config file types always serialize")
}
