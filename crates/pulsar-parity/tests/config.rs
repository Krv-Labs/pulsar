//! `pulsar-config` must resolve the shipped configs exactly as Python
//! `load_config` did, including Python-emitted YAML (`config_to_yaml`).

use pulsar_config::PulsarConfig;
use pulsar_parity::{golden_dir, load, Manifest};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
struct ConfigCase {
    source: String,
    resolved: Value,
}

/// The same shape `generate.py::_resolved_config` (Python generator, see `tests/fixtures/golden/README.md`) writes.
fn resolved(cfg: &PulsarConfig) -> Value {
    let pre = &cfg.preprocessing;
    let cg = &cfg.cosmic_graph;
    json!({
        "run_name": cfg.run_name,
        "data": cfg.data,
        "drop_columns": pre.drop_columns,
        "impute": pre.impute.iter().map(|(c, s)| {
            (c.clone(), json!({"method": s.method, "seed": s.seed}))
        }).collect::<serde_json::Map<_, _>>(),
        "encode": pre.encode.iter().map(|(c, s)| {
            (c.clone(), json!({"method": s.method, "max_categories": s.max_categories}))
        }).collect::<serde_json::Map<_, _>>(),
        "projection": {
            "method": cfg.projection.method,
            "dimensions": cfg.projection.dimensions,
            "seeds": cfg.projection.seeds,
            "center": cfg.projection.center,
        },
        "epsilons": cfg.ball_mapper.epsilons,
        "cosmic_graph": {
            "construction_threshold": cg.construction_threshold,
            "neighborhood": cg.neighborhood,
            "sparsify": cg.sparsify,
            "sparsify_epsilon": cg.sparsify_epsilon,
            "sparsify_seed": cg.sparsify_seed,
            "sparsify_sketch_dim": cg.sparsify_sketch_dim,
            "sparsify_sample_count": cg.sparsify_sample_count,
            "sparsify_pcg_tol": cg.sparsify_pcg_tol,
            "sparsify_max_iter": cg.sparsify_max_iter,
            "construction": cg.construction,
            "minhash_d": cg.minhash_d,
            "minhash_seed": cg.minhash_seed,
        },
        "n_reps": cfg.n_reps,
    })
}

#[test]
fn configs_resolve_like_python() {
    let manifest = Manifest::load();
    assert!(
        !manifest.config_cases.is_empty(),
        "manifest lists no config cases"
    );
    let repo = golden_dir().join("../../..");
    for name in &manifest.config_cases {
        let case: ConfigCase = load("config", name);
        let cfg = pulsar_config::from_path(repo.join(&case.source))
            .unwrap_or_else(|e| panic!("{name} ({}) failed to load:\n{e}", case.source));
        assert_eq!(
            resolved(&cfg),
            case.resolved,
            "{name}: resolved config differs"
        );

        // And the Rust emitter round-trips it losslessly.
        let again = pulsar_config::from_yaml_str(&pulsar_config::to_yaml(&cfg))
            .unwrap_or_else(|e| panic!("{name}: emitted YAML failed to reload:\n{e}"));
        assert_eq!(again, cfg, "{name}: YAML round-trip changed the config");
    }
}
