"""Freeze Pulsar v0.5 (Python) behaviour into golden fixtures for the Rust port.

Writes ``tests/fixtures/golden/<case>/*.json`` and ``tests/fixtures/golden/
manifest.json``. The Rust parity harness (``crates/pulsar-parity``) reads these
files; every Python-only stage that is ported must reproduce them within the
tolerances declared in the manifest (see #60).

Run from the repo root against the last Python release:

    uv run maturin develop --release
    uv run python scripts/golden/generate.py

Regenerate only to capture an *intentional* behaviour change of the Python
reference. Never regenerate to silence a parity failure.
"""

from __future__ import annotations

import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pulsar import StandardScaler, ThemaRS, cosmic_clusters, pca_grid
from pulsar._pulsar import find_stable_thresholds_sparse
from pulsar.config import PulsarConfig, config_to_yaml, load_config
from pulsar.mcp.thresholds import (
    agent_threshold_options,
    prepare_threshold_graph_from_edges,
    structural_breakpoints,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "fixtures" / "golden"
PH_NUM_BINS = 256  # pulsar_core::DEFAULT_NUM_BINS

# Comparison policy per artifact, consumed by the Rust harness.
#   exact     — integers / index sets must match exactly
#   rtol      — floats compared with relative tolerance (abs floor 1e-12)
#   partition — labels compared up to permutation (adjusted Rand index >= min_ari)
#   info      — reference only (e.g. RNG-seeded sklearn output); not gating
TOLERANCES: dict[str, dict[str, Any]] = {
    "preprocessed": {"mode": "rtol", "rtol": 1e-12},
    "scaled": {"mode": "rtol", "rtol": 1e-12},
    "embeddings": {"mode": "rtol", "rtol": 1e-9},
    "pca_embeddings": {
        "mode": "rtol",
        "rtol": 1e-6,
        "note": "randomized SVD (Halko); a re-implementation may need per-column sign alignment",
    },
    "ball_maps": {"mode": "exact"},
    "cosmic_exact": {"mode": "rtol", "rtol": 1e-12},
    "cosmic_minhash": {"mode": "rtol", "rtol": 1e-12},
    "stability": {"mode": "rtol", "rtol": 1e-12},
    "graph": {"mode": "exact"},
    "threshold_options": {"mode": "rtol", "rtol": 1e-9},
    "breakpoints": {"mode": "rtol", "rtol": 1e-9},
    "clusters_agglomerative": {"mode": "partition", "min_ari": 1.0},
    "clusters_spectral": {
        "mode": "partition",
        "min_ari": 0.9,
        "note": "sklearn SpectralClustering is RNG-seeded (random_state=None)",
    },
}


def _write(case_dir: Path, name: str, payload: Any) -> None:
    path = case_dir / f"{name}.json"
    # json.dumps writes shortest round-trip float reprs; the Rust reader must
    # parse with serde_json's `float_roundtrip` feature to be bit-exact.
    path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")


def _matrix(a: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.float64)
    return {"shape": list(a.shape), "data": a.ravel(order="C").tolist()}


def _edges(edges: list[tuple[int, int, float]]) -> list[list[float]]:
    return [[int(i), int(j), float(w)] for i, j, w in sorted(edges)]


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _run_case(
    name: str, data: pd.DataFrame, config: dict[str, Any], n_clusters: int
) -> dict:
    case_dir = OUT / name
    case_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(case_dir / "input.csv", index=False)
    (case_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    exact_cfg = {
        **config,
        "cosmic_graph": {**config["cosmic_graph"], "construction": "exact"},
    }
    model = ThemaRS(exact_cfg).fit(data=data)

    pre = model.preprocessed_data
    _write(
        case_dir,
        "preprocessed",
        {"columns": list(pre.columns), **_matrix(pre.to_numpy(np.float64))},
    )

    X = pre.to_numpy(np.float64)
    X_scaled = np.asarray(StandardScaler().fit_transform(X))
    _write(case_dir, "scaled", _matrix(X_scaled))

    proj = model.config.projection
    # jl_grid / pca_grid return seed-outer, dimension-inner order.
    grid = [(d, s) for s in proj.seeds for d in proj.dimensions]
    _write(
        case_dir,
        "embeddings",
        {
            "method": proj.method,
            "center": proj.center,
            "grid": [
                {"dimension": d, "seed": s, **_matrix(e)}
                for (d, s), e in zip(grid, model._embeddings)
            ],
        },
    )
    pca = pca_grid(
        np.ascontiguousarray(X_scaled), list(proj.dimensions), list(proj.seeds)
    )
    _write(
        case_dir,
        "pca_embeddings",
        {
            "grid": [
                {"dimension": d, "seed": s, **_matrix(e)}
                for (d, s), e in zip(grid, pca)
            ]
        },
    )

    eps = model.config.ball_mapper.epsilons
    _write(
        case_dir,
        "ball_maps",
        [
            {
                "embedding_index": k // len(eps),
                "epsilon": float(bm.eps),
                "nodes": [sorted(int(p) for p in node) for node in bm.nodes],
                "edges": sorted([int(a), int(b)] for a, b in bm.edges),
            }
            for k, bm in enumerate(model.ball_maps)
        ],
    )

    n = int(model.cosmic_rust.n)
    _write(
        case_dir,
        "cosmic_exact",
        {"n": n, "edges": _edges(model.cosmic_rust.weighted_edges())},
    )

    minhash = ThemaRS(config).fit(data=data)
    spec = minhash.config.cosmic_graph
    _write(
        case_dir,
        "cosmic_minhash",
        {
            "n": n,
            "minhash_d": spec.minhash_d,
            "minhash_seed": spec.minhash_seed,
            "edges": _edges(minhash.cosmic_rust.weighted_edges()),
        },
    )

    edges = model.weighted_edges(threshold=0.0)
    stability = find_stable_thresholds_sparse(n, edges, num_bins=PH_NUM_BINS)
    thresholds = [float(t) for t in stability.thresholds]
    counts = [int(c) for c in stability.component_counts]
    _write(
        case_dir,
        "stability",
        {
            "num_bins": PH_NUM_BINS,
            "optimal_threshold": float(stability.optimal_threshold),
            "thresholds": thresholds,
            "component_counts": counts,
            "plateaus": [
                {
                    "start": float(p.start_threshold),
                    "end": float(p.end_threshold),
                    "component_count": int(p.component_count),
                }
                for p in stability.plateaus
            ],
        },
    )
    _write(
        case_dir,
        "graph",
        {
            "construction_threshold": float(model.resolved_construction_threshold),
            "weight_scale": float(model._weight_scale),
            "edges": sorted([int(u), int(v)] for u, v in model.cosmic_graph.edges()),
        },
    )

    tgraph = prepare_threshold_graph_from_edges(n, edges)
    options = agent_threshold_options(
        tgraph,
        stability.top_k_plateaus(10),
        thresholds,
        counts,
        policy="balanced",
        max_candidates=7,
    )
    _write(case_dir, "threshold_options", _jsonable(options))
    _write(
        case_dir,
        "breakpoints",
        _jsonable(structural_breakpoints(tgraph, thresholds, counts)),
    )

    for method in ("agglomerative", "spectral"):
        labels = cosmic_clusters(
            model.cosmic_graph, method=method, n_clusters=n_clusters
        )
        _write(
            case_dir,
            f"clusters_{method}",
            {"n_clusters": n_clusters, "labels": [int(x) for x in labels]},
        )

    return {
        "n_rows": int(len(data)),
        "n_features": int(X.shape[1]),
        "n_ball_maps": len(model.ball_maps),
    }


def _penguins() -> tuple[pd.DataFrame, dict[str, Any]]:
    data = pd.read_csv(ROOT / "demos" / "penguins" / "penguins.csv")
    numeric = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    config = {
        "run": {"name": "golden_penguins"},
        "preprocessing": {
            "drop_columns": ["rowid", "year"],
            "impute": {
                **{c: {"method": "fill_mean"} for c in numeric[:2]},
                "flipper_length_mm": {"method": "fill_median"},
                "body_mass_g": {"method": "sample_normal", "seed": 42},
                "sex": {"method": "sample_categorical", "seed": 7},
            },
            "encode": {c: {"method": "one_hot"} for c in ["species", "island", "sex"]},
        },
        "sweep": {
            "projection": {
                "method": "jl",
                "dimensions": {"values": [2, 3]},
                "seed": {"values": [42, 7]},
            },
            "ball_mapper": {"epsilon": {"values": [0.6, 1.0]}},
        },
        "cosmic_graph": {
            "construction": "minhash",
            "minhash_d": 256,
            "minhash_seed": 42,
            "construction_threshold": "auto",
        },
    }
    return data, config


def _blobs() -> tuple[pd.DataFrame, dict[str, Any]]:
    # Known answer: three well-separated isotropic blobs in 4-D.
    rng = np.random.default_rng(20261016)
    centers = np.array([[0, 0, 0, 0], [8, 8, 0, 0], [0, 8, 8, 8]], dtype=np.float64)
    X = np.vstack([c + rng.normal(scale=0.5, size=(50, 4)) for c in centers])
    data = pd.DataFrame(X, columns=["x0", "x1", "x2", "x3"])
    data["blob"] = np.repeat([0, 1, 2], 50)
    config = {
        "run": {"name": "golden_blobs"},
        "preprocessing": {"drop_columns": ["blob"]},
        "sweep": {
            "projection": {
                "method": "jl",
                "dimensions": {"values": [2, 3]},
                "seed": {"values": [42, 7]},
            },
            "ball_mapper": {"epsilon": {"values": [0.4, 0.8]}},
        },
        "cosmic_graph": {
            "construction": "minhash",
            "minhash_d": 256,
            "minhash_seed": 42,
            "construction_threshold": "auto",
        },
    }
    return data, config


CONFIG_SOURCES = (
    "params.yaml.sample",
    "demos/energy/coal_params.yaml",
    "demos/ehr/physionet_params.yaml",
    "demos/mmlu/mmlu_params.yaml",
    "tests/fixtures/golden/penguins/config.yaml",
    "tests/fixtures/golden/blobs/config.yaml",
)


def _resolved_config(cfg: PulsarConfig) -> dict[str, Any]:
    cg = cfg.cosmic_graph
    return {
        "run_name": cfg.run_name,
        "data": cfg.data,
        "drop_columns": list(cfg.drop_columns),
        "impute": {
            c: {"method": s.method, "seed": s.seed} for c, s in cfg.impute.items()
        },
        "encode": {
            c: {"method": s.method, "max_categories": s.max_categories}
            for c, s in cfg.encode.items()
        },
        "projection": {
            "method": cfg.projection.method,
            "dimensions": list(cfg.projection.dimensions),
            "seeds": list(cfg.projection.seeds),
            "center": cfg.projection.center,
        },
        "epsilons": list(cfg.ball_mapper.epsilons),
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
    }


def _config_cases() -> list[str]:
    """Freeze ``load_config`` on the shipped configs plus Python-emitted YAML."""
    out = OUT / "config"
    out.mkdir(parents=True, exist_ok=True)
    emitted = out / "emitted_sample.yaml"
    emitted.write_text(config_to_yaml(load_config(str(ROOT / "params.yaml.sample"))))
    sources = [*CONFIG_SOURCES, emitted.relative_to(ROOT).as_posix()]
    cases = []
    for source in sources:
        path = Path(source)
        if path.name == "config.yaml":
            name = path.parent.name
        else:
            name = path.name.replace(".yaml", "").replace(".", "_")
        resolved = _resolved_config(load_config(str(ROOT / source)))
        _write(out, name, {"source": source, "resolved": resolved})
        cases.append(name)
    return cases


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {}
    for name, factory, k in (("penguins", _penguins, 3), ("blobs", _blobs, 3)):
        data, config = factory()
        cases[name] = _run_case(name, data, config, k)
        print(f"{name}: {cases[name]}")
    config_cases = _config_cases()
    print(f"config: {config_cases}")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "reference": {
            "package": "thema-pulsar",
            "version": version("thema-pulsar"),
            "generator_commit": commit,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": version("scikit-learn"),
            "scipy": version("scipy"),
            "platform": f"{platform.system()}-{platform.machine()}",
        },
        "ingest": {
            "note": "input.csv was read with pandas.read_csv defaults; the Rust reader must treat these tokens as null",
            "null_tokens": sorted(pd._libs.parsers.STR_NA_VALUES),
        },
        "matrix_layout": "row-major; {shape: [rows, cols], data: [...]}",
        "edge_layout": "[i, j, weight] with i < j, sorted",
        "tolerances": TOLERANCES,
        "cases": cases,
        "config_cases": config_cases,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    )


if __name__ == "__main__":
    main()
