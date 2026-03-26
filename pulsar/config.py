"""
PulsarConfig — W&B sweep-inspired YAML config loader.

Sweep parameters support three specification styles:
  values: [2, 3, 5]               → use list directly
  range: {min: 0.1, max: 1.0, steps: 10} → np.linspace
  distribution: {type: uniform, min: 0.1, max: 1.0} → (reserved for random search)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Parameter grid helpers
# ---------------------------------------------------------------------------

def _expand_param(spec: Any) -> list:
    """Expand a sweep parameter spec into a flat list of values."""
    if isinstance(spec, list):
        return spec
    if not isinstance(spec, dict):
        return [spec]
    if "values" in spec:
        return list(spec["values"])
    if "range" in spec:
        r = spec["range"]
        return np.linspace(r["min"], r["max"], int(r.get("steps", 10))).tolist()
    if "distribution" in spec:
        d = spec["distribution"]
        dist_type = d.get("type", "uniform")
        if dist_type == "uniform":
            # Return the bounds; caller decides how to sample
            return [d["min"], d["max"]]
        raise ValueError(f"Unsupported distribution type: {dist_type!r}")
    raise ValueError(f"Cannot expand parameter spec: {spec!r}")


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ImputeSpec:
    method: str   # sample_normal | sample_categorical | fill_mean | fill_median | fill_mode
    seed: int = 42


@dataclass
class PCASpec:
    dimensions: list[int] = field(default_factory=lambda: [2])
    seeds: list[int] = field(default_factory=lambda: [42])


@dataclass
class BallMapperSpec:
    epsilons: list[float] = field(default_factory=lambda: [0.5])


@dataclass
class CosmicGraphSpec:
    threshold: float = 0.0
    neighborhood: str = "node"


@dataclass
class PulsarConfig:
    data: str
    impute: dict[str, ImputeSpec]
    drop_columns: list[str]
    pca: PCASpec
    ball_mapper: BallMapperSpec
    cosmic_graph: CosmicGraphSpec
    n_reps: int = 4
    run_name: str = ""


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path_or_dict: str | dict) -> PulsarConfig:
    """Load a PulsarConfig from a YAML file path or a raw dict."""
    if isinstance(path_or_dict, str):
        with open(path_or_dict) as f:
            raw = yaml.safe_load(f)
    else:
        raw = dict(path_or_dict)

    # run section (optional)
    run_section = raw.get("run", {})
    run_name = run_section.get("name", "")
    data_path = run_section.get("data") or raw.get("data", "")

    # preprocessing
    pre = raw.get("preprocessing", {})
    drop_columns = pre.get("drop_columns", []) or []
    impute_raw = pre.get("impute", {}) or {}
    impute: dict[str, ImputeSpec] = {}
    for col, spec in impute_raw.items():
        impute[col] = ImputeSpec(
            method=spec["method"],
            seed=int(spec.get("seed", 42)),
        )

    # sweep section
    sweep = raw.get("sweep", {})

    pca_raw = sweep.get("pca", {})
    pca = PCASpec(
        dimensions=[int(d) for d in _expand_param(pca_raw.get("dimensions", [2]))],
        seeds=[int(s) for s in _expand_param(pca_raw.get("seed", [42]))],
    )

    bm_raw = sweep.get("ball_mapper", {})
    ball_mapper = BallMapperSpec(
        epsilons=[float(e) for e in _expand_param(bm_raw.get("epsilon", [0.5]))],
    )

    # cosmic_graph section
    cg_raw = raw.get("cosmic_graph", {})
    cosmic_graph = CosmicGraphSpec(
        threshold=float(cg_raw.get("threshold", 0.0)),
        neighborhood=str(cg_raw.get("neighborhood", "node")),
    )

    # output section
    output = raw.get("output", {})
    n_reps = int(output.get("n_reps", 4))

    return PulsarConfig(
        data=data_path,
        impute=impute,
        drop_columns=drop_columns,
        pca=pca,
        ball_mapper=ball_mapper,
        cosmic_graph=cosmic_graph,
        n_reps=n_reps,
        run_name=run_name,
    )
