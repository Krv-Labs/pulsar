"""
PulsarConfig — W&B sweep-inspired YAML config loader.

Sweep parameters support three specification styles:
  values: [2, 3, 5]               → use list directly
  range: {min: 0.1, max: 1.0, steps: 10} → np.linspace
  distribution: {type: uniform, min: 0.1, max: 1.0} → (reserved for random search)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal
import numpy as np
import yaml


ALLOWED_COSMIC_GRAPH_KEYS = frozenset(
    {
        "construction_threshold",
        "neighborhood",
        "sparsify",
        "sparsify_epsilon",
        "sparsify_seed",
        "sparsify_sketch_dim",
        "sparsify_sample_count",
        "sparsify_pcg_tol",
        "sparsify_max_iter",
        "construction",
        "minhash_d",
        "minhash_seed",
    }
)
COSMIC_GRAPH_CONSTRUCTION_METHODS = ("minhash", "exact")
LEGACY_COSMIC_GRAPH_THRESHOLD_KEY = "threshold"
LEGACY_COSMIC_GRAPH_THRESHOLD_MESSAGE = (
    "Unsupported legacy key cosmic_graph.threshold. "
    "Use cosmic_graph.construction_threshold instead."
)
THRESHOLD_RANGE_MESSAGE = "Threshold must be finite and between 0.0 and 1.0."


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


def _validate_cosmic_graph_keys(cosmic_graph: Any) -> None:
    if cosmic_graph is None:
        return
    if not isinstance(cosmic_graph, dict):
        raise ValueError("cosmic_graph must be a mapping")
    if LEGACY_COSMIC_GRAPH_THRESHOLD_KEY in cosmic_graph:
        raise ValueError(LEGACY_COSMIC_GRAPH_THRESHOLD_MESSAGE)
    unknown = sorted(set(cosmic_graph) - ALLOWED_COSMIC_GRAPH_KEYS)
    if unknown:
        raise ValueError(
            f"Unsupported cosmic_graph key(s): {unknown}. "
            f"Valid keys: {sorted(ALLOWED_COSMIC_GRAPH_KEYS)}"
        )


def normalize_construction_threshold(value: Any) -> float | Literal["auto"]:
    if value == "auto":
        return "auto"
    threshold = float(value)
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError(THRESHOLD_RANGE_MESSAGE)
    return threshold


def _normalize_construction(value: Any) -> Literal["minhash", "exact"]:
    method = str(value)
    if method not in COSMIC_GRAPH_CONSTRUCTION_METHODS:
        raise ValueError(
            f"cosmic_graph.construction must be one of "
            f"{list(COSMIC_GRAPH_CONSTRUCTION_METHODS)}, got {value!r}."
        )
    return method  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ImputeSpec:
    method: (
        str  # sample_normal | sample_categorical | fill_mean | fill_median | fill_mode
    )
    seed: int = 42


@dataclass
class EncodeSpec:
    method: str  # one_hot
    max_categories: int | None = None  # None = warn at 50; set = hard error at limit


@dataclass
class PCASpec:
    dimensions: list[int] = field(default_factory=lambda: [2])
    seeds: list[int] = field(default_factory=lambda: [42])


@dataclass
class ProjectionSpec:
    method: Literal["jl", "pca"] = "jl"
    dimensions: list[int] = field(default_factory=lambda: [2])
    seeds: list[int] = field(default_factory=lambda: [42])
    center: bool = True


@dataclass
class BallMapperSpec:
    epsilons: list[float] = field(default_factory=lambda: [0.5])


@dataclass
class CosmicGraphSpec:
    construction_threshold: float | Literal["auto"] = "auto"
    neighborhood: str = "node"
    # Spectral sparsification is an opt-in hook, not a default. It runs *after* the
    # (already sparse) cosmic graph is built, so it is pure additional cost on the
    # construction path; its value is a leverage-aware, epsilon-controlled graph
    # that preserves spectrum/distances for downstream spectral analysis. See
    # ThemaRS.spectral_sparsify.
    # WARNING: it is SLOW on large datasets (solves a preconditioned-CG system per
    # JL sketch row) — do not enable it for routine structural analysis.
    sparsify: bool = False
    sparsify_epsilon: float = 1.0
    sparsify_seed: int = 42
    sparsify_sketch_dim: int | None = None
    sparsify_sample_count: int | None = None
    sparsify_pcg_tol: float = 1e-6
    sparsify_max_iter: int = 1000
    # Cosmic-graph construction method:
    #   "minhash" (default) — approximate; edge weights are unbiased MinHash Jaccard
    #     estimates of each point's ball-set, replacing the O(Σ|B_c|²) pair
    #     materialization with an O(d·M) sketch. Sub-quadratic and constant-memory.
    #   "exact" — the bit-identical sparse pseudo-Laplacian backbone. Choose it when
    #     exact, reproducible co-occurrence weights matter more than speed/memory.
    construction: Literal["minhash", "exact"] = "minhash"
    # MinHash signature depth (only used when construction == "minhash"). Edge weights
    # are unbiased Jaccard estimates with Var = J(1−J)/d, so accuracy is the only knob
    # and is size-independent (see pulsar.mcp.minhash_advisor). `minhash_seed` makes the
    # randomized construction reproducible. Defaults need no tuning.
    minhash_d: int = 256
    minhash_seed: int = 42


@dataclass
class PulsarConfig:
    data: str
    impute: dict[str, ImputeSpec]
    encode: dict[str, EncodeSpec]
    drop_columns: list[str]
    pca: PCASpec = field(default_factory=PCASpec)
    projection: ProjectionSpec = field(default_factory=ProjectionSpec)
    ball_mapper: BallMapperSpec = field(default_factory=BallMapperSpec)
    cosmic_graph: CosmicGraphSpec = field(default_factory=CosmicGraphSpec)
    n_reps: int = 4
    run_name: str = ""

    def __post_init__(self) -> None:
        default_projection = ProjectionSpec()
        if self.projection == default_projection and self.pca != PCASpec():
            self.projection = ProjectionSpec(
                method="jl",
                dimensions=list(self.pca.dimensions),
                seeds=list(self.pca.seeds),
                center=True,
            )


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

    encode_raw = pre.get("encode", {}) or {}
    encode: dict[str, EncodeSpec] = {}
    for col, spec in encode_raw.items():
        encode[col] = EncodeSpec(
            method=spec["method"],
            max_categories=spec.get("max_categories"),
        )

    # sweep section
    sweep = raw.get("sweep", {})

    projection_raw = sweep.get("projection")
    pca_raw = sweep.get("pca", {})
    if projection_raw is None:
        projection_raw = {
            "method": "jl",
            "dimensions": pca_raw.get("dimensions", [2]),
            "seed": pca_raw.get("seed", [42]),
            "center": True,
        }
    method = str(projection_raw.get("method", "jl")).lower()
    if method not in {"jl", "pca"}:
        raise ValueError("sweep.projection.method must be 'jl' or 'pca'")
    projection = ProjectionSpec(
        method=method,  # type: ignore[arg-type]
        dimensions=[
            int(d) for d in _expand_param(projection_raw.get("dimensions", [2]))
        ],
        seeds=[int(s) for s in _expand_param(projection_raw.get("seed", [42]))],
        center=bool(projection_raw.get("center", True)),
    )
    pca = PCASpec(
        dimensions=list(projection.dimensions),
        seeds=list(projection.seeds),
    )

    bm_raw = sweep.get("ball_mapper", {})
    ball_mapper = BallMapperSpec(
        epsilons=[float(e) for e in _expand_param(bm_raw.get("epsilon", [0.5]))],
    )

    # cosmic_graph section
    cg_raw = raw.get("cosmic_graph", {})
    _validate_cosmic_graph_keys(cg_raw)
    threshold_raw = cg_raw.get("construction_threshold", "auto")
    construction_threshold = normalize_construction_threshold(threshold_raw)
    cosmic_graph = CosmicGraphSpec(
        construction_threshold=construction_threshold,
        neighborhood=str(cg_raw.get("neighborhood", "node")),
        sparsify=bool(cg_raw.get("sparsify", False)),
        sparsify_epsilon=float(cg_raw.get("sparsify_epsilon", 1.0)),
        sparsify_seed=int(cg_raw.get("sparsify_seed", 42)),
        sparsify_sketch_dim=(
            None
            if cg_raw.get("sparsify_sketch_dim") is None
            else int(cg_raw.get("sparsify_sketch_dim"))
        ),
        sparsify_sample_count=(
            None
            if cg_raw.get("sparsify_sample_count") is None
            else int(cg_raw.get("sparsify_sample_count"))
        ),
        sparsify_pcg_tol=float(cg_raw.get("sparsify_pcg_tol", 1e-6)),
        sparsify_max_iter=int(cg_raw.get("sparsify_max_iter", 1000)),
        construction=_normalize_construction(cg_raw.get("construction", "minhash")),
        minhash_d=int(cg_raw.get("minhash_d", 256)),
        minhash_seed=int(cg_raw.get("minhash_seed", 42)),
    )

    # output section
    output = raw.get("output", {})
    n_reps = int(output.get("n_reps", 4))

    return PulsarConfig(
        data=data_path,
        impute=impute,
        encode=encode,
        drop_columns=drop_columns,
        pca=pca,
        projection=projection,
        ball_mapper=ball_mapper,
        cosmic_graph=cosmic_graph,
        n_reps=n_reps,
        run_name=run_name,
    )


def config_to_yaml(cfg: PulsarConfig) -> str:
    """Serialize a PulsarConfig to a reproducible YAML string.

    Inverse of ``load_config``; every field is written explicitly so the
    resulting YAML can recreate the exact same pipeline run.
    """
    # Preprocessing: drop_columns + impute + encode
    drop_line = str(cfg.drop_columns) if cfg.drop_columns else "[]"

    impute_block = ""
    if cfg.impute:
        impute_block = "\n  impute:"
        for col, spec in cfg.impute.items():
            impute_block += f"\n    {col}: {{method: {spec.method}, seed: {spec.seed}}}"

    encode_block = ""
    if cfg.encode:
        encode_block = "\n  encode:"
        for col, spec in cfg.encode.items():
            parts = f"method: {spec.method}"
            if spec.max_categories is not None:
                parts += f", max_categories: {spec.max_categories}"
            encode_block += f"\n    {col}: {{{parts}}}"

    # Construction threshold
    construction_threshold = cfg.cosmic_graph.construction_threshold
    threshold_str = (
        f'"{construction_threshold}"'
        if construction_threshold == "auto"
        else str(construction_threshold)
    )
    sparsify_sketch_dim = (
        "null"
        if cfg.cosmic_graph.sparsify_sketch_dim is None
        else str(cfg.cosmic_graph.sparsify_sketch_dim)
    )
    sparsify_sample_count = (
        "null"
        if cfg.cosmic_graph.sparsify_sample_count is None
        else str(cfg.cosmic_graph.sparsify_sample_count)
    )

    return f"""run:
  name: {cfg.run_name or "experiment"}
  data: {cfg.data}
preprocessing:
  drop_columns: {drop_line}{impute_block}{encode_block}
sweep:
  projection:
    method: {cfg.projection.method}
    dimensions:
      values: {list(cfg.projection.dimensions)}
    seed:
      values: {list(cfg.projection.seeds)}
    center: {str(cfg.projection.center).lower()}
  # Legacy mirror of sweep.projection (dims/seeds) for backward compatibility.
  # sweep.projection is the source of truth; the loader ignores sweep.pca when
  # sweep.projection is present. Kept in sync here — do not hand-edit only one.
  pca:
    dimensions:
      values: {list(cfg.projection.dimensions)}
    seed:
      values: {list(cfg.projection.seeds)}
  ball_mapper:
    epsilon:
      range:
        min: {min(cfg.ball_mapper.epsilons):.4f}
        max: {max(cfg.ball_mapper.epsilons):.4f}
        steps: {len(cfg.ball_mapper.epsilons)}
cosmic_graph:
  construction_threshold: {threshold_str}
  # sparsify: opt-in spectral sparsifier. SLOW on large N (per-JL-sketch CG
  # solves) and runs after the already-sparse graph is built — leave false
  # unless you need a spectrum-preserving graph for downstream spectral analysis.
  sparsify: {str(cfg.cosmic_graph.sparsify).lower()}
  sparsify_epsilon: {cfg.cosmic_graph.sparsify_epsilon}
  sparsify_seed: {cfg.cosmic_graph.sparsify_seed}
  sparsify_sketch_dim: {sparsify_sketch_dim}
  sparsify_sample_count: {sparsify_sample_count}
  sparsify_pcg_tol: {cfg.cosmic_graph.sparsify_pcg_tol}
  sparsify_max_iter: {cfg.cosmic_graph.sparsify_max_iter}
  # construction: "minhash" (approximate, fast, constant-memory; default) or
  # "exact" (bit-identical sparse pseudo-Laplacian backbone).
  construction: {cfg.cosmic_graph.construction}
  minhash_d: {cfg.cosmic_graph.minhash_d}
  minhash_seed: {cfg.cosmic_graph.minhash_seed}
output:
  n_reps: {cfg.n_reps}
"""
