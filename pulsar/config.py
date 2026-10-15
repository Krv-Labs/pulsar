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
    method: str  # sample_normal | sample_categorical | fill_mean | fill_median | fill_mode | fill_value
    seed: int = 42
    value: float | int | str | None = None
    indicator: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.indicator, (bool, np.bool_)):
            raise ValueError(
                f"ImputeSpec indicator must be true or false (bool), got {self.indicator!r}."
            )
        self.indicator = bool(self.indicator)
        if self.method == "fill_value":
            if self.value is None:
                raise ValueError(
                    "ImputeSpec with method 'fill_value' requires 'value'."
                )
            if isinstance(self.value, (bool, np.bool_)):
                raise ValueError(
                    f"ImputeSpec 'value' must be finite numeric or string, got bool {self.value!r}."
                )
            if isinstance(self.value, (int, float, np.number)):
                try:
                    fval = float(self.value)
                except OverflowError as exc:
                    raise ValueError(
                        f"ImputeSpec 'value' must be finite, got {self.value!r}."
                    ) from exc
                if not math.isfinite(fval):
                    raise ValueError(
                        f"ImputeSpec 'value' must be finite, got {self.value!r}."
                    )
                if isinstance(self.value, np.floating):
                    self.value = float(self.value)
                elif isinstance(self.value, np.integer):
                    self.value = int(self.value)
            elif isinstance(self.value, np.str_):
                self.value = str(self.value)
            elif not isinstance(self.value, str):
                raise ValueError(
                    f"ImputeSpec 'value' must be finite numeric or str, got {type(self.value).__name__}."
                )
        else:
            if self.value is not None:
                raise ValueError(
                    f"ImputeSpec 'value' is only valid with method 'fill_value', got method {self.method!r}."
                )


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


def _validate_missing_indicator_groups(
    groups: Any,
) -> dict[str, list[str]]:
    if groups is None:
        return {}
    if not isinstance(groups, dict):
        raise ValueError(
            "preprocessing.missing_indicator_groups must be a mapping of group_name -> list of column names."
        )
    normalized: dict[str, list[str]] = {}
    seen_columns: dict[str, str] = {}
    for group_name, cols in groups.items():
        if not isinstance(group_name, str) or not group_name.strip():
            raise ValueError(
                "missing_indicator_groups group names must be non-empty strings."
            )
        group_key = str(group_name)
        if not isinstance(cols, (list, tuple)):
            raise ValueError(
                f"missing_indicator_groups[{group_key!r}] must be a list of column names."
            )
        col_list: list[str] = []
        for col in cols:
            if not isinstance(col, str) or not col.strip():
                raise ValueError(
                    f"missing_indicator_groups[{group_key!r}] column names must be non-empty strings."
                )
            col_str = str(col)
            if col_str in seen_columns:
                raise ValueError(
                    f"Column '{col_str}' appears in more than one missing_indicator_group "
                    f"({seen_columns[col_str]!r} and {group_key!r})."
                )
            seen_columns[col_str] = group_key
            col_list.append(col_str)
        normalized[group_key] = col_list
    return normalized


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
    missing_indicator_groups: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = str(self.data)
        self.run_name = str(self.run_name)
        self.n_reps = int(self.n_reps)
        self.drop_columns = [str(c) for c in self.drop_columns]
        self.impute = {str(k): v for k, v in self.impute.items()}
        self.encode = {str(k): v for k, v in self.encode.items()}
        self.missing_indicator_groups = _validate_missing_indicator_groups(
            self.missing_indicator_groups
        )
        grouped_cols = {
            c for cols in self.missing_indicator_groups.values() for c in cols
        }
        drop_set = set(self.drop_columns)
        per_col_flag_names = {
            f"{col}_was_missing"
            for col, spec in self.impute.items()
            if spec.indicator and col not in grouped_cols and col not in drop_set
        }
        configured_cols = set(self.impute) | set(self.encode) | drop_set | grouped_cols
        seen_group_flags: set[str] = set()
        for group_name in self.missing_indicator_groups:
            flag_name = f"{group_name}_was_missing"
            if flag_name in per_col_flag_names:
                raise ValueError(
                    f"missing indicator name collision: group {group_name!r} flag "
                    f"'{flag_name}' collides with per-column indicator."
                )
            if flag_name in configured_cols:
                raise ValueError(
                    f"missing indicator name collision: group {group_name!r} flag "
                    f"'{flag_name}' collides with configured column '{flag_name}'."
                )
            if flag_name in seen_group_flags:
                raise ValueError(
                    f"missing indicator name collision: duplicate group flag '{flag_name}'."
                )
            seen_group_flags.add(flag_name)

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


def _parse_preprocessing(
    pre: dict,
) -> tuple[list[str], dict[str, ImputeSpec], dict[str, EncodeSpec]]:
    drop_columns = [str(c) for c in (pre.get("drop_columns", []) or [])]
    _validate_missing_indicator_groups(pre.get("missing_indicator_groups"))
    impute = {
        str(col): ImputeSpec(
            method=str(spec["method"]),
            seed=int(spec.get("seed", 42)),
            value=spec.get("value"),
            indicator=True if spec.get("indicator") is None else spec["indicator"],
        )
        for col, spec in (pre.get("impute") or {}).items()
    }
    encode = {
        str(col): EncodeSpec(
            method=str(spec["method"]),
            max_categories=_optional_int(spec.get("max_categories")),
        )
        for col, spec in (pre.get("encode") or {}).items()
    }
    return drop_columns, impute, encode


def _parse_sweep(
    sweep: dict,
) -> tuple[ProjectionSpec, PCASpec, BallMapperSpec]:
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
    return projection, pca, ball_mapper


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)  # type: ignore[arg-type]


def _parse_cosmic_graph(cg_raw: dict) -> CosmicGraphSpec:
    _validate_cosmic_graph_keys(cg_raw)
    threshold_raw = cg_raw.get("construction_threshold", "auto")
    return CosmicGraphSpec(
        construction_threshold=normalize_construction_threshold(threshold_raw),
        neighborhood=str(cg_raw.get("neighborhood", "node")),
        sparsify=bool(cg_raw.get("sparsify", False)),
        sparsify_epsilon=float(cg_raw.get("sparsify_epsilon", 1.0)),
        sparsify_seed=int(cg_raw.get("sparsify_seed", 42)),
        sparsify_sketch_dim=_optional_int(cg_raw.get("sparsify_sketch_dim")),
        sparsify_sample_count=_optional_int(cg_raw.get("sparsify_sample_count")),
        sparsify_pcg_tol=float(cg_raw.get("sparsify_pcg_tol", 1e-6)),
        sparsify_max_iter=int(cg_raw.get("sparsify_max_iter", 1000)),
        construction=_normalize_construction(cg_raw.get("construction", "minhash")),
        minhash_d=int(cg_raw.get("minhash_d", 256)),
        minhash_seed=int(cg_raw.get("minhash_seed", 42)),
    )


def load_config(path_or_dict: str | dict) -> PulsarConfig:
    """Load a PulsarConfig from a YAML file path or a raw dict."""
    if isinstance(path_or_dict, str):
        with open(path_or_dict, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    else:
        raw = dict(path_or_dict)

    run_section = raw.get("run", {})
    run_name = run_section.get("name", "")
    data_path = run_section.get("data") or raw.get("data", "")

    pre_raw = raw.get("preprocessing", {}) or {}
    drop_columns, impute, encode = _parse_preprocessing(pre_raw)
    missing_indicator_groups = _validate_missing_indicator_groups(
        pre_raw.get("missing_indicator_groups")
    )
    projection, pca, ball_mapper = _parse_sweep(raw.get("sweep", {}))
    cosmic_graph = _parse_cosmic_graph(raw.get("cosmic_graph", {}))
    n_reps = int(raw.get("output", {}).get("n_reps", 4))

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
        missing_indicator_groups=missing_indicator_groups,
    )


def _coerce_yaml_scalar(val: Any) -> Any:
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return float(val)
    return str(val)


def _format_preprocessing_yaml(cfg: PulsarConfig) -> dict[str, Any]:
    pre: dict[str, Any] = {
        "drop_columns": [str(c) for c in cfg.drop_columns],
    }
    if cfg.missing_indicator_groups:
        pre["missing_indicator_groups"] = {
            str(name): [str(c) for c in cols]
            for name, cols in cfg.missing_indicator_groups.items()
        }
    if cfg.impute:
        impute_dict: dict[str, dict[str, Any]] = {}
        for col, spec in cfg.impute.items():
            entry: dict[str, Any] = {
                "method": str(spec.method),
                "seed": int(spec.seed),
            }
            if spec.value is not None:
                entry["value"] = _coerce_yaml_scalar(spec.value)
            if not spec.indicator:
                entry["indicator"] = False
            impute_dict[str(col)] = entry
        pre["impute"] = impute_dict
    if cfg.encode:
        encode_dict: dict[str, dict[str, Any]] = {}
        for col, spec in cfg.encode.items():
            entry = {"method": str(spec.method)}
            if spec.max_categories is not None:
                entry["max_categories"] = int(spec.max_categories)
            encode_dict[str(col)] = entry
        pre["encode"] = encode_dict
    return pre


def _yaml_nullable(value: object) -> str:
    return "null" if value is None else str(value)


def config_to_yaml(cfg: PulsarConfig) -> str:
    """Serialize a PulsarConfig to a reproducible YAML string.

    Inverse of ``load_config``; every field is written explicitly so the
    resulting YAML can recreate the exact same pipeline run.
    """
    epsilons = [float(e) for e in cfg.ball_mapper.epsilons]
    if (
        len(epsilons) >= 2
        and np.linspace(epsilons[0], epsilons[-1], len(epsilons)).tolist() == epsilons
    ):
        epsilon_spec: dict[str, Any] = {
            "range": {
                "min": float(epsilons[0]),
                "max": float(epsilons[-1]),
                "steps": int(len(epsilons)),
            }
        }
    else:
        epsilon_spec = {"values": epsilons}

    dims = [int(d) for d in cfg.projection.dimensions]
    seeds = [int(s) for s in cfg.projection.seeds]
    construction_threshold = cfg.cosmic_graph.construction_threshold
    threshold_val: str | float = (
        "auto" if construction_threshold == "auto" else float(construction_threshold)
    )

    doc: dict[str, Any] = {
        "run": {
            "name": str(cfg.run_name),
            "data": str(cfg.data),
        },
        "preprocessing": _format_preprocessing_yaml(cfg),
        "sweep": {
            "projection": {
                "method": str(cfg.projection.method),
                "dimensions": {"values": list(dims)},
                "seed": {"values": list(seeds)},
                "center": bool(cfg.projection.center),
            },
            "pca": {
                "dimensions": {"values": list(dims)},
                "seed": {"values": list(seeds)},
            },
            "ball_mapper": {
                "epsilon": epsilon_spec,
            },
        },
        "cosmic_graph": {
            "construction_threshold": threshold_val,
            "neighborhood": str(cfg.cosmic_graph.neighborhood),
            "sparsify": bool(cfg.cosmic_graph.sparsify),
            "sparsify_epsilon": float(cfg.cosmic_graph.sparsify_epsilon),
            "sparsify_seed": int(cfg.cosmic_graph.sparsify_seed),
            "sparsify_sketch_dim": _optional_int(cfg.cosmic_graph.sparsify_sketch_dim),
            "sparsify_sample_count": _optional_int(
                cfg.cosmic_graph.sparsify_sample_count
            ),
            "sparsify_pcg_tol": float(cfg.cosmic_graph.sparsify_pcg_tol),
            "sparsify_max_iter": int(cfg.cosmic_graph.sparsify_max_iter),
            "construction": str(cfg.cosmic_graph.construction),
            "minhash_d": int(cfg.cosmic_graph.minhash_d),
            "minhash_seed": int(cfg.cosmic_graph.minhash_seed),
        },
        "output": {
            "n_reps": int(cfg.n_reps),
        },
    }
    return yaml.safe_dump(doc, sort_keys=False)
