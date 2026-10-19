"""
PulsarConfig — W&B sweep-inspired YAML config loader.

Sweep parameters support three specification styles:
  values: [2, 3, 5]               → use list directly
  range: {min: 0.1, max: 1.0, steps: 10} → np.linspace
  distribution: {type: uniform, min: 0.1, max: 1.0} → (reserved for random search)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import itertools
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


def _validate_fill_value(value: Any) -> float | int | str:
    if value is None:
        raise ValueError("ImputeSpec with method 'fill_value' requires 'value'.")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"ImputeSpec 'value' must be finite numeric or string, got bool {value!r}."
        )
    if isinstance(value, (int, float, np.number)):
        try:
            fval = float(value)
        except OverflowError as exc:
            raise ValueError(
                f"ImputeSpec 'value' must be finite, got {value!r}."
            ) from exc
        if not math.isfinite(fval):
            raise ValueError(f"ImputeSpec 'value' must be finite, got {value!r}.")
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        return value
    if isinstance(value, np.str_):
        return str(value)
    if not isinstance(value, str):
        raise ValueError(
            f"ImputeSpec 'value' must be finite numeric or str, got {type(value).__name__}."
        )
    return value


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
            self.value = _validate_fill_value(self.value)
        elif self.value is not None:
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


def _validate_indicator_group_columns(
    group_key: str, cols: Any, seen_columns: dict[str, str]
) -> list[str]:
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
    return col_list


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
        normalized[group_key] = _validate_indicator_group_columns(
            group_key, cols, seen_columns
        )
    return normalized


def _check_single_group_flag(
    group_name: str,
    per_col_flag_names: set[str],
    configured_cols: set[str],
    seen_group_flags: set[str],
) -> None:
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


_STOCHASTIC_IMPUTE_METHODS = frozenset({"sample_normal", "sample_categorical"})


def _check_config_group_flag_collisions(cfg: PulsarConfig) -> None:
    grouped_cols = {c for cols in cfg.missing_indicator_groups.values() for c in cols}
    drop_set = set(cfg.drop_columns)
    per_col_flag_names: set[str] = set()
    for col, spec in cfg.impute.items():
        if col in grouped_cols or col in drop_set:
            continue
        all_specs = cfg.impute_candidates.get(col) or [spec]
        if any(s.indicator for s in all_specs):
            per_col_flag_names.add(f"{col}_was_missing")
    configured_cols = set(cfg.impute) | set(cfg.encode) | drop_set | grouped_cols
    seen_group_flags: set[str] = set()
    for group_name in cfg.missing_indicator_groups:
        _check_single_group_flag(
            group_name, per_col_flag_names, configured_cols, seen_group_flags
        )


def _normalize_impute_inputs(
    impute: dict[str, Any],
    impute_candidates: dict[str, list[ImputeSpec]] | None,
) -> tuple[dict[str, ImputeSpec], dict[str, list[ImputeSpec]]]:
    primary: dict[str, ImputeSpec] = {}
    candidates: dict[str, list[ImputeSpec]] = {}
    raw_candidates = impute_candidates or {}

    for raw_col, val in (impute or {}).items():
        col = str(raw_col)
        if isinstance(val, (list, tuple)):
            if not val:
                raise ValueError(f"impute[{col!r}] candidate list must not be empty.")
            specs = [v for v in val]
            primary[col] = specs[0]
            candidates[col] = specs
        else:
            primary[col] = val
            candidates[col] = [val]

    for raw_col, spec_list in raw_candidates.items():
        col = str(raw_col)
        if not spec_list:
            raise ValueError(
                f"impute_candidates[{col!r}] candidate list must not be empty."
            )
        specs = list(spec_list)
        candidates[col] = specs
        primary.setdefault(col, specs[0])

    return primary, candidates


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
    n_imputations: int = 1
    impute_candidates: dict[str, list[ImputeSpec]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = str(self.data)
        self.run_name = str(self.run_name)
        self.n_reps = int(self.n_reps)
        self.n_imputations = int(self.n_imputations)
        if self.n_imputations < 1:
            raise ValueError(f"n_imputations must be >= 1, got {self.n_imputations}.")
        self.drop_columns = [str(c) for c in self.drop_columns]
        self.impute, self.impute_candidates = _normalize_impute_inputs(
            self.impute, self.impute_candidates
        )
        self.encode = {str(k): v for k, v in self.encode.items()}
        self.missing_indicator_groups = _validate_missing_indicator_groups(
            self.missing_indicator_groups
        )
        _check_config_group_flag_collisions(self)

        default_projection = ProjectionSpec()
        if self.projection == default_projection and self.pca != PCASpec():
            self.projection = ProjectionSpec(
                method="jl",
                dimensions=list(self.pca.dimensions),
                seeds=list(self.pca.seeds),
                center=True,
            )

    def preprocessing_variants(self) -> list[PulsarConfig]:
        """Expand this config into one PulsarConfig per imputation ensemble realization."""
        return expand_preprocessing_variants(self)


def _select_column_variant_spec(
    col: str,
    specs: list[ImputeSpec],
    variant_index: int,
    used_stochastic_seeds: dict[tuple[str, str, bool], set[int]],
) -> ImputeSpec:
    if variant_index < len(specs):
        chosen = specs[variant_index]
    else:
        stochastic_specs = [s for s in specs if s.method in _STOCHASTIC_IMPUTE_METHODS]
        if stochastic_specs:
            offset = variant_index - len(specs)
            chosen = stochastic_specs[offset % len(stochastic_specs)]
        else:
            chosen = specs[variant_index % len(specs)]

    if chosen.method in _STOCHASTIC_IMPUTE_METHODS:
        key = (col, chosen.method, chosen.indicator)
        seen_seeds = used_stochastic_seeds.setdefault(key, set())
        seed = int(chosen.seed)
        while seed in seen_seeds:
            seed += 1
        seen_seeds.add(seed)
        if seed != chosen.seed:
            chosen = replace(chosen, seed=seed)
    return chosen


def expand_preprocessing_variants(cfg: PulsarConfig) -> list[PulsarConfig]:
    """Expand multi-imputation candidates and ``n_imputations`` into concrete configs.

    Each returned ``PulsarConfig`` has a single ``ImputeSpec`` per column in
    ``variant.impute`` so ``preprocess_dataframe(data, variant)`` can execute it
    directly. Duplicate deterministic combinations are deduplicated automatically.
    """
    candidates_map: dict[str, list[ImputeSpec]] = {
        col: list(cfg.impute_candidates.get(col) or [spec])
        for col, spec in cfg.impute.items()
    }
    max_candidates = max((len(v) for v in candidates_map.values()), default=1)
    target_variants = max(int(cfg.n_imputations), max_candidates)
    if target_variants <= 1 or not candidates_map:
        return [cfg]

    used_stochastic_seeds: dict[tuple[str, str, bool], set[int]] = {}
    seen_signatures: set[tuple[tuple[str, str, int, Any, bool], ...]] = set()
    variants: list[PulsarConfig] = []

    for k in range(target_variants):
        variant_impute: dict[str, ImputeSpec] = {}
        sig_items: list[tuple[str, str, int, Any, bool]] = []
        for col, specs in candidates_map.items():
            chosen = _select_column_variant_spec(col, specs, k, used_stochastic_seeds)
            variant_impute[col] = chosen
            effective_seed = (
                int(chosen.seed) if chosen.method in _STOCHASTIC_IMPUTE_METHODS else 0
            )
            sig_items.append(
                (
                    col,
                    chosen.method,
                    effective_seed,
                    chosen.value,
                    chosen.indicator,
                )
            )
        signature = tuple(sig_items)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        variants.append(
            replace(
                cfg,
                impute=variant_impute,
                impute_candidates={c: [s] for c, s in variant_impute.items()},
                n_imputations=1,
            )
        )

    return variants or [cfg]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _parse_impute_spec_entry(spec: Any, col: str) -> list[ImputeSpec]:
    if isinstance(spec, list):
        if not spec:
            raise ValueError(
                f"preprocessing.impute[{col!r}] candidate list must not be empty."
            )
        expanded: list[ImputeSpec] = []
        for item in spec:
            expanded.extend(_parse_impute_spec_entry(item, col))
        return expanded
    if not isinstance(spec, dict):
        raise ValueError(
            f"preprocessing.impute[{col!r}] must be a mapping or list of mappings, "
            f"got {type(spec).__name__}."
        )
    methods = [str(m) for m in _expand_param(spec["method"])]
    seeds = [int(s) for s in _expand_param(spec.get("seed", 42))]
    raw_ind = spec.get("indicator", True)
    ind_values = (
        [True]
        if raw_ind is None
        else [True if v is None else v for v in _expand_param(raw_ind)]
    )
    val = spec.get("value")
    specs: list[ImputeSpec] = []
    for method, seed, indicator in itertools.product(methods, seeds, ind_values):
        specs.append(
            ImputeSpec(
                method=method,
                seed=seed,
                value=val if method == "fill_value" else spec.get("value"),
                indicator=indicator,
            )
        )
    if not specs:
        raise ValueError(
            f"preprocessing.impute[{col!r}] expanded to zero candidate specs."
        )
    return specs


def _parse_preprocessing(
    pre: dict,
) -> tuple[
    list[str],
    dict[str, ImputeSpec],
    dict[str, list[ImputeSpec]],
    dict[str, EncodeSpec],
    int,
]:
    drop_columns = [str(c) for c in (pre.get("drop_columns", []) or [])]
    _validate_missing_indicator_groups(pre.get("missing_indicator_groups"))
    raw_n_imp = pre.get("n_imputations", 1)
    if isinstance(raw_n_imp, bool) or raw_n_imp is None:
        raise ValueError(
            f"preprocessing.n_imputations must be an integer >= 1, got {raw_n_imp!r}."
        )
    n_imputations = int(raw_n_imp)
    if n_imputations < 1:
        raise ValueError(
            f"preprocessing.n_imputations must be >= 1, got {n_imputations}."
        )
    impute_candidates: dict[str, list[ImputeSpec]] = {
        str(col): _parse_impute_spec_entry(spec, str(col))
        for col, spec in (pre.get("impute") or {}).items()
    }
    impute: dict[str, ImputeSpec] = {
        col: specs[0] for col, specs in impute_candidates.items()
    }
    encode = {
        str(col): EncodeSpec(
            method=str(spec["method"]),
            max_categories=_optional_int(spec.get("max_categories")),
        )
        for col, spec in (pre.get("encode") or {}).items()
    }
    return drop_columns, impute, impute_candidates, encode, n_imputations


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
    (
        drop_columns,
        impute,
        impute_candidates,
        encode,
        n_imputations,
    ) = _parse_preprocessing(pre_raw)
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
        n_imputations=n_imputations,
        impute_candidates=impute_candidates,
    )


def _coerce_yaml_scalar(val: Any) -> Any:
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return float(val)
    return str(val)


def _format_impute_entry(spec: ImputeSpec) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "method": str(spec.method),
        "seed": int(spec.seed),
    }
    if spec.value is not None:
        entry["value"] = _coerce_yaml_scalar(spec.value)
    if not spec.indicator:
        entry["indicator"] = False
    return entry


def _format_encode_entry(spec: EncodeSpec) -> dict[str, Any]:
    entry: dict[str, Any] = {"method": str(spec.method)}
    if spec.max_categories is not None:
        entry["max_categories"] = int(spec.max_categories)
    return entry


def _format_preprocessing_yaml(cfg: PulsarConfig) -> dict[str, Any]:
    pre: dict[str, Any] = {
        "drop_columns": [str(c) for c in cfg.drop_columns],
    }
    if cfg.n_imputations > 1:
        pre["n_imputations"] = int(cfg.n_imputations)
    if cfg.missing_indicator_groups:
        pre["missing_indicator_groups"] = {
            str(name): [str(c) for c in cols]
            for name, cols in cfg.missing_indicator_groups.items()
        }
    if cfg.impute:
        formatted_impute: dict[str, Any] = {}
        for col, spec in cfg.impute.items():
            col_str = str(col)
            candidates = cfg.impute_candidates.get(col_str) or [spec]
            if len(candidates) > 1:
                formatted_impute[col_str] = [
                    _format_impute_entry(s) for s in candidates
                ]
            else:
                formatted_impute[col_str] = _format_impute_entry(candidates[0])
        pre["impute"] = formatted_impute
    if cfg.encode:
        pre["encode"] = {
            str(col): _format_encode_entry(spec) for col, spec in cfg.encode.items()
        }
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
