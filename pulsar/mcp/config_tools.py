from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import yaml

from pulsar.config import (
    ALLOWED_COSMIC_GRAPH_KEYS,
    LEGACY_COSMIC_GRAPH_THRESHOLD_KEY,
    LEGACY_COSMIC_GRAPH_THRESHOLD_MESSAGE,
    THRESHOLD_RANGE_MESSAGE,
    config_to_yaml,
    load_config,
    normalize_construction_threshold,
)
from pulsar.mcp.errors import classify_path, mcp_error
import numpy as np
from pulsar.analysis.characterization import NumericProfile
from pulsar.mcp.preprocessing import _recommend_preprocessing_block, _preprocessing_block_to_yaml



_ALLOWED_TOP_LEVEL = {"run", "preprocessing", "sweep", "cosmic_graph", "output"}
_ALLOWED_PREPROCESSING = {"drop_columns", "impute", "encode"}
_ALLOWED_SWEEP = {"pca", "ball_mapper"}
_ALLOWED_PCA = {"dimensions", "seed"}
_ALLOWED_BALL_MAPPER = {"epsilon"}
_ALLOWED_OUTPUT = {"n_reps"}
_ALLOWED_COSMIC_GRAPH = ALLOWED_COSMIC_GRAPH_KEYS
_SUPPORTED_IMPUTE_METHODS = {
    "sample_normal",
    "sample_categorical",
    "fill_mean",
    "fill_median",
    "fill_mode",
}
_SUPPORTED_ENCODE_METHODS = {"one_hot"}
_IMPUTE_METHOD_ALIASES = {
    "mean": "fill_mean",
    "median": "fill_median",
    "mode": "fill_mode",
}


@dataclass
class ValidationIssue:
    path: str
    message: str
    expected: str | None = None
    received: Any = None
    suggestion: str | None = None
    example_fix: str | None = None


@dataclass
class ValidationReport:
    ok: bool
    normalized_yaml: str | None
    resolved_dataset_path: str | None
    issues: list[ValidationIssue]
    error_code: str | None = None
    agent_action: str | None = None


@dataclass
class ConfigOverrideResult:
    config_yaml: str
    applied_overrides: list[str]
    diff: list[dict[str, Any]]  # [{path, old, new}, ...]


def parse_yaml_mapping(config_yaml: str) -> dict[str, Any]:
    stripped = config_yaml.strip()
    if stripped.startswith("```"):
        raise ValueError(
            "config_yaml must be raw YAML, not a fenced Markdown code block"
        )
    parsed = yaml.safe_load(config_yaml)
    if not isinstance(parsed, dict):
        raise ValueError("config_yaml must be a valid YAML mapping")
    return parsed


def validate_config_yaml(
    config_yaml: str,
    *,
    dataset_path: str | None = None,
) -> ValidationReport:
    issues: list[ValidationIssue] = []

    try:
        raw = parse_yaml_mapping(config_yaml)
    except ValueError as exc:
        error_code = "YAML_NOT_RAW" if "raw YAML" in str(exc) else "CONFIG_YAML_INVALID"
        return ValidationReport(
            ok=False,
            normalized_yaml=None,
            resolved_dataset_path=dataset_path,
            issues=[ValidationIssue(path="$", message=str(exc))],
            error_code=error_code,
            agent_action="Pass raw YAML only. Do not wrap config_yaml in Markdown fences.",
        )

    _validate_known_sections(raw, issues)
    _validate_run_section(raw.get("run"), issues)
    _validate_preprocessing(raw.get("preprocessing"), issues)
    _validate_sweep(raw.get("sweep"), issues)
    _validate_cosmic_graph(raw.get("cosmic_graph"), issues)
    _validate_output(raw.get("output"), issues)

    if dataset_path:
        raw.setdefault("run", {})["data"] = dataset_path

    run_data = raw.get("run", {}).get("data") or raw.get("data")
    if not run_data:
        issues.append(
            ValidationIssue(
                path="run.data",
                message="Dataset path is missing",
                expected="A readable CSV or Parquet path via run.data or dataset_id",
                suggestion="Pass dataset_id or set run.data in the YAML",
            )
        )
    elif not Path(run_data).expanduser().exists():
        issues.append(
            ValidationIssue(
                path="run.data",
                message=f"Dataset path does not exist: {run_data}",
                expected="A host-visible CSV or Parquet path",
                suggestion="Use ingest_dataset(path) on a server-visible path, then pass dataset_id",
            )
        )

    if issues:
        error_code = _infer_issue_error_code(issues)
        agent_action = (
            "Use create_config or refine_config to produce canonical Pulsar YAML, "
            "then validate_config before running the sweep."
        )
        if error_code in {"HOST_PATH_NOT_VISIBLE", "FILE_NOT_FOUND"}:
            agent_action = (
                "Use ingest_dataset on a host-visible path, then validate_config with "
                "dataset_id or set run.data to a readable host path."
            )
        return ValidationReport(
            ok=False,
            normalized_yaml=None,
            resolved_dataset_path=dataset_path or run_data,
            issues=issues,
            error_code=error_code,
            agent_action=agent_action,
        )

    try:
        cfg = load_config(raw)
    except (TypeError, ValueError, KeyError) as exc:
        return ValidationReport(
            ok=False,
            normalized_yaml=None,
            resolved_dataset_path=dataset_path or run_data,
            issues=[ValidationIssue(path="$", message=str(exc))],
            error_code="CONFIG_VALIDATION_FAILED",
            agent_action="Fix the reported config issue, then re-run validate_config.",
        )

    normalized_yaml = config_to_yaml(cfg)
    return ValidationReport(
        ok=True,
        normalized_yaml=normalized_yaml,
        resolved_dataset_path=cfg.data,
        issues=[],
    )


_VALID_OVERRIDE_KEYS = frozenset(
    {
        "run_name",
        "dataset_path",
        "pca_dims",
        "pca_seeds",
        "epsilon_values",
        "epsilon_range",
        "construction_threshold",
        "n_reps",
        "drop_columns",
        "impute",
        "encode",
        "raw",
    }
)


_DOTTED_PATH_ROOTS = frozenset(
    {"run", "sweep", "cosmic_graph", "output", "preprocessing"}
)


def _set_dotted_path(raw: dict[str, Any], path: str, value: Any) -> Any:
    """Walk dotted ``path`` into ``raw``, set the leaf, return the old value."""
    parts = path.split(".")
    cursor = raw
    for part in parts[:-1]:
        existing = cursor.get(part)
        if not isinstance(existing, dict):
            existing = {}
            cursor[part] = existing
        cursor = existing
    old = cursor.get(parts[-1])
    cursor[parts[-1]] = value
    return old


def flatten_overrides(nested: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat = {}
    for k, v in nested.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(flatten_overrides(v, f"{key}."))
        else:
            flat[key] = v
    return flat


def apply_overrides(
    config_yaml: str, overrides: dict[str, Any]
) -> ConfigOverrideResult:
    flat = flatten_overrides(overrides)

    prefix_map = {
        "sweep.pca_dims": "pca_dims",
        "sweep.pca_seeds": "pca_seeds",
        "sweep.epsilon_values": "epsilon_values",
        "sweep.epsilon_range": "epsilon_range",
        "cosmic_graph.construction_threshold": "construction_threshold",
        "output.n_reps": "n_reps",
        "preprocessing.drop_columns": "drop_columns",
        "preprocessing.impute": "impute",
        "preprocessing.encode": "encode",
        "run.run_name": "run_name",
        "run.dataset_path": "dataset_path",
    }

    normalized_overrides = {}
    original_keys = {}

    for k, v in flat.items():
        if k in prefix_map:
            flat_key = prefix_map[k]
            normalized_overrides[flat_key] = v
            original_keys[flat_key] = k
        elif k in _VALID_OVERRIDE_KEYS:
            normalized_overrides[k] = v
            original_keys[k] = k
        else:
            normalized_overrides[k] = v
            original_keys[k] = k

    dotted_overrides = {
        key: value for key, value in normalized_overrides.items() if "." in key
    }
    flat_overrides = {
        key: value for key, value in normalized_overrides.items() if "." not in key
    }

    unknown_dotted_roots = {
        key.split(".", 1)[0]
        for key in dotted_overrides
        if key.split(".", 1)[0] not in _DOTTED_PATH_ROOTS
    }
    if unknown_dotted_roots:
        raise ValueError(
            f"Unknown dotted override root(s): {sorted(unknown_dotted_roots)}. "
            f"Valid roots: {sorted(_DOTTED_PATH_ROOTS)}"
        )
    unknown_keys = set(flat_overrides.keys()) - _VALID_OVERRIDE_KEYS
    if unknown_keys:
        raise ValueError(
            f"Unknown override key(s): {sorted(unknown_keys)}. "
            f"Valid keys: {sorted(_VALID_OVERRIDE_KEYS)}, or use dotted "
            f"YAML paths under {sorted(_DOTTED_PATH_ROOTS)}."
        )

    raw = parse_yaml_mapping(config_yaml)
    applied: list[str] = []
    diff: list[dict[str, Any]] = []

    flat_key_to_canonical = {
        "pca_dims": "sweep.pca.dimensions.values",
        "pca_seeds": "sweep.pca.seed.values",
        "epsilon_values": "sweep.ball_mapper.epsilon.values",
        "epsilon_range": "sweep.ball_mapper.epsilon.range",
        "construction_threshold": "cosmic_graph.construction_threshold",
        "n_reps": "output.n_reps",
        "drop_columns": "preprocessing.drop_columns",
        "impute": "preprocessing.impute",
        "encode": "preprocessing.encode",
        "run_name": "run.name",
        "dataset_path": "run.data",
    }

    def _track(path: str, old: Any, new: Any) -> None:
        applied.append(path)
        diff.append({"path": path, "old": old, "new": new})
        for f_key, can_path in flat_key_to_canonical.items():
            if path == can_path:
                orig = original_keys.get(f_key)
                if orig and orig != path and orig not in applied:
                    applied.append(orig)
                break

    for path, value in dotted_overrides.items():
        old = _set_dotted_path(raw, path, value)
        _track(path, old, value)

    overrides = flat_overrides

    if "run_name" in overrides:
        old = raw.get("run", {}).get("name")
        raw.setdefault("run", {})["name"] = overrides["run_name"]
        _track("run.name", old, overrides["run_name"])
    if "dataset_path" in overrides:
        old = raw.get("run", {}).get("data")
        raw.setdefault("run", {})["data"] = overrides["dataset_path"]
        _track("run.data", old, overrides["dataset_path"])
    if "pca_dims" in overrides:
        old = raw.get("sweep", {}).get("pca", {}).get("dimensions")
        raw.setdefault("sweep", {}).setdefault("pca", {})["dimensions"] = {
            "values": list(overrides["pca_dims"])
        }
        _track("sweep.pca.dimensions.values", old, list(overrides["pca_dims"]))
    if "pca_seeds" in overrides:
        old = raw.get("sweep", {}).get("pca", {}).get("seed")
        raw.setdefault("sweep", {}).setdefault("pca", {})["seed"] = {
            "values": list(overrides["pca_seeds"])
        }
        _track("sweep.pca.seed.values", old, list(overrides["pca_seeds"]))
    if "epsilon_values" in overrides:
        old = raw.get("sweep", {}).get("ball_mapper", {}).get("epsilon")
        raw.setdefault("sweep", {}).setdefault("ball_mapper", {})["epsilon"] = {
            "values": list(overrides["epsilon_values"])
        }
        _track(
            "sweep.ball_mapper.epsilon.values", old, list(overrides["epsilon_values"])
        )
    if "epsilon_range" in overrides:
        old = raw.get("sweep", {}).get("ball_mapper", {}).get("epsilon")
        raw.setdefault("sweep", {}).setdefault("ball_mapper", {})["epsilon"] = {
            "range": dict(overrides["epsilon_range"])
        }
        _track("sweep.ball_mapper.epsilon.range", old, dict(overrides["epsilon_range"]))
    if "construction_threshold" in overrides:
        old = raw.get("cosmic_graph", {}).get("construction_threshold")
        raw.setdefault("cosmic_graph", {})["construction_threshold"] = overrides[
            "construction_threshold"
        ]
        _track(
            "cosmic_graph.construction_threshold",
            old,
            overrides["construction_threshold"],
        )
    if "n_reps" in overrides:
        old = raw.get("output", {}).get("n_reps")
        raw.setdefault("output", {})["n_reps"] = overrides["n_reps"]
        _track("output.n_reps", old, overrides["n_reps"])
    if "drop_columns" in overrides:
        old = raw.get("preprocessing", {}).get("drop_columns")
        raw.setdefault("preprocessing", {})["drop_columns"] = list(
            overrides["drop_columns"]
        )
        _track("preprocessing.drop_columns", old, list(overrides["drop_columns"]))
    if "impute" in overrides:
        old = dict(raw.get("preprocessing", {}).get("impute", {}))
        raw.setdefault("preprocessing", {}).setdefault("impute", {}).update(
            overrides["impute"]
        )
        _track("preprocessing.impute", old, overrides["impute"])
    if "encode" in overrides:
        old = dict(raw.get("preprocessing", {}).get("encode", {}))
        raw.setdefault("preprocessing", {}).setdefault("encode", {}).update(
            overrides["encode"]
        )
        _track("preprocessing.encode", old, overrides["encode"])
    if "raw" in overrides:
        _deep_merge(raw, overrides["raw"])
        _track("raw", "(deep merge)", overrides["raw"])

    cfg = load_config(raw)
    return ConfigOverrideResult(
        config_yaml=config_to_yaml(cfg), applied_overrides=applied, diff=diff
    )


def render_validation_report(report: ValidationReport) -> str:
    if not report.ok:
        details = {
            "resolved_dataset_path": report.resolved_dataset_path,
            "issues": [asdict(issue) for issue in report.issues],
        }
        if (
            report.error_code in {"HOST_PATH_NOT_VISIBLE", "FILE_NOT_FOUND"}
            and len(report.issues) == 1
            and report.issues[0].path == "run.data"
            and report.issues[0].message.startswith("Dataset path does not exist: ")
        ):
            attempted_path = report.issues[0].message.removeprefix(
                "Dataset path does not exist: "
            )
            details["path_context"] = classify_path(attempted_path)
        return mcp_error(
            "validate_config",
            "Config validation failed.",
            error_code=report.error_code or "CONFIG_VALIDATION_FAILED",
            agent_action=report.agent_action,
            details=details,
        )

    payload = {
        "status": "ok",
        "resolved_dataset_path": report.resolved_dataset_path,
        "issues": [asdict(issue) for issue in report.issues],
    }
    if report.normalized_yaml is not None:
        payload["normalized_config_yaml"] = report.normalized_yaml
    return json.dumps(payload, indent=2)


def _validate_known_sections(
    raw: dict[str, Any], issues: list[ValidationIssue]
) -> None:
    if "dataset" in raw:
        issues.append(
            ValidationIssue(
                path="dataset",
                message="Unsupported top-level section 'dataset'",
                expected="run.data",
                suggestion="Move dataset.path to run.data",
                example_fix="run:\n  data: /absolute/path/to/data.csv",
            )
        )
    for key in raw.keys():
        if key not in _ALLOWED_TOP_LEVEL and key != "data":
            issues.append(
                ValidationIssue(
                    path=key,
                    message=f"Unsupported top-level key '{key}'",
                    expected=f"One of {sorted(_ALLOWED_TOP_LEVEL)}",
                )
            )


def _validate_run_section(run: Any, issues: list[ValidationIssue]) -> None:
    if run is None:
        return
    if not isinstance(run, dict):
        issues.append(ValidationIssue(path="run", message="run must be a mapping"))


def _validate_preprocessing(pre: Any, issues: list[ValidationIssue]) -> None:
    if pre is None:
        return
    if not isinstance(pre, dict):
        issues.append(
            ValidationIssue(
                path="preprocessing", message="preprocessing must be a mapping"
            )
        )
        return
    if "drop" in pre:
        issues.append(
            ValidationIssue(
                path="preprocessing.drop",
                message="Unsupported key 'drop'",
                expected="preprocessing.drop_columns",
                suggestion="Rename preprocessing.drop to preprocessing.drop_columns",
                example_fix="preprocessing:\n  drop_columns: [col_a, col_b]",
            )
        )
    for key in pre.keys():
        if key not in _ALLOWED_PREPROCESSING:
            issues.append(
                ValidationIssue(
                    path=f"preprocessing.{key}",
                    message=f"Unsupported preprocessing key '{key}'",
                    expected=f"One of {sorted(_ALLOWED_PREPROCESSING)}",
                )
            )

    drop_columns = pre.get("drop_columns")
    if drop_columns is not None and not isinstance(drop_columns, list):
        issues.append(
            ValidationIssue(
                path="preprocessing.drop_columns",
                message="drop_columns must be a list",
                expected="list[str]",
                received=drop_columns,
            )
        )

    _validate_impute(pre.get("impute"), issues)
    _validate_encode(pre.get("encode"), issues)


def _validate_impute(impute: Any, issues: list[ValidationIssue]) -> None:
    if impute is None:
        return
    if not isinstance(impute, dict):
        issues.append(
            ValidationIssue(
                path="preprocessing.impute", message="impute must be a mapping"
            )
        )
        return
    for col, spec in impute.items():
        path = f"preprocessing.impute.{col}"
        if isinstance(spec, str):
            alias = _IMPUTE_METHOD_ALIASES.get(spec, spec)
            issues.append(
                ValidationIssue(
                    path=path,
                    message="Impute entries must be mappings, not bare strings",
                    expected="{method: fill_mean|fill_median|fill_mode|sample_normal|sample_categorical, seed?: int}",
                    received=spec,
                    suggestion=f"Use {{{{method: {alias}}}}}",
                    example_fix=f"{col}: {{method: {alias}}}",
                )
            )
            continue
        if not isinstance(spec, dict):
            issues.append(
                ValidationIssue(path=path, message="Impute spec must be a mapping")
            )
            continue
        method = spec.get("method")
        if method not in _SUPPORTED_IMPUTE_METHODS:
            issues.append(
                ValidationIssue(
                    path=f"{path}.method",
                    message=f"Unsupported impute method '{method}'",
                    expected=f"One of {sorted(_SUPPORTED_IMPUTE_METHODS)}",
                )
            )


def _validate_encode(encode: Any, issues: list[ValidationIssue]) -> None:
    if encode is None:
        return
    if not isinstance(encode, dict):
        issues.append(
            ValidationIssue(
                path="preprocessing.encode", message="encode must be a mapping"
            )
        )
        return
    for col, spec in encode.items():
        path = f"preprocessing.encode.{col}"
        if not isinstance(spec, dict):
            issues.append(
                ValidationIssue(path=path, message="Encode spec must be a mapping")
            )
            continue
        method = spec.get("method")
        if method not in _SUPPORTED_ENCODE_METHODS:
            issues.append(
                ValidationIssue(
                    path=f"{path}.method",
                    message=f"Unsupported encode method '{method}'",
                    expected=f"One of {sorted(_SUPPORTED_ENCODE_METHODS)}",
                    suggestion="Current MCP workflow only supports one_hot encoding",
                )
            )
        extra_keys = set(spec.keys()) - {"method", "max_categories"}
        if extra_keys:
            issues.append(
                ValidationIssue(
                    path=path,
                    message=f"Unsupported encode fields: {sorted(extra_keys)}",
                    expected="Only method and max_categories are supported",
                )
            )


def _validate_sweep(sweep: Any, issues: list[ValidationIssue]) -> None:
    if sweep is None:
        return
    if not isinstance(sweep, dict):
        issues.append(ValidationIssue(path="sweep", message="sweep must be a mapping"))
        return
    for legacy_key in ("n_cubes", "overlap", "policy_column"):
        if legacy_key in sweep:
            issues.append(
                ValidationIssue(
                    path=f"sweep.{legacy_key}",
                    message=f"Unsupported sweep key '{legacy_key}'",
                    expected="Use sweep.pca and sweep.ball_mapper.epsilon",
                )
            )
    for key in sweep.keys():
        if key not in _ALLOWED_SWEEP:
            issues.append(
                ValidationIssue(
                    path=f"sweep.{key}",
                    message=f"Unsupported sweep key '{key}'",
                    expected=f"One of {sorted(_ALLOWED_SWEEP)}",
                )
            )
    _validate_nested_section(sweep.get("pca"), "sweep.pca", _ALLOWED_PCA, issues)
    _validate_nested_section(
        sweep.get("ball_mapper"), "sweep.ball_mapper", _ALLOWED_BALL_MAPPER, issues
    )


def _validate_cosmic_graph(cosmic_graph: Any, issues: list[ValidationIssue]) -> None:
    if cosmic_graph is None:
        return
    if not isinstance(cosmic_graph, dict):
        issues.append(
            ValidationIssue(
                path="cosmic_graph", message="cosmic_graph must be a mapping"
            )
        )
        return
    for key in cosmic_graph.keys():
        if key == LEGACY_COSMIC_GRAPH_THRESHOLD_KEY:
            issues.append(
                ValidationIssue(
                    path="cosmic_graph.threshold",
                    message=LEGACY_COSMIC_GRAPH_THRESHOLD_MESSAGE,
                    expected="cosmic_graph.construction_threshold",
                    received=cosmic_graph[key],
                    suggestion=(
                        "Rename cosmic_graph.threshold to "
                        "cosmic_graph.construction_threshold."
                    ),
                    example_fix=(
                        "cosmic_graph:\n"
                        f"  construction_threshold: {cosmic_graph[key]!r}"
                    ),
                )
            )
            continue
        if key not in _ALLOWED_COSMIC_GRAPH:
            issues.append(
                ValidationIssue(
                    path=f"cosmic_graph.{key}",
                    message=f"Unsupported cosmic_graph key '{key}'",
                    expected=f"One of {sorted(_ALLOWED_COSMIC_GRAPH)}",
                )
            )
    if "construction_threshold" in cosmic_graph:
        try:
            normalize_construction_threshold(cosmic_graph["construction_threshold"])
        except (TypeError, ValueError):
            issues.append(
                ValidationIssue(
                    path="cosmic_graph.construction_threshold",
                    message=THRESHOLD_RANGE_MESSAGE,
                    expected='"auto" or a finite number in [0.0, 1.0]',
                    received=cosmic_graph["construction_threshold"],
                    suggestion="Use 'auto' unless you have a deliberate fixed graph cutoff.",
                    example_fix='cosmic_graph:\n  construction_threshold: "auto"',
                )
            )


def _validate_output(output: Any, issues: list[ValidationIssue]) -> None:
    if output is None:
        return
    if not isinstance(output, dict):
        issues.append(
            ValidationIssue(path="output", message="output must be a mapping")
        )
        return
    for key in output.keys():
        if key not in _ALLOWED_OUTPUT:
            issues.append(
                ValidationIssue(
                    path=f"output.{key}",
                    message=f"Unsupported output key '{key}'",
                    expected=f"One of {sorted(_ALLOWED_OUTPUT)}",
                )
            )


def _validate_nested_section(
    node: Any,
    path: str,
    allowed_keys: set[str],
    issues: list[ValidationIssue],
) -> None:
    if node is None:
        return
    if not isinstance(node, dict):
        issues.append(ValidationIssue(path=path, message=f"{path} must be a mapping"))
        return
    for key in node.keys():
        if key not in allowed_keys:
            issues.append(
                ValidationIssue(
                    path=f"{path}.{key}",
                    message=f"Unsupported key '{key}'",
                    expected=f"One of {sorted(allowed_keys)}",
                )
            )


def _deep_merge(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def _infer_issue_error_code(issues: list[ValidationIssue]) -> str:
    if len(issues) == 1 and issues[0].path == "run.data":
        if issues[0].message == "Dataset path is missing":
            return "DATASET_PATH_MISSING"
        if issues[0].message.startswith("Dataset path does not exist: "):
            attempted_path = issues[0].message.removeprefix(
                "Dataset path does not exist: "
            )
            path_context = classify_path(attempted_path)
            if path_context["looks_like_sandbox_path"]:
                return "HOST_PATH_NOT_VISIBLE"
            return "FILE_NOT_FOUND"
    return "YAML_SCHEMA_MISMATCH"


def _variance_elbow_dimension(
    pca_cum_var: list[tuple[int, float]], n_features: int
) -> int:
    if not pca_cum_var:
        if n_features <= 4:
            return max(2, n_features)
        if n_features <= 12:
            return 5
        return min(max(10, int(np.log2(max(1, n_features)) * 3)), n_features)

    points = sorted((int(dim), float(var)) for dim, var in pca_cum_var)
    for dim, var in points:
        if var >= 0.90:
            return dim
    return points[-1][0]


def _initial_pca_grid(n_features: int, pca_knee: int) -> list[int]:
    if n_features <= 0:
        return [2]
    upper = max(2, int(n_features))
    if upper <= 4:
        candidates = [2, 3, 4]
    elif pca_knee <= 4:
        candidates = [2, 3, 4, 5, 6]
    elif pca_knee <= 8:
        candidates = [4, 5, 6, 8, 10]
    elif upper <= 12:
        candidates = [2, 4, 6, 8, 10]
    else:
        candidates = [2, 5, 10, 15, 20]

    clipped = [min(dim, upper) for dim in candidates if min(dim, upper) >= 2]
    return sorted(dict.fromkeys(clipped))


def _build_initial_config_yaml(
    geo: dict[str, Any],
    *,
    data_path: str,
    run_name: str = "initial_sweep",
    processed_profile: NumericProfile | None = None,
) -> str:
    """Construct a canonical initial config from dataset geometry.

    When *processed_profile* is provided, epsilon and PCA dimensions are
    calibrated against the processed feature space (after preprocessing +
    scaling).  Otherwise falls back to raw-space geometry.
    """
    if processed_profile is not None:
        knn_mean = processed_profile.knn_k5_mean or 0.5
        pca_cum_var = processed_profile.pca_cumulative_variance
        knn_p25 = processed_profile.knn_p25
        knn_p75 = processed_profile.knn_p75
    else:
        knn_mean = geo.get("knn_k5_mean") or geo.get("knn_mean") or 0.5
        pca_cum_var = geo.get("pca_cumulative_variance", [])
        knn_p25 = knn_p75 = 0.0

    n_features = (
        processed_profile.n_features
        if processed_profile is not None
        else geo.get("n_features", 0)
    )
    pca_knee = _variance_elbow_dimension(pca_cum_var, n_features)
    pca_dims = _initial_pca_grid(n_features, pca_knee)

    if knn_p25 > 0 and knn_p75 > 0:
        eps_min = knn_p25 * 0.75
        eps_max = knn_p75 * 1.35
    else:
        eps_min = knn_mean * 0.8
        eps_max = knn_mean * 1.5

    n_samples = geo.get("n_samples", 0)
    column_profiles = geo.get("column_profiles", [])
    drop, impute, encode, _ = _recommend_preprocessing_block(column_profiles, n_samples)
    preprocessing_block = _preprocessing_block_to_yaml(drop, impute, encode)

    return f"""run:
  name: {run_name}
  data: {data_path}
{preprocessing_block}
sweep:
  pca:
    dimensions:
      values: {pca_dims}
    seed:
      values: [42, 7, 13]
  ball_mapper:
    epsilon:
      range:
        min: {eps_min:.4f}
        max: {eps_max:.4f}
        steps: 24
cosmic_graph:
  construction_threshold: auto
output:
  n_reps: 4
"""


def _suggest_resolution_pca_dims(pca_dims: list[Any]) -> list[int]:
    dims = sorted({int(dim) for dim in pca_dims if int(dim) > 1})
    if not dims:
        return [5, 8, 12, 16, 20]
    if len(dims) == 1:
        center = dims[0]
        return sorted({max(2, center - 4), max(2, center - 2), center, center + 2})

    low, high = dims[0], dims[-1]
    if high <= low:
        return dims
    step = max(1, round((high - low) / 5))
    if high - low >= 8:
        step = max(2, step)
    return list(range(low, high + 1, step))[:6]

