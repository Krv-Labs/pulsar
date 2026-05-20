from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import yaml

from pulsar.config import config_to_yaml, load_config
from pulsar.mcp.errors import classify_path, mcp_error


_ALLOWED_TOP_LEVEL = {"run", "preprocessing", "sweep", "cosmic_graph", "output"}
_ALLOWED_PREPROCESSING = {"drop_columns", "impute", "encode"}
_ALLOWED_SWEEP = {"pca", "ball_mapper"}
_ALLOWED_PCA = {"dimensions", "seed"}
_ALLOWED_BALL_MAPPER = {"epsilon"}
_ALLOWED_OUTPUT = {"n_reps"}
_ALLOWED_COSMIC_GRAPH = {"construction_threshold", "neighborhood"}
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


def apply_overrides(
    config_yaml: str, overrides: dict[str, Any]
) -> ConfigOverrideResult:
    unknown_keys = set(overrides.keys()) - _VALID_OVERRIDE_KEYS
    if unknown_keys:
        raise ValueError(
            f"Unknown override key(s): {sorted(unknown_keys)}. "
            f"Valid keys: {sorted(_VALID_OVERRIDE_KEYS)}"
        )

    raw = parse_yaml_mapping(config_yaml)
    applied: list[str] = []
    diff: list[dict[str, Any]] = []

    def _track(path: str, old: Any, new: Any) -> None:
        applied.append(path)
        diff.append({"path": path, "old": old, "new": new})

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
        if key not in _ALLOWED_COSMIC_GRAPH:
            issues.append(
                ValidationIssue(
                    path=f"cosmic_graph.{key}",
                    message=f"Unsupported cosmic_graph key '{key}'",
                    expected=f"One of {sorted(_ALLOWED_COSMIC_GRAPH)}",
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
