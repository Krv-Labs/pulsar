"""Rendering and serialization helpers for preprocessing plans."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import yaml

from pulsar.mcp.preprocessing_plan.model import PreprocessingPlan


def _readable_group_key(
    ref_cols: list[str], used_keys: set[str], fallback_gid: str
) -> str:
    """C-AMENDMENT 3: ``<col>`` if 1 member, else ``<first>_and_<k-1>_more``."""
    if len(ref_cols) == 1:
        base = ref_cols[0]
    else:
        base = f"{ref_cols[0]}_and_{len(ref_cols) - 1}_more"
    if base not in used_keys:
        return base
    return f"{base}_{fallback_gid}"


def _collect_group_references(
    plan: PreprocessingPlan,
) -> list[tuple[str, str, list[str]]]:
    """Return ``[(internal_gid, readable_yaml_key, referencing_columns), ...]``.

    Follows C-AMENDMENTS 1–3:
    - Emits only groups referenced by >=1 feature or has_date (role=='drop' with
      ``indicator=='group:<gid>'``) entry.
    - Lists ONLY those referencing columns (never non-referencing members).
    - Uses readable YAML keys (`<col>` or `<first>_and_<k-1>_more`).
    """
    refs_by_gid: dict[str, list[str]] = {}
    for col, entry in plan.entries.items():
        if not entry.indicator.startswith("group:"):
            continue
        if entry.role not in ("feature", "drop"):
            continue
        gid = entry.indicator.split(":", 1)[1]
        if gid:
            refs_by_gid.setdefault(gid, []).append(col)

    used_keys: set[str] = set()
    out: list[tuple[str, str, list[str]]] = []
    for gid, ref_cols in refs_by_gid.items():
        if not ref_cols:
            continue
        yaml_key = _readable_group_key(ref_cols, used_keys, gid)
        used_keys.add(yaml_key)
        out.append((gid, yaml_key, ref_cols))
    return out


def build_marker_legend(
    plan: PreprocessingPlan,
    *,
    preview_limit: int = 5,
) -> list[dict[str, Any]]:
    """Build ``marker_legend`` entries for all emitted missingness marker columns."""
    legend: list[dict[str, Any]] = []
    grouped_cols: set[str] = set()
    for _gid, yaml_key, ref_cols in _collect_group_references(plan):
        grouped_cols.update(ref_cols)
        legend.append(
            {
                "marker_column": f"{yaml_key}_was_missing",
                "columns_preview": list(ref_cols[:preview_limit]),
                "columns_count": len(ref_cols),
            }
        )
    for col, entry in plan.entries.items():
        if (
            entry.role == "feature"
            and entry.impute is not None
            and entry.indicator == "own"
            and col not in grouped_cols
        ):
            legend.append(
                {
                    "marker_column": f"{col}_was_missing",
                    "columns_preview": [col],
                    "columns_count": 1,
                }
            )
    return legend


def _format_plan_impute_spec(
    raw_imp: dict[str, Any], indicator_mode: str
) -> dict[str, Any]:
    spec: dict[str, Any] = {"method": raw_imp["method"]}
    if "value" in raw_imp and raw_imp["value"] is not None:
        spec["value"] = raw_imp["value"]
    if "seed" in raw_imp and raw_imp["seed"] is not None:
        spec["seed"] = int(raw_imp["seed"])
    if indicator_mode != "own":
        spec["indicator"] = False
    return spec


def plan_to_preprocessing_dict(plan: PreprocessingPlan) -> dict[str, Any]:
    """Convert *plan* to the canonical ``preprocessing:`` dict consumed by ``PulsarConfig``."""
    drop_columns: list[str] = [
        col for col, entry in plan.entries.items() if entry.role != "feature"
    ]

    missing_indicator_groups: dict[str, list[str]] = {}
    for _gid, yaml_key, ref_cols in _collect_group_references(plan):
        missing_indicator_groups[yaml_key] = list(ref_cols)

    impute_dict: dict[str, Any] = {}
    encode_dict: dict[str, dict[str, Any]] = {}

    for col, entry in plan.entries.items():
        if entry.role != "feature":
            continue
        if entry.impute_candidates and len(entry.impute_candidates) > 1:
            impute_dict[col] = [
                _format_plan_impute_spec(cand, entry.indicator)
                for cand in entry.impute_candidates
            ]
        elif entry.impute is not None:
            impute_dict[col] = _format_plan_impute_spec(entry.impute, entry.indicator)
        if entry.encode is not None:
            encode_dict[col] = {"method": entry.encode}

    pre: dict[str, Any] = {
        "drop_columns": drop_columns,
    }
    if plan.n_imputations > 1:
        pre["n_imputations"] = int(plan.n_imputations)
    if missing_indicator_groups:
        pre["missing_indicator_groups"] = missing_indicator_groups
    if impute_dict:
        pre["impute"] = impute_dict
    if encode_dict:
        pre["encode"] = encode_dict
    return pre


def plan_fingerprint(plan: PreprocessingPlan) -> str:
    """Return a 12-hex SHA-256 fingerprint of ``plan_to_preprocessing_dict(plan)``."""
    pre = plan_to_preprocessing_dict(plan)
    canonical = json.dumps(pre, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def plan_to_preprocessing_yaml(
    plan: PreprocessingPlan,
    *,
    include_header: bool = False,
) -> str:
    """Render the ``preprocessing:`` YAML block for *plan*."""
    pre_dict = plan_to_preprocessing_dict(plan)
    rendered = yaml.safe_dump(
        {"preprocessing": pre_dict},
        sort_keys=False,
        default_flow_style=False,
    ).rstrip()
    if include_header:
        fp = plan_fingerprint(plan)
        header = f"# preprocessing plan {plan.plan_id} revision {plan.revision} ({fp})"
        return f"{header}\n{rendered}"
    return rendered


def summarize_plan_counts(plan: PreprocessingPlan) -> dict[str, Any]:
    """Compute summary counts for *plan*."""
    decided_by_counts = {"default": 0, "agent": 0, "user": 0}
    n_features = 0
    n_dropped = 0
    n_open_entries = 0

    for entry in plan.entries.values():
        decided_by_counts[entry.decided_by] = (
            decided_by_counts.get(entry.decided_by, 0) + 1
        )
        if entry.role == "feature":
            n_features += 1
        else:
            n_dropped += 1
        if entry.status == "open":
            n_open_entries += 1

    n_open_questions = sum(1 for q in plan.questions.values() if q.status == "open")
    return {
        "columns": len(plan.entries),
        "feature_columns": n_features,
        "dropped": n_dropped,
        "open_entries": n_open_entries,
        "open_questions": n_open_questions,
        "decided_by": decided_by_counts,
    }
