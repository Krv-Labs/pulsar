"""Question answer application, column edit validation, and plan revision diffs."""

from __future__ import annotations

from collections.abc import Sequence
import copy
import math
from typing import Any

import numpy as np

from pulsar.analysis.preprocessing_evidence import ColumnEvidence
from pulsar.mcp.preprocessing_plan.model import (
    _EDIT_ALLOWED_KEYS,
    VALID_ENCODE_METHODS,
    VALID_IMPUTE_METHODS,
    VALID_ROLES,
    DecidedBy,
    PlanEditError,
    PlanEntry,
    PlanQuestion,
    PreprocessingPlan,
    QuestionOption,
)

_PROTECTED_ROLES: frozenset[str] = frozenset({"target", "entity", "time"})


def _validate_indicator(
    indicator: str,
    col: str,
    ev: ColumnEvidence | None,
    missing_groups: dict[str, list[str]],
) -> str:
    ind = str(indicator).strip()
    if ind == "none":
        return "none"
    if ind == "own":
        normalized = "own"
    elif ind.startswith("group:"):
        gid = ind.split(":", 1)[1].strip()
        if not gid or (gid not in missing_groups and gid != col):
            raise PlanEditError(
                "INVALID_ARGUMENT",
                f"Invalid indicator {indicator!r} for column {col!r}: group must be an evidence missingness group or {col!r}.",
                {
                    "column": col,
                    "indicator": indicator,
                    "known_groups": sorted(missing_groups.keys()),
                },
            )
        normalized = f"group:{gid}"
    else:
        raise PlanEditError(
            "INVALID_ARGUMENT",
            f"Invalid indicator {indicator!r} for column {col!r}. Expected 'none', 'own', or 'group:<gid>'.",
            {
                "column": col,
                "indicator": indicator,
                "known_groups": sorted(missing_groups.keys()),
            },
        )
    if (
        ev is not None
        and ev.n_missing == 0
        and not (ev.kind == "numeric_text" and (ev.n_non_numeric or 0) > 0)
    ):
        raise PlanEditError(
            "INVALID_ARGUMENT",
            f"Column {col!r} has no blanks; indicator must be 'none'.",
            {"column": col, "indicator": indicator, "n_missing": 0},
        )
    return normalized


def _edit_role(entry: PlanEntry, edit: dict[str, Any], col: str) -> None:
    if "role" in edit and edit["role"] is not None:
        role_val = str(edit["role"])
        if role_val not in VALID_ROLES:
            raise PlanEditError(
                "INVALID_ARGUMENT",
                (
                    f"Invalid role {role_val!r} for column {col!r}; "
                    f"valid roles are {sorted(VALID_ROLES)} (for outcomes use role 'target')."
                ),
                {"column": col, "role": role_val, "valid_roles": sorted(VALID_ROLES)},
            )
        entry.role = role_val  # type: ignore[assignment]


def _edit_encode(entry: PlanEntry, edit: dict[str, Any], col: str) -> None:
    if "encode" in edit:
        enc = edit["encode"]
        if enc is None or enc == "none" or enc == "":
            entry.encode = None
        else:
            enc_str = str(enc)
            if enc_str not in VALID_ENCODE_METHODS:
                raise PlanEditError(
                    "INVALID_ARGUMENT",
                    f"Invalid encode method {enc_str!r} for column {col!r}.",
                    {
                        "column": col,
                        "encode": enc_str,
                        "valid_methods": sorted(VALID_ENCODE_METHODS),
                    },
                )
            entry.encode = enc_str


def _validate_single_impute_dict(
    imp: dict[str, Any],
    entry: PlanEntry,
    col: str,
    ev: ColumnEvidence | None,
) -> None:
    method_now = imp.get("method")
    is_text_mode = ev is not None and (
        ev.kind in ("text", "datetime_like")
        or (ev.kind == "numeric_text" and entry.encode is not None)
    )
    is_numeric_mode = ev is not None and (
        ev.kind in ("numeric", "boolean")
        or (ev.kind == "numeric_text" and entry.encode is None)
    )
    if method_now == "fill_value":
        if "value" not in imp or imp["value"] is None:
            raise PlanEditError(
                "INVALID_ARGUMENT",
                f"impute_method 'fill_value' for column {col!r} requires 'impute_value'.",
                {"column": col},
            )
        val_now = imp["value"]
        if isinstance(val_now, (float, np.floating)) and (
            math.isnan(float(val_now)) or math.isinf(float(val_now))
        ):
            raise PlanEditError(
                "INVALID_ARGUMENT",
                f"impute_value for column {col!r} must be finite, got {val_now!r}.",
                {"column": col, "impute_value": str(val_now)},
            )
        if ev is not None:
            if is_numeric_mode:
                if isinstance(val_now, str) or not isinstance(
                    val_now, (int, float, np.number)
                ):
                    raise PlanEditError(
                        "INVALID_ARGUMENT",
                        f"Column {col!r} is {ev.kind}; impute_value for 'fill_value' must be numeric, got {val_now!r}.",
                        {"column": col, "kind": ev.kind, "impute_value": val_now},
                    )
            elif is_text_mode:
                if isinstance(val_now, (int, float, np.number)) and not isinstance(
                    val_now, bool
                ):
                    imp["value"] = str(val_now)
    elif "value" in imp:
        raise PlanEditError(
            "INVALID_ARGUMENT",
            f"impute_value is only valid with impute_method='fill_value' (column {col!r}).",
            {"column": col, "impute_method": method_now},
        )
    if ev is not None:
        if is_text_mode and method_now in (
            "fill_median",
            "fill_mean",
            "sample_normal",
        ):
            raise PlanEditError(
                "INVALID_ARGUMENT",
                f"Column {col!r} is {ev.kind}; numeric impute_method {method_now!r} cannot be used on text.",
                {"column": col, "kind": ev.kind, "impute_method": method_now},
            )
        if is_numeric_mode and method_now in (
            "fill_mode",
            "sample_categorical",
        ):
            raise PlanEditError(
                "INVALID_ARGUMENT",
                f"Column {col!r} is {ev.kind}; categorical impute_method {method_now!r} cannot be used on numbers.",
                {"column": col, "kind": ev.kind, "impute_method": method_now},
            )


def _validate_impute_spec(
    entry: PlanEntry, col: str, ev: ColumnEvidence | None
) -> None:
    if entry.impute is None:
        return
    _validate_single_impute_dict(entry.impute, entry, col, ev)
    if entry.impute_candidates:
        for cand in entry.impute_candidates:
            _validate_single_impute_dict(cand, entry, col, ev)


def _parse_impute_mapping(
    raw_imp: Any,
    col: str,
) -> dict[str, Any] | None:
    """Parse and validate an explicit ``impute`` mapping or null edit value."""
    if raw_imp is None:
        return None
    if not isinstance(raw_imp, dict):
        raise PlanEditError(
            "INVALID_ARGUMENT",
            f"impute for column {col!r} must be a mapping, list of mappings, or null.",
            {"column": col, "impute": raw_imp},
        )
    method = raw_imp.get("method")
    if method is None or method == "none":
        return None
    if method not in VALID_IMPUTE_METHODS:
        raise PlanEditError(
            "INVALID_ARGUMENT",
            f"Invalid impute method {method!r} for column {col!r}.",
            {
                "column": col,
                "impute_method": method,
                "valid_methods": sorted(VALID_IMPUTE_METHODS),
            },
        )
    imp_dict: dict[str, Any] = {"method": str(method)}
    if "value" in raw_imp and raw_imp["value"] is not None:
        if isinstance(raw_imp["value"], bool):
            raise PlanEditError(
                "INVALID_ARGUMENT",
                f"impute_value for column {col!r} cannot be a boolean.",
                {"column": col, "impute_value": raw_imp["value"]},
            )
        imp_dict["value"] = raw_imp["value"]
    if "seed" in raw_imp and raw_imp["seed"] is not None:
        if isinstance(raw_imp["seed"], bool) or not isinstance(raw_imp["seed"], int):
            raise PlanEditError(
                "INVALID_ARGUMENT",
                f"impute seed for column {col!r} must be an integer.",
                {"column": col, "seed": raw_imp["seed"]},
            )
        imp_dict["seed"] = int(raw_imp["seed"])
    if "indicator" in raw_imp:
        if not isinstance(raw_imp["indicator"], bool):
            raise PlanEditError(
                "INVALID_ARGUMENT",
                f"Impute candidate indicator for column {col!r} must be a boolean.",
                {"column": col, "indicator": raw_imp["indicator"]},
            )
        imp_dict["indicator"] = raw_imp["indicator"]
    return imp_dict


def _parse_impute_edit_value(
    raw_imp: Any,
    col: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    if isinstance(raw_imp, list):
        if not raw_imp:
            raise PlanEditError(
                "INVALID_ARGUMENT",
                f"impute candidate list for column {col!r} must not be empty.",
                {"column": col},
            )
        candidates: list[dict[str, Any]] = []
        for item in raw_imp:
            parsed = _parse_impute_mapping(item, col)
            if parsed is None:
                raise PlanEditError(
                    "INVALID_ARGUMENT",
                    f"impute candidate list for column {col!r} cannot contain null entries.",
                    {"column": col, "impute": raw_imp},
                )
            candidates.append(parsed)
        return candidates[0], (candidates if len(candidates) > 1 else None)
    if isinstance(raw_imp, dict) and isinstance(raw_imp.get("method"), list):
        methods = raw_imp["method"]
        if not methods:
            raise PlanEditError(
                "INVALID_ARGUMENT",
                f"impute method list for column {col!r} must not be empty.",
                {"column": col},
            )
        candidates = []
        for m in methods:
            spec_copy = dict(raw_imp)
            spec_copy["method"] = m
            if m != "fill_value":
                spec_copy.pop("value", None)
            parsed = _parse_impute_mapping(spec_copy, col)
            if parsed is None:
                raise PlanEditError(
                    "INVALID_ARGUMENT",
                    f"impute method list for column {col!r} cannot contain 'none'.",
                    {"column": col},
                )
            candidates.append(parsed)
        return candidates[0], (candidates if len(candidates) > 1 else None)
    single = _parse_impute_mapping(raw_imp, col)
    return single, None


def _edit_impute(
    entry: PlanEntry,
    edit: dict[str, Any],
    col: str,
    ev: ColumnEvidence | None,
) -> bool:
    has_impute_edit = False
    if "impute" in edit:
        has_impute_edit = True
        entry.impute, entry.impute_candidates = _parse_impute_edit_value(
            edit["impute"], col
        )

    if "impute_method" in edit:
        has_impute_edit = True
        method = edit["impute_method"]
        if method is None or method == "none" or method == "":
            entry.impute = None
            entry.impute_candidates = None
        elif isinstance(method, list):
            if not method:
                raise PlanEditError(
                    "INVALID_ARGUMENT",
                    f"impute_method list for column {col!r} must not be empty.",
                    {"column": col},
                )
            prev_val = entry.impute.get("value") if entry.impute else None
            candidates: list[dict[str, Any]] = []
            for m in method:
                m_str = str(m)
                if m_str not in VALID_IMPUTE_METHODS:
                    raise PlanEditError(
                        "INVALID_ARGUMENT",
                        f"Invalid impute_method {m_str!r} for column {col!r}.",
                        {
                            "column": col,
                            "impute_method": m_str,
                            "valid_methods": sorted(VALID_IMPUTE_METHODS),
                        },
                    )
                cand_dict: dict[str, Any] = {"method": m_str}
                if m_str == "fill_value" and prev_val is not None:
                    cand_dict["value"] = prev_val
                candidates.append(cand_dict)
            entry.impute = candidates[0]
            entry.impute_candidates = candidates if len(candidates) > 1 else None
        else:
            method_str = str(method)
            if method_str not in VALID_IMPUTE_METHODS:
                raise PlanEditError(
                    "INVALID_ARGUMENT",
                    f"Invalid impute_method {method_str!r} for column {col!r}.",
                    {
                        "column": col,
                        "impute_method": method_str,
                        "valid_methods": sorted(VALID_IMPUTE_METHODS),
                    },
                )
            prev_val = entry.impute.get("value") if entry.impute else None
            imp_dict: dict[str, Any] = {"method": method_str}
            if method_str == "fill_value" and prev_val is not None:
                imp_dict["value"] = prev_val
            entry.impute = imp_dict
            entry.impute_candidates = None

    if "impute_value" in edit and edit["impute_value"] is not None:
        has_impute_edit = True
        val = edit["impute_value"]
        if isinstance(val, bool):
            raise PlanEditError(
                "INVALID_ARGUMENT",
                f"impute_value for column {col!r} cannot be a boolean.",
                {"column": col, "impute_value": val},
            )
        if entry.impute is None:
            entry.impute = {"method": "fill_value", "value": val}
        specs = entry.impute_candidates or [entry.impute]
        constant_specs = [s for s in specs if s.get("method") == "fill_value"]
        if not constant_specs:
            raise PlanEditError(
                "INVALID_ARGUMENT",
                f"impute_value is only valid with impute_method='fill_value' (column {col!r}).",
                {"column": col},
            )
        for spec in constant_specs:
            spec["value"] = val

    if (
        has_impute_edit
        and entry.impute is not None
        and ev is not None
        and ev.n_missing == 0
        and not (
            ev.kind == "numeric_text"
            and (ev.n_non_numeric or 0) > 0
            and entry.encode is None
        )
    ):
        raise PlanEditError(
            "INVALID_ARGUMENT",
            f"Column {col!r} has no blanks; impute is not applicable.",
            {"column": col, "n_missing": 0},
        )

    _validate_impute_spec(entry, col, ev)
    return has_impute_edit


def _edit_indicator(
    entry: PlanEntry,
    edit: dict[str, Any],
    col: str,
    ev: ColumnEvidence | None,
    missing_groups: dict[str, list[str]],
) -> None:
    if "indicator" in edit and edit["indicator"] is not None:
        entry.indicator = _validate_indicator(
            str(edit["indicator"]), col, ev, missing_groups
        )
        for spec in entry.impute_candidates or ([entry.impute] if entry.impute else []):
            spec.pop("indicator", None)


def _apply_single_edit(
    entries: dict[str, PlanEntry],
    edit: dict[str, Any],
    *,
    source: DecidedBy,
    missing_groups: dict[str, list[str]],
    ev_by_col: dict[str, ColumnEvidence],
) -> None:
    if not isinstance(edit, dict):
        raise PlanEditError(
            "INVALID_ARGUMENT",
            "Each edit must be a mapping.",
            {"edit": edit},
        )
    unknown_keys = set(edit.keys()) - _EDIT_ALLOWED_KEYS
    if unknown_keys:
        raise PlanEditError(
            "INVALID_ARGUMENT",
            f"Unsupported edit field(s) for column {edit.get('column')!r}: {sorted(unknown_keys)}.",
            {
                "column": edit.get("column"),
                "unsupported_fields": sorted(unknown_keys),
                "allowed_fields": sorted(_EDIT_ALLOWED_KEYS),
            },
        )
    col = edit.get("column")
    if not isinstance(col, str) or col not in entries:
        raise PlanEditError(
            "INVALID_ARGUMENT",
            f"Unknown column: {col!r}.",
            {"column": col},
        )

    entry = entries[col]
    ev = ev_by_col.get(col)

    _edit_role(entry, edit, col)
    _edit_encode(entry, edit, col)
    has_impute_edit = _edit_impute(entry, edit, col, ev)
    _edit_indicator(entry, edit, col, ev, missing_groups)

    # When a column is set to a non-feature role without explicit impute/encode/indicator,
    # clear feature-only settings so dropped/id/target columns do not linger in impute/groups.
    if entry.role != "feature":
        if not has_impute_edit:
            entry.impute = None
            entry.impute_candidates = None
        if "encode" not in edit:
            entry.encode = None
        if "indicator" not in edit:
            entry.indicator = "none"

    entry.decided_by = source
    entry.status = "resolved"


def _merge_answer_entries(col: str, entries: list[PlanEntry]) -> PlanEntry:
    """Combine compatible answer treatments without losing candidate indicators."""
    primary = entries[0]
    for entry in entries[1:]:
        same_group = entry.indicator == primary.indicator or not (
            entry.indicator.startswith("group:")
            or primary.indicator.startswith("group:")
        )
        if (
            entry.role != primary.role
            or entry.encode != primary.encode
            or (entry.impute is None) != (primary.impute is None)
            or not same_group
        ):
            raise PlanEditError(
                "INVALID_ARGUMENT",
                f"Selected options for column {col!r} have incompatible roles, encoding, or missingness groups.",
                {"column": col},
            )
    if primary.impute is None:
        return primary
    candidates: list[dict[str, Any]] = []
    for entry in entries:
        for raw_spec in entry.impute_candidates or [entry.impute]:
            spec = dict(raw_spec)
            spec.setdefault("indicator", entry.indicator == "own")
            if spec not in candidates:
                candidates.append(spec)
    primary.impute = candidates[0]
    primary.impute_candidates = candidates if len(candidates) > 1 else None
    if not primary.indicator.startswith("group:"):
        primary.indicator = "own" if any(s["indicator"] for s in candidates) else "none"
    return primary


def _apply_single_answer(
    ans: Any,
    *,
    new_questions: dict[str, PlanQuestion],
    new_entries: dict[str, PlanEntry],
    incoming_protected_roles: dict[str, str],
    source: DecidedBy,
    missing_groups: dict[str, list[str]],
    ev_by_col: dict[str, ColumnEvidence],
) -> None:
    if not isinstance(ans, dict):
        raise PlanEditError(
            "INVALID_ARGUMENT",
            "Each answer must be a mapping with 'question_id' and 'option_id'.",
            {"answer": ans},
        )
    qid = ans.get("question_id")
    raw_oid = ans.get("option_ids") if "option_ids" in ans else ans.get("option_id")
    if not isinstance(qid, str) or qid not in new_questions:
        raise PlanEditError(
            "INVALID_ARGUMENT",
            f"Unknown question_id: {qid!r}.",
            {
                "question_id": qid,
                "known_question_ids": list(new_questions.keys()),
            },
        )
    q = new_questions[qid]
    option_ids = list(raw_oid) if isinstance(raw_oid, list) else [raw_oid]
    if not option_ids:
        raise PlanEditError(
            "INVALID_ARGUMENT",
            f"option_id list for question {qid!r} must not be empty.",
            {"question_id": qid, "valid_options": [o.option_id for o in q.options]},
        )
    matched_opts: list[QuestionOption] = []
    for oid_item in option_ids:
        opt = next((o for o in q.options if o.option_id == oid_item), None)
        if opt is None:
            raise PlanEditError(
                "INVALID_ARGUMENT",
                f"Unknown option_id {oid_item!r} for question {qid!r}.",
                {
                    "question_id": qid,
                    "topic": q.topic,
                    "option_id": oid_item,
                    "valid_options": [o.option_id for o in q.options],
                },
            )
        matched_opts.append(opt)

    effective_roles = {
        c: incoming_protected_roles.get(c, new_entries[c].role)
        for c in q.applies_to
        if c in new_entries
    }
    all_protected = bool(effective_roles) and all(
        r in _PROTECTED_ROLES for r in effective_roles.values()
    )
    if q.status != "open" or all_protected:
        roles_str = (
            ", ".join(f"{c}={r}" for c, r in effective_roles.items())
            or ", ".join(
                f"{c}={incoming_protected_roles.get(c, e.role)}"
                for c, e in new_entries.items()
                if incoming_protected_roles.get(c, e.role) in _PROTECTED_ROLES
            )
            or "resolved"
        )
        raise PlanEditError(
            "INVALID_ARGUMENT",
            f"question {qid} no longer applies (columns now: {roles_str})",
            {
                "question_id": qid,
                "topic": q.topic,
                "status": q.status,
                "columns": effective_roles,
            },
        )

    answers_by_col: dict[str, list[PlanEntry]] = {}
    multiple = len(matched_opts) > 1
    for matched_opt in matched_opts:
        # Every selected option starts from the same revision, so one option's
        # edits cannot become another option's implicit defaults.
        option_entries = dict(new_entries) if multiple else new_entries
        edited_columns: set[str] = set()
        for edit_spec in matched_opt.edits:
            col_name = edit_spec.get("column") if isinstance(edit_spec, dict) else None
            if (
                isinstance(col_name, str)
                and col_name in new_entries
                and incoming_protected_roles.get(col_name, new_entries[col_name].role)
                in _PROTECTED_ROLES
            ):
                continue
            if (
                multiple
                and isinstance(col_name, str)
                and col_name in new_entries
                and col_name not in edited_columns
            ):
                option_entries[col_name] = copy.deepcopy(new_entries[col_name])
            _apply_single_edit(
                option_entries,
                edit_spec,
                source=source,
                missing_groups=missing_groups,
                ev_by_col=ev_by_col,
            )
            edited_columns.add(col_name)
        if multiple:
            for col_name in edited_columns:
                answers_by_col.setdefault(col_name, []).append(option_entries[col_name])

    for col_name, entries in answers_by_col.items():
        new_entries[col_name] = _merge_answer_entries(col_name, entries)

    for col in q.applies_to:
        if (
            col in new_entries
            and incoming_protected_roles.get(col, new_entries[col].role)
            not in _PROTECTED_ROLES
        ):
            new_entries[col].decided_by = source
            new_entries[col].status = "resolved"
    q.status = "answered"
    q.answer = {
        "option_id": option_ids[0] if len(option_ids) == 1 else option_ids,
        "decided_by": source,
    }


def _sync_question_statuses(
    new_questions: dict[str, PlanQuestion],
    new_entries: dict[str, PlanEntry],
    *,
    target_role_added: bool,
    source: DecidedBy,
) -> None:
    """Mark open questions as answered when direct column edits resolve their columns."""
    for q in new_questions.values():
        if q.status != "open":
            continue
        if q.topic == "outcome_columns" and target_role_added:
            q.status = "answered"
            q.answer = {"option_id": "custom_edit", "decided_by": source}
        elif q.applies_to and all(
            new_entries[c].status == "resolved"
            and new_entries[c].decided_by != "default"
            for c in q.applies_to
            if c in new_entries
        ):
            q.status = "answered"
            q.answer = {"option_id": "custom_edit", "decided_by": source}


def apply_answers_and_edits(
    plan: PreprocessingPlan,
    answers: Sequence[dict[str, Any]] = (),
    edits: Sequence[dict[str, Any]] = (),
    source: DecidedBy = "agent",
) -> PreprocessingPlan:
    """Apply question *answers* and column *edits* to *plan*, returning a new revision."""
    if source not in ("agent", "user"):
        raise PlanEditError(
            "INVALID_ARGUMENT",
            f"Invalid source {source!r}; expected 'agent' or 'user'.",
            {"source": source, "valid_sources": ["agent", "user"]},
        )

    ans_list = list(answers or ())
    edit_list = list(edits or ())
    if not ans_list and not edit_list:
        raise PlanEditError(
            "INVALID_ARGUMENT",
            "Nothing to change: provide at least one answer or edit.",
            {"answers": [], "edits": []},
        )

    ev_by_col: dict[str, ColumnEvidence] = {c.name: c for c in plan.evidence.columns}
    new_entries: dict[str, PlanEntry] = {
        col: copy.deepcopy(entry) for col, entry in plan.entries.items()
    }
    new_questions: dict[str, PlanQuestion] = {
        qid: copy.deepcopy(q) for qid, q in plan.questions.items()
    }

    # Track columns explicitly assigned a protected role in this call's edit_list
    incoming_protected_roles: dict[str, str] = {
        str(e["column"]): str(e["role"])
        for e in edit_list
        if isinstance(e, dict)
        and isinstance(e.get("column"), str)
        and e.get("role") in _PROTECTED_ROLES
    }

    for ans in ans_list:
        _apply_single_answer(
            ans,
            new_questions=new_questions,
            new_entries=new_entries,
            incoming_protected_roles=incoming_protected_roles,
            source=source,
            missing_groups=plan.missing_groups,
            ev_by_col=ev_by_col,
        )

    target_role_added = False
    for edit_spec in edit_list:
        _apply_single_edit(
            new_entries,
            edit_spec,
            source=source,
            missing_groups=plan.missing_groups,
            ev_by_col=ev_by_col,
        )
        if isinstance(edit_spec, dict) and edit_spec.get("role") == "target":
            target_role_added = True

    _sync_question_statuses(
        new_questions,
        new_entries,
        target_role_added=target_role_added,
        source=source,
    )

    return PreprocessingPlan(
        plan_id=plan.plan_id,
        dataset_id=plan.dataset_id,
        revision=plan.revision + 1,
        entries=new_entries,
        questions=new_questions,
        missing_groups=copy.deepcopy(plan.missing_groups),
        evidence=plan.evidence,
        created_at=plan.created_at,
        cost=copy.deepcopy(plan.cost),
        target_proxies=copy.deepcopy(plan.target_proxies),
        n_imputations=plan.n_imputations,
    )


def diff_plan_entries(
    before: PreprocessingPlan, after: PreprocessingPlan
) -> list[dict[str, Any]]:
    """Return a list of ``{column, field, was, now}`` diffs between two plan revisions."""
    changes: list[dict[str, Any]] = []
    tracked_fields = (
        "role",
        "impute",
        "impute_candidates",
        "indicator",
        "encode",
        "status",
        "decided_by",
    )
    for col, old_entry in before.entries.items():
        new_entry = after.entries.get(col)
        if new_entry is None:
            continue
        for fld in tracked_fields:
            old_val = getattr(old_entry, fld)
            new_val = getattr(new_entry, fld)
            if old_val != new_val:
                changes.append(
                    {
                        "column": col,
                        "field": fld,
                        "was": copy.deepcopy(old_val),
                        "now": copy.deepcopy(new_val),
                    }
                )
    return changes
