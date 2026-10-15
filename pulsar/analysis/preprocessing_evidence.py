"""
Empirical preprocessing evidence measurement for tabular datasets.

Provides pure statistical and structural measurements of columns, missingness
masks, and inter-column relationships, plus k-NN neighborhood overlap between
alternative representations.
"""

from __future__ import annotations

import re
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

ColumnKind = Literal["numeric", "boolean", "datetime_like", "numeric_text", "text"]

_TYPE_SAMPLE_SIZE = 200
_SUM_COEFF_TOL = 1e-6
_SUM_OTHER_COEFF_TOL = 1e-4
_SUM_RESIDUAL_REL_TOL = 1e-9
_THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_DATE_SEP_BETWEEN_DIGITS_RE = re.compile(r"\d[-/:.T]\d")


@dataclass(frozen=True)
class ColumnEvidence:
    name: str
    kind: ColumnKind
    n_rows: int
    n_missing: int
    missing_fraction: float
    n_unique: int
    unique_ratio: float
    is_constant: bool
    n_zero: int | None
    observed_min: float | None
    observed_max: float | None
    integer_valued: bool | None
    nonnegative: bool | None
    minority_fraction: float | None
    max_abs_z: float | None
    missing_group: str | None
    mask_dependency: MaskDependency | None
    non_numeric_values: tuple[tuple[str, int], ...] | None = None
    n_non_numeric: int | None = None


@dataclass(frozen=True)
class MissingGroup:
    group_id: str
    columns: tuple[str, ...]
    n_missing: int
    zero_free_columns: tuple[str, ...]


@dataclass(frozen=True)
class MaskDependency:
    column: str
    partner: str
    relation: Literal["missing_when_partner_zero", "missing_when_partner_nonzero"]
    n_other_partners: int


@dataclass(frozen=True)
class ColumnRelation:
    kind: Literal["duplicate", "sum_of", "strong_correlation"]
    target: str
    sources: tuple[str, ...]
    value: float
    rows_used: int


@dataclass(frozen=True)
class PreprocessingEvidence:
    n_rows: int
    n_columns: int
    columns: tuple[ColumnEvidence, ...]
    missing_groups: tuple[MissingGroup, ...]
    mask_dependencies: tuple[MaskDependency, ...]
    relations: tuple[ColumnRelation, ...]
    cost: dict[str, Any]


def _detect_object_kind(series: pd.Series, n_observed: int) -> ColumnKind:
    """Classify a non-numeric, non-boolean, non-datetime64 column."""
    if n_observed == 0:
        return "text"

    obs = series.dropna()
    if len(obs) > _TYPE_SAMPLE_SIZE:
        idx = np.linspace(0, len(obs) - 1, _TYPE_SAMPLE_SIZE, dtype=int)
        obs = obs.iloc[idx]

    str_vals = obs.astype(str).str.strip()
    non_empty = str_vals != ""
    num_clean = str_vals.str.replace(_THOUSANDS_COMMA_RE, "", regex=True)
    num_coerced = pd.to_numeric(num_clean.where(non_empty), errors="coerce")
    num_arr = num_coerced.to_numpy(dtype=np.float64, na_value=np.nan)
    is_plain_number = np.isfinite(num_arr)

    if float(np.mean(is_plain_number)) >= 0.5:
        return "numeric_text"

    has_date_sep = str_vals.str.contains(
        _DATE_SEP_BETWEEN_DIGITS_RE, regex=True, na=False
    ).to_numpy(dtype=bool)
    if not np.any(has_date_sep & (~is_plain_number)):
        return "text"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            dt_coerced = pd.to_datetime(
                str_vals.where(non_empty), errors="coerce", format="mixed"
            )
        except TypeError:
            dt_coerced = pd.to_datetime(str_vals.where(non_empty), errors="coerce")

    is_date = (
        dt_coerced.notna().to_numpy(dtype=bool) & (~is_plain_number) & has_date_sep
    )
    if float(np.mean(is_date)) >= 0.5:
        return "datetime_like"

    return "text"


def _compute_minority_fraction_from_series(
    obs_series: pd.Series, n_observed: int
) -> float | None:
    """Compute share of the rarer value when n_unique == 2."""
    if n_observed == 0:
        return None
    counts = obs_series.value_counts(dropna=True)
    if len(counts) != 2:
        return None
    return float(counts.min() / n_observed)


def _profile_single_column(
    name: str, series: pd.Series, n_rows: int
) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None]:
    """Measure per-column facts and return (raw_fields, isna_mask, float_values).

    ``isna_mask`` is a 1-D bool array when ``0 < n_missing < n_rows``, else None.
    ``float_values`` is a 1-D float64 array when ``kind in ("numeric", "boolean")``,
    else None.
    """
    dtype = series.dtype
    is_bool_dt = pd.api.types.is_bool_dtype(dtype)
    is_td64 = pd.api.types.is_timedelta64_dtype(dtype)
    is_dt64 = pd.api.types.is_datetime64_any_dtype(dtype)
    is_complex_dt = pd.api.types.is_complex_dtype(dtype)
    is_num_dt = (
        pd.api.types.is_numeric_dtype(dtype) and not is_bool_dt and not is_complex_dt
    ) or is_td64
    is_obj_bool = False
    if not (is_bool_dt or is_num_dt or is_dt64 or is_complex_dt):
        obs_vals = series.dropna()
        if len(obs_vals) > 0 and all(isinstance(v, (bool, np.bool_)) for v in obs_vals):
            is_obj_bool = True
            is_bool_dt = True

    if is_bool_dt or is_num_dt:
        if is_td64:
            float_arr = series.dt.total_seconds().to_numpy(
                dtype=np.float64, na_value=np.nan
            )
        elif is_obj_bool:
            isna_mask = series.isna().to_numpy(dtype=bool)
            raw_obj = series.to_numpy()
            float_arr = np.empty(n_rows, dtype=np.float64)
            float_arr[isna_mask] = np.nan
            if not isna_mask.all():
                float_arr[~isna_mask] = np.asarray(
                    raw_obj[~isna_mask], dtype=np.float64
                )
        else:
            float_arr = series.to_numpy(dtype=np.float64, na_value=np.nan)
        isna_arr = np.isnan(float_arr)
        n_missing = int(isna_arr.sum())
        n_observed = n_rows - n_missing
        missing_fraction = float(n_missing / n_rows)
        obs = float_arr[~isna_arr] if n_missing > 0 else float_arr

        if n_observed == 0:
            kind: ColumnKind = "boolean" if is_bool_dt else "numeric"
            return (
                {
                    "name": name,
                    "kind": kind,
                    "n_rows": n_rows,
                    "n_missing": n_missing,
                    "missing_fraction": missing_fraction,
                    "n_unique": 0,
                    "unique_ratio": 0.0,
                    "is_constant": True,
                    "n_zero": None,
                    "observed_min": None,
                    "observed_max": None,
                    "integer_valued": None,
                    "nonnegative": None,
                    "minority_fraction": None,
                    "max_abs_z": None,
                    "non_numeric_values": None,
                    "n_non_numeric": None,
                },
                None,
                float_arr,
            )

        finite_mask = np.isfinite(obs)
        all_finite = bool(np.all(finite_mask))
        obs_finite = obs if all_finite else obs[finite_mask]
        obs_min: float | None = float(obs_finite.min()) if len(obs_finite) > 0 else None
        obs_max: float | None = float(obs_finite.max()) if len(obs_finite) > 0 else None
        n_zero = int(n_observed - np.count_nonzero(obs))
        is_01 = bool(
            is_bool_dt
            or (
                not is_td64
                and all_finite
                and obs_min is not None
                and obs_max is not None
                and obs_min >= 0.0
                and obs_max <= 1.0
                and int(np.count_nonzero((obs == 0.0) | (obs == 1.0))) == n_observed
            )
        )
        kind = "boolean" if is_01 else "numeric"

        if is_01:
            n_one = n_observed - n_zero
            n_unique = (1 if n_zero > 0 else 0) + (1 if n_one > 0 else 0)
            minority_fraction = (
                float(min(n_zero, n_one) / n_observed) if n_unique == 2 else None
            )
            integer_valued = None
        else:
            if all_finite and obs_min is not None and obs_min == obs_max:
                n_unique = 1
                minority_fraction = None
            else:
                uniques = pd.unique(obs)
                n_unique = int(len(uniques))
                if n_unique == 2:
                    c_first = int(np.count_nonzero(obs == uniques[0]))
                    minority_fraction = float(
                        min(c_first, n_observed - c_first) / n_observed
                    )
                else:
                    minority_fraction = None
            if pd.api.types.is_integer_dtype(dtype):
                integer_valued = True
            else:
                integer_valued = bool(all_finite and np.all(np.floor(obs) == obs))

        unique_ratio = float(n_unique / max(n_observed, 1))
        is_constant = bool(n_unique <= 1)
        nonnegative = bool(obs_min >= 0.0) if obs_min is not None else None

        if is_constant or len(obs_finite) < 2 or obs_min is None or obs_max is None:
            max_abs_z = None
        else:
            mean = float(obs_finite.mean())
            std = float(obs_finite.std(ddof=0))
            if std > 0.0 and np.isfinite(std):
                max_abs_z = float(max(abs(obs_max - mean), abs(obs_min - mean)) / std)
            else:
                max_abs_z = None

        partial_mask = isna_arr if 0 < n_missing < n_rows else None
        return (
            {
                "name": name,
                "kind": kind,
                "n_rows": n_rows,
                "n_missing": n_missing,
                "missing_fraction": missing_fraction,
                "n_unique": n_unique,
                "unique_ratio": unique_ratio,
                "is_constant": is_constant,
                "n_zero": n_zero,
                "observed_min": obs_min,
                "observed_max": obs_max,
                "integer_valued": integer_valued,
                "nonnegative": nonnegative,
                "minority_fraction": minority_fraction,
                "max_abs_z": max_abs_z,
                "non_numeric_values": None,
                "n_non_numeric": None,
            },
            partial_mask,
            float_arr,
        )

    isna_arr = series.isna().to_numpy(dtype=bool)
    n_missing = int(isna_arr.sum())
    n_observed = n_rows - n_missing
    missing_fraction = float(n_missing / n_rows)
    unhashable = False
    str_obs: pd.Series | None = None
    if n_observed > 0:
        try:
            n_unique = int(series.nunique(dropna=True))
        except TypeError:
            unhashable = True
            str_obs = series.dropna().astype(str)
            n_unique = int(str_obs.nunique(dropna=True))
    else:
        n_unique = 0
    unique_ratio = float(n_unique / max(n_observed, 1))
    is_constant = bool(n_unique <= 1)

    if is_dt64:
        kind = "datetime_like"
    elif unhashable:
        kind = "text"
    else:
        kind = _detect_object_kind(series, n_observed)

    if n_unique == 2:
        minority_series = str_obs if str_obs is not None else series
        minority_fraction = _compute_minority_fraction_from_series(
            minority_series, n_observed
        )
    else:
        minority_fraction = None

    non_numeric_values: tuple[tuple[str, int], ...] | None = None
    n_non_numeric: int | None = None
    if kind == "numeric_text":
        full_obs = str_obs if str_obs is not None else series.dropna().astype(str)
        stripped = full_obs.str.strip()
        non_empty = stripped[stripped != ""]
        if len(non_empty) == 0:
            non_numeric_values = ()
            n_non_numeric = 0
        else:
            cleaned = non_empty.str.replace(_THOUSANDS_COMMA_RE, "", regex=True)
            parsed = pd.to_numeric(cleaned, errors="coerce")
            bad = non_empty[parsed.isna()]
            n_non_numeric = int(len(bad))
            if n_non_numeric > 0:
                vc = bad.value_counts()
                sorted_items = sorted(
                    ((str(k), int(v)) for k, v in vc.items()),
                    key=lambda kv: (-kv[1], kv[0]),
                )
                non_numeric_values = tuple(sorted_items[:5])
            else:
                non_numeric_values = ()

    partial_mask = isna_arr if 0 < n_missing < n_rows else None

    return (
        {
            "name": name,
            "kind": kind,
            "n_rows": n_rows,
            "n_missing": n_missing,
            "missing_fraction": missing_fraction,
            "n_unique": n_unique,
            "unique_ratio": unique_ratio,
            "is_constant": is_constant,
            "n_zero": None,
            "observed_min": None,
            "observed_max": None,
            "integer_valued": None,
            "nonnegative": None,
            "minority_fraction": minority_fraction,
            "max_abs_z": None,
            "non_numeric_values": non_numeric_values,
            "n_non_numeric": n_non_numeric,
        },
        partial_mask,
        None,
    )


def _compute_relations(
    col_names: list[str],
    X_sample: np.ndarray,
    *,
    max_sum_sources: int,
    max_correlation_pairs: int,
) -> tuple[ColumnRelation, ...]:
    """Compute duplicate, sum_of, and strong_correlation relations on X_sample."""
    n_sub, p = X_sample.shape
    if n_sub < 2 or p < 2:
        return ()

    if not np.all(np.isfinite(X_sample) | np.isnan(X_sample)):
        X_sample = np.where(np.isfinite(X_sample), X_sample, np.nan)

    # Filter out columns that are constant on the sample
    obs_mask = ~np.isnan(X_sample)
    obs_counts = obs_mask.sum(axis=0)

    active_indices: list[int] = []
    for j in range(p):
        if obs_counts[j] < 2:
            continue
        col_obs = X_sample[obs_mask[:, j], j]
        if float(col_obs.min()) < float(col_obs.max()):
            active_indices.append(j)

    if len(active_indices) < 2:
        return ()

    # 1. Exact duplicates on all sampled rows (NaN-aware)
    duplicates: list[ColumnRelation] = []
    duplicate_targets: set[int] = set()
    reported_pairs: set[tuple[int, int]] = set()
    hash_to_first: dict[bytes, list[int]] = {}

    for j in active_indices:
        col = X_sample[:, j]
        # Canonicalize -0.0 -> 0.0 and NaN -> canonical NaN bit pattern
        norm = np.where(np.isnan(col), np.nan, col + 0.0)
        key = norm.tobytes()
        bucket = hash_to_first.get(key)
        if bucket is None:
            hash_to_first[key] = [j]
        else:
            matched_src: int | None = None
            for cand_i in bucket:
                if np.array_equal(X_sample[:, cand_i], col, equal_nan=True):
                    matched_src = cand_i
                    break
            if matched_src is not None:
                duplicates.append(
                    ColumnRelation(
                        kind="duplicate",
                        target=col_names[j],
                        sources=(col_names[matched_src],),
                        value=0.0,
                        rows_used=int(obs_counts[j]),
                    )
                )
                duplicate_targets.add(j)
                for prev_i in bucket:
                    pair = (min(prev_i, j), max(prev_i, j))
                    reported_pairs.add(pair)
            bucket.append(j)

    # 2. Vectorized pairwise Pearson correlation matrix over active columns
    idx_arr = np.array(active_indices, dtype=int)
    X_act = X_sample[:, idx_arr]
    M = (~np.isnan(X_act)).astype(np.float64)
    n_act = len(active_indices)

    # Pre-standardize each active column using its marginal observed mean/std
    col_n = np.maximum(M.sum(axis=0, keepdims=True), 1.0)
    X_zero = np.where(M > 0.0, X_act, 0.0)
    col_mean = X_zero.sum(axis=0, keepdims=True) / col_n
    X_cent = np.where(M > 0.0, X_act - col_mean, 0.0)
    col_std = np.sqrt(np.maximum((X_cent**2).sum(axis=0, keepdims=True) / col_n, 1e-30))
    Z = X_cent / col_std

    N_mat = M.T @ M
    S_mat = Z.T @ M
    P_mat = Z.T @ Z
    Q_mat = (Z**2).T @ M

    N_safe = np.maximum(N_mat, 1.0)
    mean_x = S_mat / N_safe
    mean_y = S_mat.T / N_safe
    cov_xy = P_mat - N_mat * mean_x * mean_y
    var_x = Q_mat - N_mat * (mean_x**2)
    var_y = Q_mat.T - N_mat * (mean_y**2)
    denom = np.sqrt(np.maximum(var_x * var_y, 0.0))

    valid_corr = (
        (N_mat >= 3.0)
        & (var_x > 1e-12 * N_safe)
        & (var_y > 1e-12 * N_safe)
        & (denom > 0.0)
    )
    R_mat = np.where(
        valid_corr, np.clip(cov_xy / np.where(denom > 0.0, denom, 1.0), -1.0, 1.0), 0.0
    )
    np.fill_diagonal(R_mat, 0.0)
    abs_R = np.abs(R_mat)

    # 3. sum_of relations on non-duplicate active columns
    sum_relations: list[ColumnRelation] = []
    non_dup_positions = [
        pos
        for pos, orig_j in enumerate(active_indices)
        if orig_j not in duplicate_targets
    ]
    eligible_cand_mask = np.ones(n_act, dtype=bool)
    if duplicate_targets:
        for pos, orig_j in enumerate(active_indices):
            if orig_j in duplicate_targets:
                eligible_cand_mask[pos] = False

    if len(non_dup_positions) >= 3 and max_sum_sources >= 2:
        for t_pos in non_dup_positions:
            orig_t = active_indices[t_pos]
            n_cands = int(eligible_cand_mask.sum()) - (
                1 if eligible_cand_mask[t_pos] else 0
            )
            if n_cands < 2:
                continue

            neg_scores = np.where(eligible_cand_mask, -abs_R[t_pos], 1.0)
            neg_scores[t_pos] = 1.0
            k_take = min(max_sum_sources, n_cands)
            if n_act > k_take:
                kth_val = float(np.partition(neg_scores, k_take - 1)[k_take - 1])
                strict_better = np.flatnonzero(neg_scores < kth_val)
                need = k_take - len(strict_better)
                ties = np.flatnonzero(neg_scores == kth_val)[:need]
                top_pool = (
                    np.concatenate([strict_better, ties])
                    if len(strict_better) > 0
                    else ties
                )
            else:
                top_pool = np.flatnonzero(neg_scores <= 0.0)
            order = np.lexsort((top_pool, neg_scores[top_pool]))
            top_cands = top_pool[order]
            if len(top_cands) < 2:
                continue

            y_col = X_act[:, t_pos]
            X_cands = X_act[:, top_cands]
            joint_mask = (~np.isnan(y_col)) & np.all(~np.isnan(X_cands), axis=1)
            n_joint = int(joint_mask.sum())
            if n_joint <= 2 * len(top_cands):
                continue

            try:
                coeffs, _, _, _ = np.linalg.lstsq(
                    X_cands[joint_mask], y_col[joint_mask], rcond=None
                )
            except np.linalg.LinAlgError:
                continue
            unit_mask = np.abs(coeffs - 1.0) <= _SUM_COEFF_TOL
            other_mask = ~unit_mask
            if int(unit_mask.sum()) < 2:
                continue
            if (
                np.any(other_mask)
                and float(np.max(np.abs(coeffs[other_mask]))) > _SUM_OTHER_COEFF_TOL
            ):
                continue

            chosen_positions = top_cands[unit_mask].tolist()
            X_chosen = X_act[:, chosen_positions]
            recheck_mask = (~np.isnan(y_col)) & np.all(~np.isnan(X_chosen), axis=1)
            n_recheck = int(recheck_mask.sum())
            if n_recheck <= 2 * len(top_cands):
                continue

            y_obs = y_col[recheck_mask]
            pred = np.sum(X_chosen[recheck_mask], axis=1)
            max_abs_res = float(np.max(np.abs(y_obs - pred)))
            max_abs_target = float(np.max(np.abs(y_obs))) if len(y_obs) > 0 else 0.0
            if max_abs_res <= _SUM_RESIDUAL_REL_TOL * (1.0 + max_abs_target):
                chosen_orig = sorted(active_indices[pos] for pos in chosen_positions)
                sum_relations.append(
                    ColumnRelation(
                        kind="sum_of",
                        target=col_names[orig_t],
                        sources=tuple(col_names[idx] for idx in chosen_orig),
                        value=max_abs_res,
                        rows_used=n_recheck,
                    )
                )
                eligible_cand_mask[t_pos] = False
                for src_idx in chosen_orig:
                    pair = (min(src_idx, orig_t), max(src_idx, orig_t))
                    reported_pairs.add(pair)

    # 4. strong_correlation relations: top max_correlation_pairs by |r| desc
    corr_relations: list[ColumnRelation] = []
    if max_correlation_pairs > 0:
        eligible_pair_mask = valid_corr.copy()
        if reported_pairs:
            orig_to_pos = {orig: pos for pos, orig in enumerate(active_indices)}
            for orig_a, orig_b in reported_pairs:
                pa = orig_to_pos.get(orig_a)
                pb = orig_to_pos.get(orig_b)
                if pa is not None and pb is not None:
                    eligible_pair_mask[pa, pb] = False
                    eligible_pair_mask[pb, pa] = False

        iu, ju = np.triu_indices(n_act, k=1)
        valid_pairs = eligible_pair_mask[iu, ju]
        iu_v = iu[valid_pairs]
        ju_v = ju[valid_pairs]
        n_valid_pairs = len(iu_v)
        if n_valid_pairs > 0:
            r_vals = abs_R[iu_v, ju_v]
            neg_r = -r_vals
            k_corr = min(max_correlation_pairs, n_valid_pairs)
            if n_valid_pairs > k_corr:
                kth_val = float(np.partition(neg_r, k_corr - 1)[k_corr - 1])
                strict_better = np.flatnonzero(neg_r < kth_val)
                need = k_corr - len(strict_better)
                ties = np.flatnonzero(neg_r == kth_val)[:need]
                sel = (
                    np.concatenate([strict_better, ties])
                    if len(strict_better) > 0
                    else ties
                )
            else:
                sel = np.arange(n_valid_pairs)

            order = np.lexsort((ju_v[sel], iu_v[sel], neg_r[sel]))
            top_sel = sel[order]
            for idx_pair in top_sel.tolist():
                a_pos = int(iu_v[idx_pair])
                b_pos = int(ju_v[idx_pair])
                orig_i = active_indices[a_pos]
                orig_j = active_indices[b_pos]
                corr_relations.append(
                    ColumnRelation(
                        kind="strong_correlation",
                        target=col_names[orig_j],
                        sources=(col_names[orig_i],),
                        value=float(r_vals[idx_pair]),
                        rows_used=int(N_mat[a_pos, b_pos]),
                    )
                )

    return tuple(duplicates + sum_relations + corr_relations)


def measure_preprocessing_evidence(
    df: pd.DataFrame,
    *,
    sample_rows: int = 5000,
    seed: int = 0,
    max_sum_sources: int = 6,
    max_correlation_pairs: int = 20,
    max_relation_columns: int = 2000,
    exclude_columns: Sequence[str] = (),
) -> PreprocessingEvidence:
    """Measure per-column statistics, missing groups, mask dependencies, and relations."""
    t0 = time.perf_counter()

    if exclude_columns:
        excluded_set = {str(c) for c in exclude_columns}
        keep_mask = [str(col) not in excluded_set for col in df.columns]
        df = df.loc[:, keep_mask]

    n_rows, n_cols = df.shape
    if n_rows == 0 and n_cols == 0:
        raise ValueError("DataFrame has no rows and no columns.")
    if n_rows == 0:
        raise ValueError("DataFrame has no rows.")
    if n_cols == 0:
        raise ValueError("DataFrame has no columns.")

    raw_cols = list(df.columns)
    seen_raw_keys: set[tuple[type, Any]] = set()
    dup_raw_keys: set[tuple[type, Any]] = set()
    dup_raw: list[Any] = []
    for col in raw_cols:
        key: tuple[type, Any] = (type(col), col)
        try:
            in_seen = key in seen_raw_keys
        except TypeError:
            key = (type(col), repr(col))
            in_seen = key in seen_raw_keys
        if in_seen:
            if key not in dup_raw_keys:
                dup_raw_keys.add(key)
                dup_raw.append(col)
        else:
            seen_raw_keys.add(key)
    if dup_raw:
        raise ValueError(f"DataFrame has duplicate column names: {dup_raw!r}.")

    seen_str: dict[str, Any] = {}
    for col in raw_cols:
        col_s = str(col)
        if col_s in seen_str:
            prev_col = seen_str[col_s]
            raise ValueError(
                f"DataFrame column names must be unique after str(): "
                f"{prev_col!r} and {col!r} both map to {col_s!r}."
            )
        seen_str[col_s] = col

    raw_profiles: list[dict[str, Any]] = []
    partial_packed: dict[str, bytes] = {}
    col_order_index: dict[str, int] = {}
    numeric_col_names: list[str] = []
    numeric_col_arrays: list[np.ndarray] = []

    zero_partner_buckets: dict[bytes, list[str]] = {}
    nonzero_partner_buckets: dict[bytes, list[str]] = {}

    for idx, col in enumerate(raw_cols):
        col_name = str(col)
        col_order_index[col_name] = idx
        raw, partial_mask, float_arr = _profile_single_column(
            col_name, df.iloc[:, idx], n_rows
        )
        raw_profiles.append(raw)

        if partial_mask is not None:
            partial_packed[col_name] = np.packbits(partial_mask).tobytes()

        if float_arr is not None:
            numeric_col_names.append(col_name)
            numeric_col_arrays.append(float_arr)
            if raw["n_missing"] == 0 and 0 < (raw["n_zero"] or 0) < n_rows:
                zero_mask = float_arr == 0.0
                z_bytes = np.packbits(zero_mask).tobytes()
                nz_bytes = np.packbits(~zero_mask).tobytes()
                zero_partner_buckets.setdefault(z_bytes, []).append(col_name)
                nonzero_partner_buckets.setdefault(nz_bytes, []).append(col_name)

    # Missing groups: >=2 partially missing columns with identical NaN masks
    mask_buckets: dict[bytes, list[str]] = {}
    for col_name, packed in partial_packed.items():
        mask_buckets.setdefault(packed, []).append(col_name)

    raw_by_name = {r["name"]: r for r in raw_profiles}
    multi_buckets = [cols for cols in mask_buckets.values() if len(cols) >= 2]
    multi_buckets.sort(
        key=lambda cols: (
            -int(raw_by_name[cols[0]]["n_missing"]),
            col_order_index[cols[0]],
        )
    )

    missing_groups: list[MissingGroup] = []
    col_to_group_id: dict[str, str] = {}
    for g_idx, cols in enumerate(multi_buckets, start=1):
        gid = f"mg{g_idx}"
        n_miss = int(raw_by_name[cols[0]]["n_missing"])
        zero_free = tuple(
            c
            for c in cols
            if raw_by_name[c]["kind"] == "numeric"
            and raw_by_name[c]["nonnegative"] is True
            and raw_by_name[c]["n_zero"] == 0
        )
        missing_groups.append(
            MissingGroup(
                group_id=gid,
                columns=tuple(cols),
                n_missing=n_miss,
                zero_free_columns=zero_free,
            )
        )
        for c in cols:
            col_to_group_id[c] = gid

    # Mask dependencies: exact match between partially-missing NaN mask and
    # partner == 0 or partner != 0 on a fully-observed numeric/boolean column
    mask_dependencies: list[MaskDependency] = []
    col_to_mask_dep: dict[str, MaskDependency] = {}
    for r in raw_profiles:
        col_name = r["name"]
        packed = partial_packed.get(col_name)
        if packed is None:
            continue
        z_matches = zero_partner_buckets.get(packed, [])
        nz_matches = nonzero_partner_buckets.get(packed, [])
        if not z_matches and not nz_matches:
            continue

        if z_matches and (
            not nz_matches
            or col_order_index[z_matches[0]] <= col_order_index[nz_matches[0]]
        ):
            dep = MaskDependency(
                column=col_name,
                partner=z_matches[0],
                relation="missing_when_partner_zero",
                n_other_partners=len(z_matches) - 1,
            )
        else:
            dep = MaskDependency(
                column=col_name,
                partner=nz_matches[0],
                relation="missing_when_partner_nonzero",
                n_other_partners=len(nz_matches) - 1,
            )
        mask_dependencies.append(dep)
        col_to_mask_dep[col_name] = dep

    # Assemble ColumnEvidence in df column order
    column_evidences = tuple(
        ColumnEvidence(
            name=r["name"],
            kind=r["kind"],
            n_rows=r["n_rows"],
            n_missing=r["n_missing"],
            missing_fraction=r["missing_fraction"],
            n_unique=r["n_unique"],
            unique_ratio=r["unique_ratio"],
            is_constant=r["is_constant"],
            n_zero=r["n_zero"],
            observed_min=r["observed_min"],
            observed_max=r["observed_max"],
            integer_valued=r["integer_valued"],
            nonnegative=r["nonnegative"],
            minority_fraction=r["minority_fraction"],
            max_abs_z=r["max_abs_z"],
            missing_group=col_to_group_id.get(r["name"]),
            mask_dependency=col_to_mask_dep.get(r["name"]),
            non_numeric_values=r["non_numeric_values"],
            n_non_numeric=r["n_non_numeric"],
        )
        for r in raw_profiles
    )

    # Relations over deterministic row sample
    steps_skipped: list[dict[str, str]] = []
    rows_sampled = min(n_rows, max(int(sample_rows), 1))
    n_num_cols = len(numeric_col_names)

    if n_rows < 2:
        steps_skipped.append(
            {"step": "relations", "reason": "fewer than 2 rows in dataset"}
        )
        relations: tuple[ColumnRelation, ...] = ()
        numeric_considered = 0
    elif n_num_cols > max_relation_columns:
        steps_skipped.append(
            {
                "step": "relations",
                "reason": (
                    f"numeric/boolean column count ({n_num_cols}) exceeds "
                    f"max_relation_columns ({max_relation_columns})"
                ),
            }
        )
        relations = ()
        numeric_considered = 0
    elif n_num_cols < 2:
        steps_skipped.append(
            {
                "step": "relations",
                "reason": "fewer than 2 numeric or boolean columns",
            }
        )
        relations = ()
        numeric_considered = n_num_cols
    else:
        numeric_considered = n_num_cols
        X_full = np.column_stack(numeric_col_arrays)
        if n_rows > rows_sampled:
            rng = np.random.default_rng(seed)
            sample_idx = np.sort(rng.choice(n_rows, size=rows_sampled, replace=False))
            X_sample = X_full[sample_idx]
        else:
            X_sample = X_full
        relations = _compute_relations(
            numeric_col_names,
            X_sample,
            max_sum_sources=max_sum_sources,
            max_correlation_pairs=max_correlation_pairs,
        )

    elapsed_ms = float((time.perf_counter() - t0) * 1000.0)
    cost: dict[str, Any] = {
        "rows_sampled": rows_sampled,
        "numeric_columns_considered": numeric_considered,
        "steps_skipped": steps_skipped,
        "elapsed_ms": elapsed_ms,
    }

    return PreprocessingEvidence(
        n_rows=n_rows,
        n_columns=n_cols,
        columns=column_evidences,
        missing_groups=tuple(missing_groups),
        mask_dependencies=tuple(mask_dependencies),
        relations=relations,
        cost=cost,
    )


def _zscore_columns(X: np.ndarray) -> np.ndarray:
    """Z-score each column (ddof=0), mapping zero-variance columns to 0."""
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, ddof=0, keepdims=True)
    centered = X - mean
    safe_std = np.where(std > 0.0, std, 1.0)
    return np.where(std > 0.0, centered / safe_std, 0.0)


def _knn_neighbor_indices(X_scaled: np.ndarray, k: int) -> np.ndarray:
    """Return (n_rows, k) array of Euclidean k-NN indices excluding self."""
    from sklearn.neighbors import NearestNeighbors

    n_rows = X_scaled.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
    _, idx = nn.fit(X_scaled).kneighbors(X_scaled, n_neighbors=k + 1)

    row_ids = np.arange(n_rows)[:, None]
    is_self = idx == row_ids
    has_self = np.any(is_self, axis=1)
    if not np.all(has_self):
        is_self[~has_self, -1] = True
    return idx[~is_self].reshape(n_rows, k)


def neighborhood_overlap(X_a: np.ndarray, X_b: np.ndarray, *, k: int = 10) -> float:
    """Compute mean per-row Jaccard overlap of Euclidean k-NN sets after column z-scoring."""
    arr_a = np.asarray(X_a, dtype=np.float64)
    arr_b = np.asarray(X_b, dtype=np.float64)

    if arr_a.ndim != 2 or arr_b.ndim != 2 or arr_a.shape != arr_b.shape:
        raise ValueError(
            f"X_a and X_b must have the same shape (2D), got {arr_a.shape} and {arr_b.shape}."
        )
    n_rows, n_cols = arr_a.shape
    if n_rows < 2:
        raise ValueError(f"X_a and X_b must have at least 2 rows, got {n_rows}.")
    if n_cols < 1:
        raise ValueError("X_a and X_b must have at least 1 column.")
    if (
        not isinstance(k, (int, np.integer))
        or isinstance(k, bool)
        or k < 1
        or k >= n_rows
    ):
        raise ValueError(
            f"k must be an integer with 1 <= k < n_rows ({n_rows}), got {k}."
        )
    if not (np.all(np.isfinite(arr_a)) and np.all(np.isfinite(arr_b))):
        raise ValueError("X_a and X_b must contain only finite values.")

    Z_a = _zscore_columns(arr_a)
    Z_b = _zscore_columns(arr_b)

    nbrs_a = _knn_neighbor_indices(Z_a, int(k))
    nbrs_b = _knn_neighbor_indices(Z_b, int(k))

    intersections = np.sum(
        nbrs_a[:, :, None] == nbrs_b[:, None, :], axis=(1, 2), dtype=np.float64
    )
    unions = 2.0 * float(k) - intersections
    jaccard = intersections / unions
    return float(np.mean(jaccard))
