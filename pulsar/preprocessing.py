"""
Python-side preprocessing utilities for Pulsar.

Complements the Rust impute_column (f64 only) with Python implementations
for data types Rust cannot handle directly (string/object columns).

Also provides ``preprocess_dataframe`` — the single shared preprocessing
function used by both ``ThemaRS.fit()`` and ``ThemaRS.fit_multi()``.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from pulsar._pulsar import impute_column
from pulsar.config import ImputeSpec, PulsarConfig

# Numeric imputation methods — these require the column to be numeric.
_NUMERIC_IMPUTE_METHODS = frozenset({"fill_mean", "fill_median", "sample_normal"})


# ---------------------------------------------------------------------------
# PreprocessLayout — frozen transform contract for fit_multi()
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessLayout:
    """Records the output schema of a preprocessing pass.

    Used by ``fit_multi()`` to enforce that every dataset produces the
    exact same feature layout after preprocessing.
    """

    feature_names: tuple[str, ...]  # exact output columns, in order
    vocab: dict[str, tuple[str, ...]]  # categorical vocabularies (frozen)
    n_rows: int  # row count (must match across datasets)


# ---------------------------------------------------------------------------
# String imputation (Python-side, in-place)
# ---------------------------------------------------------------------------


def impute_string_column(df: pd.DataFrame, col: str, spec: ImputeSpec) -> None:
    """Impute a string/object column in-place.

    The Rust ``impute_column`` only handles f64 arrays, so string/object
    columns require a Python implementation.  Only ``fill_mode``,
    ``sample_categorical``, and ``fill_value`` are meaningful for string data.

    Raises:
        ValueError: If ``spec.method`` is numeric-only
            (``fill_mean``, ``fill_median``, ``sample_normal``).
        ValueError: If the column is entirely missing (except ``fill_value``).
    """
    missing_mask = df[col].isna()
    if not missing_mask.any():
        return

    if spec.method == "fill_value":
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            if spec.value not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories([spec.value])
        elif not isinstance(spec.value, str) and not is_numeric_dtype(df[col]):
            df[col] = df[col].astype(object)
        df.loc[missing_mask, col] = spec.value
        return

    observed = df.loc[~missing_mask, col]
    if observed.empty:
        raise ValueError(
            f"Column '{col}' is all-missing; cannot impute without observed values."
        )

    if spec.method == "fill_mode":
        mode = observed.mode()
        if not mode.empty:
            df.loc[missing_mask, col] = mode.iloc[0]

    elif spec.method == "sample_categorical":
        counts = observed.value_counts()
        categories = counts.index.tolist()
        weights = counts.values.tolist()
        rng = random.Random(spec.seed)
        n_missing = int(missing_mask.sum())
        fills = rng.choices(categories, weights=weights, k=n_missing)
        df.loc[missing_mask, col] = fills

    else:
        raise ValueError(
            f"Column '{col}' is string/object dtype; method '{spec.method}' requires "
            f"numeric data. Use 'fill_mode', 'sample_categorical', or 'fill_value' for string columns."
        )


# ---------------------------------------------------------------------------
# Core preprocessing function
# ---------------------------------------------------------------------------


def preprocess_dataframe(
    data: pd.DataFrame,
    cfg: PulsarConfig,
    *,
    vocab: dict[str, list[Any]] | None = None,
    expected_layout: PreprocessLayout | None = None,
) -> tuple[pd.DataFrame, PreprocessLayout]:
    """Preprocess a DataFrame according to ``cfg``.

    Sequencing: group indicators (pre-drop) → drop → coerce →
    all-missing guard → impute flags → impute → encode → NaN check →
    numeric check → layout check.

    Args:
        data: Raw input DataFrame (not mutated).
        cfg: Pipeline configuration.
        vocab: Shared categorical vocabularies for ``fit_multi()``.
            When provided, one-hot encoding uses these categories instead
            of inferring from the current dataset.
        expected_layout: If provided, the output must match this layout
            exactly (column names and row count).  Used by ``fit_multi()``
            to enforce consistency across datasets.

    Returns:
        ``(df_out, layout)`` where ``df_out`` is all-numeric with no NaN
        and ``layout`` records the output schema.

    Raises:
        ValueError: On NaN remaining, all-missing columns, coercion
            failure, cardinality violation, or layout mismatch.
        TypeError: On non-numeric columns remaining after preprocessing.
    """
    n_rows_input = len(data)

    # 0. Validate group indicator collisions and capture pre-drop masks for dropped group columns
    drop_set = set(cfg.drop_columns)
    grouped_columns = {
        c for cols in cfg.missing_indicator_groups.values() for c in cols
    }
    per_col_flag_names = {
        f"{col}_was_missing"
        for col, spec in cfg.impute.items()
        if spec.indicator and col not in grouped_columns and col not in drop_set
    }
    input_col_names = {str(c) for c in data.columns}
    seen_group_flags: set[str] = set()
    pre_drop_group_masks: dict[str, pd.Series] = {}
    for group_name, group_cols in cfg.missing_indicator_groups.items():
        flag_name = f"{group_name}_was_missing"
        if flag_name in input_col_names:
            raise ValueError(
                f"missing indicator name collision: group {group_name!r} flag "
                f"'{flag_name}' collides with existing input column '{flag_name}'."
            )
        if flag_name in per_col_flag_names:
            raise ValueError(
                f"missing indicator name collision: group {group_name!r} flag "
                f"'{flag_name}' collides with per-column indicator '{flag_name}'."
            )
        if flag_name in seen_group_flags:
            raise ValueError(
                f"missing indicator name collision: duplicate group flag '{flag_name}'."
            )
        seen_group_flags.add(flag_name)
        dropped_present = [c for c in group_cols if c in data.columns and c in drop_set]
        if dropped_present:
            pre_drop_group_masks[group_name] = data[dropped_present].isna().any(axis=1)

    # 1. Drop unwanted columns
    drop = [c for c in cfg.drop_columns if c in data.columns]
    df = data.drop(columns=drop).copy()

    # 2. Coerce configured numeric-impute columns
    for col, spec in cfg.impute.items():
        if col not in df.columns:
            continue
        is_numeric_fill_value = (
            spec.method == "fill_value"
            and isinstance(spec.value, (int, float, np.number))
            and not isinstance(spec.value, (bool, np.bool_))
        )
        if spec.method not in _NUMERIC_IMPUTE_METHODS and not is_numeric_fill_value:
            continue
        if is_numeric_dtype(df[col]):
            continue
        # Attempt coercion
        original_notna = df[col].notna().sum()
        coerced = pd.to_numeric(df[col], errors="coerce")
        coerced_notna = coerced.notna().sum()
        if original_notna > 0 and coerced_notna < original_notna * 0.5:
            # More than half of non-null values couldn't be parsed
            bad_mask = df[col].notna() & coerced.isna()
            sample = df.loc[bad_mask, col].head(5).tolist()
            raise ValueError(
                f"Column '{col}' is configured for numeric imputation "
                f"({spec.method}) but contains non-numeric values. "
                f"Sample: {sample!r}. Clean the data or use a string "
                f"imputation method."
            )
        df[col] = coerced.astype(np.float64)

    # 3. All-missing guard for impute/encode targets
    for col in list(cfg.impute) + list(cfg.encode):
        if col not in df.columns:
            continue
        if col in cfg.impute and cfg.impute[col].method == "fill_value":
            continue
        if df[col].isna().all():
            raise ValueError(
                f"Column '{col}' is all-missing; cannot impute/encode "
                f"without observed values."
            )

    # 4. Imputation indicator flags (group masks combine pre-drop dropped cols and post-coercion kept cols)
    for group_name, group_cols in cfg.missing_indicator_groups.items():
        kept_present = [c for c in group_cols if c in df.columns]
        pre_mask = pre_drop_group_masks.get(group_name)
        if pre_mask is None and not kept_present:
            continue
        if pre_mask is not None and kept_present:
            combined_mask = pre_mask | df[kept_present].isna().any(axis=1)
        elif pre_mask is not None:
            combined_mask = pre_mask
        else:
            combined_mask = df[kept_present].isna().any(axis=1)
        df[f"{group_name}_was_missing"] = combined_mask.astype(np.float64)

    for col, spec in cfg.impute.items():
        if col in df.columns and spec.indicator and col not in grouped_columns:
            df[f"{col}_was_missing"] = df[col].isna().astype(np.float64)

    # 5. Impute (Rust for numeric, Python for string)
    for col, spec in cfg.impute.items():
        if col not in df.columns:
            continue
        if is_numeric_dtype(df[col]):
            arr = df[col].to_numpy(dtype=np.float64, na_value=np.nan)
            if spec.method == "fill_value":
                if isinstance(spec.value, (str, bool)) or not isinstance(
                    spec.value, (int, float, np.number)
                ):
                    raise ValueError(
                        f"fill_value for numeric column '{col}' must be numeric, "
                        f"got {spec.value!r}."
                    )
                df[col] = impute_column(
                    arr, "fill_value", spec.seed, value=float(spec.value)
                )
            else:
                df[col] = impute_column(arr, spec.method, spec.seed)
        else:
            impute_string_column(df, col, spec)

    # 6. Encode categoricals
    for col, spec in cfg.encode.items():
        if col not in df.columns:
            continue
        if spec.method == "one_hot":
            if vocab is not None and col in vocab:
                n_cats = len(vocab[col])
                df[col] = pd.Categorical(df[col], categories=vocab[col])
            else:
                n_cats = df[col].nunique(dropna=True)

            # Cardinality check
            if spec.max_categories is not None and n_cats > spec.max_categories:
                raise ValueError(
                    f"Column '{col}' has {n_cats} categories, exceeding "
                    f"max_categories={spec.max_categories}. Reduce cardinality, "
                    f"drop the column, or increase max_categories."
                )
            if n_cats > 50 and spec.max_categories is None:
                warnings.warn(
                    f"One-hot encoding column '{col}' with {n_cats} categories "
                    f"will add {n_cats} dimensions to the feature space, which "
                    f"may distort Euclidean distances after scaling.",
                    stacklevel=2,
                )

            df = pd.get_dummies(df, columns=[col], prefix=col, dtype=np.float64)
        else:
            raise ValueError(
                f"Unsupported encode method '{spec.method}' for column '{col}'. "
                f"Supported: 'one_hot'."
            )

    # 7. NaN check (replaces silent dropna)
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        detail = ", ".join(f"'{c}' ({int(nan_cols[c])} rows)" for c in nan_cols.index)
        raise ValueError(
            f"NaN values remain after imputation in {len(nan_cols)} column(s): "
            f"{detail}. Add imputation rules in preprocessing.impute, or drop "
            f"them via drop_columns."
        )

    # 8. Numeric type check
    non_numeric = [
        (c, str(df[c].dtype)) for c in df.columns if not is_numeric_dtype(df[c])
    ]
    if non_numeric:
        detail = ", ".join(f"'{c}' (dtype={dt})" for c, dt in non_numeric)
        raise TypeError(
            f"Non-numeric columns remain after preprocessing: {detail}. "
            f"Encode them via preprocessing.encode in params.yaml, or drop "
            f"them via drop_columns."
        )

    # 9. Row count assertion (defense-in-depth)
    if len(df) != n_rows_input:
        raise RuntimeError(
            f"Preprocessing changed row count from {n_rows_input} to {len(df)}. "
            f"This is a bug in the preprocessing pipeline."
        )

    df = df.reset_index(drop=True)

    # 10. Build layout
    built_vocab: dict[str, tuple[str, ...]] = {}
    if vocab is not None:
        built_vocab = {k: tuple(v) for k, v in vocab.items()}
    layout = PreprocessLayout(
        feature_names=tuple(df.columns.tolist()),
        vocab=built_vocab,
        n_rows=len(df),
    )

    # 11. Layout enforcement
    if expected_layout is not None:
        if layout.feature_names != expected_layout.feature_names:
            expected_set = set(expected_layout.feature_names)
            actual_set = set(layout.feature_names)
            missing = sorted(expected_set - actual_set)
            extra = sorted(actual_set - expected_set)
            parts = []
            if missing:
                parts.append(f"  Missing columns: {missing}")
            if extra:
                parts.append(f"  Extra columns: {extra}")
            raise ValueError(
                "Feature layout does not match the reference dataset.\n"
                + "\n".join(parts)
            )
        if layout.n_rows != expected_layout.n_rows:
            raise ValueError(
                f"Row count {layout.n_rows} does not match the reference "
                f"dataset ({expected_layout.n_rows} rows)."
            )

    return df, layout
