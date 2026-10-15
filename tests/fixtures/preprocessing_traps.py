"""
Synthetic clinical/claims fixture reproducing every preprocessing postmortem trap
with exact, documented known answers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_preprocessing_traps(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Construct a synthetic tabular dataset containing known preprocessing traps.

    Known answers for default ``n=400, seed=0``:

    - **Dimensions**: ``n_rows = 400``, ``n_columns = 21``.
    - **Identifiers & constants**:
      - ``patient_id``: ``kind="text"``, ``n_missing=0``, ``n_unique=400``,
        ``unique_ratio=1.0``, ``is_constant=False``.
      - ``cohort``: ``kind="text"``, ``n_missing=0``, ``n_unique=1``,
        ``is_constant=True``.
      - ``sex``: ``kind="boolean"``, ``n_missing=0``, ``n_unique=1``,
        ``is_constant=True``, ``max_abs_z=None``.
    - **Shared missingness group 1 (mortality block, ``group_id="mg1"``)**:
      - ``died``: ``kind="boolean"``, ``n_missing=0``, ``n_unique=2``,
        ``n_zero=360``, ``minority_fraction=0.10`` (40 True, 360 False).
      - ``death_date``: ``kind="datetime_like"``, ISO date strings when ``died``
        is True and NaN when ``died`` is False; ``n_missing=360``
        (``missing_fraction=0.90``), ``missing_group="mg1"``,
        ``mask_dependency=MaskDependency("death_date", "died", "missing_when_partner_zero", 0)``.
      - ``days_to_death``: ``kind="numeric"``, positive integers 1..365 when
        ``died`` is True and NaN when ``died`` is False; ``n_missing=360``
        (``missing_fraction=0.90``), ``n_zero=0``, ``integer_valued=True``,
        ``nonnegative=True``, ``missing_group="mg1"``,
        ``mask_dependency=MaskDependency("days_to_death", "died", "missing_when_partner_zero", 0)``.
      - Group ``mg1``: ``columns=("death_date", "days_to_death")``,
        ``n_missing=360``, ``zero_free_columns=("days_to_death",)``.
    - **Shared missingness group 2 (inpatient utilization block, ``group_id="mg2"``)**:
      - ``n_inpatient_claims``: ``kind="numeric"``, ``n_missing=200``
        (``missing_fraction=0.50``), ``n_zero=0``, ``observed_min >= 1.0``,
        ``integer_valued=True``, ``nonnegative=True``, ``missing_group="mg2"``.
      - ``inpatient_payment``: ``kind="numeric"``, identical NaN mask
        (``n_missing=200``), ``n_zero=5`` (contains explicit zero payments),
        ``nonnegative=True``, ``missing_group="mg2"``.
      - ``inpatient_days``: ``kind="numeric"``, identical NaN mask
        (``n_missing=200``), ``n_zero=0``, ``observed_min >= 1.0``,
        ``integer_valued=True``, ``nonnegative=True``, ``missing_group="mg2"``.
      - Group ``mg2``:
        ``columns=("n_inpatient_claims", "inpatient_payment", "inpatient_days")``,
        ``n_missing=200``,
        ``zero_free_columns=("n_inpatient_claims", "inpatient_days")``.
    - **Exact linear sum block**:
      - ``cost_2008``, ``cost_2009``, ``cost_2010``: complete nonnegative
        ``kind="numeric"`` columns.
      - ``cost_total``: exact sum ``cost_2008 + cost_2009 + cost_2010``,
        producing ``ColumnRelation(kind="sum_of", target="cost_total",
        sources=("cost_2008", "cost_2009", "cost_2010"), ...)``.
    - **Strongly correlated binary pair**:
      - ``chf_2008``, ``chf_ever``: ``kind="boolean"``, complete 0/1 indicators
        with Pearson ``|r| > 0.95`` reported under ``strong_correlation``.
    - **Discrete code & rare binary indicator**:
      - ``state_code``: ``kind="numeric"``, complete integers ``1..52``,
        ``n_unique=min(52, n)``, ``integer_valued=True``, ``nonnegative=True``.
      - ``rare_flag``: ``kind="boolean"``, complete 0/1 with a single ``1`` at
        row 0; ``n_unique=2``, ``n_zero=n - 1``, ``minority_fraction=1 / n``,
        ``max_abs_z=sqrt(n - 1)``.
    - **Unstructured / dirty / MCAR columns**:
      - ``race``: ``kind="text"``, 4 categories including a rare level,
        ``n_missing = n // 20`` (5% missing), ``missing_group=None``.
      - ``bmi``: ``kind="numeric"``, continuous normal values with
        ``n_missing = n // 10`` (10% MCAR), ``integer_valued=False``,
        ``missing_group=None``.
      - ``lab_text``: ``kind="numeric_text"``, numeric strings with literal
        ``"N/A"`` sentinels, ``n_missing=0``.
      - ``age``: ``kind="numeric"``, complete nonnegative integers 65..95,
        ``n_missing=0``.
    """
    if n < 20:
        raise ValueError(f"make_preprocessing_traps requires n >= 20, got {n}.")

    rng = np.random.default_rng(seed)

    # 1. Identifiers and constants
    patient_id = [f"PAT_{i:05d}" for i in range(n)]
    cohort = ["COHORT_2008"] * n
    sex = np.ones(n, dtype=np.int64)

    # 2. Inpatient utilization block (~50% missing together)
    n_inp_missing = n // 2
    inp_perm = rng.permutation(n)
    inp_missing_idx = inp_perm[:n_inp_missing]
    inp_observed_idx = inp_perm[n_inp_missing:]

    n_inpatient_claims = rng.integers(1, 9, size=n).astype(np.float64)
    inpatient_payment = np.round(rng.uniform(500.0, 25000.0, size=n), 2)
    # Force 5 observed rows to have exact zero payment (e.g. denied/waived claims)
    inpatient_payment[inp_observed_idx[:5]] = 0.0
    inpatient_days = (n_inpatient_claims * rng.integers(1, 5, size=n)).astype(
        np.float64
    )

    n_inpatient_claims[inp_missing_idx] = np.nan
    inpatient_payment[inp_missing_idx] = np.nan
    inpatient_days[inp_missing_idx] = np.nan

    # 3. Mortality block (died ~10%; death_date & days_to_death NaN iff not died)
    n_died = max(2, n // 10)
    died_perm = rng.permutation(n)
    died_idx = died_perm[:n_died]
    died = np.zeros(n, dtype=bool)
    died[died_idx] = True

    months = rng.integers(1, 13, size=n)
    days = rng.integers(1, 29, size=n)
    death_date: list[str | None] = [
        f"2008-{int(months[i]):02d}-{int(days[i]):02d}" if died[i] else None
        for i in range(n)
    ]
    days_to_death = np.where(
        died, rng.integers(1, 366, size=n).astype(np.float64), np.nan
    )

    # 4. Cost sum block (cost_total = cost_2008 + cost_2009 + cost_2010)
    cost_2008 = np.round(rng.uniform(100.0, 5000.0, size=n), 2)
    cost_2009 = np.round(rng.uniform(100.0, 5000.0, size=n), 2)
    cost_2010 = np.round(rng.uniform(100.0, 5000.0, size=n), 2)
    cost_total = cost_2008 + cost_2009 + cost_2010

    # 5. Chronic condition binary pair (chf_2008 / chf_ever)
    n_chf_2008 = n // 4
    chf_perm = rng.permutation(n)
    chf_2008 = np.zeros(n, dtype=np.int64)
    chf_2008[chf_perm[:n_chf_2008]] = 1
    chf_ever = chf_2008.copy()
    chf_ever[chf_perm[n_chf_2008 : n_chf_2008 + max(1, n // 80)]] = 1

    # 6. Discrete state code (1..52) and rare binary flag (single 1)
    state_code = ((np.arange(n, dtype=np.int64) % 52) + 1)[rng.permutation(n)]
    rare_flag = np.zeros(n, dtype=np.int64)
    rare_flag[0] = 1

    # 7. Race categorical with rare level and 5% MCAR
    race_levels = ["White", "Black", "Hispanic"]
    race_vals: list[str | None] = [
        race_levels[int(x)] for x in rng.choice(3, size=n, p=[0.70, 0.18, 0.12])
    ]
    n_race_missing = n // 20
    race_perm = rng.permutation(n)
    for idx in race_perm[:3]:
        race_vals[int(idx)] = "Other"
    for idx in race_perm[3 : 3 + n_race_missing]:
        race_vals[int(idx)] = None

    # 8. BMI normal numeric with 10% MCAR
    bmi = np.round(rng.normal(27.5, 4.5, size=n), 2)
    bmi[0] = 27.35  # Guarantee non-integer value on an observed row
    n_bmi_missing = n // 10
    bmi_missing_idx = rng.choice(np.arange(1, n), size=n_bmi_missing, replace=False)
    bmi[bmi_missing_idx] = np.nan

    # 9. Lab text: numeric strings with literal "N/A" sentinels
    lab_raw = rng.normal(100.0, 12.0, size=n)
    lab_text = [f"{v:.1f}" for v in lab_raw]
    for idx in rng.choice(n, size=max(1, n // 25), replace=False):
        lab_text[int(idx)] = "N/A"

    # 10. Age: complete numeric
    age = rng.integers(65, 96, size=n).astype(np.float64)

    return pd.DataFrame(
        {
            "patient_id": pd.Series(patient_id, dtype=object),
            "cohort": pd.Series(cohort, dtype=object),
            "sex": sex,
            "n_inpatient_claims": n_inpatient_claims,
            "inpatient_payment": inpatient_payment,
            "inpatient_days": inpatient_days,
            "died": died,
            "death_date": pd.Series(death_date, dtype=object),
            "days_to_death": days_to_death,
            "cost_2008": cost_2008,
            "cost_2009": cost_2009,
            "cost_2010": cost_2010,
            "cost_total": cost_total,
            "chf_2008": chf_2008,
            "chf_ever": chf_ever,
            "state_code": state_code,
            "rare_flag": rare_flag,
            "race": pd.Series(race_vals, dtype=object),
            "bmi": bmi,
            "lab_text": pd.Series(lab_text, dtype=object),
            "age": age,
        }
    )
