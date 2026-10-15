"""
Unit, known-answer, edge-case, and performance tests for
pulsar.analysis.preprocessing_evidence.
"""

from __future__ import annotations

import inspect
import time

import numpy as np
import pandas as pd
import pytest

import pulsar.analysis.preprocessing_evidence as pe_module
from pulsar.analysis.preprocessing_evidence import (
    ColumnEvidence,
    ColumnRelation,
    MaskDependency,
    MissingGroup,
    PreprocessingEvidence,
    measure_preprocessing_evidence,
    neighborhood_overlap,
)
from tests.fixtures.preprocessing_traps import make_preprocessing_traps


def test_traps_known_answers_columns_and_kinds():
    """Verify every column kind and statistical known answer on make_preprocessing_traps."""
    df = make_preprocessing_traps(n=400, seed=0)
    ev = measure_preprocessing_evidence(df)

    assert isinstance(ev, PreprocessingEvidence)
    assert ev.n_rows == 400
    assert ev.n_columns == 21
    assert [c.name for c in ev.columns] == list(df.columns)

    by_name: dict[str, ColumnEvidence] = {c.name: c for c in ev.columns}

    # 1. patient_id: unique text identifier
    pid = by_name["patient_id"]
    assert pid.kind == "text"
    assert pid.n_rows == 400
    assert pid.n_missing == 0
    assert pid.missing_fraction == 0.0
    assert pid.n_unique == 400
    assert pid.unique_ratio == 1.0
    assert pid.is_constant is False
    assert pid.n_zero is None
    assert pid.observed_min is None
    assert pid.observed_max is None
    assert pid.integer_valued is None
    assert pid.nonnegative is None
    assert pid.minority_fraction is None
    assert pid.max_abs_z is None
    assert pid.missing_group is None
    assert pid.mask_dependency is None

    # 2. cohort & sex: constants
    cohort = by_name["cohort"]
    assert cohort.kind == "text"
    assert cohort.n_unique == 1
    assert cohort.is_constant is True
    assert cohort.max_abs_z is None

    sex = by_name["sex"]
    assert sex.kind == "boolean"
    assert sex.n_unique == 1
    assert sex.is_constant is True
    assert sex.n_zero == 0
    assert sex.observed_min == 1.0
    assert sex.observed_max == 1.0
    assert sex.integer_valued is None
    assert sex.nonnegative is True
    assert sex.minority_fraction is None
    assert sex.max_abs_z is None

    # 3. Inpatient block
    claims = by_name["n_inpatient_claims"]
    assert claims.kind == "numeric"
    assert claims.n_missing == 200
    assert claims.missing_fraction == 0.5
    assert claims.n_zero == 0
    assert claims.observed_min is not None and claims.observed_min >= 1.0
    assert claims.integer_valued is True
    assert claims.nonnegative is True
    assert claims.missing_group == "mg2"
    assert claims.mask_dependency is None

    payment = by_name["inpatient_payment"]
    assert payment.kind == "numeric"
    assert payment.n_missing == 200
    assert payment.missing_fraction == 0.5
    assert payment.n_zero == 5
    assert payment.observed_min == 0.0
    assert payment.nonnegative is True
    assert payment.missing_group == "mg2"

    days = by_name["inpatient_days"]
    assert days.kind == "numeric"
    assert days.n_missing == 200
    assert days.n_zero == 0
    assert days.integer_valued is True
    assert days.nonnegative is True
    assert days.missing_group == "mg2"

    # 4. Mortality block
    died = by_name["died"]
    assert died.kind == "boolean"
    assert died.n_missing == 0
    assert died.n_unique == 2
    assert died.n_zero == 360
    assert died.minority_fraction == pytest.approx(0.10)
    assert died.integer_valued is None
    assert died.nonnegative is True

    death_date = by_name["death_date"]
    assert death_date.kind == "datetime_like"
    assert death_date.n_missing == 360
    assert death_date.missing_fraction == pytest.approx(0.90)
    assert death_date.missing_group == "mg1"

    days_to_death = by_name["days_to_death"]
    assert days_to_death.kind == "numeric"
    assert days_to_death.n_missing == 360
    assert days_to_death.missing_fraction == pytest.approx(0.90)
    assert days_to_death.n_zero == 0
    assert days_to_death.integer_valued is True
    assert days_to_death.nonnegative is True
    assert days_to_death.missing_group == "mg1"

    # 5. state_code: discrete integer code 1..52
    state = by_name["state_code"]
    assert state.kind == "numeric"
    assert state.n_missing == 0
    assert state.n_unique == 52
    assert state.integer_valued is True
    assert state.nonnegative is True
    assert state.observed_min == 1.0
    assert state.observed_max == 52.0

    # 6. rare_flag: single 1 out of 400
    rare = by_name["rare_flag"]
    assert rare.kind == "boolean"
    assert rare.n_missing == 0
    assert rare.n_unique == 2
    assert rare.n_zero == 399
    assert rare.minority_fraction == pytest.approx(1.0 / 400.0)
    assert rare.max_abs_z == pytest.approx(float(np.sqrt(399.0)))

    # 7. race, bmi, lab_text, age
    race = by_name["race"]
    assert race.kind == "text"
    assert race.n_missing == 20
    assert race.missing_fraction == pytest.approx(0.05)
    assert race.n_unique == 4
    assert race.missing_group is None
    assert race.mask_dependency is None

    bmi = by_name["bmi"]
    assert bmi.kind == "numeric"
    assert bmi.n_missing == 40
    assert bmi.missing_fraction == pytest.approx(0.10)
    assert bmi.integer_valued is False
    assert bmi.nonnegative is True
    assert bmi.missing_group is None
    assert bmi.mask_dependency is None

    lab = by_name["lab_text"]
    assert lab.kind == "numeric_text"
    assert lab.n_missing == 0
    assert lab.is_constant is False
    assert lab.non_numeric_values == (("N/A", 16),)
    assert lab.n_non_numeric == 16
    assert pid.non_numeric_values is None
    assert pid.n_non_numeric is None

    age = by_name["age"]
    assert age.kind == "numeric"
    assert age.n_missing == 0
    assert age.missing_fraction == 0.0
    assert age.n_zero == 0
    assert age.integer_valued is True
    assert age.nonnegative is True
    assert age.non_numeric_values is None
    assert age.n_non_numeric is None


def test_traps_known_answers_groups_dependencies_and_relations():
    """Verify missing_groups, mask_dependencies, and relations on make_preprocessing_traps."""
    df = make_preprocessing_traps(n=400, seed=0)
    ev = measure_preprocessing_evidence(df)

    # Missing groups ordered by n_missing desc, then first-column position
    assert ev.missing_groups == (
        MissingGroup(
            group_id="mg1",
            columns=("death_date", "days_to_death"),
            n_missing=360,
            zero_free_columns=("days_to_death",),
        ),
        MissingGroup(
            group_id="mg2",
            columns=("n_inpatient_claims", "inpatient_payment", "inpatient_days"),
            n_missing=200,
            zero_free_columns=("n_inpatient_claims", "inpatient_days"),
        ),
    )

    # Mask dependencies: death_date and days_to_death -> died
    expected_deps = (
        MaskDependency(
            column="death_date",
            partner="died",
            relation="missing_when_partner_zero",
            n_other_partners=0,
        ),
        MaskDependency(
            column="days_to_death",
            partner="died",
            relation="missing_when_partner_zero",
            n_other_partners=0,
        ),
    )
    assert ev.mask_dependencies == expected_deps
    by_name = {c.name: c for c in ev.columns}
    assert by_name["death_date"].mask_dependency == expected_deps[0]
    assert by_name["days_to_death"].mask_dependency == expected_deps[1]

    # Sum relation: cost_total = cost_2008 + cost_2009 + cost_2010
    sum_rels = [r for r in ev.relations if r.kind == "sum_of"]
    assert len(sum_rels) == 1
    cost_rel = sum_rels[0]
    assert isinstance(cost_rel, ColumnRelation)
    assert cost_rel.target == "cost_total"
    assert cost_rel.sources == ("cost_2008", "cost_2009", "cost_2010")
    assert cost_rel.rows_used == 400
    assert cost_rel.value <= 1e-9 * (1.0 + float(df["cost_total"].max()))

    # Strong correlation: chf_2008 and chf_ever is the top correlated pair
    corr_rels = [r for r in ev.relations if r.kind == "strong_correlation"]
    assert len(corr_rels) > 0
    top_corr = corr_rels[0]
    assert {top_corr.target, *top_corr.sources} == {"chf_2008", "chf_ever"}
    assert top_corr.value > 0.95
    assert top_corr.rows_used == 400

    # Pairs already in sum_of must not appear in strong_correlation
    for r in corr_rels:
        pair_set = {r.target, *r.sources}
        for src in ("cost_2008", "cost_2009", "cost_2010"):
            assert pair_set != {"cost_total", src}

    # Cost dictionary structure
    assert ev.cost["rows_sampled"] == 400
    assert ev.cost["numeric_columns_considered"] > 0
    assert ev.cost["steps_skipped"] == []
    assert isinstance(ev.cost["elapsed_ms"], float)
    assert ev.cost["elapsed_ms"] >= 0.0


def test_duplicate_and_nonzero_partner_mask_dependency():
    """Verify duplicate detection (NaN-aware) and missing_when_partner_nonzero."""
    df = make_preprocessing_traps(n=100, seed=7)
    df["cost_total_copy"] = df["cost_total"].copy()
    df["bmi_copy"] = df["bmi"].copy()
    # Alive indicator (1 when not died, 0 when died)
    df["alive"] = (~df["died"]).astype(int)
    # Column missing when alive == 1 (i.e. missing_when_partner_nonzero if only alive present)
    df_sub = pd.DataFrame(
        {
            "alive": df["alive"],
            "alive_dup": df["alive"].copy(),
            "postmortem_score": df["days_to_death"],
            "cost_2008": df["cost_2008"],
            "cost_2008_dup": df["cost_2008"].copy(),
            "bmi": df["bmi"],
            "bmi_dup": df["bmi"].copy(),
        }
    )
    ev = measure_preprocessing_evidence(df_sub)

    dup_rels = [r for r in ev.relations if r.kind == "duplicate"]
    dup_map = {r.target: r.sources for r in dup_rels}
    assert dup_map["alive_dup"] == ("alive",)
    assert dup_map["cost_2008_dup"] == ("cost_2008",)
    assert dup_map["bmi_dup"] == ("bmi",)
    for r in dup_rels:
        assert r.value == 0.0

    # Mask dependency with missing_when_partner_nonzero and n_other_partners == 1
    assert ev.mask_dependencies == (
        MaskDependency(
            column="postmortem_score",
            partner="alive",
            relation="missing_when_partner_nonzero",
            n_other_partners=1,
        ),
    )


def test_datetime64_dtype_and_plain_number_not_datetime():
    """Verify native datetime64 is datetime_like and numeric strings are numeric_text."""
    df = pd.DataFrame(
        {
            "native_dt": pd.date_range("2025-01-01", periods=10, freq="D"),
            "iso_str": [f"2025-01-{i:02d}" for i in range(1, 11)],
            "num_str": [str(2000 + i) for i in range(10)],
            "float_str": [f"{i}.5" if i < 8 else "missing" for i in range(10)],
        }
    )
    ev = measure_preprocessing_evidence(df)
    kinds = {c.name: c.kind for c in ev.columns}
    assert kinds["native_dt"] == "datetime_like"
    assert kinds["iso_str"] == "datetime_like"
    assert kinds["num_str"] == "numeric_text"
    assert kinds["float_str"] == "numeric_text"


def test_edge_cases_and_error_messages():
    """Verify mandatory edge cases and error message matching."""
    # 0 rows
    with pytest.raises(ValueError, match="no rows"):
        measure_preprocessing_evidence(pd.DataFrame({"a": [], "b": []}))

    # 0 columns
    with pytest.raises(ValueError, match="no columns"):
        measure_preprocessing_evidence(pd.DataFrame(index=range(5)))

    # 0 rows and 0 columns matches both
    with pytest.raises(ValueError, match="no rows"):
        measure_preprocessing_evidence(pd.DataFrame())
    with pytest.raises(ValueError, match="no columns"):
        measure_preprocessing_evidence(pd.DataFrame())

    # exclude_columns leaving 0 columns
    with pytest.raises(ValueError, match="no columns"):
        measure_preprocessing_evidence(
            pd.DataFrame({"a": [1.0, 2.0]}), exclude_columns=["a", "nonexistent"]
        )

    # 1 row works (no groups / no relations)
    df_one = pd.DataFrame({"a": [1.0], "b": [np.nan], "c": [np.nan], "txt": ["hello"]})
    ev_one = measure_preprocessing_evidence(df_one)
    assert ev_one.n_rows == 1
    assert ev_one.n_columns == 4
    assert ev_one.missing_groups == ()
    assert ev_one.mask_dependencies == ()
    assert ev_one.relations == ()
    assert len(ev_one.cost["steps_skipped"]) == 1

    # All-NaN columns (numeric and object) are never in missing_groups
    df_nan = pd.DataFrame(
        {
            "all_nan_num1": [np.nan, np.nan, np.nan, np.nan],
            "all_nan_num2": [np.nan, np.nan, np.nan, np.nan],
            "all_nan_obj": pd.Series([None, None, None, None], dtype=object),
            "valid": [1.0, 2.0, 3.0, 4.0],
        }
    )
    ev_nan = measure_preprocessing_evidence(df_nan)
    assert ev_nan.missing_groups == ()
    by_name = {c.name: c for c in ev_nan.columns}
    for name in ("all_nan_num1", "all_nan_num2", "all_nan_obj"):
        col = by_name[name]
        assert col.n_missing == 4
        assert col.missing_fraction == 1.0
        assert col.n_unique == 0
        assert col.unique_ratio == 0.0
        assert col.is_constant is True
        assert col.observed_min is None
        assert col.observed_max is None
        assert col.max_abs_z is None
        assert col.missing_group is None

    # exclude_columns ignores unknown names and drops known names
    ev_excl = measure_preprocessing_evidence(
        df_nan, exclude_columns=["all_nan_obj", "unknown_col_xyz"]
    )
    assert [c.name for c in ev_excl.columns] == [
        "all_nan_num1",
        "all_nan_num2",
        "valid",
    ]

    # max_relation_columns budget skip
    df_many = pd.DataFrame({f"c{i}": np.arange(10, dtype=float) + i for i in range(6)})
    ev_skip = measure_preprocessing_evidence(df_many, max_relation_columns=4)
    assert ev_skip.relations == ()
    assert any(s["step"] == "relations" for s in ev_skip.cost["steps_skipped"])


def test_neighborhood_overlap_known_answers_and_errors():
    """Verify neighborhood_overlap known answers and input validation error messages."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 5))

    # Identical matrices -> 1.0
    assert neighborhood_overlap(X, X.copy(), k=10) == pytest.approx(1.0)

    # Positive affine column scaling and shifting leaves z-scored k-NN invariant -> 1.0
    X_affine = X * np.array([2.0, 0.5, 10.0, 1.0, 3.0]) + np.array(
        [100.0, -50.0, 0.0, 7.0, -3.0]
    )
    assert neighborhood_overlap(X, X_affine, k=10) == pytest.approx(1.0)

    # Zero-variance column is mapped to 0 without NaN
    X_const = X.copy()
    X_const[:, 0] = 5.0
    score = neighborhood_overlap(X, X_const, k=8)
    assert 0.0 <= score <= 1.0

    # Validation errors
    with pytest.raises(ValueError, match="same shape"):
        neighborhood_overlap(X, X[:, :4], k=5)

    with pytest.raises(ValueError, match="same shape"):
        neighborhood_overlap(X[:10], X[:12], k=5)

    with pytest.raises(ValueError, match="at least 2 rows"):
        neighborhood_overlap(X[:1], X[:1], k=1)

    with pytest.raises(ValueError, match="k must be"):
        neighborhood_overlap(X, X, k=0)

    with pytest.raises(ValueError, match="k must be"):
        neighborhood_overlap(X, X, k=60)

    with pytest.raises(ValueError, match="k must be"):
        neighborhood_overlap(X, X, k=True)

    X_bad = X.copy()
    X_bad[3, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        neighborhood_overlap(X, X_bad, k=5)

    X_inf = X.copy()
    X_inf[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        neighborhood_overlap(X_inf, X, k=5)


def test_no_policy_words_in_evidence_module():
    """Verify preprocessing_evidence.py contains no policy words ('should', 'recommend')."""
    source = inspect.getsource(pe_module).lower()
    assert "should" not in source
    assert "recommend" not in source


def test_timing_20000_by_300_under_20_seconds():
    """20,000 x 300 synthetic DataFrame completes measure_preprocessing_evidence in < 20s."""
    rng = np.random.default_rng(123)
    n_rows, n_cols = 20_000, 300
    data = rng.standard_normal((n_rows, n_cols))

    # Inject shared missingness group on columns 0..3
    shared_mask = rng.random(n_rows) < 0.3
    data[shared_mask, 0:4] = np.nan

    # Inject exact sum on column 10 = col 7 + col 8 + col 9
    data[:, 10] = data[:, 7] + data[:, 8] + data[:, 9]

    # Inject duplicate on column 12 = column 11
    data[:, 12] = data[:, 11]

    df = pd.DataFrame(data, columns=[f"f_{i}" for i in range(n_cols)])

    t0 = time.perf_counter()
    ev = measure_preprocessing_evidence(df, sample_rows=5000, seed=0)
    elapsed = time.perf_counter() - t0

    assert elapsed < 20.0, f"Expected < 20s, took {elapsed:.2f}s"
    assert ev.n_rows == n_rows
    assert ev.n_columns == n_cols
    assert len(ev.missing_groups) == 1
    assert ev.missing_groups[0].columns == ("f_0", "f_1", "f_2", "f_3")
    assert any(
        r.kind == "duplicate" and r.target == "f_12" and r.sources == ("f_11",)
        for r in ev.relations
    )
    assert any(
        r.kind == "sum_of" and r.target == "f_10" and r.sources == ("f_7", "f_8", "f_9")
        for r in ev.relations
    )


def test_inf_handling_in_stats_and_relations():
    """Item 1: ±inf values do not crash lstsq and are excluded from min/max/max_abs_z."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=50)
    b = rng.normal(size=50)
    t = a + b
    t[0] = np.inf
    df = pd.DataFrame({"a": a, "b": b, "t": t, "c": rng.normal(size=50)})
    ev = measure_preprocessing_evidence(df)
    cols = {c.name: c for c in ev.columns}
    finite_t = t[1:]
    assert cols["t"].observed_min == pytest.approx(float(finite_t.min()))
    assert cols["t"].observed_max == pytest.approx(float(finite_t.max()))
    assert cols["t"].max_abs_z is not None and np.isfinite(cols["t"].max_abs_z)
    assert any(
        r.kind == "sum_of" and r.target == "t" and r.sources == ("a", "b")
        for r in ev.relations
    )

    # Also check ±inf in source column and all-inf column
    a2 = a.copy()
    a2[0] = -np.inf
    df2 = pd.DataFrame(
        {
            "a": a2,
            "b": b,
            "t": a + b,
            "all_inf": np.full(50, np.inf),
        }
    )
    ev2 = measure_preprocessing_evidence(df2)
    cols2 = {c.name: c for c in ev2.columns}
    assert cols2["all_inf"].observed_min is None
    assert cols2["all_inf"].observed_max is None
    assert cols2["all_inf"].max_abs_z is None
    assert any(
        r.kind == "sum_of" and r.target == "t" and r.sources == ("a", "b")
        for r in ev2.relations
    )


def test_duplicate_column_names_raises():
    """Item 6: duplicate column labels raise ValueError listing them."""
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
    with pytest.raises(ValueError, match=r"duplicate column names.*'a'"):
        measure_preprocessing_evidence(df)


def test_str_colliding_column_names_raises():
    """Item 7: distinct column labels that stringify identically raise ValueError."""
    df = pd.DataFrame([[1, 2], [3, 4]], columns=[1, "1"])
    with pytest.raises(ValueError, match=r"column names must be unique after str\(\)"):
        measure_preprocessing_evidence(df)


def test_kind_detection_thousands_commas_and_false_dates():
    """Item 9: thousands commas -> numeric_text; short tokens/numbers -> text; real dates -> datetime_like."""
    df = pd.DataFrame(
        {
            "commas": ["1,234", "2,345", "1,234", "2,345"],
            "months_short": ["Jan", "Feb", "Mar", "Apr"],
            "ordinals": ["1st", "2nd", "3rd", "4th"],
            "months_long": ["May", "June", "May", "June"],
            "rel_words": ["now", "today", "now", "today"],
            "space_nums": ["3 4", "5 6", "3 4", "5 6"],
            "prefixed_date": ["A-2009-01", "B-2009-02", "C-2009-03", "D-2009-04"],
            "pt_ids": ["PT00001", "PT00002", "PT00003", "PT00004"],
            "yyyy_mm": ["2009-04", "2009-05", "2009-06", "2009-07"],
            "iso_dt": [
                "2009-04-04 10:00",
                "2009-04-05 11:00",
                "2009-04-06 12:00",
                "2009-04-07 13:00",
            ],
            "slash_dt": ["04/05/2009", "04/06/2009", "04/07/2009", "04/08/2009"],
        }
    )
    ev = measure_preprocessing_evidence(df)
    kinds = {c.name: c.kind for c in ev.columns}
    assert kinds["commas"] == "numeric_text"
    assert kinds["months_short"] == "text"
    assert kinds["ordinals"] == "text"
    assert kinds["months_long"] == "text"
    assert kinds["rel_words"] == "text"
    assert kinds["space_nums"] == "text"
    assert kinds["prefixed_date"] == "text"
    assert kinds["pt_ids"] == "text"
    assert kinds["yyyy_mm"] == "datetime_like"
    assert kinds["iso_dt"] == "datetime_like"
    assert kinds["slash_dt"] == "datetime_like"


def test_unhashable_list_column_and_timedelta64_column():
    """Item 13: unhashable list values fall back to str (kind='text'); timedelta64 -> kind='numeric' via total_seconds."""
    df = pd.DataFrame(
        {
            "lists": [[1, 2], [3], [1, 2], None],
            "td": pd.to_timedelta([1, 2, None, 4], unit="D"),
        }
    )
    ev = measure_preprocessing_evidence(df)
    cols = {c.name: c for c in ev.columns}

    assert cols["lists"].kind == "text"
    assert cols["lists"].n_missing == 1
    assert cols["lists"].n_unique == 2
    assert cols["lists"].minority_fraction == pytest.approx(1.0 / 3.0)

    assert cols["td"].kind == "numeric"
    assert cols["td"].n_missing == 1
    assert cols["td"].n_unique == 3
    assert cols["td"].observed_min == pytest.approx(86400.0)
    assert cols["td"].observed_max == pytest.approx(345600.0)
    assert cols["td"].integer_valued is True
    assert cols["td"].nonnegative is True


def test_timing_5000_by_1500_with_5pct_nan_under_6_seconds():
    """Item 10: 5,000 x 1,500 numeric DataFrame with 5% NaN completes in < 6s."""
    rng = np.random.default_rng(0)
    n_rows, n_cols = 5000, 1500
    X = rng.normal(size=(n_rows, n_cols))
    X[rng.random(X.shape) < 0.05] = np.nan
    df = pd.DataFrame(X, columns=[f"c{i}" for i in range(n_cols)])

    t0 = time.perf_counter()
    ev = measure_preprocessing_evidence(df)
    elapsed = time.perf_counter() - t0

    assert elapsed < 6.0, f"Expected < 6s, took {elapsed:.2f}s"
    assert ev.n_rows == n_rows
    assert ev.n_columns == n_cols
    assert len(ev.relations) == 20
