"""
Numerical correctness and reference parity tests for
pulsar.analysis.preprocessing_evidence.

Compares vectorized implementations against transparent brute-force
NumPy/SciPy reference implementations.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import pearsonr

from pulsar.analysis.preprocessing_evidence import (
    measure_preprocessing_evidence,
    neighborhood_overlap,
)
from tests.fixtures.preprocessing_traps import make_preprocessing_traps


def _reference_neighborhood_overlap(X_a: np.ndarray, X_b: np.ndarray, k: int) -> float:
    """Brute-force pure-NumPy reference implementation of neighborhood_overlap."""

    def zscore_ref(X: np.ndarray) -> np.ndarray:
        out = np.zeros_like(X, dtype=np.float64)
        for j in range(X.shape[1]):
            col = X[:, j].astype(np.float64)
            mu = float(np.mean(col))
            sd = float(np.std(col, ddof=0))
            if sd > 0.0:
                out[:, j] = (col - mu) / sd
        return out

    def brute_knn_sets(Z: np.ndarray) -> list[set[int]]:
        n = Z.shape[0]
        diff = Z[:, None, :] - Z[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        nbr_sets: list[set[int]] = []
        for i in range(n):
            order = np.argsort(dist[i], kind="mergesort")[:k]
            nbr_sets.append(set(int(x) for x in order))
        return nbr_sets

    Z_a = zscore_ref(X_a)
    Z_b = zscore_ref(X_b)
    sets_a = brute_knn_sets(Z_a)
    sets_b = brute_knn_sets(Z_b)

    jaccards = [len(sa & sb) / len(sa | sb) for sa, sb in zip(sets_a, sets_b)]
    return float(np.mean(jaccards))


def test_neighborhood_overlap_matches_brute_force_reference():
    """Verify neighborhood_overlap matches brute-force distance-matrix + set-Jaccard reference."""
    rng = np.random.default_rng(2026)
    for k in (1, 5, 10, 15):
        X_a = rng.standard_normal((80, 7))
        X_b = X_a + 0.35 * rng.standard_normal((80, 7))
        # Include a constant column in X_b to verify zero-variance handling
        X_b[:, 2] = -4.0

        actual = neighborhood_overlap(X_a, X_b, k=k)
        expected = _reference_neighborhood_overlap(X_a, X_b, k=k)
        assert actual == pytest.approx(expected, abs=1e-12)


def test_column_statistics_match_naive_reference():
    """Verify per-column numerical statistics match naive per-column numpy/pandas reference."""
    df = make_preprocessing_traps(n=240, seed=19)
    ev = measure_preprocessing_evidence(df)

    for col_ev in ev.columns:
        s = df[col_ev.name]
        na_mask = s.isna().to_numpy(dtype=bool)
        n_rows = len(s)
        n_missing = int(na_mask.sum())
        n_obs = n_rows - n_missing

        assert col_ev.n_rows == n_rows
        assert col_ev.n_missing == n_missing
        assert col_ev.missing_fraction == pytest.approx(n_missing / n_rows)

        n_unique_ref = int(s.dropna().nunique())
        assert col_ev.n_unique == n_unique_ref
        assert col_ev.unique_ratio == pytest.approx(n_unique_ref / max(n_obs, 1))
        assert col_ev.is_constant == (n_unique_ref <= 1)

        if n_unique_ref == 2:
            vc = s.dropna().value_counts()
            assert col_ev.minority_fraction == pytest.approx(float(vc.min() / n_obs))
        else:
            assert col_ev.minority_fraction is None

        if col_ev.kind in ("numeric", "boolean") and n_obs > 0:
            obs = s.dropna().to_numpy(dtype=np.float64)
            assert col_ev.n_zero == int(np.sum(obs == 0.0))
            assert col_ev.observed_min == pytest.approx(float(np.min(obs)))
            assert col_ev.observed_max == pytest.approx(float(np.max(obs)))
            assert col_ev.nonnegative is bool(np.min(obs) >= 0.0)
            sd = float(np.std(obs, ddof=0))
            if sd > 0.0:
                expected_z = float(np.max(np.abs(obs - np.mean(obs))) / sd)
                assert col_ev.max_abs_z == pytest.approx(expected_z, rel=1e-12)
            else:
                assert col_ev.max_abs_z is None


def test_packed_masks_and_correlations_match_brute_force_reference():
    """Verify packed-bit mask groups, mask dependencies, and BLAS correlations match brute force."""
    df = make_preprocessing_traps(n=300, seed=99)
    ev = measure_preprocessing_evidence(df, sample_rows=5000, max_correlation_pairs=50)

    # 1. Brute-force missing groups (O(p^2 * n) pairwise comparison)
    cols = list(df.columns)
    na_masks = {c: df[c].isna().to_numpy(dtype=bool) for c in cols}
    partial_cols = [c for c in cols if 0 < int(na_masks[c].sum()) < len(df)]

    visited: set[str] = set()
    brute_groups: list[tuple[str, ...]] = []
    for c in partial_cols:
        if c in visited:
            continue
        grp = tuple(
            other
            for other in partial_cols
            if np.array_equal(na_masks[c], na_masks[other])
        )
        for member in grp:
            visited.add(member)
        if len(grp) >= 2:
            brute_groups.append(grp)

    brute_groups.sort(key=lambda g: (-int(na_masks[g[0]].sum()), cols.index(g[0])))
    assert [g.columns for g in ev.missing_groups] == brute_groups

    # 2. Brute-force mask dependencies
    by_name = {c.name: c for c in ev.columns}
    full_num_cols = [
        c
        for c in cols
        if int(na_masks[c].sum()) == 0 and by_name[c].kind in ("numeric", "boolean")
    ]
    for p_col in partial_cols:
        z_partners = [
            fc
            for fc in full_num_cols
            if np.array_equal(na_masks[p_col], df[fc].to_numpy(dtype=np.float64) == 0.0)
        ]
        nz_partners = [
            fc
            for fc in full_num_cols
            if np.array_equal(na_masks[p_col], df[fc].to_numpy(dtype=np.float64) != 0.0)
        ]
        dep = by_name[p_col].mask_dependency
        if not z_partners and not nz_partners:
            assert dep is None
        elif z_partners:
            assert dep is not None
            assert dep.partner == z_partners[0]
            assert dep.relation == "missing_when_partner_zero"
            assert dep.n_other_partners == len(z_partners) - 1
        else:
            assert dep is not None
            assert dep.partner == nz_partners[0]
            assert dep.relation == "missing_when_partner_nonzero"
            assert dep.n_other_partners == len(nz_partners) - 1

    # 3. Brute-force scipy.stats.pearsonr parity for all reported strong_correlation pairs
    for rel in ev.relations:
        if rel.kind == "strong_correlation":
            c1 = df[rel.sources[0]].to_numpy(dtype=np.float64)
            c2 = df[rel.target].to_numpy(dtype=np.float64)
            valid = (~np.isnan(c1)) & (~np.isnan(c2))
            assert rel.rows_used == int(valid.sum())
            r_ref, _ = pearsonr(c1[valid], c2[valid])
            assert rel.value == pytest.approx(abs(float(r_ref)), rel=1e-11, abs=1e-12)
        elif rel.kind == "sum_of":
            target_arr = df[rel.target].to_numpy(dtype=np.float64)
            src_arrs = np.column_stack(
                [df[s].to_numpy(dtype=np.float64) for s in rel.sources]
            )
            valid = (~np.isnan(target_arr)) & np.all(~np.isnan(src_arrs), axis=1)
            assert rel.rows_used == int(valid.sum())
            res_ref = float(
                np.max(np.abs(target_arr[valid] - np.sum(src_arrs[valid], axis=1)))
            )
            assert rel.value == pytest.approx(res_ref, abs=1e-12)
