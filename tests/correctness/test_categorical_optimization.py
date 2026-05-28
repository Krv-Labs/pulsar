import math
import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Any

from pulsar.mcp.interpreter import (
    _compute_categorical_rows,
    _strongest_neighbor,
    dossier_to_markdown,
    ClusterProfile,
    TopologicalDossier,
    _EPS,
)
from pulsar.mcp.payloads import _evidence_metadata_summary


# Reference naive implementation of the original categorical rows computation to serve as the regression comparison baseline.
def reference_compute_categorical_rows(
    data: pd.DataFrame,
    categorical_cols: list[str],
    clusters: pd.Series,
    adjacency: dict[int, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    global_categorical_stats: dict[str, dict[str, Any]] = {}
    cluster_values = sorted(int(cid) for cid in clusters.unique())
    for column in categorical_cols:
        encoded = data[column].fillna("__MISSING__").astype(str)
        contingency = pd.crosstab(clusters, encoded)
        p_value = 1.0
        association = 0.0
        column_failures: list[str] = []
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            try:
                chi2, p_value, _dof, _expected = stats.chi2_contingency(contingency)
                association = float(chi2 / max(contingency.values.sum(), 1))
            except ValueError as exc:
                column_failures.append(f"chi2_contingency: {exc}")
        global_categorical_stats[column] = {
            "p_value": float(p_value),
            "association": association,
            "failure_reasons": column_failures,
        }

    # Dummy q-value assignment for simplicity
    for column in categorical_cols:
        global_categorical_stats[column]["q_value"] = 1.0

    rows: list[dict[str, Any]] = []
    n_total = len(data)
    for cluster_id in cluster_values:
        cluster_mask = clusters == cluster_id
        rest_mask = clusters != cluster_id
        cluster_size = int(cluster_mask.sum())
        rest_size = int(rest_mask.sum())

        # Original strongest-neighbor logic via O(N) scan (the path the
        # optimized implementation replaces with precomputed value counts).
        neighbor_id = _strongest_neighbor(adjacency, cluster_id)
        neighbor_mask = clusters == neighbor_id if neighbor_id is not None else None
        neighbor_size = int(neighbor_mask.sum()) if neighbor_mask is not None else 0

        for column in categorical_cols:
            global_values = data[column].fillna("__MISSING__").astype(str)
            cluster_values_series = global_values.loc[cluster_mask]
            rest_values_series = global_values.loc[rest_mask]
            cluster_counts = cluster_values_series.value_counts()
            for value, count in cluster_counts.items():
                global_count = int((global_values == value).sum())
                count_rest = int((rest_values_series == value).sum())
                neighbor_prevalence = 0.0
                if neighbor_mask is not None and neighbor_size:
                    neighbor_prevalence = float(
                        ((global_values.loc[neighbor_mask] == value).sum())
                        / neighbor_size
                    )
                prevalence_cluster = (count / cluster_size) if cluster_size else 0.0
                prevalence_rest = (count_rest / rest_size) if rest_size else 0.0
                prevalence_global = (global_count / n_total) if n_total else 0.0
                lift = prevalence_cluster / max(prevalence_global, _EPS)
                log_lift = float(math.log(max(lift, _EPS)))

                p_cv = count / max(n_total, 1)
                p_c = cluster_size / max(n_total, 1)
                p_v = global_count / max(n_total, 1)
                mi_contrib = float(
                    p_cv * math.log(max(p_cv, _EPS) / max(p_c * p_v, _EPS))
                )
                fisher_p = 1.0
                try:
                    fisher_p = float(
                        stats.fisher_exact(
                            [
                                [count, max(cluster_size - count, 0)],
                                [count_rest, max(rest_size - count_rest, 0)],
                            ]
                        )[1]
                    )
                except ValueError:
                    pass

                rows.append(
                    {
                        "column": column,
                        "value": str(value),
                        "cluster_id": int(cluster_id),
                        "count": int(count),
                        "count_cluster": int(count),
                        "count_rest": int(count_rest),
                        "cluster_size": int(cluster_size),
                        "global_count": int(global_count),
                        "prevalence_cluster": float(prevalence_cluster * 100.0),
                        "prevalence_rest": float(prevalence_rest * 100.0),
                        "prevalence_global": float(prevalence_global * 100.0),
                        "lift": float(lift),
                        "log_lift": float(log_lift),
                        "specificity": float(prevalence_cluster - prevalence_rest),
                        "global_recall": float(
                            (count / global_count) * 100.0 if global_count else 0.0
                        ),
                        "neighbor_specificity": float(
                            prevalence_cluster - neighbor_prevalence
                        ),
                        "mi_contrib": float(mi_contrib),
                        "fisher_p": float(fisher_p),
                        "fisher_q": 1.0,
                        "global_p": float(global_categorical_stats[column]["p_value"]),
                        "global_q": 1.0,
                        "aggregate_score": 0.0,
                        "percentile_score": 0.0,
                        "signal_tier": "noise",
                        "in_cluster_prevalence": float(prevalence_cluster * 100.0),
                        "concentration": float(prevalence_cluster * 100.0),
                        "failure_reasons": [],
                    }
                )

    return rows, global_categorical_stats


def test_stage1_precomputation_identity():
    """Verify that optimized precomputed logic is bit-identical to the original logic under force_fisher=True."""
    rng = np.random.default_rng(12345)
    n = 100
    df = pd.DataFrame(
        {
            "color": rng.choice(["red", "blue", "green"], size=n),
            "shape": rng.choice(["circle", "square"], size=n),
        }
    )
    clusters = pd.Series(rng.choice([0, 1, 2], size=n), name="cluster")
    # Non-empty adjacency so the neighbor precompute path is exercised and
    # verified against the reference O(N) scan (not just the no-neighbor case).
    adjacency = {
        0: [{"cluster_id": 1, "weight": 2.0}],
        1: [{"cluster_id": 2, "weight": 1.5}],
        2: [{"cluster_id": 0, "weight": 1.0}],
    }

    # Run reference (naive/original)
    ref_rows, _ = reference_compute_categorical_rows(
        df, ["color", "shape"], clusters, adjacency
    )

    # Run optimized under force_fisher=True
    opt_rows, _, _ = _compute_categorical_rows(
        df, ["color", "shape"], clusters, adjacency, force_fisher=True
    )

    # Sort both results to guarantee matching alignment
    ref_sorted = sorted(
        ref_rows, key=lambda x: (x["cluster_id"], x["column"], x["value"])
    )
    opt_sorted = sorted(
        opt_rows, key=lambda x: (x["cluster_id"], x["column"], x["value"])
    )

    assert len(ref_sorted) == len(opt_sorted)

    for r, o in zip(ref_sorted, opt_sorted):
        assert r["column"] == o["column"]
        assert r["value"] == o["value"]
        assert r["cluster_id"] == o["cluster_id"]
        assert r["count"] == o["count"]
        assert r["count_rest"] == o["count_rest"]
        assert r["global_count"] == o["global_count"]
        assert np.isclose(r["prevalence_cluster"], o["prevalence_cluster"], rtol=1e-9)
        assert np.isclose(r["prevalence_rest"], o["prevalence_rest"], rtol=1e-9)
        assert np.isclose(r["prevalence_global"], o["prevalence_global"], rtol=1e-9)
        assert np.isclose(r["lift"], o["lift"], rtol=1e-9)
        assert np.isclose(r["log_lift"], o["log_lift"], rtol=1e-9)
        assert np.isclose(r["mi_contrib"], o["mi_contrib"], rtol=1e-9)
        assert np.isclose(r["fisher_p"], o["fisher_p"], rtol=1e-9)
        assert np.isclose(
            r["neighbor_specificity"], o["neighbor_specificity"], rtol=1e-9, atol=1e-12
        )


def test_stage2_cardinality_gating(caplog):
    """Verify high-cardinality noise is gated out while preserving low-cardinality features."""
    rng = np.random.default_rng(99)
    n = 200
    df = pd.DataFrame(
        {
            "unique_id": [
                f"id_{i}" for i in range(n)
            ],  # Cardinality = 200 (Ratio = 1.0 > 0.05)
            "gender": rng.choice(["M", "F"], size=n),  # Cardinality = 2 (Ratio = 0.01)
        }
    )
    clusters = pd.Series(rng.choice([0, 1], size=n), name="cluster")
    adjacency = {0: [], 1: []}

    with caplog.at_level(logging.WARNING):
        rows, stats_dict, gated_cols = _compute_categorical_rows(
            df,
            ["unique_id", "gender"],
            clusters,
            adjacency,
            max_cardinality_ratio=0.05,
            min_cardinality_floor=10,
        )

    # unique_id should be gated because 200 > max(10, 0.05 * 200) = 10
    # gender should NOT be gated because 2 <= 10
    gated_names = [g["column"] for g in gated_cols]
    assert "unique_id" in gated_names
    assert "gender" not in gated_names

    # Verify logging warning occurred for gated columns
    assert any(
        "gated" in record.message and "unique_id" in record.message
        for record in caplog.records
    )

    # Verify that only preserved column rows are present
    assert all(row["column"] == "gender" for row in rows)


def test_gated_columns_are_surfaced_to_agent():
    """Gating must be visible in the agent-facing payload, not only in logs."""
    metadata = {
        "categorical_columns_screened": 1,
        "categorical_columns_gated": [
            {"column": "patient_id", "cardinality": 20000, "reason": "high_cardinality"}
        ],
        "stats_failures": {},
    }
    summary = _evidence_metadata_summary(metadata)
    assert summary["categorical_columns_screened"] == 1
    assert summary["categorical_columns_gated"][0]["column"] == "patient_id"


def test_stage3_chisquared_fallback():
    """Verify conditional fallback based on expected cell values and test_method registration."""
    # Scenario A: Highly balanced and large cell counts (expected cell values >= 5) -> Chi-Squared
    df_large = pd.DataFrame({"feature": (["A"] * 100 + ["B"] * 100) * 2})
    # Total n = 400.
    # Group into clusters
    clusters_large = pd.Series(([0] * 200) + ([1] * 200), name="cluster")
    adjacency = {0: [], 1: []}

    rows, _, _ = _compute_categorical_rows(
        df_large,
        ["feature"],
        clusters_large,
        adjacency,
        force_fisher=False,
    )

    # Expected count per cell:
    # Cluster 0 size = 200. Rest size = 200.
    # Global count of "A" = 200. Rest count of "A" = 100.
    # E00 = 200 * 200 / 400 = 100 >= 5
    # All expected cells are 100, which is >= 5, so it must use chi2!
    assert len(rows) > 0
    for row in rows:
        assert row["test_method"] == "chi2"

    # Scenario B: Low cell counts (expected cell values < 5) -> Fallback to Fisher
    df_small = pd.DataFrame({"feature": ["A"] * 2 + ["B"] * 98})
    clusters_small = pd.Series(([0] * 50) + ([1] * 50), name="cluster")

    rows_small, _, _ = _compute_categorical_rows(
        df_small,
        ["feature"],
        clusters_small,
        adjacency,
        force_fisher=False,
    )

    # Expected counts for "A":
    # Global count of "A" = 2.
    # E00 = 50 * 2 / 100 = 1.0 < 5.
    # So "A" value rows should fall back to fisher!
    has_fisher = False
    for row in rows_small:
        if row["value"] == "A":
            assert row["test_method"] == "fisher"
            has_fisher = True
    assert has_fisher


def test_stage3_mixed_methods_share_one_fdr_pool():
    """A single run that yields both chi2 and fisher rows must still produce
    valid, finite q-values pooled across the mixed p-values."""
    # "common" has large balanced cells (chi2); "rare" has tiny cells (fisher).
    feature = (["common"] * 180 + ["rare"] * 2 + ["filler"] * 18) * 2
    df = pd.DataFrame({"feature": feature})
    n = len(df)
    clusters = pd.Series(([0] * (n // 2)) + ([1] * (n - n // 2)), name="cluster")
    adjacency = {0: [], 1: []}

    rows, _, _ = _compute_categorical_rows(
        df, ["feature"], clusters, adjacency, force_fisher=False
    )

    methods = {row["test_method"] for row in rows}
    assert "chi2" in methods and "fisher" in methods, (
        f"expected both test methods in one run, got {methods}"
    )
    # Every row, regardless of which test produced fisher_p, gets a finite q.
    for row in rows:
        assert 0.0 <= row["fisher_q"] <= 1.0
        assert math.isfinite(row["fisher_p"])


def _feature(idx: int) -> dict:
    return {
        "column": f"f{idx}",
        "sparkline": "",
        "mean": 0.0,
        "global_mean": 0.0,
        "z_score": 0.0,
        "homogeneity": 1.0,
    }


def test_dossier_warns_when_oversized():
    """detail='full' renders are non-destructive but must warn past ~500 rows."""
    # Two clusters, 300 numeric features each -> 600 total feature-cluster rows.
    clusters = [
        ClusterProfile(
            cluster_id=cid,
            size=100,
            size_pct=50.0,
            numeric_features=[_feature(i) for i in range(300)],
        )
        for cid in (0, 1)
    ]
    big = TopologicalDossier(
        n_total=200, n_clusters=2, clusters=clusters, global_stats={}
    )
    md = dossier_to_markdown(big)
    assert "[!WARNING]" in md
    assert "get_cluster_profile" in md and "get_feature_signal" in md

    # A small dossier stays clean (no warning, no nag).
    small = TopologicalDossier(
        n_total=10,
        n_clusters=1,
        clusters=[
            ClusterProfile(
                cluster_id=0,
                size=10,
                size_pct=100.0,
                numeric_features=[_feature(0)],
            )
        ],
        global_stats={},
    )
    assert "[!WARNING]" not in dossier_to_markdown(small)
