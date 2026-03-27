"""
Pulsar demo — Longitudinal ICU Vital Signs (True Temporal Approach)
====================================================================

This demo showcases the full TemporalCosmicGraph for longitudinal time-series data,
where the same patients are observed across multiple time steps.

Dataset: Simulated ICU vital sign trajectories
- ~500 patients observed over 72 hours (hourly measurements)
- 8 vital signs per time step
- 5 patient archetypes with distinct temporal patterns

Approach:
1. Generate synthetic ICU trajectories with known temporal patterns
2. Build TemporalCosmicGraph with 3D tensor W[i, j, t]
3. Compare different aggregation strategies (persistence, trend, volatility)
4. Show how different aggregations reveal different patient groupings

This demonstrates the "true longitudinal" approach — essential when:
- Temporal trajectory patterns are clinically meaningful
- Patients with similar current state may have different trajectories
- Early warning / trend detection is important

Usage:
    uv run python demos/physionet.py
    uv run python demos/physionet.py --n-patients 200 --n-hours 48
"""

from __future__ import annotations

import argparse
import time

import networkx as nx
import numpy as np
import pandas as pd

from pulsar.config import load_config
from pulsar.temporal import TemporalCosmicGraph

# ---------------------------------------------------------------------------
# Simulated ICU Data Generator
# ---------------------------------------------------------------------------

VITAL_SIGNS = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "map",
    "respiratory_rate",
    "spo2",
    "temperature",
    "gcs",
]

NORMAL_RANGES = {
    "heart_rate": (60, 100),
    "systolic_bp": (90, 140),
    "diastolic_bp": (60, 90),
    "map": (70, 105),
    "respiratory_rate": (12, 20),
    "spo2": (95, 100),
    "temperature": (36.5, 37.5),
    "gcs": (14, 15),
}

ABNORMAL_RANGES = {
    "heart_rate": (100, 140),
    "systolic_bp": (70, 90),
    "diastolic_bp": (40, 60),
    "map": (50, 70),
    "respiratory_rate": (22, 35),
    "spo2": (85, 94),
    "temperature": (38.0, 39.5),
    "gcs": (8, 12),
}


def generate_patient_trajectory(
    archetype: str,
    n_hours: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate vital sign trajectory for a single patient.

    Parameters
    ----------
    archetype : str
        One of: "stable", "improving", "deteriorating", "fluctuating", "sudden_event"
    n_hours : int
        Number of hourly observations
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    np.ndarray
        Shape (n_hours, 8) — vital signs at each hour
    """
    trajectory = np.zeros((n_hours, len(VITAL_SIGNS)))

    for v_idx, vital in enumerate(VITAL_SIGNS):
        normal_low, normal_high = NORMAL_RANGES[vital]
        abnormal_low, abnormal_high = ABNORMAL_RANGES[vital]

        normal_mean = (normal_low + normal_high) / 2
        normal_std = (normal_high - normal_low) / 4

        abnormal_mean = (abnormal_low + abnormal_high) / 2

        if archetype == "stable":
            # Stay in normal range throughout
            base = rng.uniform(normal_low, normal_high)
            noise = rng.normal(0, normal_std * 0.2, n_hours)
            trajectory[:, v_idx] = base + noise

        elif archetype == "improving":
            # Start abnormal, trend toward normal
            start_val = rng.uniform(abnormal_low, abnormal_high)
            end_val = rng.uniform(normal_low, normal_high)

            # Exponential decay toward normal
            decay_rate = rng.uniform(0.03, 0.08)
            t = np.arange(n_hours)
            trend = start_val + (end_val - start_val) * (1 - np.exp(-decay_rate * t))
            noise = rng.normal(0, normal_std * 0.3, n_hours)
            trajectory[:, v_idx] = trend + noise

        elif archetype == "deteriorating":
            # Start normal, trend toward abnormal
            start_val = rng.uniform(normal_low, normal_high)
            end_val = rng.uniform(abnormal_low, abnormal_high)

            # Gradual linear + accelerating deterioration
            t = np.linspace(0, 1, n_hours)
            trend = start_val + (end_val - start_val) * (t**1.5)
            noise = rng.normal(0, normal_std * 0.3, n_hours)
            trajectory[:, v_idx] = trend + noise

        elif archetype == "fluctuating":
            # High volatility, swinging between normal and abnormal
            base = rng.uniform(normal_low, normal_high)
            amplitude = (abnormal_mean - normal_mean) * 0.8
            frequency = rng.uniform(0.05, 0.15)
            phase = rng.uniform(0, 2 * np.pi)

            t = np.arange(n_hours)
            oscillation = amplitude * np.sin(2 * np.pi * frequency * t + phase)
            noise = rng.normal(0, normal_std * 0.5, n_hours)
            trajectory[:, v_idx] = base + oscillation + noise

        elif archetype == "sudden_event":
            # Stable then acute change (cardiac arrest, hemorrhage, etc.)
            event_hour = rng.integers(n_hours // 3, 2 * n_hours // 3)

            # Before event: stable normal
            pre_event = rng.uniform(normal_low, normal_high)
            trajectory[:event_hour, v_idx] = pre_event + rng.normal(
                0, normal_std * 0.2, event_hour
            )

            # After event: acute change then partial recovery or continued abnormal
            post_event_start = rng.uniform(abnormal_low, abnormal_high)
            # Recovery target: between abnormal and normal
            recovery_low = min(abnormal_mean, normal_mean)
            recovery_high = max(abnormal_mean, normal_mean)
            post_event_end = rng.uniform(recovery_low, recovery_high)

            n_post = n_hours - event_hour
            t = np.linspace(0, 1, n_post)
            recovery = post_event_start + (post_event_end - post_event_start) * (
                1 - np.exp(-2 * t)
            )
            trajectory[event_hour:, v_idx] = recovery + rng.normal(
                0, normal_std * 0.4, n_post
            )

        # Clip to physiological bounds
        if vital == "spo2":
            trajectory[:, v_idx] = np.clip(trajectory[:, v_idx], 70, 100)
        elif vital == "gcs":
            trajectory[:, v_idx] = np.clip(trajectory[:, v_idx], 3, 15)
        elif vital == "temperature":
            trajectory[:, v_idx] = np.clip(trajectory[:, v_idx], 34, 42)
        else:
            trajectory[:, v_idx] = np.clip(trajectory[:, v_idx], 0, 300)

    return trajectory


def generate_icu_dataset(
    n_patients: int = 500,
    n_hours: int = 72,
    seed: int = 42,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    """
    Generate synthetic ICU vital sign dataset.

    Returns
    -------
    snapshots : list[np.ndarray]
        List of T arrays, each of shape (n_patients, n_vitals)
    labels : pd.DataFrame
        Patient metadata including archetype
    """
    rng = np.random.default_rng(seed)

    # Archetype distribution
    archetypes = {
        "stable": 0.40,
        "improving": 0.20,
        "deteriorating": 0.15,
        "fluctuating": 0.15,
        "sudden_event": 0.10,
    }

    archetype_names = list(archetypes.keys())
    archetype_probs = list(archetypes.values())

    patient_archetypes = rng.choice(archetype_names, size=n_patients, p=archetype_probs)

    # Generate all trajectories: shape (n_patients, n_hours, n_vitals)
    all_trajectories = np.zeros((n_patients, n_hours, len(VITAL_SIGNS)))

    for i in range(n_patients):
        all_trajectories[i] = generate_patient_trajectory(
            patient_archetypes[i], n_hours, rng
        )

    # Convert to list of snapshots (one per time step)
    snapshots = [all_trajectories[:, t, :].copy() for t in range(n_hours)]

    # Create labels dataframe
    labels = pd.DataFrame(
        {
            "patient_id": np.arange(n_patients),
            "archetype": patient_archetypes,
            "age": rng.normal(65, 12, n_patients).clip(18, 95).astype(int),
            "sex": rng.choice(["M", "F"], n_patients),
            "admission_type": rng.choice(
                ["medical", "surgical", "trauma"], n_patients, p=[0.5, 0.35, 0.15]
            ),
        }
    )

    return snapshots, labels


# ---------------------------------------------------------------------------
# Main Demo
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Longitudinal ICU demo - true temporal approach"
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=500,
        help="Number of patients (default: 500)",
    )
    parser.add_argument(
        "--n-hours",
        type=int,
        default=72,
        help="Number of hours to simulate (default: 72)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Longitudinal ICU Demo — True Temporal Approach")
    print("=" * 70)
    print()

    # Generate data
    print("[data] Generating ICU trajectories...")
    print(
        f"       {args.n_patients} patients × {args.n_hours} hours × {len(VITAL_SIGNS)} vitals"
    )
    t0 = time.perf_counter()
    snapshots, labels = generate_icu_dataset(
        n_patients=args.n_patients,
        n_hours=args.n_hours,
        seed=args.seed,
    )
    print(f"[data] Generated in {time.perf_counter() - t0:.2f}s")
    print()

    print("[data] Archetype distribution:")
    for archetype, count in labels["archetype"].value_counts().items():
        print(f"       {archetype}: {count} ({100 * count / len(labels):.1f}%)")
    print()

    # Build config
    config = load_config(
        {
            "run": {"name": "longitudinal_icu_demo"},
            "preprocessing": {"drop_columns": [], "impute": {}},
            "sweep": {
                "pca": {
                    "dimensions": {"values": [3, 5, 7]},
                    "seed": {"values": [42, 7, 13]},
                },
                "ball_mapper": {
                    "epsilon": {"range": {"min": 0.5, "max": 2.5, "steps": 15}},
                },
            },
            "cosmic_graph": {"threshold": 0.0},
            "output": {"n_reps": 3},
        }
    )

    n_maps = (
        len(config.pca.dimensions)
        * len(config.pca.seeds)
        * len(config.ball_mapper.epsilons)
    )
    print(
        f"[grid] {len(config.pca.dimensions)} dims × {len(config.pca.seeds)} seeds × {len(config.ball_mapper.epsilons)} epsilons = {n_maps} ball maps per time step"
    )
    print(
        f"[grid] Total: {n_maps} × {args.n_hours} time steps = {n_maps * args.n_hours} ball maps"
    )
    print()

    # Build TemporalCosmicGraph
    print("[temporal] Building TemporalCosmicGraph...")
    t0 = time.perf_counter()
    tcg = TemporalCosmicGraph.from_snapshots(snapshots, config, threshold=0.0)
    t_build = time.perf_counter() - t0
    print(f"[temporal] Built in {t_build:.2f}s")
    print(f"[temporal] Tensor shape: {tcg.shape} (n × n × T)")
    print()

    # Compute aggregations
    print("[aggregation] Computing temporal aggregations...")
    t0 = time.perf_counter()

    agg_persistence = tcg.persistence_graph(threshold=0.1)
    agg_mean = tcg.mean_graph()
    agg_recency = tcg.recency_graph(decay=0.9)
    agg_volatility = tcg.volatility_graph()
    agg_trend = tcg.trend_graph()
    agg_change = tcg.change_point_graph()

    t_agg = time.perf_counter() - t0
    print(f"[aggregation] Computed 6 aggregations in {t_agg:.3f}s")
    print()

    # Summary statistics for each aggregation
    print("Aggregation Statistics:")
    print("-" * 70)
    print(f"{'Aggregation':<20} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print("-" * 70)

    for name, agg in [
        ("Persistence", agg_persistence),
        ("Mean", agg_mean),
        ("Recency (λ=0.9)", agg_recency),
        ("Volatility", agg_volatility),
        ("Trend", agg_trend),
        ("Change-point", agg_change),
    ]:
        # Exclude diagonal
        mask = ~np.eye(agg.shape[0], dtype=bool)
        vals = agg[mask]
        print(
            f"{name:<20} {vals.min():>10.4f} {vals.max():>10.4f} "
            f"{vals.mean():>10.4f} {vals.std():>10.4f}"
        )
    print("-" * 70)
    print()

    # Convert to NetworkX graphs and analyze clusters
    print("Cluster Analysis by Aggregation Strategy:")
    print("-" * 70)

    def analyze_graph(G: nx.Graph, name: str, labels_df: pd.DataFrame) -> None:
        """Analyze connected components and archetype distribution."""
        components = list(nx.connected_components(G))
        n_clusters = len(components)

        # Assign cluster IDs
        cluster_labels = {}
        for cluster_id, component in enumerate(components):
            for node in component:
                cluster_labels[node] = cluster_id

        # Cross-tabulate
        cluster_df = pd.DataFrame(
            {
                "patient_id": list(cluster_labels.keys()),
                "cluster": list(cluster_labels.values()),
            }
        )
        cluster_df = cluster_df.merge(
            labels_df[["patient_id", "archetype"]], on="patient_id"
        )

        print(f"\n{name}:")
        print(f"  Edges: {G.number_of_edges()}, Components: {n_clusters}")

        if n_clusters > 1 and n_clusters <= 20:
            # Show archetype distribution in top clusters
            cluster_sizes = (
                cluster_df.groupby("cluster").size().sort_values(ascending=False)
            )
            print("  Top clusters by archetype composition:")

            for cluster_id in cluster_sizes.head(5).index:
                members = cluster_df[cluster_df["cluster"] == cluster_id]
                size = len(members)
                arch_dist = members["archetype"].value_counts()
                dominant = arch_dist.head(2)
                arch_str = ", ".join([f"{a}:{c}" for a, c in dominant.items()])
                print(f"    Cluster {cluster_id}: {size} patients — {arch_str}")

    # Analyze each aggregation
    for name, agg, threshold in [
        ("Persistence (τ=0.1)", agg_persistence, 0.3),
        ("Mean", agg_mean, 0.2),
        ("Recency (λ=0.9)", agg_recency, 0.2),
        ("Volatility", agg_volatility, 0.02),
        ("Trend (positive)", agg_trend, 0.005),
    ]:
        # Build graph from aggregation matrix
        G = nx.Graph()
        G.add_nodes_from(range(args.n_patients))
        n = agg.shape[0]

        for i in range(n):
            for j in range(i + 1, n):
                w = agg[i, j]
                if name.startswith("Volatility"):
                    # High volatility = similar unstable trajectory
                    if w > threshold:
                        G.add_edge(i, j, weight=float(w))
                elif name.startswith("Trend"):
                    # Both positive trend = converging patients
                    if w > threshold:
                        G.add_edge(i, j, weight=float(w))
                else:
                    if w > threshold:
                        G.add_edge(i, j, weight=float(w))

        analyze_graph(G, name, labels)

    print()
    print("-" * 70)
    print()

    # Key insights
    print("=" * 70)
    print("Key Insights from Temporal Analysis")
    print("=" * 70)
    print()
    print("1. PERSISTENCE graph clusters patients with STABLE similarity over time.")
    print("   → 'Stable' archetype patients should cluster together")
    print("   → Useful for identifying robust phenotype subgroups")
    print()
    print("2. VOLATILITY graph clusters patients with UNSTABLE relationships.")
    print("   → 'Fluctuating' patients should appear together")
    print("   → Useful for identifying patients needing closer monitoring")
    print()
    print("3. TREND graph clusters patients whose similarity is CHANGING.")
    print("   → 'Improving' and 'Deteriorating' patients separate here")
    print("   → Positive trend = converging trajectories")
    print("   → Useful for trajectory-based patient matching")
    print()
    print("4. RECENCY graph emphasizes CURRENT similarity over history.")
    print("   → Useful for real-time clinical decision support")
    print("   → Patients may cluster differently than in persistence graph")
    print()
    print("5. CHANGE-POINT graph highlights SUDDEN state transitions.")
    print("   → 'Sudden event' patients should be detectable")
    print("   → Useful for event detection and anomaly identification")
    print()
    print("=" * 70)
    print("Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
