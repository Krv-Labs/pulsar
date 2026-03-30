"""
Quick experiment runner — try different param configs and compare results.

Usage:
    uv run python helpers/sweep_experiment.py

Edit the EXPERIMENTS dict below to try different configurations.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from pulsar import ThemaRS, cosmic_clusters
from pulsar.config import load_config

from diagnose import load_data, diagnose_distances


DATA_DIR = Path("data")

# ============================================================
# EDIT THESE to try different configurations
# ============================================================
EXPERIMENTS = {
    # Informed by diagnostics: post-PCA distances range from ~0.06 (5d) to ~0.43 (50d).
    # We need epsilons in the P10-P90 range of each PCA dim's distance distribution.

    "mid_dim_tight_eps": {
        # Focus on mid-range PCA dims where structure is clearest
        # Epsilon spans P10-P90 for 10d-50d range
        "sweep": {
            "pca": {
                "dimensions": {"values": [10, 20, 30, 50]},
                "seed": {"values": [42, 7, 13, 99, 123]},
            },
            "ball_mapper": {
                "epsilon": {"range": {"min": 0.10, "max": 0.35, "steps": 20}},
            },
        },
        "cosmic_graph": {"threshold": "auto"},
    },
    "high_dim_wide_eps": {
        # Higher PCA dims preserve more structure, wider epsilon
        "sweep": {
            "pca": {
                "dimensions": {"values": [20, 50, 100, 150]},
                "seed": {"values": [42, 7, 13, 99, 123]},
            },
            "ball_mapper": {
                "epsilon": {"range": {"min": 0.15, "max": 0.50, "steps": 20}},
            },
        },
        "cosmic_graph": {"threshold": "auto"},
    },
    "balanced": {
        # Mix of dims, epsilon centered on median distances
        "sweep": {
            "pca": {
                "dimensions": {"values": [10, 20, 50]},
                "seed": {"values": [42, 7, 13, 99, 123, 456, 789]},
            },
            "ball_mapper": {
                "epsilon": {"range": {"min": 0.08, "max": 0.40, "steps": 25}},
            },
        },
        "cosmic_graph": {"threshold": "auto"},
    },
    "aggressive": {
        # Lots of diversity: many dims, many seeds, fine epsilon grid
        "sweep": {
            "pca": {
                "dimensions": {"values": [5, 10, 20, 50, 100]},
                "seed": {"values": [42, 7, 13, 99, 123, 456, 789, 1000]},
            },
            "ball_mapper": {
                "epsilon": {"range": {"min": 0.05, "max": 0.45, "steps": 30}},
            },
        },
        "cosmic_graph": {"threshold": "auto"},
    },
}


def run_experiment(name: str, config_dict: dict, embeddings_sub: np.ndarray):
    """Run one experiment and return summary stats."""
    config_dict.setdefault("run", {"name": name})

    cfg = load_config(config_dict)
    n_maps = len(cfg.pca.dimensions) * len(cfg.pca.seeds) * len(cfg.ball_mapper.epsilons)

    df_emb = pd.DataFrame(embeddings_sub)

    t0 = time.perf_counter()
    model = ThemaRS(cfg).fit(data=df_emb)
    elapsed = time.perf_counter() - t0

    G = model.cosmic_graph
    W = model.weighted_adjacency
    n = W.shape[0]
    upper = W[np.triu_indices(n, k=1)]
    nonzero = upper[upper > 0]

    components = list(nx.connected_components(G))
    comp_sizes = sorted([len(c) for c in components], reverse=True)

    # Ball map diversity
    nodes_arr = np.array([bm.n_nodes() for bm in model.ball_maps])

    result = {
        "name": name,
        "n_maps": n_maps,
        "time_s": elapsed,
        "n_edges": G.number_of_edges(),
        "density": G.number_of_edges() / (n * (n - 1) // 2),
        "threshold": model.resolved_threshold,
        "nonzero_pct": 100 * len(nonzero) / len(upper) if len(upper) > 0 else 0,
        "weight_median": float(np.median(nonzero)) if len(nonzero) > 0 else 0,
        "weight_std": float(np.std(nonzero)) if len(nonzero) > 0 else 0,
        "n_components": len(components),
        "giant_size": comp_sizes[0],
        "giant_pct": 100 * comp_sizes[0] / n,
        "singletons": sum(1 for s in comp_sizes if s == 1),
        "bm_nodes_std": float(nodes_arr.std()),
        "bm_nodes_range": f"{nodes_arr.min()}-{nodes_arr.max()}",
    }

    return result, model


def main():
    print("Loading data...")
    _, _, embeddings_sub, df_sub = load_data(n_subsample=5000)
    print(f"Subsample: {embeddings_sub.shape}\n")

    # Show distance stats first
    diagnose_distances(embeddings_sub)

    results = []
    models = {}

    for name, config in EXPERIMENTS.items():
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {name}")
        print(f"{'='*60}")

        try:
            result, model = run_experiment(name, config, embeddings_sub)
            results.append(result)
            models[name] = model

            print(f"  Time: {result['time_s']:.1f}s")
            print(f"  Edges: {result['n_edges']:,} ({result['density']:.1%} density)")
            print(f"  Threshold: {result['threshold']:.4f}")
            print(f"  Components: {result['n_components']} "
                  f"(giant={result['giant_size']:,}, {result['giant_pct']:.0f}%)")
            print(f"  Ball map node range: {result['bm_nodes_range']} "
                  f"(std={result['bm_nodes_std']:.1f})")
        except Exception as e:
            print(f"  FAILED: {e}")

    # Summary table
    if results:
        df_results = pd.DataFrame(results)
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(df_results.to_string(index=False))

        # Save
        df_results.to_csv(DATA_DIR / "experiment_results.csv", index=False)
        print(f"\nSaved: {DATA_DIR / 'experiment_results.csv'}")

    return models


if __name__ == "__main__":
    models = main()
