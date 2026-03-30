"""
v2: Now with actually correct epsilon ranges.

The BallMapper uses SQUARED Euclidean comparison internally but eps is
compared as eps^2 against L2_sq. So eps IS the actual radius in Euclidean space.

Post-PCA distances at 10d: P50=0.162, at 50d: P50=0.379
n_nodes ~= n_points means eps is TOO SMALL (every point is its own ball).
We need eps large enough that each ball covers ~10-100 points.

Target: n_nodes should be ~50-500 (much less than n_points).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import networkx as nx

from pulsar import ThemaRS
from pulsar.config import load_config
from pulsar._pulsar import pca_grid, BallMapper

from diagnose import load_data


def test_epsilon_directly(embeddings_sub, pca_dim, epsilons, seed=42):
    """Directly test BallMapper at various epsilons to find the sweet spot."""
    pca_results = pca_grid(embeddings_sub.astype(np.float64), [pca_dim], [seed])
    X = np.ascontiguousarray(pca_results[0])

    print(f"\n  PCA {pca_dim}d ({X.shape}):")
    for eps in epsilons:
        bm = BallMapper(eps)
        bm.fit(X)
        coverage = sum(len(members) for members in bm.nodes) / X.shape[0]
        print(f"    eps={eps:.2f} -> {bm.n_nodes():5d} balls, "
              f"{bm.n_edges():5d} edges, "
              f"coverage={coverage:.2f}")


def main():
    N = 1500
    print(f"Loading data (n={N})...")
    _, _, embeddings_sub, df_sub = load_data(n_subsample=N)
    n = len(df_sub)
    print(f"Subsample: {embeddings_sub.shape}")

    # Step 1: Find the right epsilon range by direct BallMapper testing
    print("\n" + "="*60)
    print("STEP 1: Finding epsilon sweet spot per PCA dimension")
    print("="*60)

    epsilons = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75,
                1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]

    for dim in [5, 10, 20, 50, 100]:
        test_epsilon_directly(embeddings_sub, dim, epsilons)

    # Step 2: Run full sweeps with informed epsilon ranges
    print("\n" + "="*60)
    print("STEP 2: Full sweeps with corrected epsilon ranges")
    print("="*60)

    experiments = {
        # Based on where n_nodes is in the 50-500 range from step 1
        "informed_low_dim": {
            "sweep": {
                "pca": {
                    "dimensions": {"values": [5, 10, 20]},
                    "seed": {"values": [42, 7, 13, 99, 123]},
                },
                "ball_mapper": {
                    "epsilon": {"range": {"min": 0.5, "max": 5.0, "steps": 15}},
                },
            },
            "cosmic_graph": {"threshold": "auto"},
        },
        "informed_high_dim": {
            "sweep": {
                "pca": {
                    "dimensions": {"values": [20, 50, 100]},
                    "seed": {"values": [42, 7, 13, 99, 123]},
                },
                "ball_mapper": {
                    "epsilon": {"range": {"min": 1.0, "max": 8.0, "steps": 15}},
                },
            },
            "cosmic_graph": {"threshold": "auto"},
        },
        "informed_mix": {
            "sweep": {
                "pca": {
                    "dimensions": {"values": [10, 20, 50]},
                    "seed": {"values": [42, 7, 13, 99, 123, 456, 789]},
                },
                "ball_mapper": {
                    "epsilon": {"range": {"min": 0.5, "max": 6.0, "steps": 20}},
                },
            },
            "cosmic_graph": {"threshold": "auto"},
        },
    }

    results = []
    for name, config in experiments.items():
        config.setdefault("run", {"name": name})
        cfg = load_config(config)
        n_maps = len(cfg.pca.dimensions) * len(cfg.pca.seeds) * len(cfg.ball_mapper.epsilons)
        df_emb = pd.DataFrame(embeddings_sub)

        print(f"\n--- {name} ({n_maps} maps) ---")
        t0 = time.perf_counter()
        model = ThemaRS(cfg).fit(data=df_emb)
        elapsed = time.perf_counter() - t0

        G = model.cosmic_graph
        W = model.weighted_adjacency
        upper = W[np.triu_indices(n, k=1)]
        nonzero = upper[upper > 0]
        components = list(nx.connected_components(G))
        comp_sizes = sorted([len(c) for c in components], reverse=True)
        nodes_arr = np.array([bm.n_nodes() for bm in model.ball_maps])

        print(f"  Time: {elapsed:.1f}s")
        print(f"  Edges: {G.number_of_edges():,} ({G.number_of_edges()/(n*(n-1)//2):.2%} density)")
        print(f"  Threshold: {model.resolved_threshold:.4f}")
        print(f"  Components: {len(components)} (giant={comp_sizes[0]}, "
              f"{100*comp_sizes[0]/n:.0f}%)")
        print(f"  Singletons: {sum(1 for s in comp_sizes if s == 1)}")
        print(f"  Ball map nodes: {nodes_arr.min()}-{nodes_arr.max()} "
              f"(median={int(np.median(nodes_arr))})")
        if len(nonzero) > 0:
            print(f"  Weights: median={np.median(nonzero):.4f}, "
                  f"P95={np.percentile(nonzero, 95):.4f}")

        results.append({
            "name": name, "n_maps": n_maps, "time": f"{elapsed:.1f}s",
            "edges": G.number_of_edges(),
            "components": len(components), "giant": comp_sizes[0],
            "singletons": sum(1 for s in comp_sizes if s == 1),
            "bm_range": f"{nodes_arr.min()}-{nodes_arr.max()}",
            "threshold": f"{model.resolved_threshold:.4f}",
        })

    print(f"\n{'='*60}")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
