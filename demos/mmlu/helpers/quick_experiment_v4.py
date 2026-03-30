"""
v4: Correct epsilon ranges based on StandardScaler discovery.

StandardScaler normalizes each feature to mean=0, std=1, which multiplies
distances by ~20x compared to raw normalized embeddings.

Post-scale+PCA sweet spots:
  5d:  eps 1.0-2.0 (69-424 balls)
  10d: eps 2.0-5.0 (12-728 balls)
  20d: eps 4.0-6.0 (~189 balls)
  50d: eps 8.0-12.0 (~30 balls)
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import networkx as nx

from pulsar import ThemaRS, cosmic_clusters
from pulsar.config import load_config

from diagnose import load_data


def run_and_report(name, config, embeddings_sub, n):
    config.setdefault("run", {"name": name})
    cfg = load_config(config)
    n_maps = len(cfg.pca.dimensions) * len(cfg.pca.seeds) * len(cfg.ball_mapper.epsilons)
    df_emb = pd.DataFrame(embeddings_sub)

    print(f"\n{'='*60}")
    print(f"{name} ({n_maps} maps)")
    print(f"  dims={cfg.pca.dimensions}, seeds={len(cfg.pca.seeds)}")
    print(f"  eps=[{cfg.ball_mapper.epsilons[0]:.1f}..{cfg.ball_mapper.epsilons[-1]:.1f}] "
          f"({len(cfg.ball_mapper.epsilons)} values)")
    print(f"  threshold={cfg.cosmic_graph.threshold}")
    print(f"{'='*60}")

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

    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  Graph: {G.number_of_edges():,} edges "
          f"({G.number_of_edges()/(n*(n-1)//2):.2%} density)")
    print(f"  Threshold: {model.resolved_threshold:.4f}")
    print(f"  Components: {len(components)}")
    print(f"    Giant: {comp_sizes[0]:,} ({100*comp_sizes[0]/n:.0f}%)")
    if len(comp_sizes) > 1:
        print(f"    Top 10: {comp_sizes[:10]}")
    print(f"    Singletons: {sum(1 for s in comp_sizes if s == 1)}")
    print(f"  Ball maps: nodes {nodes_arr.min()}-{nodes_arr.max()} "
          f"(median={int(np.median(nodes_arr))}, std={nodes_arr.std():.0f})")
    if len(nonzero) > 0:
        print(f"  Weights: median={np.median(nonzero):.4f}, "
              f"P95={np.percentile(nonzero, 95):.4f}, "
              f"max={nonzero.max():.4f}")

    # If decent structure, try clustering
    if 2 <= len(components) <= 50 and comp_sizes[0] > n * 0.5:
        for k in [4, 6, 8, 10]:
            labels = cosmic_clusters(G, method="agglomerative", n_clusters=k)
            sizes = np.bincount(labels)
            min_size = sizes.min()
            print(f"  Clustering k={k}: sizes={list(sizes)}, min={min_size}")

    return model


def main():
    N = 1500
    print(f"Loading data (n={N})...")
    _, _, embeddings_sub, df_sub = load_data(n_subsample=N)
    n = len(df_sub)
    print(f"Subsample: {embeddings_sub.shape}")

    experiments = {
        "A_correct_range": {
            "sweep": {
                "pca": {
                    "dimensions": {"values": [5, 10, 20]},
                    "seed": {"values": [42, 7, 13, 99, 123]},
                },
                "ball_mapper": {
                    "epsilon": {"values": [
                        0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0,
                    ]},
                },
            },
            "cosmic_graph": {"threshold": "auto"},
        },
        "B_wide_dim_range": {
            "sweep": {
                "pca": {
                    "dimensions": {"values": [5, 10, 20, 50]},
                    "seed": {"values": [42, 7, 13, 99, 123]},
                },
                "ball_mapper": {
                    "epsilon": {"values": [
                        0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0,
                    ]},
                },
            },
            "cosmic_graph": {"threshold": "auto"},
        },
        "C_more_seeds": {
            "sweep": {
                "pca": {
                    "dimensions": {"values": [5, 10, 20]},
                    "seed": {"values": [42, 7, 13, 99, 123, 456, 789, 1000, 2023, 3141]},
                },
                "ball_mapper": {
                    "epsilon": {"values": [
                        0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0,
                    ]},
                },
            },
            "cosmic_graph": {"threshold": "auto"},
        },
        "D_fine_grained": {
            "sweep": {
                "pca": {
                    "dimensions": {"values": [5, 10, 20]},
                    "seed": {"values": [42, 7, 13, 99, 123]},
                },
                "ball_mapper": {
                    "epsilon": {"values": [
                        0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5,
                        1.7, 2.0, 2.3, 2.7, 3.0, 3.5, 4.0, 5.0,
                    ]},
                },
            },
            "cosmic_graph": {"threshold": "auto"},
        },
    }

    for name, config in experiments.items():
        run_and_report(name, config, embeddings_sub, n)


if __name__ == "__main__":
    main()
