"""
v3: Per-dimension epsilon targeting.

The sweet spot is where n_balls is ~50-500 with meaningful edges.
We target that range for each PCA dim separately.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import networkx as nx

from pulsar import ThemaRS
from pulsar.config import load_config

from diagnose import load_data


def main():
    N = 1500
    print(f"Loading data (n={N})...")
    _, _, embeddings_sub, df_sub = load_data(n_subsample=N)
    n = len(df_sub)
    print(f"Subsample: {embeddings_sub.shape}")

    # From Step 1 findings, the sweet spots are:
    #   5d:  eps 0.05-0.10  (target: 26-224 balls)
    #  10d:  eps 0.08-0.15  (target: 39-295 balls)
    #  20d:  eps 0.12-0.22  (target: 117-599 balls)
    #  50d:  eps 0.25-0.35  (target: ~350 balls)
    # 100d:  eps 0.40-0.55  (target: ~90 balls)
    #
    # Strategy: use per-dim epsilon lists that stay in the sweet spot

    experiments = {
        "per_dim_sweet_spot": {
            # Use values lists with eps targeted per dimension.
            # Since config only supports one global epsilon list,
            # we pick values that span all sweet spots.
            # eps 0.05-0.55 with fine steps in the 0.05-0.25 range
            "sweep": {
                "pca": {
                    "dimensions": {"values": [5, 10, 20, 50]},
                    "seed": {"values": [42, 7, 13, 99, 123]},
                },
                "ball_mapper": {
                    "epsilon": {"values": [
                        0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                        0.12, 0.14, 0.16, 0.18, 0.20,
                        0.25, 0.30, 0.35, 0.40,
                    ]},
                },
            },
            "cosmic_graph": {"threshold": "auto"},
        },
        "low_dim_focused": {
            # Only dims where we get good ball diversity
            "sweep": {
                "pca": {
                    "dimensions": {"values": [5, 10, 20]},
                    "seed": {"values": [42, 7, 13, 99, 123, 456, 789, 1000]},
                },
                "ball_mapper": {
                    "epsilon": {"values": [
                        0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                        0.12, 0.14, 0.16, 0.18, 0.20, 0.22,
                    ]},
                },
            },
            "cosmic_graph": {"threshold": "auto"},
        },
        "low_dim_manual_thresh_005": {
            "sweep": {
                "pca": {
                    "dimensions": {"values": [5, 10, 20]},
                    "seed": {"values": [42, 7, 13, 99, 123, 456, 789, 1000]},
                },
                "ball_mapper": {
                    "epsilon": {"values": [
                        0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                        0.12, 0.14, 0.16, 0.18, 0.20, 0.22,
                    ]},
                },
            },
            "cosmic_graph": {"threshold": 0.005},
        },
        "low_dim_manual_thresh_01": {
            "sweep": {
                "pca": {
                    "dimensions": {"values": [5, 10, 20]},
                    "seed": {"values": [42, 7, 13, 99, 123, 456, 789, 1000]},
                },
                "ball_mapper": {
                    "epsilon": {"values": [
                        0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                        0.12, 0.14, 0.16, 0.18, 0.20, 0.22,
                    ]},
                },
            },
            "cosmic_graph": {"threshold": 0.01},
        },
        "low_dim_manual_thresh_05": {
            "sweep": {
                "pca": {
                    "dimensions": {"values": [5, 10, 20]},
                    "seed": {"values": [42, 7, 13, 99, 123, 456, 789, 1000]},
                },
                "ball_mapper": {
                    "epsilon": {"values": [
                        0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                        0.12, 0.14, 0.16, 0.18, 0.20, 0.22,
                    ]},
                },
            },
            "cosmic_graph": {"threshold": 0.05},
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

        print(f"  Time: {elapsed:.1f}s | Threshold: {model.resolved_threshold:.4f}")
        print(f"  Edges: {G.number_of_edges():,} ({G.number_of_edges()/(n*(n-1)//2):.2%})")
        print(f"  Components: {len(components)} "
              f"(giant={comp_sizes[0]}, {100*comp_sizes[0]/n:.0f}%) "
              f"singletons={sum(1 for s in comp_sizes if s == 1)}")
        print(f"  Ball nodes: {nodes_arr.min()}-{nodes_arr.max()} "
              f"(median={int(np.median(nodes_arr))}, std={nodes_arr.std():.0f})")
        if len(nonzero) > 0:
            print(f"  Weights: median={np.median(nonzero):.4f}, "
                  f"P95={np.percentile(nonzero, 95):.4f}, "
                  f"max={nonzero.max():.4f}")

        # If we have a decent graph, show component distribution
        if 2 < len(components) < n * 0.9:
            print(f"  Component sizes (top 20): {comp_sizes[:20]}")

        results.append({
            "name": name, "n_maps": n_maps,
            "edges": G.number_of_edges(),
            "density": f"{G.number_of_edges()/(n*(n-1)//2):.2%}",
            "components": len(components),
            "giant": comp_sizes[0],
            "giant%": f"{100*comp_sizes[0]/n:.0f}%",
            "singletons": sum(1 for s in comp_sizes if s == 1),
            "threshold": f"{model.resolved_threshold:.4f}",
            "bm_median": int(np.median(nodes_arr)),
        })

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
