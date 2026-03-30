"""
Fast experiment runner — uses 1500-point subsample for quick iteration.

Usage:
    uv run python helpers/quick_experiment.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import networkx as nx

from pulsar import ThemaRS, cosmic_clusters
from pulsar.config import load_config
from pulsar._pulsar import pca_grid

from diagnose import load_data


def run_one(name, config_dict, embeddings_sub):
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

    nodes_arr = np.array([bm.n_nodes() for bm in model.ball_maps])

    return {
        "name": name,
        "n_maps": n_maps,
        "time": f"{elapsed:.1f}s",
        "edges": G.number_of_edges(),
        "density": f"{G.number_of_edges() / (n*(n-1)//2):.2%}",
        "threshold": f"{model.resolved_threshold:.4f}",
        "nonzero%": f"{100*len(nonzero)/len(upper):.1f}",
        "wt_median": f"{np.median(nonzero):.4f}" if len(nonzero) > 0 else "N/A",
        "wt_p95": f"{np.percentile(nonzero, 95):.4f}" if len(nonzero) > 0 else "N/A",
        "components": len(components),
        "giant": comp_sizes[0],
        "giant%": f"{100*comp_sizes[0]/n:.0f}",
        "singletons": sum(1 for s in comp_sizes if s == 1),
        "bm_nodes": f"{nodes_arr.min()}-{nodes_arr.max()}",
    }, model


def main():
    N = 1500  # Small for fast iteration
    print(f"Loading data (n={N})...")
    _, _, embeddings_sub, df_sub = load_data(n_subsample=N)
    print(f"Subsample: {embeddings_sub.shape}")

    # Check post-PCA distances at this scale
    from scipy.spatial.distance import pdist
    sample = embeddings_sub[:500]
    print("\nPost-PCA distance ranges (P10 / P50 / P90):")
    for dim in [5, 10, 20, 50, 100]:
        if dim >= sample.shape[1]:
            continue
        pca_results = pca_grid(sample.astype(np.float64), [dim], [42])
        d = pdist(np.ascontiguousarray(pca_results[0]), metric="euclidean")
        print(f"  {dim:3d}d: {np.percentile(d,10):.3f} / {np.percentile(d,50):.3f} / {np.percentile(d,90):.3f}")

    experiments = {
        "A_mid_dim": {
            "sweep": {
                "pca": {
                    "dimensions": {"values": [10, 20, 50]},
                    "seed": {"values": [42, 7, 13, 99, 123]},
                },
                "ball_mapper": {
                    "epsilon": {"range": {"min": 0.10, "max": 0.35, "steps": 15}},
                },
            },
            "cosmic_graph": {"threshold": "auto"},
        },
        "B_high_dim": {
            "sweep": {
                "pca": {
                    "dimensions": {"values": [20, 50, 100]},
                    "seed": {"values": [42, 7, 13, 99, 123]},
                },
                "ball_mapper": {
                    "epsilon": {"range": {"min": 0.15, "max": 0.45, "steps": 15}},
                },
            },
            "cosmic_graph": {"threshold": "auto"},
        },
        "C_narrow_sweet_spot": {
            # Target the P25-P75 range for each dim
            "sweep": {
                "pca": {
                    "dimensions": {"values": [10, 20, 50]},
                    "seed": {"values": [42, 7, 13, 99, 123, 456, 789]},
                },
                "ball_mapper": {
                    "epsilon": {"range": {"min": 0.12, "max": 0.30, "steps": 20}},
                },
            },
            "cosmic_graph": {"threshold": "auto"},
        },
        "D_very_high_dim": {
            # Preserve max variance, let Ball Mapper find fine structure
            "sweep": {
                "pca": {
                    "dimensions": {"values": [50, 100, 150, 200]},
                    "seed": {"values": [42, 7, 13, 99, 123]},
                },
                "ball_mapper": {
                    "epsilon": {"range": {"min": 0.25, "max": 0.60, "steps": 15}},
                },
            },
            "cosmic_graph": {"threshold": "auto"},
        },
        "E_manual_threshold_low": {
            # Same as C but with a low manual threshold to see what structure exists
            "sweep": {
                "pca": {
                    "dimensions": {"values": [10, 20, 50]},
                    "seed": {"values": [42, 7, 13, 99, 123, 456, 789]},
                },
                "ball_mapper": {
                    "epsilon": {"range": {"min": 0.12, "max": 0.30, "steps": 20}},
                },
            },
            "cosmic_graph": {"threshold": 0.01},
        },
        "F_manual_threshold_mid": {
            "sweep": {
                "pca": {
                    "dimensions": {"values": [10, 20, 50]},
                    "seed": {"values": [42, 7, 13, 99, 123, 456, 789]},
                },
                "ball_mapper": {
                    "epsilon": {"range": {"min": 0.12, "max": 0.30, "steps": 20}},
                },
            },
            "cosmic_graph": {"threshold": 0.05},
        },
    }

    results = []
    for name, config in experiments.items():
        print(f"\n--- {name} ---")
        try:
            result, model = run_one(name, config, embeddings_sub)
            results.append(result)
            print(f"  {result['time']} | edges={result['edges']} ({result['density']}) | "
                  f"components={result['components']} (giant={result['giant']}, {result['giant%']}%) | "
                  f"singletons={result['singletons']} | bm_nodes={result['bm_nodes']}")
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
