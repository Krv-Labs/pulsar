"""
Diagnostics for the MMLU × Pulsar sweep.

Loads cached embeddings, runs Pulsar with given params, and prints
everything you need to understand why the graph looks the way it does.

Usage:
    uv run python helpers/diagnose.py
    uv run python helpers/diagnose.py --config helpers/params_v2.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

from pulsar import ThemaRS, cosmic_clusters
from pulsar.config import load_config


DATA_DIR = Path("data")
EMBED_PATH = DATA_DIR / "mmlu_embeddings_all.npy"
QUESTIONS_PATH = DATA_DIR / "mmlu_questions.csv"


def load_data(n_subsample: int = 5000, seed: int = 42):
    """Load cached embeddings + stratified subsample."""
    embeddings_all = np.load(EMBED_PATH)
    df_mmlu = pd.read_csv(QUESTIONS_PATH)

    rng = np.random.default_rng(seed)
    indices = []
    for subj in df_mmlu["subject"].unique():
        subj_idx = df_mmlu.index[df_mmlu["subject"] == subj].tolist()
        n_take = max(10, int(n_subsample * len(subj_idx) / len(df_mmlu)))
        n_take = min(n_take, len(subj_idx))
        chosen = rng.choice(subj_idx, size=n_take, replace=False)
        indices.extend(chosen.tolist())

    if len(indices) > n_subsample:
        indices = sorted(rng.choice(indices, size=n_subsample, replace=False))
    else:
        indices = sorted(indices)

    return (
        embeddings_all,
        df_mmlu,
        embeddings_all[indices],
        df_mmlu.iloc[indices].reset_index(drop=True),
    )


def run_sweep(embeddings_sub: np.ndarray, config_path: str) -> ThemaRS:
    """Run Pulsar sweep and return fitted model."""
    import yaml

    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)
    cfg = load_config(raw_cfg)

    n_maps = len(cfg.pca.dimensions) * len(cfg.pca.seeds) * len(cfg.ball_mapper.epsilons)
    print(f"\n{'='*60}")
    print(f"CONFIG: {config_path}")
    print(f"  PCA dims: {cfg.pca.dimensions}")
    print(f"  PCA seeds: {cfg.pca.seeds}")
    print(f"  Epsilons: {len(cfg.ball_mapper.epsilons)} values "
          f"[{cfg.ball_mapper.epsilons[0]:.3f} .. {cfg.ball_mapper.epsilons[-1]:.3f}]")
    print(f"  Total ball maps: {n_maps}")
    print(f"  Threshold: {cfg.cosmic_graph.threshold}")
    print(f"{'='*60}")

    df_emb = pd.DataFrame(embeddings_sub)
    t0 = time.perf_counter()
    model = ThemaRS(cfg).fit(data=df_emb)
    elapsed = time.perf_counter() - t0
    print(f"\nSweep completed in {elapsed:.1f}s")
    return model


def diagnose_graph(model: ThemaRS):
    """Print all diagnostics about the cosmic graph."""
    G = model.cosmic_graph
    W = model.weighted_adjacency
    n = W.shape[0]
    max_edges = n * (n - 1) // 2

    n_edges = G.number_of_edges()
    density = n_edges / max_edges if max_edges > 0 else 0

    print(f"\n--- GRAPH STRUCTURE ---")
    print(f"Nodes: {G.number_of_nodes():,}")
    print(f"Edges: {n_edges:,} / {max_edges:,} possible ({density:.1%} density)")
    print(f"Resolved threshold: {model.resolved_threshold:.6f}")

    # Weight distribution
    upper = W[np.triu_indices(n, k=1)]
    nonzero = upper[upper > 0]
    print(f"\n--- WEIGHT DISTRIBUTION ---")
    print(f"Total pairs: {len(upper):,}")
    print(f"Non-zero pairs: {len(nonzero):,} ({100*len(nonzero)/len(upper):.1f}%)")
    if len(nonzero) > 0:
        for label, fn in [("Min", np.min), ("P5", lambda x: np.percentile(x, 5)),
                          ("P25", lambda x: np.percentile(x, 25)),
                          ("Median", np.median), ("Mean", np.mean),
                          ("P75", lambda x: np.percentile(x, 75)),
                          ("P95", lambda x: np.percentile(x, 95)),
                          ("Max", np.max), ("Std", np.std)]:
            print(f"  {label:>6}: {fn(nonzero):.6f}")

    # Stability
    if model.stability_result is not None:
        sr = model.stability_result
        print(f"\n--- STABILITY ANALYSIS ---")
        print(f"Optimal threshold: {sr.optimal_threshold:.6f}")
        print(f"Plateaus: {len(sr.plateaus)}")
        for i, p in enumerate(sr.plateaus[:8]):
            length = p.start_threshold - p.end_threshold
            print(f"  [{i}] {p.start_threshold:.4f} -> {p.end_threshold:.4f} "
                  f"({p.component_count} components, length={length:.4f})")

    # Connected components
    components = list(nx.connected_components(G))
    comp_sizes = sorted([len(c) for c in components], reverse=True)
    print(f"\n--- CONNECTED COMPONENTS ---")
    print(f"Count: {len(components)}")
    print(f"Sizes (top 15): {comp_sizes[:15]}")
    if len(comp_sizes) > 1:
        print(f"Giant component: {comp_sizes[0]:,} ({100*comp_sizes[0]/n:.1f}% of nodes)")
        print(f"Singletons: {sum(1 for s in comp_sizes if s == 1)}")


def diagnose_ball_maps(model: ThemaRS):
    """Analyze ball map diversity."""
    bms = model.ball_maps
    nodes = np.array([bm.n_nodes() for bm in bms])
    edges = np.array([bm.n_edges() for bm in bms])
    epsilons = np.array([bm.eps for bm in bms])

    print(f"\n--- BALL MAP DIVERSITY ({len(bms)} maps) ---")
    for label, arr in [("Nodes", nodes), ("Edges", edges), ("Epsilon", epsilons)]:
        print(f"  {label:>8} — min: {arr.min():.1f}, median: {np.median(arr):.1f}, "
              f"max: {arr.max():.1f}, std: {arr.std():.1f}")

    # How many unique node counts?
    print(f"  Unique node counts: {len(np.unique(nodes))}")
    print(f"  Unique edge counts: {len(np.unique(edges))}")

    # If most maps are identical, the sweep isn't doing anything
    if nodes.std() < 1.0:
        print("  *** WARNING: Ball maps are nearly identical. Epsilon range needs widening. ***")


def diagnose_distances(embeddings_sub: np.ndarray, n_sample: int = 1000):
    """Analyze pairwise distances to inform epsilon selection."""
    idx = np.random.default_rng(42).choice(len(embeddings_sub),
                                           size=min(n_sample, len(embeddings_sub)),
                                           replace=False)
    sample = embeddings_sub[idx]

    dists = pdist(sample, metric="euclidean")
    print(f"\n--- PAIRWISE DISTANCES ({len(idx)}-point sample, Euclidean) ---")
    for label, fn in [("Min", np.min), ("P5", lambda x: np.percentile(x, 5)),
                      ("P25", lambda x: np.percentile(x, 25)),
                      ("Median", np.median), ("Mean", np.mean),
                      ("P75", lambda x: np.percentile(x, 75)),
                      ("P95", lambda x: np.percentile(x, 95)),
                      ("Max", np.max)]:
        print(f"  {label:>6}: {fn(dists):.4f}")

    # What epsilon range makes sense?
    p10 = np.percentile(dists, 10)
    p50 = np.percentile(dists, 50)
    p90 = np.percentile(dists, 90)
    print(f"\n  Suggested epsilon range: {p10:.3f} to {p90:.3f}")
    print(f"  (P10 to P90 of pairwise distances)")

    # Also check after PCA to various dims
    from pulsar import pca_grid
    for dim in [5, 10, 20, 50]:
        if dim >= sample.shape[1]:
            continue
        pca_results = pca_grid(sample.astype(np.float64), [dim], [42])
        pca_dists = pdist(np.ascontiguousarray(pca_results[0]), metric="euclidean")
        p10_pca = np.percentile(pca_dists, 10)
        p50_pca = np.percentile(pca_dists, 50)
        p90_pca = np.percentile(pca_dists, 90)
        print(f"  After PCA to {dim}d: P10={p10_pca:.3f}, P50={p50_pca:.3f}, P90={p90_pca:.3f}")


def plot_diagnostics(model: ThemaRS, embeddings_sub: np.ndarray, save_dir: Path):
    """Generate diagnostic plots."""
    W = model.weighted_adjacency
    n = W.shape[0]
    upper = W[np.triu_indices(n, k=1)]
    nonzero = upper[upper > 0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Weight distribution
    ax = axes[0, 0]
    if len(nonzero) > 0:
        ax.hist(nonzero, bins=100, alpha=0.7, color="#4C72B0", edgecolor="white")
        ax.axvline(model.resolved_threshold, color="red", linestyle="--",
                   label=f"threshold={model.resolved_threshold:.4f}")
        ax.set_title("Non-Zero Edge Weight Distribution")
        ax.legend()

    # 2. All weights (including zeros)
    ax = axes[0, 1]
    ax.hist(upper, bins=100, alpha=0.7, color="#55A868", edgecolor="white")
    ax.axvline(model.resolved_threshold, color="red", linestyle="--",
               label=f"threshold={model.resolved_threshold:.4f}")
    ax.set_title("ALL Pair Weight Distribution")
    ax.legend()

    # 3. Ball map node count vs epsilon
    ax = axes[1, 0]
    bms = model.ball_maps
    eps_vals = [bm.eps for bm in bms]
    node_vals = [bm.n_nodes() for bm in bms]
    ax.scatter(eps_vals, node_vals, s=10, alpha=0.5)
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Number of nodes (balls)")
    ax.set_title("Ball Map Complexity vs Epsilon")

    # 4. Component count vs threshold (stability curve)
    if model.stability_result is not None:
        ax = axes[1, 1]
        sr = model.stability_result
        thresholds = np.array(sr.thresholds)
        counts = np.array(sr.component_counts)
        ax.plot(thresholds, counts, color="#C44E52", linewidth=1.5)
        ax.axvline(sr.optimal_threshold, color="blue", linestyle="--",
                   label=f"auto={sr.optimal_threshold:.4f}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Connected Components")
        ax.set_title("Stability Curve")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "diagnostics.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_dir / 'diagnostics.png'}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose MMLU × Pulsar sweep")
    parser.add_argument("--config", default="mmlu_params.yaml",
                        help="Path to params YAML")
    parser.add_argument("--n-subsample", type=int, default=5000)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print("Loading data...")
    embeddings_all, df_mmlu, embeddings_sub, df_sub = load_data(args.n_subsample)
    print(f"Subsample: {embeddings_sub.shape}")

    # Distance diagnostics (independent of Pulsar)
    diagnose_distances(embeddings_sub)

    # Run sweep
    model = run_sweep(embeddings_sub, args.config)

    # Diagnose
    diagnose_graph(model)
    diagnose_ball_maps(model)

    if not args.no_plot:
        plot_diagnostics(model, embeddings_sub, DATA_DIR)


if __name__ == "__main__":
    main()
