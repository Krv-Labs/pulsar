"""
Diagnostics for the MMLU x Pulsar sweep.

This script is meant to answer one question quickly:
"Is this config producing a useful cosmic graph, and if not, what should I change?"

Usage:
    uv run python helpers/diagnose.py
    uv run python helpers/diagnose.py --config mmlu_params.yaml --n-subsample 3000
    uv run python helpers/diagnose.py --no-plot --distance-sample 600
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from scipy.spatial.distance import pdist

from pulsar import ThemaRS
from pulsar._pulsar import StandardScaler, pca_grid
from pulsar.config import load_config


DATA_DIR = Path("data")
EMBED_PATH = DATA_DIR / "mmlu_embeddings_all.npy"
QUESTIONS_PATH = DATA_DIR / "mmlu_questions.csv"
DEFAULT_DISTANCE_SAMPLE = 600
DISTANCE_PERCENTILES = (0, 5, 25, 50, 75, 95, 100)
SUMMARY_LABELS = {
    0: "min",
    5: "p05",
    25: "p25",
    50: "p50",
    75: "p75",
    95: "p95",
    100: "max",
}

console = Console()


@dataclass
class ConfigSummary:
    path: str
    dimensions: list[int]
    seeds: list[int]
    epsilons: list[float]
    threshold: str | float
    n_maps: int


@dataclass
class DistanceSummary:
    sample_size: int
    raw: dict[str, float]
    scaled: dict[str, float]
    pca: dict[int, dict[str, float]]
    scale_ratio: float


@dataclass
class GraphSummary:
    n_nodes: int
    n_edges: int
    density: float
    resolved_threshold: float
    total_pairs: int
    nonzero_pairs: int
    nonzero_fraction: float
    weight_stats: dict[str, float]
    component_count: int
    component_sizes: list[int]
    giant_fraction: float
    singleton_count: int


@dataclass
class BallMapSummary:
    n_maps: int
    node_stats: dict[str, float]
    edge_stats: dict[str, float]
    epsilon_stats: dict[str, float]
    unique_node_counts: int
    unique_edge_counts: int


def summarise(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {label: 0.0 for label in SUMMARY_LABELS.values()}

    percentiles = np.percentile(values, DISTANCE_PERCENTILES)
    summary = {
        SUMMARY_LABELS[p]: float(v) for p, v in zip(DISTANCE_PERCENTILES, percentiles)
    }
    summary["mean"] = float(np.mean(values))
    summary["std"] = float(np.std(values))
    return summary


def format_summary(summary: dict[str, float], precision: int = 4) -> str:
    keys = ["min", "p05", "p25", "p50", "p75", "p95", "max"]
    return " | ".join(
        f"{key}={summary[key]:.{precision}f}" for key in keys if key in summary
    )


def load_cfg(config_path: str) -> tuple[ConfigSummary, object]:
    import yaml

    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)
    cfg = load_config(raw_cfg)

    summary = ConfigSummary(
        path=config_path,
        dimensions=list(cfg.pca.dimensions),
        seeds=list(cfg.pca.seeds),
        epsilons=list(cfg.ball_mapper.epsilons),
        threshold=cfg.cosmic_graph.threshold,
        n_maps=len(cfg.pca.dimensions)
        * len(cfg.pca.seeds)
        * len(cfg.ball_mapper.epsilons),
    )
    return summary, cfg


def load_data(n_subsample: int = 5000, seed: int = 42):
    """Load cached embeddings and a stratified subsample."""
    embeddings_all = np.load(EMBED_PATH, mmap_mode="r")
    df_mmlu = pd.read_csv(QUESTIONS_PATH)

    rng = np.random.default_rng(seed)
    subject_to_indices = df_mmlu.groupby("subject", sort=False).indices
    indices: list[int] = []

    for subj_idx in subject_to_indices.values():
        subj_idx_arr = np.asarray(subj_idx, dtype=np.int64)
        n_take = max(10, int(n_subsample * len(subj_idx_arr) / len(df_mmlu)))
        n_take = min(n_take, len(subj_idx_arr))
        chosen = rng.choice(subj_idx_arr, size=n_take, replace=False)
        indices.extend(chosen.tolist())

    if len(indices) > n_subsample:
        indices = rng.choice(
            np.asarray(indices, dtype=np.int64), size=n_subsample, replace=False
        ).tolist()

    indices_arr = np.sort(np.asarray(indices, dtype=np.int64))
    embeddings_sub = np.asarray(embeddings_all[indices_arr], dtype=np.float64)
    df_sub = df_mmlu.iloc[indices_arr].reset_index(drop=True)

    return embeddings_all, df_mmlu, embeddings_sub, df_sub


def diagnose_distances(
    embeddings_sub: np.ndarray,
    dims: list[int],
    n_sample: int,
) -> DistanceSummary:
    """Estimate the distance scale that Ball Mapper actually sees."""
    sample_size = min(n_sample, len(embeddings_sub))
    idx = np.random.default_rng(42).choice(
        len(embeddings_sub), size=sample_size, replace=False
    )
    sample = np.ascontiguousarray(embeddings_sub[idx], dtype=np.float64)

    raw_dists = pdist(sample, metric="euclidean")
    scaled = np.asarray(StandardScaler().fit_transform(sample), dtype=np.float64)
    scaled_dists = pdist(scaled, metric="euclidean")

    unique_dims = sorted(dim for dim in set(dims) if dim < scaled.shape[1])
    pca_embeddings = pca_grid(scaled, unique_dims, [42]) if unique_dims else []

    pca_stats: dict[int, dict[str, float]] = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=24),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            "Computing post-scale PCA distance ranges", total=max(len(unique_dims), 1)
        )
        if not unique_dims:
            progress.advance(task)
        for dim, embedding in zip(unique_dims, pca_embeddings):
            pca_dists = pdist(np.ascontiguousarray(embedding), metric="euclidean")
            pca_stats[dim] = summarise(pca_dists)
            progress.advance(task)

    raw_summary = summarise(raw_dists)
    scaled_summary = summarise(scaled_dists)
    scale_ratio = scaled_summary["p50"] / max(raw_summary["p50"], 1e-12)

    return DistanceSummary(
        sample_size=sample_size,
        raw=raw_summary,
        scaled=scaled_summary,
        pca=pca_stats,
        scale_ratio=scale_ratio,
    )


def run_sweep(embeddings_sub: np.ndarray, cfg) -> tuple[ThemaRS, float]:
    df_emb = pd.DataFrame(embeddings_sub)
    t0 = time.perf_counter()
    with console.status(
        "[bold cyan]Running Pulsar sweep...[/bold cyan]", spinner="dots"
    ):
        model = ThemaRS(cfg).fit(data=df_emb)
    elapsed = time.perf_counter() - t0
    return model, elapsed


def diagnose_graph(model: ThemaRS) -> GraphSummary:
    G = model.cosmic_graph
    W = np.asarray(model.weighted_adjacency)
    n = W.shape[0]
    max_edges = n * (n - 1) // 2

    upper = W[np.triu_indices(n, k=1)]
    nonzero = upper[upper > 0]
    components = list(nx.connected_components(G))
    component_sizes = sorted((len(component) for component in components), reverse=True)
    giant_fraction = (component_sizes[0] / n) if component_sizes else 0.0

    return GraphSummary(
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
        density=(G.number_of_edges() / max_edges) if max_edges else 0.0,
        resolved_threshold=float(model.resolved_threshold),
        total_pairs=len(upper),
        nonzero_pairs=len(nonzero),
        nonzero_fraction=(len(nonzero) / len(upper)) if len(upper) else 0.0,
        weight_stats=summarise(nonzero),
        component_count=len(components),
        component_sizes=component_sizes,
        giant_fraction=giant_fraction,
        singleton_count=sum(1 for size in component_sizes if size == 1),
    )


def diagnose_ball_maps(model: ThemaRS) -> BallMapSummary:
    bms = model.ball_maps
    nodes = np.asarray([bm.n_nodes() for bm in bms], dtype=np.float64)
    edges = np.asarray([bm.n_edges() for bm in bms], dtype=np.float64)
    epsilons = np.asarray([bm.eps for bm in bms], dtype=np.float64)

    return BallMapSummary(
        n_maps=len(bms),
        node_stats=summarise(nodes),
        edge_stats=summarise(edges),
        epsilon_stats=summarise(epsilons),
        unique_node_counts=len(np.unique(nodes)),
        unique_edge_counts=len(np.unique(edges)),
    )


def render_header(cfg_summary: ConfigSummary, n_rows: int, n_subjects: int) -> None:
    table = Table(title="MMLU Diagnose", show_header=True, header_style="bold cyan")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Config", cfg_summary.path)
    table.add_row("Questions", f"{n_rows:,} from {n_subjects} subjects")
    table.add_row("PCA dims", ", ".join(str(dim) for dim in cfg_summary.dimensions))
    table.add_row("Seeds", f"{len(cfg_summary.seeds)} seeds")
    table.add_row(
        "Epsilons",
        f"{len(cfg_summary.epsilons)} values [{cfg_summary.epsilons[0]:.2f} .. {cfg_summary.epsilons[-1]:.2f}]",
    )
    table.add_row("Ball maps", f"{cfg_summary.n_maps:,}")
    table.add_row("Threshold", str(cfg_summary.threshold))
    console.print(table)


def render_distance_summary(summary: DistanceSummary) -> None:
    table = Table(
        title=f"Distance Scale ({summary.sample_size}-point sample)",
        header_style="bold cyan",
    )
    table.add_column("Space", style="bold")
    table.add_column("Summary")
    table.add_row("Raw embeddings", format_summary(summary.raw))
    table.add_row("After StandardScaler", format_summary(summary.scaled))
    for dim, stats in summary.pca.items():
        table.add_row(f"After StandardScaler + PCA {dim}d", format_summary(stats))
    console.print(table)
    console.print(
        Panel.fit(
            (
                f"Median pairwise distance grows by [bold]{summary.scale_ratio:.2f}x[/bold] after StandardScaler. "
                "Calibrate epsilon from the post-scale spaces, not the raw embedding space."
            ),
            title="Distance Insight",
            border_style="cyan",
        )
    )


def render_graph_summary(summary: GraphSummary) -> None:
    table = Table(title="Cosmic Graph Summary", header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Nodes", f"{summary.n_nodes:,}")
    table.add_row("Edges", f"{summary.n_edges:,} ({summary.density:.2%} density)")
    table.add_row("Resolved threshold", f"{summary.resolved_threshold:.6f}")
    table.add_row(
        "Non-zero pairs",
        f"{summary.nonzero_pairs:,} / {summary.total_pairs:,} ({summary.nonzero_fraction:.2%})",
    )
    table.add_row(
        "Weight distribution", format_summary(summary.weight_stats, precision=6)
    )
    table.add_row("Connected components", str(summary.component_count))
    table.add_row(
        "Giant component",
        f"{summary.component_sizes[0]:,} ({summary.giant_fraction:.1%})"
        if summary.component_sizes
        else "0",
    )
    table.add_row("Singletons", f"{summary.singleton_count:,}")
    if summary.component_sizes:
        table.add_row(
            "Top component sizes",
            ", ".join(f"{size:,}" for size in summary.component_sizes[:10]),
        )
    console.print(table)


def render_ball_map_summary(summary: BallMapSummary, n_points: int) -> None:
    table = Table(
        title=f"Ball Mapper Diversity ({summary.n_maps} maps)", header_style="bold cyan"
    )
    table.add_column("Metric", style="bold")
    table.add_column("Summary")
    table.add_row("Ball counts", format_summary(summary.node_stats, precision=1))
    table.add_row("Edge counts", format_summary(summary.edge_stats, precision=1))
    table.add_row("Epsilons", format_summary(summary.epsilon_stats, precision=2))
    table.add_row("Unique node counts", str(summary.unique_node_counts))
    table.add_row("Unique edge counts", str(summary.unique_edge_counts))
    table.add_row(
        "Median balls / points", f"{summary.node_stats['p50'] / max(n_points, 1):.2%}"
    )
    console.print(table)


def build_suggestions(
    cfg_summary: ConfigSummary,
    distance_summary: DistanceSummary,
    graph_summary: GraphSummary,
    ball_map_summary: BallMapSummary,
) -> list[str]:
    suggestions: list[str] = []
    median_balls = ball_map_summary.node_stats["p50"]
    p95_weight = graph_summary.weight_stats.get("p95", 0.0)
    zero_threshold = (
        cfg_summary.threshold == 0
        or cfg_summary.threshold == 0.0
        or cfg_summary.threshold == "0.0"
    )

    if distance_summary.scale_ratio > 3:
        suggestions.append(
            "Distance scale changes materially after StandardScaler. If an epsilon range came from raw embeddings, re-tune it in the post-scale PCA spaces."
        )

    if median_balls > graph_summary.n_nodes * 0.80:
        suggestions.append(
            "Median ball count is close to the number of points. Epsilons are too small, so many maps are near-singletons. Shift the epsilon range upward."
        )
    elif median_balls < max(10, graph_summary.n_nodes * 0.01):
        suggestions.append(
            "Median ball count is extremely low. Epsilons are too large and covers are collapsing. Shift the epsilon range downward."
        )

    if ball_map_summary.unique_node_counts < max(5, ball_map_summary.n_maps // 20):
        suggestions.append(
            "Ball-map diversity is weak. The sweep is not exploring enough structural variation. Broaden the epsilon range or drop redundant PCA dimensions."
        )

    if graph_summary.density < 0.001 or graph_summary.giant_fraction < 0.50:
        suggestions.append(
            "The cosmic graph is too sparse or too fragmented. Lower the threshold and/or increase epsilon until a meaningful giant component appears."
        )
    elif graph_summary.density > 0.50:
        if zero_threshold:
            suggestions.append(
                "The graph is intentionally dense because `threshold: 0.0` keeps all weighted edges. That is compatible with downstream spectral clustering on smooth text embeddings. For visualization or threshold diagnostics, inspect a higher percentile slice rather than changing the clustering config by default."
            )
        else:
            suggestions.append(
                "The cosmic graph is extremely dense. You may be washing out structure. Reduce epsilon or use a stronger threshold for visualization and diagnostics."
            )

    if (
        cfg_summary.threshold == "auto"
        and graph_summary.resolved_threshold > p95_weight
        and p95_weight > 0
    ):
        suggestions.append(
            "Auto-threshold landed above the 95th percentile of non-zero weights. For smooth text embeddings, that usually over-prunes the graph. Try `threshold: 0.0` and cluster on the weighted adjacency instead."
        )

    if not suggestions:
        suggestions.append(
            "This looks healthy: the sweep has variation, the graph retains connectivity, and the threshold is not obviously over-pruning. Use this as the baseline config."
        )

    return suggestions


def render_suggestions(suggestions: list[str]) -> None:
    body = "\n".join(f"- {suggestion}" for suggestion in suggestions)
    console.print(Panel.fit(body, title="Suggestions", border_style="green"))


def plot_diagnostics(model: ThemaRS, save_dir: Path, show_plot: bool) -> Path:
    import matplotlib.pyplot as plt

    W = np.asarray(model.weighted_adjacency)
    n = W.shape[0]
    upper = W[np.triu_indices(n, k=1)]
    nonzero = upper[upper > 0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")

    ax = axes[0, 0]
    if len(nonzero) > 0:
        ax.hist(nonzero, bins=100, alpha=0.75, color="#4C72B0", edgecolor="white")
        ax.axvline(
            model.resolved_threshold,
            color="#C44E52",
            linestyle="--",
            label=f"threshold={model.resolved_threshold:.4f}",
        )
        ax.set_title("Non-zero Edge Weights")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No non-zero weights", ha="center", va="center")
        ax.set_title("Non-zero Edge Weights")

    ax = axes[0, 1]
    ax.hist(upper, bins=100, alpha=0.75, color="#55A868", edgecolor="white")
    ax.axvline(
        model.resolved_threshold,
        color="#C44E52",
        linestyle="--",
        label=f"threshold={model.resolved_threshold:.4f}",
    )
    ax.set_title("All Pair Weights")
    ax.legend()

    ax = axes[1, 0]
    eps_vals = [bm.eps for bm in model.ball_maps]
    node_vals = [bm.n_nodes() for bm in model.ball_maps]
    ax.scatter(eps_vals, node_vals, s=14, alpha=0.6, color="#8172B3")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Number of balls")
    ax.set_title("Ball-map Complexity vs Epsilon")

    ax = axes[1, 1]
    if model.stability_result is not None:
        sr = model.stability_result
        thresholds = np.asarray(sr.thresholds)
        counts = np.asarray(sr.component_counts)
        ax.plot(thresholds, counts, color="#C44E52", linewidth=1.5)
        ax.axvline(
            sr.optimal_threshold,
            color="#4C72B0",
            linestyle="--",
            label=f"auto={sr.optimal_threshold:.4f}",
        )
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Connected components")
        ax.set_title("Stability Curve")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No stability result", ha="center", va="center")
        ax.set_title("Stability Curve")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    output_path = save_dir / "diagnostics.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose whether a Pulsar config is producing a useful MMLU cosmic graph."
    )
    parser.add_argument(
        "--config", default="mmlu_params.yaml", help="Path to params YAML."
    )
    parser.add_argument(
        "--n-subsample",
        type=int,
        default=5000,
        help="Stratified sample size to diagnose.",
    )
    parser.add_argument(
        "--distance-sample",
        type=int,
        default=DEFAULT_DISTANCE_SAMPLE,
        help="Sample size used for pairwise-distance diagnostics.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for the stratified subsample."
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip generating diagnostics.png."
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the matplotlib figure after saving it.",
    )
    args = parser.parse_args()

    cfg_summary, cfg = load_cfg(args.config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=24),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading cached MMLU data", total=4)
        progress.advance(task)
        _, df_mmlu, embeddings_sub, _ = load_data(args.n_subsample, seed=args.seed)
        progress.advance(task)
        distance_summary = diagnose_distances(
            embeddings_sub, cfg_summary.dimensions, args.distance_sample
        )
        progress.advance(task)
        model, sweep_elapsed = run_sweep(embeddings_sub, cfg)
        progress.advance(task)

    graph_summary = diagnose_graph(model)
    ball_map_summary = diagnose_ball_maps(model)
    suggestions = build_suggestions(
        cfg_summary, distance_summary, graph_summary, ball_map_summary
    )

    render_header(cfg_summary, len(embeddings_sub), df_mmlu["subject"].nunique())
    console.print(f"[bold]Sweep runtime:[/bold] {sweep_elapsed:.1f}s")
    render_distance_summary(distance_summary)
    render_graph_summary(graph_summary)
    render_ball_map_summary(ball_map_summary, len(embeddings_sub))
    render_suggestions(suggestions)

    if not args.no_plot:
        with console.status(
            "[bold cyan]Saving diagnostic plots...[/bold cyan]", spinner="dots"
        ):
            output_path = plot_diagnostics(model, DATA_DIR, show_plot=args.show_plot)
        console.print(f"[bold]Saved:[/bold] {output_path}")


if __name__ == "__main__":
    main()
