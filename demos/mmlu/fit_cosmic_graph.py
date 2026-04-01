"""Run MMLU cosmic graph fitting outside Jupyter.

This script mirrors the notebook's Step 6 fit stage while avoiding notebook
kernel overhead. It loads cached embedding artifacts from
data/mmlu_embedding_artifacts_metadata.json, runs ThemaRS.fit_multi(), and
writes a summary plus optional weighted adjacency output.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pulsar import ThemaRS
from pulsar.config import load_config


DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = DEMO_DIR / "mmlu_params.yaml"
DEFAULT_METADATA_INDEX = DEMO_DIR / "data" / "mmlu_embedding_artifacts_metadata.json"
DEFAULT_OUTPUT_DIR = DEMO_DIR / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a fused cosmic graph from cached MMLU embedding artifacts "
            "without running the notebook."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to Pulsar YAML config (default: demos/mmlu/mmlu_params.yaml).",
    )
    parser.add_argument(
        "--metadata-index",
        type=Path,
        default=DEFAULT_METADATA_INDEX,
        help="Path to embedding artifact metadata JSON index.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for summary/output files.",
    )
    parser.add_argument(
        "--max-representations",
        type=int,
        default=None,
        help="Optional cap on number of embedding artifacts to fuse.",
    )
    parser.add_argument(
        "--ballmap-batch-size",
        type=int,
        default=None,
        help="Optional BallMapper batch size to reduce peak memory usage.",
    )
    parser.add_argument(
        "--rayon-workers",
        type=int,
        default=None,
        help="Optional cap for Rayon worker threads in Rust operations.",
    )
    parser.add_argument(
        "--save-adjacency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save weighted adjacency to a compressed .npz file (default: true).",
    )
    return parser.parse_args()


def load_pulsar_config(config_path: Path):
    with config_path.open("r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle)
    return load_config(raw_cfg)


def maps_per_representation(cfg: Any) -> int:
    return (
        len(cfg.pca.dimensions)
        * len(cfg.pca.seeds)
        * len(cfg.ball_mapper.epsilons)
    )


def load_embedding_datasets(
    metadata_index_path: Path,
    max_representations: int | None,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], dict[str, Any]]:
    with metadata_index_path.open("r", encoding="utf-8") as handle:
        index_payload = json.load(handle)

    artifacts = index_payload.get("artifacts", [])
    if not artifacts:
        raise RuntimeError(f"No artifacts listed in {metadata_index_path}")

    selected = sorted(artifacts, key=lambda row: row["artifact_key"])
    if max_representations is not None:
        if max_representations <= 0:
            raise ValueError("--max-representations must be positive")
        selected = selected[:max_representations]

    datasets: list[pd.DataFrame] = []
    expected_rows: int | None = None

    for row in selected:
        cache_path = Path(row["cache_path"])
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing embedding cache for {row['artifact_key']}: {cache_path}"
            )

        emb = np.load(cache_path, mmap_mode="r")
        if emb.ndim != 2:
            raise ValueError(
                f"Expected 2D embedding matrix for {row['artifact_key']}, got shape {emb.shape}"
            )

        if expected_rows is None:
            expected_rows = int(emb.shape[0])
        elif int(emb.shape[0]) != expected_rows:
            raise ValueError(
                "All embedding artifacts must have identical row counts; "
                f"found {emb.shape[0]} and {expected_rows}."
            )

        datasets.append(pd.DataFrame(emb))

    if expected_rows is None:
        raise RuntimeError("No embedding datasets were loaded")

    return datasets, selected, index_payload


def build_summary(
    *,
    cfg: Any,
    selected_artifacts: list[dict[str, Any]],
    index_payload: dict[str, Any],
    elapsed_sec: float,
    model: ThemaRS,
    adjacency_path: Path | None,
) -> dict[str, Any]:
    graph = model.cosmic_graph
    n_nodes = int(graph.number_of_nodes())
    n_edges = int(graph.number_of_edges())
    max_edges = n_nodes * (n_nodes - 1) // 2
    density = (n_edges / max_edges) if max_edges else 0.0

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sweep": {
            "pca_dimensions": list(cfg.pca.dimensions),
            "pca_seeds": list(cfg.pca.seeds),
            "epsilons": list(cfg.ball_mapper.epsilons),
            "maps_per_representation": maps_per_representation(cfg),
        },
        "metadata_index": {
            "path": str(index_payload.get("path", "")),
            "run_mode": index_payload.get("run_mode"),
            "rows": index_payload.get("rows"),
            "target_artifacts": index_payload.get("target_artifacts"),
            "completed_artifacts": index_payload.get("completed_artifacts"),
        },
        "fit": {
            "representations_fused": len(selected_artifacts),
            "total_ball_maps": maps_per_representation(cfg) * len(selected_artifacts),
            "elapsed_sec": elapsed_sec,
            "resolved_threshold": float(model.resolved_threshold),
            "graph_nodes": n_nodes,
            "graph_edges": n_edges,
            "graph_density": density,
        },
        "artifacts": [row["artifact_key"] for row in selected_artifacts],
        "outputs": {
            "weighted_adjacency": str(adjacency_path) if adjacency_path else None,
        },
    }


def main() -> None:
    args = parse_args()

    cfg_path = args.config.resolve()
    metadata_index_path = args.metadata_index.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_pulsar_config(cfg_path)
    n_maps_per_rep = maps_per_representation(cfg)

    datasets, selected_artifacts, index_payload = load_embedding_datasets(
        metadata_index_path,
        args.max_representations,
    )

    print(f"Loaded config: {cfg_path}")
    print(f"Loaded metadata index: {metadata_index_path}")
    print(f"Representations to fuse: {len(selected_artifacts)}")
    print(
        "Sweep per representation: "
        f"{len(cfg.pca.dimensions)} dims x {len(cfg.pca.seeds)} seeds "
        f"x {len(cfg.ball_mapper.epsilons)} epsilons = {n_maps_per_rep} ball maps"
    )
    print(f"Total ball maps fused: {n_maps_per_rep * len(selected_artifacts)}")

    previous_stage = ""

    def progress_callback(stage: str, fraction: float) -> None:
        nonlocal previous_stage
        if stage == previous_stage:
            return
        previous_stage = stage
        print(f"[{fraction:6.1%}] {stage}")

    t0 = time.perf_counter()
    model = ThemaRS(cfg).fit_multi(
        datasets=datasets,
        progress_callback=progress_callback,
        store_ball_maps=False,
        ballmap_batch_size=args.ballmap_batch_size,
        rayon_workers=args.rayon_workers,
    )
    elapsed_sec = time.perf_counter() - t0

    adjacency_path: Path | None = None
    if args.save_adjacency:
        adjacency_path = output_dir / "mmlu_cosmic_weighted_adjacency.npz"
        np.savez_compressed(adjacency_path, weighted_adjacency=model.weighted_adjacency)

    selected_keys_path = output_dir / "mmlu_cosmic_selected_artifacts.txt"
    selected_keys_path.write_text(
        "\n".join(row["artifact_key"] for row in selected_artifacts) + "\n",
        encoding="utf-8",
    )

    summary = build_summary(
        cfg=cfg,
        selected_artifacts=selected_artifacts,
        index_payload=index_payload,
        elapsed_sec=elapsed_sec,
        model=model,
        adjacency_path=adjacency_path,
    )

    summary_path = output_dir / "mmlu_cosmic_fit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nFit complete")
    print(f"Elapsed: {elapsed_sec:.2f}s")
    print(f"Resolved threshold: {model.resolved_threshold:.6f}")
    print(
        f"Graph: {summary['fit']['graph_nodes']:,} nodes, "
        f"{summary['fit']['graph_edges']:,} edges"
    )
    print(f"Summary: {summary_path}")
    if adjacency_path is not None:
        print(f"Weighted adjacency: {adjacency_path}")
    print(f"Selected artifact keys: {selected_keys_path}")


if __name__ == "__main__":
    main()