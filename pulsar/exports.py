from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from pulsar.pipeline import ThemaRS

__all__ = [
    "clean_data",
    "cosmic_edges",
    "snapshot_edges",
    "node_table",
    "group_table",
    "export_dataset_bundle",
    "clean_embedding_at_center",
]

EDGE_SCHEMA = pa.schema(
    [
        ("source_id", pa.uint32(), False),
        ("target_id", pa.uint32(), False),
        ("weight", pa.float32(), False),
    ]
)

GROUP_SCHEMA = pa.schema(
    [
        ("group_id", pa.uint32(), False),
        ("name", pa.string(), False),
        ("desc", pa.string(), False),
        ("member_ids", pa.list_(pa.uint32()), False),
    ]
)

NODE_FIELD_TYPES = {
    "node_id": pa.uint32(),
    "group_id": pa.uint32(),
    "val": pa.float32(),
    "archetype": pa.string(),
    "ex": pa.float32(),
    "ey": pa.float32(),
    "ez": pa.float32(),
    "is_live": pa.bool_(),
}
NODE_RESERVED_FIELDS = set(NODE_FIELD_TYPES)


def _write_parquet(df: pd.DataFrame, schema: pa.Schema, path: Path) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False).cast(schema)
    pq.write_table(table, path)


def _cast_known_fields(
    table: pa.Table, field_types: dict[str, pa.DataType]
) -> pa.Table:
    """Cast only the named fields to fixed types, leaving other (e.g. passthrough) columns as inferred."""
    schema = table.schema
    for i, field in enumerate(schema):
        if field.name in field_types:
            schema = schema.set(
                i, pa.field(field.name, field_types[field.name], nullable=False)
            )
    return table.cast(schema)


def _node_passthrough_columns(
    raw_columns: list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return graph/nodes passthrough names without clobbering canonical fields."""
    used = set(NODE_RESERVED_FIELDS)
    passthrough: dict[str, str] = {}
    renamed: dict[str, str] = {}

    for col in raw_columns:
        output_col = col
        if output_col in used:
            base = f"raw_{col}"
            output_col = base
            suffix = 1
            while output_col in used or output_col in raw_columns:
                output_col = f"{base}_{suffix}"
                suffix += 1
            renamed[col] = output_col
        used.add(output_col)
        passthrough[output_col] = col

    return passthrough, renamed


def clean_data(model: ThemaRS) -> pd.DataFrame:
    """Preprocessed (imputed) table, row-aligned with the fitted graph.

    Returns the single representation produced by fit()'s preprocessing
    stage. TODO(#27): once fit_multi() generates multiple imputed
    representations of the same points by default, TODO(#28): use phil's
    ECT-based topological center selection to pick among them here.
    """
    if model.data is None:
        raise RuntimeError("Call fit() first")
    return model.preprocessed_data


def clean_embedding_at_center(model: ThemaRS) -> np.ndarray:
    """Scale/project clean_data() and return the embedding of the graph's center node."""
    G = model.cosmic_graph
    if len(G) == 0:
        raise RuntimeError("Cosmic graph has no nodes.")

    # 1. Scale & project the clean (preprocessed) data
    X_clean = clean_data(model).to_numpy(dtype=np.float64)
    from pulsar._pulsar import StandardScaler

    scaler = StandardScaler()
    X_clean_scaled = np.array(scaler.fit_transform(X_clean))

    from pulsar.pipeline import projection_grid

    clean_embs = projection_grid(X_clean_scaled, model.config)
    if not clean_embs:
        raise RuntimeError("Failed to generate embeddings.")

    # 2. Find topological center (highest weighted degree)
    center_idx = max(G.nodes, key=lambda n: G.degree(n, weight="weight"))

    # 3. Extract first projected embedding vector of the center node
    return clean_embs[0][center_idx]


def _weighted_edges_df(model: ThemaRS, threshold: float) -> pd.DataFrame:
    """Undirected edges (i, j) where i < j. Columns: source_id (UInt32), target_id (UInt32), weight (Float32)."""
    if model.cosmic_rust is None:
        raise RuntimeError("Call fit() first")
    edges = model.weighted_edges(threshold=threshold)
    df = pd.DataFrame(edges, columns=["source_id", "target_id", "weight"])
    df["source_id"] = df["source_id"].astype("uint32")
    df["target_id"] = df["target_id"].astype("uint32")
    df["weight"] = df["weight"].astype("float32")
    return df


def cosmic_edges(
    model: ThemaRS,
    *,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Export the full exploration graph. See `_weighted_edges_df` for the schema."""
    return _weighted_edges_df(model, threshold)


def snapshot_edges(
    model: ThemaRS,
    *,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Export the thresholded snapshot graph (default threshold = resolved_construction_threshold)."""
    cutoff = (
        model.resolved_construction_threshold if threshold is None else float(threshold)
    )
    return _weighted_edges_df(model, cutoff)


def _node_weighted_degree(
    model: ThemaRS, n: int, *, threshold: float | None = None
) -> np.ndarray:
    """Per-node sum of incident snapshot-edge weights."""
    val = np.zeros(n, dtype=np.float32)
    snapshot_df = snapshot_edges(model, threshold=threshold)
    if len(snapshot_df) > 0:
        sources = snapshot_df["source_id"].to_numpy()
        targets = snapshot_df["target_id"].to_numpy()
        weights = snapshot_df["weight"].to_numpy()
        np.add.at(val, sources, weights)
        np.add.at(val, targets, weights)
    return val


def _node_archetypes(
    group_ids: np.ndarray, cluster_names: dict[int, str] | None
) -> list[str]:
    """Semantic cluster name per node, falling back to "Cluster {group_id}"."""
    return [
        cluster_names[gid]
        if (cluster_names and gid in cluster_names)
        else f"Cluster {gid}"
        for gid in (int(g) for g in group_ids)
    ]


def _layout_coordinates(
    model: ThemaRS, n: int, layout: Literal["projection", "spectral", "zeros"]
) -> np.ndarray:
    """(n, 3) float32 layout coordinates for the requested layout mode."""
    if layout == "zeros":
        return np.zeros((n, 3), dtype=np.float32)

    if layout == "projection":
        embeddings = getattr(model, "_embeddings", None)
        if not embeddings:
            return np.zeros((n, 3), dtype=np.float32)
        fallback = embeddings[0]
        if fallback.shape[1] >= 3:
            return fallback[:, :3].astype(np.float32)
        padded = np.zeros((fallback.shape[0], 3), dtype=np.float32)
        padded[:, : fallback.shape[1]] = fallback
        return padded

    if layout == "spectral":
        coords = np.zeros((n, 3), dtype=np.float32)
        pos = nx.spectral_layout(model.cosmic_graph, dim=3, weight="weight")
        for i, xyz in pos.items():
            coords[i, : len(xyz)] = xyz
        return coords

    raise ValueError(f"Unsupported layout: {layout!r}")


def _normalize_keys(d: dict[Any, Any] | None) -> dict[int, Any] | None:
    """Normalize dictionary keys to integers where possible, supporting both int and str keys."""
    if d is None:
        return None
    normalized = {}
    for k, v in d.items():
        try:
            normalized[int(k)] = v
        except (ValueError, TypeError):
            normalized[k] = v
    return normalized


def node_table(
    model: ThemaRS,
    *,
    cluster_labels: pd.Series | np.ndarray,
    cluster_names: dict[int, str] | None = None,
    edges_threshold: float | None = None,
    layout: Literal["projection", "spectral", "zeros"] = "projection",
    extra_columns: dict[str, str] | list[str] | None = None,
) -> pd.DataFrame:
    """Build nodes.parquet-shaped frame.

    Columns:
        node_id: UInt32 (0 .. n-1)
        group_id: UInt32 (cluster label)
        val: Float32 (weighted degree on snapshot graph)
        archetype: Utf8 (semantic cluster name or "Cluster {group_id}")
        ex, ey, ez: Float32 (layout coords)
        is_live: Boolean (always true)
    """
    if model.cosmic_rust is None:
        raise RuntimeError("Call fit() first")
    n = model.cosmic_rust.n

    cluster_names = _normalize_keys(cluster_names)

    node_ids = np.arange(n, dtype=np.uint32)
    if isinstance(cluster_labels, pd.Series):
        group_ids = cluster_labels.to_numpy().astype(np.uint32)
    else:
        group_ids = np.ascontiguousarray(cluster_labels).astype(np.uint32)

    coords = _layout_coordinates(model, n, layout)

    df = pd.DataFrame(
        {
            "node_id": node_ids,
            "group_id": group_ids,
            "val": _node_weighted_degree(model, n, threshold=edges_threshold),
            "archetype": _node_archetypes(group_ids, cluster_names),
            "ex": coords[:, 0],
            "ey": coords[:, 1],
            "ez": coords[:, 2],
            "is_live": np.ones(n, dtype=bool),
        }
    )

    if extra_columns:
        raw_df = model.data
        column_map = (
            extra_columns
            if isinstance(extra_columns, dict)
            else _node_passthrough_columns(extra_columns)[0]
        )
        for output_col, source_col in column_map.items():
            if source_col in raw_df.columns:
                df[output_col] = raw_df[source_col].to_numpy()

    return df


def group_table(
    model: ThemaRS,
    *,
    cluster_labels: pd.Series | np.ndarray,
    cluster_names: dict[int, str] | None = None,
    cluster_descriptions: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Build groups.parquet-shaped frame.

    Columns:
        group_id: UInt32
        name: Utf8
        desc: Utf8
        member_ids: List<UInt32>
    """
    if model.cosmic_rust is None:
        raise RuntimeError("Call fit() first")
    if isinstance(cluster_labels, pd.Series):
        group_ids_arr = cluster_labels.to_numpy().astype(np.uint32)
    else:
        group_ids_arr = np.ascontiguousarray(cluster_labels).astype(np.uint32)

    cluster_names = _normalize_keys(cluster_names)
    cluster_descriptions = _normalize_keys(cluster_descriptions)

    unique_gids = sorted(list(set(int(gid) for gid in group_ids_arr)))

    records = []
    for gid in unique_gids:
        member_ids = np.where(group_ids_arr == gid)[0].tolist()
        name = (
            cluster_names[gid]
            if (cluster_names and gid in cluster_names)
            else f"Cluster {gid}"
        )
        desc = (
            cluster_descriptions[gid]
            if (cluster_descriptions and gid in cluster_descriptions)
            else ""
        )
        records.append(
            {
                "group_id": np.uint32(gid),
                "name": str(name),
                "desc": str(desc),
                "member_ids": member_ids,
            }
        )

    df = pd.DataFrame(records)
    if len(df) == 0:
        df = pd.DataFrame(columns=["group_id", "name", "desc", "member_ids"])
        df["group_id"] = df["group_id"].astype("uint32")
    return df


def export_dataset_bundle(
    model: ThemaRS,
    output_dir: Path | str,
    slug: str,
    *,
    cluster_labels: pd.Series | np.ndarray,
    cluster_names: dict[int, str] | None = None,
    cluster_descriptions: dict[int, str] | None = None,
    edges_threshold: float | None = None,
    layout: Literal["projection", "spectral", "zeros"] = "projection",
    include_clean: bool = False,
    write_manifest: bool = True,
) -> dict[str, Any]:
    """Write the full bundle under `{output_dir}/{slug}/`.

    Bundle layout:
        {slug}/
            tabular/
                raw.parquet
                clean.parquet  (optional)
            graph/
                nodes.parquet
                edges.parquet
                cosmic.parquet
                groups.parquet
            export_manifest.json  (optional)
    """
    base_path = Path(output_dir) / slug
    tabular_path = base_path / "tabular"
    graph_path = base_path / "graph"

    tabular_path.mkdir(parents=True, exist_ok=True)
    graph_path.mkdir(parents=True, exist_ok=True)

    if model.cosmic_rust is None:
        raise RuntimeError("Call fit() first")
    n = model.cosmic_rust.n

    cluster_names = _normalize_keys(cluster_names)
    cluster_descriptions = _normalize_keys(cluster_descriptions)

    # 1. Export tabular/raw.parquet
    raw_df = model.data.copy()
    if len(raw_df) != n:
        raise RuntimeError(f"raw data has {len(raw_df)} rows but graph has {n} nodes")
    extra_cols, renamed_node_columns = _node_passthrough_columns(list(raw_df.columns))
    if source_node_id := renamed_node_columns.get("node_id"):
        raw_df.rename(columns={"node_id": source_node_id}, inplace=True)
    raw_df.insert(0, "node_id", np.arange(n, dtype=np.uint32))
    raw_df.to_parquet(tabular_path / "raw.parquet", index=False)

    # 2. Export tabular/clean.parquet (optional)
    clean_path = tabular_path / "clean.parquet"
    if include_clean:
        clean_df = clean_data(model).copy()
        if len(clean_df) != n:
            raise RuntimeError(
                f"clean_data has {len(clean_df)} rows but graph has {n} nodes"
            )
        if "node_id" in clean_df.columns:
            clean_renames = _node_passthrough_columns(list(clean_df.columns))[1]
            clean_df.rename(columns={"node_id": clean_renames["node_id"]}, inplace=True)
        clean_df.insert(0, "node_id", np.arange(n, dtype=np.uint32))
        clean_df.to_parquet(clean_path, index=False)
    else:
        clean_path.unlink(missing_ok=True)

    # 3. Export graph/cosmic.parquet
    cosmic_df = cosmic_edges(model, threshold=0.0)
    _write_parquet(cosmic_df, EDGE_SCHEMA, graph_path / "cosmic.parquet")

    # 4. Export graph/edges.parquet
    edges_df = snapshot_edges(model, threshold=edges_threshold)
    _write_parquet(edges_df, EDGE_SCHEMA, graph_path / "edges.parquet")

    # 5. Export graph/nodes.parquet
    nodes_df = node_table(
        model,
        cluster_labels=cluster_labels,
        cluster_names=cluster_names,
        edges_threshold=edges_threshold,
        layout=layout,
        extra_columns=extra_cols,
    )
    nodes_table = _cast_known_fields(
        pa.Table.from_pandas(nodes_df, preserve_index=False), NODE_FIELD_TYPES
    )
    pq.write_table(nodes_table, graph_path / "nodes.parquet")

    # 6. Export graph/groups.parquet
    groups_df = group_table(
        model,
        cluster_labels=cluster_labels,
        cluster_names=cluster_names,
        cluster_descriptions=cluster_descriptions,
    )
    _write_parquet(groups_df, GROUP_SCHEMA, graph_path / "groups.parquet")

    # 7. Write export_manifest.json (optional)
    manifest_data = {}
    manifest_path = base_path / "export_manifest.json"
    if write_manifest:
        try:
            import importlib.metadata as importlib_metadata

            version = importlib_metadata.version("thema-pulsar")
        except Exception:
            version = "0.1.0"

        config_payload = json.dumps(
            asdict(model.config), sort_keys=True, separators=(",", ":")
        )
        config_hash = hashlib.sha256(config_payload.encode("utf-8")).hexdigest()

        manifest_data = {
            "pulsar_version": version,
            "config_hash": config_hash,
            "thresholds": {
                "resolved_construction_threshold": float(
                    model.resolved_construction_threshold
                ),
                "edges_threshold": float(edges_threshold)
                if edges_threshold is not None
                else float(model.resolved_construction_threshold),
            },
            "phil_config": {
                "enabled": include_clean,
            },
            "counts": {
                "nodes": int(model.cosmic_rust.n),
                "edges": int(len(edges_df)),
                "cosmic_edges": int(len(cosmic_df)),
                "groups": int(len(groups_df)),
            },
            "node_passthrough_column_renames": renamed_node_columns,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=2)
    else:
        manifest_path.unlink(missing_ok=True)

    return manifest_data
