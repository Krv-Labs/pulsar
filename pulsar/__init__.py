"""Pulsar — a Rust-backed implementation of the Thema pipeline.

Public names are resolved lazily (PEP 562). Importing this package used to pull
in scikit-learn, SciPy, pandas, pyarrow, and NetworkX eagerly — about 1.6s —
which every consumer paid whether or not it touched them. The CLI
(`pulsar install`, `pulsar status`) needs none of that stack, but lives under
this package, so `pulsar --help` paid the full cost too.

Attribute access is unchanged: `from pulsar import ThemaRS`, `import pulsar;
pulsar.PCA`, and `from pulsar import *` all behave as before, and the first
touch of a name imports its module. Only the timing moves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# name -> module that defines it. Kept explicit rather than derived so a typo
# surfaces as a normal AttributeError at the boundary instead of a confusing
# ImportError from deep inside a submodule.
_EXPORTS = {
    "__version__": "pulsar._version",
    # Preprocessing
    "impute_column": "pulsar._pulsar",
    "StandardScaler": "pulsar._pulsar",
    # Dimensionality reduction
    "JLProjection": "pulsar._pulsar",
    "jl_grid": "pulsar._pulsar",
    "PCA": "pulsar._pulsar",
    "pca_grid": "pulsar._pulsar",
    # Ball Mapper
    "BallMapper": "pulsar._pulsar",
    "ball_mapper_grid": "pulsar._pulsar",
    # Pseudo-Laplacian accumulation
    "accumulate_pseudo_laplacians": "pulsar._pulsar",
    "accumulate_pseudo_laplacians_sparse": "pulsar._pulsar",
    "SparsePseudoLaplacian": "pulsar._pulsar",
    "accumulate_temporal_pseudo_laplacians": "pulsar._pulsar",
    # Cosmic Graph
    "CosmicGraph": "pulsar._pulsar",
    # Threshold stability
    "find_stable_thresholds": "pulsar._pulsar",
    "find_stable_thresholds_sparse": "pulsar._pulsar",
    # Representations
    "TemporalCosmicGraph": "pulsar.representations",
    "CosmicTrajectory": "pulsar.representations",
    # Config
    "PulsarConfig": "pulsar.config",
    "load_config": "pulsar.config",
    "config_to_yaml": "pulsar.config",
    # Pipeline
    "ThemaRS": "pulsar.pipeline",
    # Hooks / utilities
    "label_points": "pulsar.analysis",
    "membership_matrix": "pulsar.analysis",
    "cosmic_clusters": "pulsar.analysis",
    "graph_to_dataframe": "pulsar.analysis",
    "unclustered_points": "pulsar.analysis",
    "cosmic_to_networkx": "pulsar.analysis",
    # Analysis
    "characterize_dataset": "pulsar.analysis",
    "ColumnProfile": "pulsar.analysis",
    "DatasetProfile": "pulsar.analysis",
}

# The one name whose exported spelling differs from its source spelling.
_RENAMED = {
    "normalize_temporal_laplacian": (
        "pulsar._pulsar",
        "py_normalize_temporal_laplacian",
    )
}

__all__ = [*_EXPORTS, *_RENAMED]


def __getattr__(name: str):
    import importlib

    if name in _RENAMED:
        module_name, attribute = _RENAMED[name]
    elif name in _EXPORTS:
        module_name, attribute = _EXPORTS[name], name
    else:
        raise AttributeError(f"module 'pulsar' has no attribute {name!r}")

    value = getattr(importlib.import_module(module_name), attribute)
    # Cache on the module so subsequent lookups skip __getattr__ entirely.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    # Re-stated eagerly for type checkers and IDE completion, which cannot
    # follow the lazy lookup above. Never executed at runtime. The redundant
    # `X as X` spelling is the PEP 484 re-export convention — without it these
    # read as unused imports, since __all__ is built from _EXPORTS at runtime
    # and no static analyser can see the names in it.
    from pulsar._pulsar import (
        PCA as PCA,
        BallMapper as BallMapper,
        CosmicGraph as CosmicGraph,
        JLProjection as JLProjection,
        SparsePseudoLaplacian as SparsePseudoLaplacian,
        StandardScaler as StandardScaler,
        accumulate_pseudo_laplacians as accumulate_pseudo_laplacians,
        accumulate_pseudo_laplacians_sparse as accumulate_pseudo_laplacians_sparse,
        accumulate_temporal_pseudo_laplacians as accumulate_temporal_pseudo_laplacians,
        ball_mapper_grid as ball_mapper_grid,
        find_stable_thresholds as find_stable_thresholds,
        find_stable_thresholds_sparse as find_stable_thresholds_sparse,
        impute_column as impute_column,
        jl_grid as jl_grid,
        pca_grid as pca_grid,
        # The one export whose name differs from its source, so the redundant
        # alias trick does not apply.
        py_normalize_temporal_laplacian as normalize_temporal_laplacian,  # noqa: F401
    )
    from pulsar._version import __version__ as __version__
    from pulsar.analysis import (
        ColumnProfile as ColumnProfile,
        DatasetProfile as DatasetProfile,
        characterize_dataset as characterize_dataset,
        cosmic_clusters as cosmic_clusters,
        cosmic_to_networkx as cosmic_to_networkx,
        graph_to_dataframe as graph_to_dataframe,
        label_points as label_points,
        membership_matrix as membership_matrix,
        unclustered_points as unclustered_points,
    )
    from pulsar.config import (
        PulsarConfig as PulsarConfig,
        config_to_yaml as config_to_yaml,
        load_config as load_config,
    )
    from pulsar.pipeline import ThemaRS as ThemaRS
    from pulsar.representations import (
        CosmicTrajectory as CosmicTrajectory,
        TemporalCosmicGraph as TemporalCosmicGraph,
    )
