"""
Rich progress bar helpers for ThemaRS.fit() and fit_multi().

Requires the 'rich' package (already included in the 'demos' dependency group).
Install with: pip install rich
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pulsar.pipeline import ThemaRS


def fit_with_progress(
    model: "ThemaRS",
    data: pd.DataFrame | None = None,
    **fit_kwargs,
) -> "ThemaRS":
    """Run model.fit() with a transient rich progress bar.

    The bar disappears on completion, keeping notebook output clean.
    Uses the model's progress_callback mechanism — zero overhead on Rust stages.

    Args:
        model: Unfitted ThemaRS instance.
        data: Input DataFrame (optional if config specifies a data path).
        **fit_kwargs: Forwarded to model.fit() (e.g. _precomputed_embeddings).

    Returns:
        The fitted model (for method chaining).

    Raises:
        ImportError: If 'rich' is not installed.

    Example::

        from pulsar.pipeline import ThemaRS
        from pulsar.runtime.progress import fit_with_progress

        model = fit_with_progress(ThemaRS("params.yaml"))
        graph = model.cosmic_graph
    """
    try:
        from rich.progress import (
            BarColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )
    except ImportError:
        raise ImportError(
            "fit_with_progress() requires the 'rich' package. "
            "Install with: pip install rich"
        ) from None

    with Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        transient=True,  # bar disappears on completion — clean notebook output
    ) as progress:
        task = progress.add_task("Starting...", total=1.0)

        def callback(stage: str, fraction: float) -> None:
            progress.update(task, completed=fraction, description=f"[bold cyan]{stage}")

        return model.fit(data=data, progress_callback=callback, **fit_kwargs)


def fit_multi_with_progress(
    model: "ThemaRS",
    datasets: list[pd.DataFrame],
) -> "ThemaRS":
    """Run model.fit_multi() with a transient rich progress bar.

    Args:
        model: Unfitted ThemaRS instance.
        datasets: List of DataFrames (same points, different representations).

    Returns:
        The fitted model (for method chaining).

    Raises:
        ImportError: If 'rich' is not installed.
    """
    try:
        from rich.progress import (
            BarColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )
    except ImportError:
        raise ImportError(
            "fit_multi_with_progress() requires the 'rich' package. "
            "Install with: pip install rich"
        ) from None

    with Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Starting...", total=1.0)

        def callback(stage: str, fraction: float) -> None:
            progress.update(task, completed=fraction, description=f"[bold cyan]{stage}")

        return model.fit_multi(datasets, progress_callback=callback)
