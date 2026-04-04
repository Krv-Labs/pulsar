"""
FastMCP Server for Pulsar.

Exposes "Thick Tools" for topological data analysis and interpretation.
"""

import asyncio
import dataclasses
from dataclasses import dataclass
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError

from pulsar.config import config_to_yaml, load_config
from pulsar.preprocessing import preprocess_dataframe
from pulsar.runtime.fingerprint import pca_fingerprint
from pulsar.pipeline import ThemaRS
from pulsar.mcp.interpreter import (
    resolve_clusters,
    build_dossier,
    dossier_to_markdown,
    compare_clusters,
    comparison_to_markdown,
)
from pulsar.mcp.errors import mcp_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defensive patch: strip unknown kwargs from non-compliant MCP clients.
# Some clients (e.g. Gemini CLI) inject orchestration fields like
# ``wait_for_previous`` into tool calls.  FastMCP's Pydantic validation
# rejects these.  Patching FunctionTool.run at the class level filters
# them out *before* validation — one patch protects every tool.
# ---------------------------------------------------------------------------
from fastmcp.tools.function_tool import FunctionTool  # noqa: E402

_original_function_tool_run = FunctionTool.run


async def _lenient_function_tool_run(self, arguments, context=None):
    if arguments:
        valid_keys = set(self.parameters.get("properties", {}).keys())
        arguments = {k: v for k, v in arguments.items() if k in valid_keys}
    return await _original_function_tool_run(self, arguments, context)


FunctionTool.run = _lenient_function_tool_run


# ---------------------------------------------------------------------------
# Preprocessing recommendation helpers
# ---------------------------------------------------------------------------


def _try_float(s: str) -> bool:
    """Return True if s can be parsed as a float."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _looks_like_dirty_numeric(sample_values: List[str]) -> bool:
    """Return True if majority of sample values parse as float.

    Used to detect columns where string sentinels (e.g. 'N/A') caused pandas
    to cast an otherwise numeric column to object dtype.
    """
    if not sample_values:
        return False
    parseable = sum(1 for v in sample_values if _try_float(v))
    return parseable / len(sample_values) > 0.5


def _recommend_preprocessing_block(
    column_profiles: List[Any],
    n_samples: int,
) -> tuple[list[str], dict[str, Any], dict[str, Any], list[tuple[str, str, str]]]:
    """Apply decision tree to column profiles and return preprocessing components.

    Returns:
        (drop_columns, impute_dict, encode_dict, rationale_rows)
        where rationale_rows is list of (column, decision_label, rationale).
    """
    drop: list[str] = []
    impute: dict[str, Any] = {}
    encode: dict[str, Any] = {}
    rationale: list[tuple[str, str, str]] = []

    for cp in column_profiles:
        name = cp["name"] if isinstance(cp, dict) else cp.name
        is_numeric = cp["is_numeric"] if isinstance(cp, dict) else cp.is_numeric
        n_unique = cp["n_unique"] if isinstance(cp, dict) else cp.n_unique
        n_missing = cp["n_missing"] if isinstance(cp, dict) else cp.n_missing
        missing_pct = cp["missing_pct"] if isinstance(cp, dict) else cp.missing_pct
        sample_values = (
            cp["sample_values"] if isinstance(cp, dict) else cp.sample_values
        )

        # Rule 1: All-missing
        if missing_pct >= 100.0:
            drop.append(name)
            rationale.append((name, "drop", "All values are missing — cannot impute"))
            continue

        if not is_numeric:
            # Rule 2: Dirty numeric — object column where values are mostly parseable as float
            if _looks_like_dirty_numeric(sample_values):
                if missing_pct >= 30:
                    impute[name] = {"method": "sample_normal", "seed": 42}
                    rationale.append(
                        (
                            name,
                            "impute: sample_normal",
                            f"Dirty numeric ({missing_pct:.0f}% missing); string sentinels detected — coercion will rescue it",
                        )
                    )
                else:
                    impute[name] = {"method": "fill_mean", "seed": 42}
                    rationale.append(
                        (
                            name,
                            "impute: fill_mean",
                            f"Dirty numeric ({missing_pct:.0f}% missing); string sentinels detected — coercion will rescue it",
                        )
                    )
                continue

            # Rule 3: High-cardinality ID
            if n_samples > 0 and n_unique / n_samples > 0.9:
                drop.append(name)
                rationale.append(
                    (
                        name,
                        "drop",
                        f"ID-like column ({n_unique}/{n_samples} unique) — no topological signal",
                    )
                )
                continue

            # Rule 4: Too many categories to safely one-hot
            if n_unique > 50:
                drop.append(name)
                rationale.append(
                    (
                        name,
                        "drop",
                        f"{n_unique} categories would add {n_unique} dimensions — distorts Euclidean distance",
                    )
                )
                continue

            # Rule 5: Medium cardinality — cap at actual count so the
            # validation gate (n_cats > max_categories) passes on first try.
            if n_unique > 20:
                encode[name] = {"method": "one_hot", "max_categories": n_unique}
                if n_missing > 0:
                    impute[name] = {"method": "sample_categorical", "seed": 42}
                rationale.append(
                    (
                        name,
                        f"encode: one_hot (max {n_unique})",
                        f"{n_unique} categories — cap set to actual count to avoid validation failure",
                    )
                )
                continue

            # Rule 6: Binary — fill_mode is safe
            if n_unique == 2:
                encode[name] = {"method": "one_hot"}
                if n_missing > 0:
                    impute[name] = {"method": "fill_mode", "seed": 42}
                rationale.append(
                    (
                        name,
                        "encode: one_hot",
                        f"Binary column ({n_unique} values)"
                        + (
                            f"; impute: fill_mode ({n_missing} missing)"
                            if n_missing > 0
                            else ""
                        ),
                    )
                )
                continue

            # Rule 7: Low cardinality categorical
            encode[name] = {"method": "one_hot"}
            if n_missing > 0:
                impute[name] = {"method": "sample_categorical", "seed": 42}
            rationale.append(
                (
                    name,
                    "encode: one_hot",
                    f"{n_unique} unique values — safe cardinality"
                    + (
                        f"; impute: sample_categorical ({n_missing} missing)"
                        if n_missing > 0
                        else ""
                    ),
                )
            )
            continue

        # Numeric column
        if missing_pct >= 30:
            impute[name] = {"method": "sample_normal", "seed": 42}
            rationale.append(
                (
                    name,
                    "impute: sample_normal",
                    f"Numeric, {missing_pct:.0f}% missing — sample_normal preserves distribution shape",
                )
            )
        elif missing_pct > 0:
            impute[name] = {"method": "fill_mean", "seed": 42}
            rationale.append(
                (
                    name,
                    "impute: fill_mean",
                    f"Numeric, {missing_pct:.0f}% missing — low missingness, mean fill is stable",
                )
            )
        else:
            rationale.append(
                (name, "no action", "Numeric, complete — no preprocessing needed")
            )

    # Safety net: catch columns with missing values that slipped through
    # (e.g. non-numeric columns routed to drop by rules 3-4, or edge cases
    # in the dirty-numeric detection).
    for cp in column_profiles:
        name = cp["name"] if isinstance(cp, dict) else cp.name
        n_missing = cp["n_missing"] if isinstance(cp, dict) else cp.n_missing
        col_is_numeric = cp["is_numeric"] if isinstance(cp, dict) else cp.is_numeric
        if n_missing > 0 and name not in impute and name not in drop:
            if col_is_numeric:
                impute[name] = {"method": "fill_mean", "seed": 42}
            else:
                impute[name] = {"method": "fill_mode", "seed": 42}
            rationale.append(
                (
                    name,
                    f"impute: {'fill_mean' if col_is_numeric else 'fill_mode'}",
                    f"Safety net — {n_missing} missing values not covered by primary rules",
                )
            )

    return drop, impute, encode, rationale


def _preprocessing_block_to_yaml(
    drop: list[str],
    impute: dict[str, Any],
    encode: dict[str, Any],
) -> str:
    """Render drop/impute/encode dicts as a preprocessing: YAML block."""
    lines = ["preprocessing:"]
    lines.append(f"  drop_columns: {json.dumps(drop)}")
    if impute:
        lines.append("  impute:")
        for col, spec in impute.items():
            lines.append(
                f"    {col}: {{method: {spec['method']}, seed: {spec.get('seed', 42)}}}"
            )
    else:
        lines.append("  impute: {}")
    if encode:
        lines.append("  encode:")
        for col, spec in encode.items():
            parts = f"method: {spec['method']}"
            if "max_categories" in spec:
                parts += f", max_categories: {spec['max_categories']}"
            lines.append(f"    {col}: {{{parts}}}")
    else:
        lines.append("  encode: {}")
    return "\n".join(lines)


def _rationale_table(rows: list[tuple[str, str, str]]) -> str:
    """Render rationale rows as a markdown table."""
    lines = ["| Column | Decision | Rationale |", "|---|---|---|"]
    for col, decision, reason in rows:
        lines.append(f"| `{col}` | {decision} | {reason} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Initialize FastMCP
mcp = FastMCP(
    "Pulsar",
    instructions=(
        "You are a topological cartographer. Your job is to reveal the shape of the data, not impose one.\n\n"
        "# Your Mental Model\n"
        "- **Manifold Recovery, Not Clustering:** You are mapping the continuous geometric structure of the dataset. "
        "Do not force data into neat, disconnected clusters if they do not organically exist.\n"
        "- **The Cosmic Graph:** This is your primary artifact. It is the aggregated topological signal of the data across multiple spatial scales.\n\n"
        "# Core Principles\n"
        "1. **Dimensionality vs. Scale:** Higher dimensions compress Euclidean distance into a narrow band. This requires a larger epsilon "
        "to capture local neighborhoods, which rapidly leads to a massive, uninformative 'blob' (a fully connected component). "
        "Manage dimensionality (e.g., PCA dims) tightly before aggressively tuning your scale (epsilon) parameters.\n"
        "2. **Respect Categoricals:** Categorical variables (e.g., island, group) add rigid, orthogonal structure to the manifold. "
        "They are not noise. Never drop them simply to force spatial separation. If the reality of the data is partitioned, the topology must reflect that.\n"
        "3. **Variance Curve:** Look at the cumulative variance curve from characterize_dataset. If Dim 3 captures 94% variance, using Dim 5 "
        "adds almost no signal but drastically dilutes distances (curse of dimensionality).\n"
        "4. **Good Graph vs Convenient Graph:** A 'good' graph has structural balance. Look for a `giant_fraction` (the largest component) "
        "typically between 20% and 60%. These metrics apply to the **final aggregated ensemble graph**. It is not necessarily the graph that perfectly fits a preconceived notion of 'K=3 clusters'.\n"
        "5. **Embrace the Grid (Multi-Scale):** Pulsar is an ensemble method. Do NOT try to find the 'one true epsilon' or 'one best PCA dimension'. "
        "The graph gains expressive power by accumulating topology across a filtration. Always prefer ranges for epsilon and arrays for PCA dimensions. "
        "(Exception: You may use single values only for isolated diagnostic runs to 'find the floor' of a massive blob.)\n\n"
        "# The Scientific Posture\n"
        "Treat the cosmic graph as empirical evidence, not a score to maximize.\n"
        "- **Form a hypothesis** before adjusting any configuration.\n"
        "- **Change ONE parameter at a time.** Note: An entire Epsilon Range or PCA Array counts as ONE parameter change.\n"
        "- **Observe the exact structural impact** on the cosmic graph (use the diff provided by run_topological_sweep), and re-assess your hypothesis.\n\n"
        "# Workflow\n"
        "1. characterize_dataset -> suggest_initial_config (includes preprocessing) -> validate_preprocessing_config\n"
        "2. run_topological_sweep -> diagnose_cosmic_graph\n"
        "3. **Iterate:** Change ONE parameter (as defined above). After each sweep, review the diff.\n"
        "4. **Blob Recovery:** If run 1 produces a massive blob (`giant_fraction` > 80%), call `explain_suggestion` to understand the initial reasoning. "
        "Usually, you must shift your epsilon range lower OR reduce the PCA dimension array to break the blob.\n"
        "5. **Shatter Recovery:** If the graph is fragmented (`component_count` > 20 and `giant_fraction` < 5%), your epsilon range is too small or threshold too aggressive. "
        "Increase epsilon bounds modestly (10-20%) or reduce the threshold (e.g., change 'auto' to a float like 0.3) before any other change.\n"
        "6. generate_cluster_dossier -> compare_clusters_tool\n\n"
        "# Preprocessing Error Handling\n"
        "When `run_topological_sweep` returns a preprocessing error (ValueError, TypeError):\n"
        "1. Call `repair_preprocessing_config(error_message, config_yaml, dataset_geometry)` — pass the exact error text, the failing YAML, and the JSON from `characterize_dataset`.\n"
        "2. Replace your `config_yaml` with the patched version from the **Patched Config** block.\n"
        "3. Call `validate_preprocessing_config(patched_config_yaml)` — must return PASS before re-running the sweep.\n"
        "4. If validation surfaces a new error, repeat from step 1. Preprocessing errors rarely stack more than 2 levels.\n"
        "5. **Never call `run_topological_sweep` on a config that `validate_preprocessing_config` has rejected.**\n\n"
        "## Cold Start Preprocessing\n"
        "`suggest_initial_config` already embeds preprocessing recommendations. Always call "
        "`validate_preprocessing_config` before the first `run_topological_sweep` to surface any issues without burning sweep time. "
        "If you want a standalone explanation of each column decision, call `recommend_preprocessing(dataset_geometry)`.\n\n"
        "## Preprocessing Rules\n"
        "- Never hand-write impute/encode YAML from raw column stats — use `recommend_preprocessing` or `repair_preprocessing_config`.\n"
        "- Valid impute methods: `fill_mean`, `fill_median`, `fill_mode`, `sample_normal`, `sample_categorical`.\n"
        "- Valid encode methods: `one_hot` only.\n"
        "- Object columns with `n_unique > 50` should be dropped, not encoded — one-hot at that cardinality adds 50+ orthogonal dimensions and destroys Euclidean distance structure.\n"
        "- Use `sample_categorical` (not `fill_mode`) for missing categoricals unless the column is binary — `fill_mode` collapses all missing values to a single point, distorting the manifold.\n"
    ),
)


@dataclass
class SweepRecord:
    timestamp: float
    config_yaml: str
    metrics: dict


# Session state: stores (model, data, clusters) per session
@dataclasses.dataclass
class _PulsarSession:
    """Session state for a single MCP client."""

    model: Optional[ThemaRS] = None
    data: Optional[pd.DataFrame] = None
    clusters: Optional[pd.Series] = None
    embeddings: Optional[list] = None  # cached PCA output from last fit
    pca_fingerprint: Optional[str] = None  # SHA256 of (data_path, dims, seeds, n_rows)
    sweep_history: list[SweepRecord] = dataclasses.field(default_factory=list)


# Global session storage, keyed by session_id (or "default" for STDIO)
# STDIO transport assumption: mcp.run() with no transport argument defaults to STDIO.
# Under STDIO, each process serves exactly one client. _sessions will contain at most
# one entry, keyed "default". If ported to SSE/WebSocket (multi-client), replace this
# with a bounded LRU dict capped at MAX_SESSIONS.
_sessions: Dict[str, _PulsarSession] = {}
_MAX_SESSIONS = 1


def _session_key(ctx: Context | None) -> str:
    """Get the session key from context (session_id or 'default' for STDIO)."""
    if ctx is None:
        return "default"
    return ctx.session_id or "default"


def _get_session(ctx: Context) -> _PulsarSession:
    """Get or create session state for the current client."""
    key = _session_key(ctx)
    if key not in _sessions:
        if len(_sessions) >= _MAX_SESSIONS:
            logger.warning(
                "_sessions has %d entries; expected %d under STDIO transport. "
                "If using a multi-client transport, this server needs LRU eviction.",
                len(_sessions),
                _MAX_SESSIONS,
            )
        _sessions[key] = _PulsarSession()
    return _sessions[key]


def _format_epsilon(cfg: dict) -> str:
    """Format epsilon config as a display string, handling both range and values shapes."""
    eps_node = cfg.get("sweep", {}).get("ball_mapper", {}).get("epsilon", {})
    if "range" in eps_node:
        r = eps_node["range"]
        return f"[{r.get('min', 0):.3f}, {r.get('max', 0):.3f}]"
    elif "values" in eps_node:
        return str(eps_node["values"])
    return "n/a"


def _validate_config_path(path: str) -> None:
    """Validate that config file exists and is a YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    if not path.endswith((".yaml", ".yml")):
        raise ValueError(f"Config must be a YAML file (*.yaml or *.yml), got: {path}")


def _auto_save_config(cfg) -> str:
    """Save resolved config to disk for reproducibility. Returns saved path."""
    name = cfg.run_name or "pulsar"
    if cfg.data and os.path.isfile(cfg.data):
        save_dir = os.path.dirname(os.path.abspath(cfg.data))
    else:
        save_dir = os.getcwd()
    save_path = os.path.join(save_dir, f"{name}_params.yaml")
    with open(save_path, "w") as f:
        f.write(config_to_yaml(cfg))
    logger.info("Config saved to %s", save_path)
    return save_path


@mcp.tool()
async def suggest_initial_config(dataset_geometry: str, ctx: Context) -> str:
    """
    Generate an initial configuration YAML based on the raw dataset geometry.

    Args:
        dataset_geometry: The raw JSON string from characterize_dataset.

    Returns:
        A starter config_yaml string.
    """
    try:
        geo = json.loads(dataset_geometry)

        # Handle both raw DatasetProfile and manual summaries
        knn_mean = geo.get("knn_k5_mean") or geo.get("knn_mean") or 0.5
        pca_cum_var = geo.get("pca_cumulative_variance", [])

        # Heuristic for PCA knee if not provided
        pca_knee = 3
        if pca_cum_var:
            # Simple elbow heuristic: find where variance > 90%
            for dim, var in pca_cum_var:
                if var >= 0.90:
                    pca_knee = dim
                    break

        # Generate an ensemble array around the knee to encourage multi-scale mapping
        pca_dims = [max(2, pca_knee - 1), pca_knee, pca_knee + 1]

        eps_min = knn_mean * 0.8
        eps_max = knn_mean * 1.5

        # Build preprocessing block from column profiles
        n_samples = geo.get("n_samples", 0)
        column_profiles = geo.get("column_profiles", [])
        drop, impute, encode, _ = _recommend_preprocessing_block(
            column_profiles, n_samples
        )
        preprocessing_block = _preprocessing_block_to_yaml(drop, impute, encode)

        yaml_str = f"""run:
  name: initial_sweep
  data: "FILL_THIS_IN_WITH_PATH"
{preprocessing_block}
sweep:
  pca:
    dimensions:
      values: {pca_dims}
    seed:
      values: [42]
  ball_mapper:
    epsilon:
      range:
        min: {eps_min:.4f}
        max: {eps_max:.4f}
        steps: 15
cosmic_graph:
  threshold: auto
output:
  n_reps: 4
"""
        return yaml_str
    except Exception as e:
        return mcp_error("suggest_initial_config", str(e))


@mcp.tool()
async def explain_suggestion(
    config_yaml: str, dataset_geometry: str, ctx: Context
) -> str:
    """
    Explains the mathematical reasoning behind a specific parameter suggestion based on raw geometry.

    Args:
        config_yaml: The YAML config to explain.
        dataset_geometry: JSON string of the dataset geometry summary.

    Returns:
        A Markdown explanation of WHY these parameters were chosen.
    """
    try:
        geo = json.loads(dataset_geometry)
        config_dict = yaml.safe_load(config_yaml)

        pca_dims = (
            config_dict.get("sweep", {})
            .get("pca", {})
            .get("dimensions", {})
            .get("values", [])
        )
        eps_str = _format_epsilon(config_dict)

        n_samples = geo.get("n_samples", "unknown")
        cum_var_map = {int(d): v for d, v in geo.get("pca_cumulative_variance", [])}
        knn_mean = geo.get("knn_k5_mean") or geo.get("knn_mean")

        explanation = "### Parameter Reasoning\n\n"

        # 1. PCA Reasoning
        pca_reasons = []
        for dim in pca_dims:
            var = cum_var_map.get(int(dim))
            if var is not None:
                next_var = cum_var_map.get(int(dim) + 1)
                benefit = f" (+{next_var - var:.1%})" if next_var else ""
                pca_reasons.append(f"Dim {dim} captures {var:.1%} variance{benefit}.")

        if pca_reasons:
            explanation += f"- **PCA Dimensions {pca_dims}**: " + " ".join(pca_reasons)
            explanation += " An array of dimensions is used rather than a single point estimate to prevent dimension collapse and ensure the CosmicGraph captures topology across varying geometric resolutions.\n"
            if isinstance(n_samples, int):
                pts_per_dim = n_samples / max(pca_dims)
                explanation += f" With N={n_samples}, this maintains ~{pts_per_dim:.1f} points per dimension, ensuring sufficient manifold density.\n"
            else:
                explanation += "\n"
        else:
            explanation += f"- **PCA Dimensions {pca_dims}**: Chosen as a multi-scale array around the variance curve elbow to aggregate varying topological resolutions (values not provided in geo summary).\n"

        # 2. Epsilon Reasoning
        if knn_mean:
            eps_node = (
                config_dict.get("sweep", {}).get("ball_mapper", {}).get("epsilon", {})
            )
            if "range" in eps_node:
                r = eps_node["range"]
                e_min, e_max = r.get("min", 0), r.get("max", 0)
                explanation += f"- **Epsilon Range {eps_str}**: Anchored at knn_mean={knn_mean:.4f}. The range spans {e_min / knn_mean:.2f}x to {e_max / knn_mean:.2f}x the mean distance. This filtration sweeps from local neighborhoods to global structures, aggregating multi-scale persistent homology into the final graph.\n"
            else:
                explanation += f"- **Epsilon {eps_str}**: Evaluated relative to knn_mean={knn_mean:.4f}. Note: A single epsilon limits the graph to a single scale. Consider sweeping a range to capture ensemble topology.\n"
        else:
            explanation += f"- **Epsilon {eps_str}**: Search window centered around k-NN mean (knn_mean not provided in summary).\n"

        return explanation
    except Exception as e:
        return mcp_error("explain_suggestion", str(e))


@mcp.tool()
async def get_experiment_history(ctx: Context) -> str:
    """
    Returns a markdown table of all topological sweeps run in the current session.
    Use this to reason about your trajectory across multiple iterations.

    Returns:
        Markdown table of history. Returns an empty table if no sweeps have been run.
    """
    session = _get_session(ctx)
    if not session.sweep_history:
        return "No experiments run yet in this session.\n\n| Run | PCA Dims | Epsilon Range | Nodes | Edges | Components | Giant Fraction |\n|---|---|---|---|---|---|---|"

    lines = [
        "| Run | PCA Dims | Epsilon Range | Nodes | Edges | Components | Giant Fraction |"
    ]
    lines.append("|---|---|---|---|---|---|---|")

    for i, record in enumerate(session.sweep_history):
        cfg = yaml.safe_load(record.config_yaml)
        pca = str(
            cfg.get("sweep", {}).get("pca", {}).get("dimensions", {}).get("values", [])
        )
        eps = _format_epsilon(cfg)

        m = record.metrics
        lines.append(
            f"| {i + 1} | {pca} | {eps} | {m.get('n_nodes')} | {m.get('n_edges')} | {m.get('component_count')} | {m.get('giant_fraction', 0):.2%} |"
        )

    return "\n".join(lines)


@mcp.tool()
async def run_topological_sweep(
    config_path: str = "",
    config_yaml: str = "",
    ctx: Context = None,
) -> str:
    """
    Run the Pulsar topological sweep pipeline on a dataset.

    Returns a markdown diff of parameter and metric changes compared to your previous run,
    followed by the full execution summary.

    Args:
        config_path: Path to a params.yaml file on disk.
        config_yaml: Inline YAML string (preferred).
    """
    if config_yaml:
        config_dict = yaml.safe_load(config_yaml)
        if not isinstance(config_dict, dict):
            raise ToolError("config_yaml must be a valid YAML mapping")
        current_yaml = config_yaml
    elif config_path:
        _validate_config_path(config_path)
        with open(config_path) as f:
            current_yaml = f.read()
    else:
        raise ToolError("Provide either config_path or config_yaml.")

    session = _get_session(ctx)

    try:
        if config_yaml:
            logger.info("Starting topological sweep from inline YAML")
            model = ThemaRS(config_dict)
        else:
            logger.info("Starting topological sweep for: %s", config_path)
            model = ThemaRS(config_path)

        cfg = model.config

        precomputed = None
        if session.embeddings is not None and session.data is not None:
            fingerprint = pca_fingerprint(cfg, len(session.data))
            if fingerprint == session.pca_fingerprint:
                precomputed = session.embeddings
                logger.info("Reusing cached PCA embeddings (fingerprint match)")

        loop = asyncio.get_running_loop()

        def progress_callback(stage: str, fraction: float) -> None:
            if ctx is None:
                return
            try:
                asyncio.run_coroutine_threadsafe(
                    ctx.report_progress(progress=fraction, total=1.0, message=stage),
                    loop,
                )
            except RuntimeError:
                pass

        await asyncio.to_thread(
            model.fit,
            _precomputed_embeddings=precomputed,
            progress_callback=progress_callback,
        )

        session.model = model
        session.data = model.data  # raw pre-preprocessing DataFrame

        if precomputed is None:
            session.embeddings = model._embeddings
            session.pca_fingerprint = pca_fingerprint(cfg, len(model.data))

        saved_path = _auto_save_config(cfg)
        graph = model.cosmic_graph

        # Calculate metrics for diff
        from pulsar.mcp.diagnostics import diagnose_model

        current_metrics_obj = diagnose_model(model)
        current_metrics = dataclasses.asdict(current_metrics_obj)

        # Build Diff block
        diff_block = (
            "### Experiment Diff\n\n| Parameter | Previous | Current |\n|---|---|---|\n"
        )
        if not session.sweep_history:
            diff_block += "| (Initial Run) | - | - |\n"
        else:
            prev_record = session.sweep_history[-1]
            prev_cfg = yaml.safe_load(prev_record.config_yaml)
            curr_cfg = yaml.safe_load(current_yaml)

            p_pca = str(
                prev_cfg.get("sweep", {})
                .get("pca", {})
                .get("dimensions", {})
                .get("values", [])
            )
            c_pca = str(
                curr_cfg.get("sweep", {})
                .get("pca", {})
                .get("dimensions", {})
                .get("values", [])
            )
            if p_pca != c_pca:
                diff_block += f"| pca_dims | {p_pca} | {c_pca} |\n"

            p_eps = _format_epsilon(prev_cfg)
            c_eps = _format_epsilon(curr_cfg)
            if p_eps != c_eps:
                diff_block += f"| epsilon | {p_eps} | {c_eps} |\n"

            pm = prev_record.metrics
            cm = current_metrics
            diff_block += (
                f"| Edges | {pm.get('n_edges', 0):,} | {cm.get('n_edges', 0):,} |\n"
            )
            diff_block += f"| Components | {pm.get('component_count', 0)} | {cm.get('component_count', 0)} |\n"
            diff_block += f"| Giant Fraction | {pm.get('giant_fraction', 0):.2%} | {cm.get('giant_fraction', 0):.2%} |\n"

        # Record history
        session.sweep_history.append(
            SweepRecord(time.time(), current_yaml, current_metrics)
        )

        result = f"{diff_block}\n\n"
        result += (
            "### Execution Summary\n"
            "Successfully ran topological sweep.\n"
            f"- Nodes: {graph.number_of_nodes()}\n"
            f"- Edges: {graph.number_of_edges()}\n"
            f"- Resolution: {model.resolved_threshold:.4f}\n"
            f"- Data Shape: {session.data.shape}\n"
            f"- Config saved: {saved_path}"
        )
        return result

    except Exception as e:
        logger.error(f"Error running sweep: {e}")
        return mcp_error("run_topological_sweep", str(e))


@mcp.tool()
async def generate_cluster_dossier(
    method: str = "auto",
    max_k: int = 15,
    ctx: Context = None,
) -> str:
    """
    Generate a statistical dossier of the topological clusters.
    """
    session = _get_session(ctx)

    if session.model is None or session.data is None:
        raise ToolError("No model found. Run run_topological_sweep() first.")

    if method not in ("auto", "spectral", "components"):
        raise ToolError(
            f"method must be 'auto', 'spectral', or 'components', got '{method}'"
        )

    try:
        clusters = resolve_clusters(session.model, method=method, max_k=max_k)
        session.clusters = clusters

        dossier = build_dossier(session.model, session.data, clusters)
        markdown = dossier_to_markdown(dossier)
        return markdown

    except Exception as e:
        logger.error(f"Error generating dossier: {e}")
        return mcp_error("generate_cluster_dossier", str(e))


@mcp.tool()
async def compare_clusters_tool(
    cluster_a: int, cluster_b: int, ctx: Context = None
) -> str:
    """
    Perform pairwise statistical tests between two clusters.
    """
    session = _get_session(ctx)

    if session.data is None or session.clusters is None:
        raise ToolError(
            "No data or clusters found. Run generate_cluster_dossier() first."
        )

    try:
        results = compare_clusters(session.data, session.clusters, cluster_a, cluster_b)
        if not results:
            return mcp_error(
                "compare_clusters_tool",
                f"No results for comparison between clusters {cluster_a} and {cluster_b}.",
            )

        markdown = comparison_to_markdown(cluster_a, cluster_b, results)
        return markdown

    except Exception as e:
        logger.error(f"Error comparing clusters: {e}")
        return mcp_error("compare_clusters_tool", str(e))


@mcp.tool()
async def export_labeled_data(
    cluster_names: Dict[int, str], output_path: str, ctx: Context
) -> str:
    """
    Assign semantic names to clusters and export the labeled dataset to CSV.
    """
    session = _get_session(ctx)

    if session.data is None or session.clusters is None:
        raise ToolError(
            "No data or clusters found. Run generate_cluster_dossier() first."
        )

    actual_ids = set(session.clusters.unique().tolist())
    provided_ids = set(int(k) for k in cluster_names.keys())
    missing_ids = actual_ids - provided_ids
    if missing_ids:
        raise ToolError(
            f"cluster IDs {sorted(missing_ids)} have no name mapping. Provide all {len(actual_ids)} cluster IDs."
        )

    try:
        df = session.data.copy()
        df["topological_cluster_id"] = session.clusters
        names_map = {int(k): v for k, v in cluster_names.items()}
        df["topological_cluster_name"] = df["topological_cluster_id"].map(names_map)
        df.to_csv(output_path, index=False)
        return f"Successfully exported labeled data to {output_path}"

    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return mcp_error("export_labeled_data", str(e))


@mcp.tool()
async def characterize_dataset(csv_path: str, ctx: Context) -> str:
    """
    Probes dataset geometry to return raw facts (N, features, variance curve, k-NN mean).
    Use this signal to form a hypothesis and pass your summary into suggest_initial_config.
    """
    try:
        from pulsar.analysis.characterization import characterize_dataset as _char

        # Read once, reuse for both session state and characterization.
        session = _get_session(ctx)
        df = await asyncio.to_thread(pd.read_csv, csv_path)
        session.data = df

        result = await asyncio.to_thread(_char, csv_path, dataframe=df)
        return json.dumps(dataclasses.asdict(result), indent=2)
    except Exception as e:
        logger.error(f"Error analyzing dataset: {e}")
        return mcp_error("characterize_dataset", str(e))


@mcp.tool()
async def diagnose_cosmic_graph(ctx: Context) -> str:
    """
    Diagnose the fitted cosmic graph quality by returning pure GraphMetrics.
    Interpret these metrics (e.g. density, component distribution) given N.
    """
    session = _get_session(ctx)

    if session.model is None:
        raise ToolError("No model found. Run run_topological_sweep() first.")

    try:
        from pulsar.mcp.diagnostics import diagnose_model

        result = diagnose_model(session.model)
        return json.dumps(dataclasses.asdict(result), indent=2)
    except Exception as e:
        logger.error(f"Error diagnosing graph: {e}")
        return mcp_error("diagnose_cosmic_graph", str(e))


@mcp.tool()
async def recommend_preprocessing(dataset_geometry: str, ctx: Context) -> str:
    """
    Analyze column profiles from characterize_dataset and return a complete
    preprocessing block with impute/encode/drop recommendations and per-column rationale.

    Call this after characterize_dataset to get an expert-recommended preprocessing:
    YAML block before running run_topological_sweep.

    Args:
        dataset_geometry: The raw JSON string from characterize_dataset.

    Returns:
        Markdown rationale table + ready-to-paste preprocessing: YAML block.
    """
    try:
        geo = json.loads(dataset_geometry)
        n_samples = geo.get("n_samples", 0)
        column_profiles = geo.get("column_profiles", [])

        if not column_profiles:
            return mcp_error(
                "recommend_preprocessing",
                "No column_profiles found in dataset_geometry. Pass the full JSON from characterize_dataset.",
            )

        drop, impute, encode, rationale = _recommend_preprocessing_block(
            column_profiles, n_samples
        )

        result = "## Preprocessing Recommendation\n\n"
        result += _rationale_table(rationale)
        result += "\n\n### Recommended preprocessing block\n\n```yaml\n"
        result += _preprocessing_block_to_yaml(drop, impute, encode)
        result += "\n```\n"
        result += "\nCopy this block into your config_yaml, then call `validate_preprocessing_config` to confirm before running `run_topological_sweep`."
        return result

    except Exception as e:
        logger.error(f"Error in recommend_preprocessing: {e}")
        return mcp_error("recommend_preprocessing", str(e))


@mcp.tool()
async def repair_preprocessing_config(
    error_message: str,
    config_yaml: str,
    dataset_geometry: str,
    ctx: Context,
) -> str:
    """
    Given a preprocessing error from run_topological_sweep, produce a corrected
    config_yaml with a change log of what was fixed and why.

    Handles: NaN remaining, non-numeric columns, coercion failure, all-missing
    columns, and cardinality violations.

    Args:
        error_message: The full error text from the failed sweep.
        config_yaml: The config_yaml that caused the error.
        dataset_geometry: The raw JSON string from characterize_dataset.

    Returns:
        Markdown with error classification, change log table, and patched config_yaml.
    """
    try:
        config_dict = yaml.safe_load(config_yaml)
        if not isinstance(config_dict, dict):
            return mcp_error(
                "repair_preprocessing_config",
                "config_yaml must be a valid YAML mapping.",
            )

        geo = json.loads(dataset_geometry)
        # Build a name → profile dict for O(1) lookup
        profiles_by_name: dict[str, Any] = {}
        for cp in geo.get("column_profiles", []):
            pname = cp["name"] if isinstance(cp, dict) else cp.name
            profiles_by_name[pname] = cp

        pre = config_dict.setdefault("preprocessing", {})
        drop_list: list[str] = pre.setdefault("drop_columns", [])
        impute_dict: dict[str, Any] = pre.setdefault("impute", {})
        encode_dict: dict[str, Any] = pre.setdefault("encode", {})

        changes: list[tuple[str, str, str, str]] = []  # (col, old, new, rationale)

        def _get_profile_field(col: str, field: str, default: Any = None) -> Any:
            cp = profiles_by_name.get(col)
            if cp is None:
                return default
            return (
                cp.get(field, default)
                if isinstance(cp, dict)
                else getattr(cp, field, default)
            )

        # Pattern 1: Coercion failure — "configured for numeric imputation"
        m = re.search(
            r"Column '([^']+)' is configured for numeric imputation \(([^)]+)\)",
            error_message,
        )
        if m:
            col, old_method = m.group(1), m.group(2)
            n_unique = _get_profile_field(col, "n_unique", 999)
            new_method = "fill_mode" if n_unique <= 10 else "sample_categorical"
            impute_dict[col] = {"method": new_method, "seed": 42}
            changes.append(
                (
                    col,
                    f"impute: {old_method}",
                    f"impute: {new_method}",
                    "Column contains non-numeric values; switched to string imputation",
                )
            )

        # Pattern 2: NaN remaining
        elif "NaN values remain after imputation" in error_message:
            nan_cols = re.findall(r"'([^']+)' \(\d+ rows\)", error_message)
            for col in nan_cols:
                is_numeric = _get_profile_field(col, "is_numeric", True)
                n_missing = _get_profile_field(col, "n_missing", 0)
                new_method = "fill_mean" if is_numeric else "sample_categorical"
                impute_dict[col] = {"method": new_method, "seed": 42}
                changes.append(
                    (
                        col,
                        "no impute rule",
                        f"impute: {new_method}",
                        f"{'Numeric' if is_numeric else 'Categorical'} column with {n_missing} missing rows",
                    )
                )

        # Pattern 3: Non-numeric columns remaining
        elif "Non-numeric columns remain" in error_message:
            bad_cols = re.findall(r"'([^']+)' \(dtype=\w+\)", error_message)
            for col in bad_cols:
                n_unique = _get_profile_field(col, "n_unique", 999)
                if n_unique > 50:
                    drop_list.append(col)
                    changes.append(
                        (
                            col,
                            "no rule",
                            "drop_columns",
                            f"{n_unique} categories — one-hot would add {n_unique} dimensions",
                        )
                    )
                else:
                    encode_dict[col] = {"method": "one_hot"}
                    changes.append(
                        (
                            col,
                            "no rule",
                            "encode: one_hot",
                            f"{n_unique} unique values — safe to one-hot",
                        )
                    )

        # Pattern 4: All-missing
        elif "is all-missing" in error_message:
            m2 = re.search(r"Column '([^']+)' is all-missing", error_message)
            if m2:
                col = m2.group(1)
                drop_list.append(col)
                changes.append(
                    (
                        col,
                        "impute/encode",
                        "drop_columns",
                        "Column is entirely null — cannot impute or encode",
                    )
                )

        # Pattern 5: Cardinality exceeded
        elif "exceeding max_categories" in error_message:
            m3 = re.search(r"Column '([^']+)' has (\d+) categories", error_message)
            if m3:
                col, n_cats = m3.group(1), int(m3.group(2))
                if n_cats > 50:
                    drop_list.append(col)
                    if col in encode_dict:
                        del encode_dict[col]
                    changes.append(
                        (
                            col,
                            "encode: one_hot",
                            "drop_columns",
                            f"{n_cats} categories is too many for one-hot encoding",
                        )
                    )
                else:
                    # Raise cap to match actual cardinality so the
                    # validation gate (n_cats > max_categories) passes.
                    encode_dict[col] = {"method": "one_hot", "max_categories": n_cats}
                    changes.append(
                        (
                            col,
                            "encode: one_hot (max_categories too low)",
                            f"encode: one_hot (max_categories={n_cats})",
                            f"Raised cap to match actual {n_cats} categories",
                        )
                    )

        else:
            return mcp_error(
                "repair_preprocessing_config",
                f"Unrecognized preprocessing error pattern. Raw error:\n{error_message}",
            )

        if not changes:
            return mcp_error(
                "repair_preprocessing_config",
                "Error was classified but no changes were needed. Review the error message manually.",
            )

        # Classify error for display
        if "NaN values remain" in error_message:
            classification = "`NaN values remain after imputation`"
        elif "Non-numeric columns" in error_message:
            classification = "`Non-numeric columns remain after preprocessing`"
        elif "configured for numeric imputation" in error_message:
            classification = "`Numeric imputation on non-numeric column`"
        elif "all-missing" in error_message:
            classification = "`All-missing column`"
        elif "exceeding max_categories" in error_message:
            classification = "`Cardinality limit exceeded`"
        else:
            classification = "`Preprocessing error`"

        result = f"## Error Classification\n{classification} — {len(changes)} change(s) made.\n\n"
        result += "## Changes Made\n\n"
        result += "| Column | Was | Now | Rationale |\n|---|---|---|---|\n"
        for col, was, now, reason in changes:
            result += f"| `{col}` | {was} | {now} | {reason} |\n"

        patched_yaml = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
        result += f"\n## Patched Config\n\n```yaml\n{patched_yaml}```\n"
        result += "\nCall `validate_preprocessing_config` with this config before re-running `run_topological_sweep`."
        return result

    except Exception as e:
        logger.error(f"Error in repair_preprocessing_config: {e}")
        return mcp_error("repair_preprocessing_config", str(e))


@mcp.tool()
async def validate_preprocessing_config(config_yaml: str, ctx: Context) -> str:
    """
    Dry-run the preprocessing stage only against session data — no PCA, no BallMapper,
    no sweep cost. Use this to confirm a config is valid before run_topological_sweep.

    Requires a prior run_topological_sweep call (to populate session data).

    Args:
        config_yaml: Inline YAML config string to validate.

    Returns:
        PASS with schema summary, or a structured error matching repair_preprocessing_config input format.
    """
    session = _get_session(ctx)

    if session.data is None:
        return mcp_error(
            "validate_preprocessing_config",
            "No data in session. Run run_topological_sweep (even with a minimal config) or characterize_dataset first to load data into the session.",
        )

    try:
        config_dict = yaml.safe_load(config_yaml)
        if not isinstance(config_dict, dict):
            return mcp_error(
                "validate_preprocessing_config",
                "config_yaml must be a valid YAML mapping.",
            )

        cfg = load_config(config_dict)
        df_out, layout = await asyncio.to_thread(
            preprocess_dataframe, session.data, cfg
        )

        col_preview = list(layout.feature_names[:8])
        if len(layout.feature_names) > 8:
            col_preview.append(f"... +{len(layout.feature_names) - 8} more")

        result = "## Preprocessing Validation: PASS\n\n"
        result += f"- Input rows: {len(session.data)} → Output rows: {layout.n_rows} (no rows dropped)\n"
        result += f"- Output features: {len(layout.feature_names)}\n"
        result += f"- Columns: {col_preview}\n"
        result += "- NaN remaining: 0\n\n"
        result += "Config is ready for `run_topological_sweep`."
        return result

    except (ValueError, TypeError) as e:
        result = "## Preprocessing Validation: FAIL\n\n"
        result += f"**Error:** {e}\n\n"
        result += "Call `repair_preprocessing_config(error_message=..., config_yaml=..., dataset_geometry=...)` to fix this automatically."
        return result
    except Exception as e:
        logger.error(f"Error in validate_preprocessing_config: {e}")
        return mcp_error("validate_preprocessing_config", str(e))


def main():
    mcp.run()


if __name__ == "__main__":
    main()
