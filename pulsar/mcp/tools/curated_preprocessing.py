"""Tenant-safe config/preprocessing helpers for the CURATED (HTTP) surface.

The full-surface versions in ``tools/preprocessing.py`` / ``tools/reporting.py`` / ``tools/meta.py``
load data through the stdio *session* layer + a global, single-tenant dataset registry
(``registry.get_dataset(dataset_id)`` — no ``user_id``). That is unsafe over the multi-tenant HTTP
surface. These wrappers preserve the engine ANALYSIS logic unchanged but load the dataframe through the
tenant-scoped object store (``load_dataset(dataset_id, store, user_id=…)``), exactly like the other
curated tools, and return the standard ``ToolResult`` contract.

Adding these to ``CURATED_TOOLS_LIST`` gives the HTTP agent the self-correction + workflow tools it
otherwise lacks (config recommendation/validation/repair, column probing, workflow guide) — so it no
longer has to guess an impute config it has no schema for.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import Literal

import yaml
from fastmcp.exceptions import ToolError

from pulsar.analysis.characterization import (
    characterize_dataset as _char,
    profile_column_details,
)
from pulsar.config import load_config
from pulsar.preprocessing import preprocess_dataframe
from pulsar.mcp.characterization import (
    probe_columns_payload,
    probe_columns_payload_to_markdown,
)
from pulsar.mcp.config_tools import (
    _build_initial_config_yaml,
    apply_overrides,
    render_validation_report,
    validate_config_yaml,
)
from pulsar.mcp.config_refs import load_config_ref, save_config_ref
from pulsar.mcp.datasets import (
    data_key,
    dataset_exists,
    load_dataset,
    load_dataset_meta,
)
from pulsar.mcp.preprocessing import (
    _calibrate_processed_space,
    _preprocessing_block_to_yaml,
    _recommend_preprocessing_block,
    build_preprocessing_recommendation_payload,
    enrich_dirty_numeric_samples,
    preprocessing_recommendation_to_markdown,
    repair_config,
)
from pulsar.mcp.store import get_object_store
from pulsar.mcp.sweep_index import load_sweep_manifest


# --------------------------------------------------------------------------- #
# Result helpers (local copy to avoid a circular import with curated.py).
# --------------------------------------------------------------------------- #
def _json_default(o):
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)!r}")


def _result(markdown: str, structured, viz=None, confidence=None) -> str:
    return json.dumps(
        {
            "markdown": markdown,
            "structured": structured,
            "vizPayload": viz,
            "confidence": confidence,
        },
        default=_json_default,
    )


def _require_dataset(dataset_id: str, user_id: str):
    """Tenant-scoped load: returns (df, data_path) or raises ToolError. user_id comes from the
    authenticated session (bound by the app), never the model."""
    store = get_object_store()
    if not dataset_exists(dataset_id, store, user_id=user_id):
        raise ToolError(f"dataset {dataset_id} not found; ingest it first.")
    df = load_dataset(dataset_id, store, user_id=user_id)
    data_path = str(store.root / data_key(user_id, dataset_id))
    return df, data_path


# --------------------------------------------------------------------------- #
# Workflow guide (stateless) — curated-accurate (names only curated tools).
# --------------------------------------------------------------------------- #
_CURATED_WORKFLOW = """# Pulsar curated workflow (H0 connected-component analysis)

1. Orient — a dataset is usually already linked. If not, `ingest_dataset` first.
   Before spending a new sweep on an existing dataset, use `list_sweeps(dataset_id)` to check for a
   completed artifact; if one matches, reuse its `artifact_ref` (datasetId + configHash) with the
   interpret tools and SKIP steps 3-4 entirely.
2. Characterize (optional) — `characterize_dataset(dataset_id)` to judge whether reliable
   connected-component (H0) structure is plausible before spending a sweep.
3. Prepare — `prepare_sweep(dataset_id)` with NO `config`. It auto-characterizes, auto-calibrates,
   and AUTO-HANDLES MISSING VALUES, returning a validated `config_hash`. Do NOT GUESS imputation
   method names or hand-write an `impute` block.
   - To EXCLUDE a column from graph construction (e.g. a label you want to predict, like `species`):
     `create_config(dataset_id)` → `refine_config(dataset_id=..., config_ref=..., overrides={"preprocessing.drop_columns": ["species"]})` →
     `prepare_sweep(dataset_id, config_ref=<the refined config_ref>)`. The excluded column never enters the graph.
   - If preparation reports a preprocessing/config error: `recommend_preprocessing(dataset_id)` for a
     valid preprocessing block, `repair_preprocessing_config(...)` to fix a specific error, and
     `validate_preprocessing_config(...)` / `validate_config(...)` to dry-run before sweeping. Inspect
     columns with `probe_columns(dataset_id, columns)`.
4. Run — `run_sweep(dataset_id)` with the prepared config; it waits server-side (~90s). Re-check long
   runs with `get_sweep_status(job_id)`.
5. Evaluate — if `structureStatus` is "no_reliable_structure", report that and STOP. If it is "caution", keep going but frame the interpretation as tentative.
6. Interpret — `diagnose_cosmic_graph` → `get_threshold_stability_curve` →
   `generate_cluster_dossier` → `get_feature_signal` / `get_cluster_profile` /
   `get_cluster_signal_matrix` / `get_topological_skeleton` / `compare_clusters`.

Enrichment stats (signal_tier, z_score, lift, specificity, homogeneity) are EXPLORATORY and
selection-biased — frame them as such; never assert causation or clinical significance."""


async def get_workflow_guide(user_id: str = "local") -> str:
    """Recommended end-to-end Pulsar workflow as markdown. Call once at the start to follow the
    opinionated procedure (characterize → prepare → run → diagnose → interpret)."""
    return _result(_CURATED_WORKFLOW, {"workflow": "curated"})


# --------------------------------------------------------------------------- #
# Config / preprocessing helpers (tenant-safe).
# --------------------------------------------------------------------------- #
async def recommend_preprocessing(
    dataset_id: str,
    user_id: str = "local",
    detail: Literal["summary", "full"] = "summary",
    rationale_limit: int = 20,
) -> str:
    """Recommend a valid preprocessing block (drop/impute/encode) from the dataset's column profiles.
    Returns the ready-to-use `preprocessing_yaml` plus rationale — use this instead of guessing a
    preprocessing/impute config."""
    if rationale_limit < 1:
        raise ToolError(f"rationale_limit must be >= 1, got '{rationale_limit}'")
    df, data_path = _require_dataset(dataset_id, user_id)

    result = await asyncio.to_thread(_char, data_path, dataframe=df)
    geo = dataclasses.asdict(result)
    n_samples = geo.get("n_samples", 0)
    column_profiles = geo.get("column_profiles", [])

    column_profiles, dirty_numeric_detection = enrich_dirty_numeric_samples(
        column_profiles, df
    )
    drop, impute, encode, rationale = _recommend_preprocessing_block(
        column_profiles, n_samples
    )
    preprocessing_yaml = _preprocessing_block_to_yaml(drop, impute, encode)

    expansion_estimate = 0
    for raw_cp in column_profiles:
        cp = (
            raw_cp
            if isinstance(raw_cp, dict)
            else {"name": raw_cp.name, "n_unique": raw_cp.n_unique}
        )
        if cp["name"] in encode:
            expansion_estimate += cp.get("n_unique", 2)

    payload = build_preprocessing_recommendation_payload(
        drop=drop,
        impute=impute,
        encode=encode,
        rationale=rationale,
        column_profiles=column_profiles,
        preprocessing_yaml=preprocessing_yaml,
        expansion_estimate=expansion_estimate,
        dirty_numeric_detection=dirty_numeric_detection,
        detail=detail,
        rationale_limit=rationale_limit,
    )
    return _result(preprocessing_recommendation_to_markdown(payload), payload)


async def validate_preprocessing_config(
    dataset_id: str,
    config_yaml: str,
    user_id: str = "local",
) -> str:
    """Dry-run a preprocessing config against the dataset (no PCA/BallMapper/sweep cost). Returns a
    PASS summary, or a structured failure whose error names the valid options — feed that to
    `repair_preprocessing_config`."""
    df, _ = _require_dataset(dataset_id, user_id)

    config_dict = yaml.safe_load(config_yaml)
    if not isinstance(config_dict, dict):
        raise ToolError("config_yaml must be a valid YAML mapping.")

    try:
        cfg = load_config(config_dict)
        df_out, layout = await asyncio.to_thread(preprocess_dataframe, df, cfg)
    except (ValueError, TypeError) as e:
        structured = {
            "status": "error",
            "valid": False,
            "error": str(e),
            "agent_action": (
                "Call repair_preprocessing_config(dataset_id=..., error_message=..., config_yaml=...) "
                "to fix this automatically, or recommend_preprocessing(dataset_id=...) for a fresh block."
            ),
        }
        return _result(f"Preprocessing config is INVALID: {e}", structured)

    input_cols = set(df.columns)
    output_names = layout.feature_names
    dummy_count = sum(
        1 for name in output_names if "_" in name and name not in input_cols
    )
    missingness_flag_count = sum(
        1 for name in output_names if name.endswith("_was_missing")
    )
    high_cardinality_encoded = [
        col for col, cats in layout.vocab.items() if len(cats) > 20
    ]
    col_preview = list(output_names[:8])
    if len(output_names) > 8:
        col_preview.append(f"... +{len(output_names) - 8} more")

    structured = {
        "status": "ok",
        "valid": True,
        "input_rows": len(df),
        "output_rows": layout.n_rows,
        "output_feature_count": len(output_names),
        "dummy_expansion_count": dummy_count,
        "missingness_flag_count": missingness_flag_count,
        "high_cardinality_encoded_columns": high_cardinality_encoded,
        "feature_names_preview": list(col_preview),
        "nan_remaining": 0,
    }
    md = (
        f"Preprocessing config PASSED — {len(df)} rows → {len(output_names)} features "
        f"({dummy_count} one-hot, {missingness_flag_count} missingness flags)."
    )
    return _result(md, structured)


async def repair_preprocessing_config(
    dataset_id: str,
    error_message: str,
    config_yaml: str,
    user_id: str = "local",
) -> str:
    """Given a sweep/validation preprocessing error and the offending config_yaml, return a corrected
    config_yaml with a change log (handles NaN-remaining, non-numeric columns, coercion failure,
    all-missing columns, cardinality violations)."""
    df, data_path = _require_dataset(dataset_id, user_id)

    result = await asyncio.to_thread(_char, data_path, dataframe=df)
    geo = dataclasses.asdict(result)
    profiles_by_name = {}
    for cp in geo.get("column_profiles", []):
        pname = cp["name"] if isinstance(cp, dict) else cp.name
        profiles_by_name[pname] = cp

    repaired_markdown = repair_config(config_yaml, error_message, profiles_by_name)
    return _result(repaired_markdown, {"source": "repair_preprocessing_config"})


async def probe_columns(
    dataset_id: str,
    columns: list[str],
    user_id: str = "local",
) -> str:
    """Detailed per-column profiles (sample values, distributions) for up to 20 columns. Use to reason
    about features / a held-out label before configuring or sweeping."""
    if len(columns) > 20:
        raise ToolError("Too many columns requested. Max 20 at a time.")
    df, _ = _require_dataset(dataset_id, user_id)

    profiles = []
    missing_columns = []
    for col in columns:
        if col not in df.columns:
            missing_columns.append(col)
            continue
        profiles.append(dataclasses.asdict(profile_column_details(df, col)))

    payload = probe_columns_payload(profiles, columns, missing_columns)
    return _result(probe_columns_payload_to_markdown(payload), payload)


# --------------------------------------------------------------------------- #
# Config build / edit / validate (tenant-safe ports of the stdio config tools).
# These give the HTTP agent the discoverable, validated config workflow — most
# importantly a clear path to EXCLUDE a column from graph construction via
# preprocessing.drop_columns (create_config → refine_config → prepare_sweep).
# --------------------------------------------------------------------------- #
async def create_config(
    dataset_id: str, intent: str = "", user_id: str = "local"
) -> str:
    """Build the canonical Pulsar config for a dataset and return a stable config_ref.

    Use the returned `config_ref` with `refine_config`; do not copy the YAML string between tools.
    The YAML remains in the payload for human review/backward compatibility only."""
    df, data_path = _require_dataset(dataset_id, user_id)

    result = await asyncio.to_thread(_char, data_path, dataframe=df)
    geo = dataclasses.asdict(result)
    run_name = intent.strip() or "initial_sweep"
    processed = await asyncio.to_thread(
        _calibrate_processed_space,
        df,
        geo["column_profiles"],
        geo["n_samples"],
        data_path,
    )
    config_yaml = _build_initial_config_yaml(
        geo, data_path=data_path, run_name=run_name, processed_profile=processed
    )

    store = get_object_store()
    config_ref, resolved_cfg = save_config_ref(
        store, user_id=user_id, dataset_id=dataset_id, config_yaml=config_yaml
    )
    structured = {
        "status": "ok",
        "datasetId": dataset_id,
        "config_ref": config_ref,
        "configHash": config_ref,
        "config_yaml": config_yaml,
        "config": resolved_cfg,
        "calibration_space": "processed" if processed is not None else "raw",
        "drop_columns": resolved_cfg.get("preprocessing", {}).get("drop_columns", []),
    }
    md = (
        f"Built config `{config_ref}` for `{dataset_id}`. To exclude columns from the graph, call "
        f'`refine_config(dataset_id="{dataset_id}", config_ref="{config_ref}", '
        f'overrides={{"preprocessing.drop_columns": [...]}})`, then pass the returned '
        f"`config_ref` to `prepare_sweep`. Current drop_columns: {structured['drop_columns']}."
    )
    return _result(md, structured)


async def refine_config(
    config_yaml: str = "",
    overrides: dict | None = None,
    dataset_id: str = "",
    config_ref: str = "",
    user_id: str = "local",
) -> str:
    """Apply validated overrides and return a new config_ref + normalized config.

    Preferred: pass `dataset_id` + `config_ref` from `create_config`. Raw `config_yaml` is accepted
    only for backward compatibility."""
    store = get_object_store()
    if config_ref:
        if not dataset_id:
            raise ToolError("dataset_id is required when using config_ref.")
        try:
            config_yaml = load_config_ref(
                store, user_id=user_id, dataset_id=dataset_id, config_ref=config_ref
            )
        except FileNotFoundError as e:
            raise ToolError(str(e)) from e
    if not config_yaml:
        raise ToolError(
            "config_yaml or config_ref is required; call create_config(dataset_id) first and pass its config_ref."
        )
    try:
        result = apply_overrides(config_yaml, overrides or {})
    except ValueError as e:
        # Unknown override key — surface the valid options to the agent (not a crash).
        return _result(
            f"Override rejected: {e}",
            {"status": "error", "valid": False, "error": str(e)},
        )
    new_ref = ""
    config = yaml.safe_load(result.config_yaml)
    if dataset_id:
        new_ref, config = save_config_ref(
            store,
            user_id=user_id,
            dataset_id=dataset_id,
            config_yaml=result.config_yaml,
        )
    structured = {
        "status": "ok",
        "datasetId": dataset_id or None,
        "config_ref": new_ref or None,
        "configHash": new_ref or None,
        "applied_overrides": result.applied_overrides,
        "diff": result.diff,
        "config_yaml": result.config_yaml,
        "config": config,
    }
    if new_ref:
        md = f'Applied overrides {result.applied_overrides}. Pass `config_ref="{new_ref}"` to `prepare_sweep`.'
    else:
        md = f"Applied overrides {result.applied_overrides}. Pass `config_yaml` to `prepare_sweep`.\n\n```yaml\n{result.config_yaml}\n```"
    return _result(md, structured)


async def get_config(dataset_id: str, config_ref: str, user_id: str = "local") -> str:
    """Return the exact tenant-scoped config params for a dataset/config_ref.

    Use this when you have a historical `configHash` from `list_sweeps` and need to inspect the
    original sweep parameters without re-running or reconstructing the config. `config_ref` and
    `configHash` are the same stable config hash returned by create_config/refine_config/prepare_sweep.
    """
    store = get_object_store()
    try:
        config_yaml = load_config_ref(
            store, user_id=user_id, dataset_id=dataset_id, config_ref=config_ref
        )
    except FileNotFoundError as e:
        raise ToolError(str(e)) from e
    config = yaml.safe_load(config_yaml)
    structured = {
        "status": "ok",
        "datasetId": dataset_id,
        "config_ref": config_ref,
        "configHash": config_ref,
        "config_yaml": config_yaml,
        "config": config,
    }
    md = f"Loaded config `{config_ref}` for dataset `{dataset_id}`."
    return _result(md, structured)


async def validate_config(
    config_yaml: str, dataset_id: str = "", user_id: str = "local"
) -> str:
    """Validate a full Pulsar config (and normalize it). Pass `dataset_id` to also check column
    references (e.g. that drop_columns names exist) against the actual data."""
    data_path = None
    if dataset_id:
        _df, data_path = _require_dataset(dataset_id, user_id)
    report = validate_config_yaml(config_yaml, dataset_path=data_path)
    return _result(
        render_validation_report(report), {"status": "ok", "validated": True}
    )


# --------------------------------------------------------------------------- #
# Completed-sweep lookup (tenant-safe, read-only) — the "did I already run this?" tool.
#
# A finished sweep persists a derived artifact at
# ``{user_id}/{dataset_id}/{config_hash}/artifact.json`` in the object store (see
# pulsar/mcp/worker.py + pulsar/artifacts.py). ``get_sweep_status`` answers "is THIS job
# done?" but needs the ephemeral ``job_id``; an agent that lost the job_id (new turn,
# compaction, re-prompt) otherwise has NO way to discover a completed sweep and would
# re-prepare/re-run the whole pipeline. This tool reads only compact sweep manifests under
# the caller's own ``user_id`` prefix and returns the ``artifact_ref`` (datasetId +
# configHash) the interpret tools need. No artifact bytes, no host paths, no mutation.
# --------------------------------------------------------------------------- #
async def list_sweeps(
    dataset_id: str = "", user_id: str = "local", limit: int = 20
) -> str:
    """List COMPLETED topological sweeps already persisted for this user (optionally one dataset).

    Use this before spending a new sweep on an existing dataset: if a completed sweep already exists
    for the dataset (and config) you want, reuse it instead of re-running the pipeline. Each entry is an
    artifact_ref — pass its `datasetId` + `configHash` straight to the interpret tools
    (diagnose_cosmic_graph, get_threshold_stability_curve, generate_cluster_dossier, etc.) with NO
    new sweep. Pass `dataset_id` to scope to one dataset; omit it to list every completed sweep for
    the user. Returns [] when nothing has been swept yet (that is the only case where you must run a
    sweep). Read-only and tenant-scoped: it can never see another user's sweeps."""
    store = get_object_store()
    root = getattr(store, "root", None)
    if root is None:
        # Non-filesystem store (e.g. a GCS prod adapter without prefix-listing wired here):
        # degrade safely rather than scan something we cannot enumerate.
        raise ToolError(
            "list_sweeps requires an enumerable object store; not available on this backend."
        )

    try:
        max_items = max(1, min(int(limit), 100))
    except (TypeError, ValueError):
        max_items = 20

    sweeps: list[dict] = []
    user_root = root / user_id
    # Sweep manifests live at {user_id}/{dataset_id}/sweeps/{config_hash}.json. They are compact
    # listing metadata; never read the heavyweight artifact.json on this path.
    manifest_paths = (
        (user_root / dataset_id / "sweeps").glob("*.json")
        if dataset_id
        else user_root.glob("*/sweeps/*.json")
    )
    for manifest_path in sorted(manifest_paths):
        rel = manifest_path.relative_to(user_root)
        ds_id, cfg_file = rel.parts[0], rel.parts[2]
        cfg_hash = cfg_file.removesuffix(".json")
        try:
            sweep = load_sweep_manifest(
                store, user_id=user_id, dataset_id=ds_id, config_hash=cfg_hash
            )
        except Exception:
            continue
        name = None
        try:
            name = load_dataset_meta(ds_id, store, user_id=user_id).get("name")
        except Exception:
            name = None
        artifact_ref = (
            sweep.get("artifactRef")
            or sweep.get("artifact_ref")
            or {
                "datasetId": ds_id,
                "configHash": cfg_hash,
                "userId": user_id,
            }
        )
        sweeps.append(
            {
                "artifactRef": artifact_ref,
                "artifact_ref": artifact_ref,
                "datasetId": ds_id,
                "configHash": cfg_hash,
                "datasetName": name,
                "n": sweep.get("n"),
                "metrics": sweep.get("metrics"),
                "structureStatus": sweep.get("structureStatus"),
                "pulsarVersion": sweep.get("pulsarVersion"),
                "createdAt": sweep.get("createdAt"),
            }
        )

    sweeps.sort(key=lambda s: s.get("createdAt") or "", reverse=True)
    returned = sweeps[:max_items]
    structured = {
        "status": "ok",
        "datasetId": dataset_id or None,
        "count": len(sweeps),
        "returned": len(returned),
        "limit": max_items,
        "sweeps": returned,
        "agent_action": (
            "If a sweep matching your target dataset/config is listed, reuse its artifact_ref with the "
            "interpret tools instead of calling prepare_sweep/run_topological_sweep again."
            if sweeps
            else "No completed sweep exists yet — prepare_sweep then run_topological_sweep."
        ),
    }
    if not returned:
        scope = f" for dataset `{dataset_id}`" if dataset_id else ""
        md = f"No completed sweeps found{scope}. Prepare and run one before interpreting."
    else:
        lines = [
            f"Found {len(sweeps)} completed sweep(s)"
            + (f" for dataset `{dataset_id}`" if dataset_id else "")
            + " — reuse these instead of re-running:",
            "",
            "| dataset | config | n | components | created |",
            "|---|---|---|---|---|",
        ]
        for s in returned:
            m = s.get("metrics") or {}
            comp = m.get("componentCount")
            comp_str = str(int(comp)) if isinstance(comp, (int, float)) else "?"
            lines.append(
                f"| `{s['datasetId']}` | `{s['configHash']}` | {s.get('n', '?')} | "
                f"{comp_str} | {s.get('createdAt', '?')} |"
            )
        md = "\n".join(lines)
    return _result(md, structured)
