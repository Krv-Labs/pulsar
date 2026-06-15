"""Sweep worker (Stage 1, build-spec §4 step 5/6).

Claims sweep jobs off the local queue, runs ``ThemaRS.fit``, persists the derived
artifact (``artifacts.dump_artifact`` → object store), and updates job status —
WITHOUT the MCP HTTP request ever holding the sweep.

Admission is row-based (claude.md: "row-based envelope, not file bytes"): an input
over ``MAX_ROWS`` fails the job with a STRUCTURED error — never a 500/SIGKILL. Any
fit exception (incl. MemoryError) is caught and recorded as a clean job failure.
A post-sweep structure check surfaces "no reliable structure detected" for noise.
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time

import pandas as pd

from pulsar.artifacts import dump_artifact
from pulsar.config import MAX_ROWS, load_config
from pulsar.mcp.jobs import FsJobQueue, get_job_queue
from pulsar.mcp.store import FsObjectStore, artifact_prefix, get_object_store


class AdmissionError(Exception):
    """Raised when a job exceeds the compute envelope (row cap, etc.)."""


def _load_data(job: dict, store=None) -> pd.DataFrame:
    # Preferred: dataset persisted in the object store (curated flow).
    ref = job.get("data_ref")
    if ref:
        import io

        return pd.read_parquet(io.BytesIO((store or get_object_store()).get(ref)))
    # Fallback: a local path (dev / direct enqueue).
    path = job.get("data_path")
    if not path:
        raise AdmissionError("job has no data_ref or data_path")
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    return ru / 1024 / 1024 if sys.platform == "darwin" else ru / 1024


def _assess_structure(model) -> str:
    """Validity gate (build-spec §4 step 7): no stable H0 plateau ⇒ 'no_reliable_structure'.

    Uses Pulsar's own H0 connected-component stability criterion
    (``_cluster_by_threshold_stability`` returns ``None`` when no threshold plateau
    yields a valid split) rather than a post-hoc metric heuristic — noise data has no
    stable plateau, real structure does.
    """
    import numpy as np

    from pulsar.mcp.interpreter import _cluster_by_threshold_stability

    W = np.asarray(model._weighted_adjacency, dtype=float)
    n = int(W.shape[0])
    try:
        ts = _cluster_by_threshold_stability(W, n)
    except Exception:
        ts = None
    if ts is None or ts.n_clusters <= 1:
        return "no_reliable_structure"
    return "ok"


def run_job(job: dict, *, queue: FsJobQueue | None = None, store: FsObjectStore | None = None):
    """Run a single claimed sweep job to completion. Returns the artifact_ref or None on failure."""
    queue = queue or get_job_queue()
    store = store or get_object_store()
    job_id = job["job_id"]
    t0 = time.time()
    try:
        df = _load_data(job, store)
        cfg = load_config(job["config"])
        max_rows = int(getattr(cfg, "max_rows", MAX_ROWS) or MAX_ROWS)
        if len(df) > max_rows:
            raise AdmissionError(
                f"row cap exceeded: {len(df)} rows > MAX_ROWS={max_rows} "
                f"(Spike-3 memory envelope; n>10k prohibitive). Downsample or raise the cap."
            )

        model = ThemaRS_fit(cfg, df)
        structure_status = _assess_structure(model)

        prefix = artifact_prefix(job["user_id"], job["dataset_id"], job["config_hash"])
        artifact = dump_artifact(
            model,
            dataset_id=job["dataset_id"],
            config_hash=job["config_hash"],
            prefix=prefix,
            store=store,
        )
        artifact["structureStatus"] = structure_status
        store.put(f"{prefix}/artifact.json", json.dumps(artifact).encode())

        artifact_ref = {
            "userId": job["user_id"],
            "datasetId": job["dataset_id"],
            "configHash": job["config_hash"],
        }
        queue.complete(
            job_id,
            artifact_ref=artifact_ref,
            structure_status=structure_status,
            peak_rss_mb=round(_peak_rss_mb(), 1),
            vcpu_ms=int((time.time() - t0) * 1000),
        )
        return artifact_ref
    except Exception as e:  # incl. AdmissionError / MemoryError → clean job failure
        queue.fail(job_id, f"{type(e).__name__}: {e}")
        return None


def ThemaRS_fit(cfg, df):
    """Isolated fit call (kept tiny so the import of the heavy pipeline is lazy)."""
    from pulsar.pipeline import ThemaRS

    return ThemaRS(cfg).fit(df)


def run_forever(poll_interval: float = 0.5) -> None:
    queue = get_job_queue()
    store = get_object_store()
    while True:
        job = queue.claim()
        if job is None:
            time.sleep(poll_interval)
            continue
        run_job(job, queue=queue, store=store)


def main() -> None:
    run_forever()


if __name__ == "__main__":
    main()
