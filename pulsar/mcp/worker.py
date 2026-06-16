"""Sweep worker (Stage 1, build-spec §4 step 5/6).

Claims sweep jobs off the local queue, runs ``ThemaRS.fit``, persists the derived
artifact (``artifacts.dump_artifact`` → object store), and updates job status —
WITHOUT the MCP HTTP request ever holding the sweep.

Admission is row-based (claude.md: "row-based envelope, not file bytes"): an input
over ``MAX_ROWS`` fails the job with a STRUCTURED error — never a 500/SIGKILL. Any
fit exception (incl. MemoryError) is caught and recorded as a clean job failure.
A post-sweep structure check surfaces "no reliable structure detected" for noise and
advisory "caution" when the plateau is real but singleton/tail mass still merits care.
"""
from __future__ import annotations

import json
import os
import resource
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

import pandas as pd

from pulsar.artifacts import dump_artifact
from pulsar.config import MAX_ROWS, load_config
from pulsar.mcp.jobs import FsJobQueue, get_job_queue
from pulsar.mcp.payloads import cluster_result_payload
from pulsar.mcp.store import FsObjectStore, artifact_prefix, get_object_store


class AdmissionError(Exception):
    """Raised when a job exceeds the compute envelope (row cap, etc.)."""


class SweepCancelled(Exception):
    """Raised when the upstream controller asks the worker to stop a sweep."""


def _worker_id() -> str:
    return os.environ.get("PULSAR_WORKER_ID") or f"{socket.gethostname()}:{os.getpid()}"


def _control_base_url() -> str | None:
    explicit = os.environ.get("SWEEP_CONTROL_BASE_URL") or os.environ.get("ISOMORPH_WEB_URL")
    if explicit:
        return explicit.rstrip("/")
    port = os.environ.get("WEB_PORT")
    return f"http://localhost:{port}" if port else None


def _control_token() -> str | None:
    return os.environ.get("INTERNAL_WORKER_TOKEN") or os.environ.get("INTERNAL_MCP_TOKEN")


def _post_heartbeat(job_id: str, worker_id: str) -> dict:
    base_url = _control_base_url()
    token = _control_token()
    if not base_url or not token:
        return {"shouldStop": False}

    body = json.dumps({"workerId": worker_id}).encode()
    req = urllib.request.Request(
        f"{base_url}/api/internal/sweeps/{job_id}/heartbeat",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode() or "{}")
    except (
        TimeoutError,
        urllib.error.HTTPError,
        urllib.error.URLError,
        json.JSONDecodeError,
    ):
        return {"shouldStop": False}


def _cancel_if_requested(job_id: str, queue: FsJobQueue, control: dict) -> None:
    if not control.get("shouldStop"):
        return
    reason = str(control.get("cancelReason") or "cancel_requested")
    queue.cancel(job_id, reason)
    raise SweepCancelled(reason)


def _raise_if_cancelled(job_id: str, queue: FsJobQueue) -> None:
    rec = queue.status(job_id)
    if rec and rec.get("status") == "cancelled":
        raise SweepCancelled(str(rec.get("cancel_reason") or "cancel_requested"))


def _start_heartbeat_loop(job_id: str, queue: FsJobQueue, worker_id: str) -> threading.Event:
    stop = threading.Event()
    try:
        interval = float(os.environ.get("SWEEP_HEARTBEAT_INTERVAL_S", "15"))
    except ValueError:
        interval = 15.0
    if interval <= 0:
        return stop

    def beat() -> None:
        while not stop.wait(interval):
            try:
                _cancel_if_requested(job_id, queue, _post_heartbeat(job_id, worker_id))
            except SweepCancelled:
                stop.set()

    thread = threading.Thread(target=beat, name=f"sweep-heartbeat-{job_id}", daemon=True)
    thread.start()
    return stop


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
    readiness = cluster_result_payload(ts).get("interpretation_readiness", {})
    return "ok" if readiness.get("status") == "ready" else "caution"


def run_job(job: dict, *, queue: FsJobQueue | None = None, store: FsObjectStore | None = None):
    """Run a single claimed sweep job to completion. Returns the artifact_ref or None on failure."""
    queue = queue or get_job_queue()
    store = store or get_object_store()
    job_id = job["job_id"]
    worker_id = _worker_id()
    t0 = time.time()
    heartbeat_stop: threading.Event | None = None
    try:
        _cancel_if_requested(job_id, queue, _post_heartbeat(job_id, worker_id))
        heartbeat_stop = _start_heartbeat_loop(job_id, queue, worker_id)
        df = _load_data(job, store)
        _raise_if_cancelled(job_id, queue)
        cfg = load_config(job["config"])
        max_rows = int(getattr(cfg, "max_rows", MAX_ROWS) or MAX_ROWS)
        if len(df) > max_rows:
            raise AdmissionError(
                f"row cap exceeded: {len(df)} rows > MAX_ROWS={max_rows} "
                f"(Spike-3 memory envelope; n>10k prohibitive). Downsample or raise the cap."
            )

        model = ThemaRS_fit(cfg, df)
        _raise_if_cancelled(job_id, queue)
        structure_status = _assess_structure(model)
        _raise_if_cancelled(job_id, queue)

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
        _raise_if_cancelled(job_id, queue)

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
    except SweepCancelled:
        return None
    except Exception as e:  # incl. AdmissionError / MemoryError → clean job failure
        queue.fail(job_id, f"{type(e).__name__}: {e}")
        return None
    finally:
        if heartbeat_stop is not None:
            heartbeat_stop.set()


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
