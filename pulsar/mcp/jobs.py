"""Local async sweep queue + status store (Stage 1, build-spec §4 step 5).

The MCP request must NOT hold a sweep: ``run_topological_sweep`` enqueues a job and
returns a ``job_id``; a separate ``worker`` process claims it, runs ``ThemaRS.fit``,
persists the derived artifact, and updates status; ``get_sweep_status`` polls.

``FsJobQueue`` is filesystem-backed: hermetic (no external service), cross-process,
good for local dev + tests. A Redis-backed ``LocalQueue`` is the documented
multi-worker alternative; the prod swap is Cloud Tasks (same enqueue/claim surface).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

QUEUED, RUNNING, DONE, ERROR, CANCELLED = (
    "queued",
    "running",
    "done",
    "error",
    "cancelled",
)


def config_hash(config: dict) -> str:
    """Stable hash of the canonicalized sweep config (build-spec §3.1)."""
    canon = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return "cfg_" + hashlib.sha256(canon.encode()).hexdigest()[:16]


def new_job_id() -> str:
    return "job_" + uuid.uuid4().hex[:16]


class FsJobQueue:
    """Filesystem job queue + status store. One JSON record per job."""

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root).expanduser().resolve()
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)

    def _job_path(self, job_id: str) -> Path:
        return self.root / "jobs" / f"{job_id}.json"

    def _write(self, rec: dict) -> None:
        p = self._job_path(rec["job_id"])
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, p)  # atomic

    def _read(self, job_id: str) -> Optional[dict]:
        p = self._job_path(job_id)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def _unlock(self, job_id: str) -> None:
        try:
            self._job_path(job_id).with_suffix(".lock").unlink()
        except FileNotFoundError:
            pass

    def enqueue(self, payload: dict) -> str:
        """Register a queued job. ``payload`` carries user_id/dataset_id/config_hash/data_path/config."""
        job_id = new_job_id()
        rec = {
            "job_id": job_id,
            "status": QUEUED,
            "created_at": time.time(),
            "claimed_at": None,
            "finished_at": None,
            "artifact_ref": None,
            "structure_status": None,
            "error": None,
            "cancel_reason": None,
            "cancelled_at": None,
            "progress_stage": "queued",
            "progress_fraction": 0.0,
            "progress_updated_at": time.time(),
            "peak_rss_mb": None,
            "vcpu_ms": None,
            **payload,
        }
        self._write(rec)
        return job_id

    def claim(self) -> Optional[dict]:
        """Atomically claim the oldest queued job (O_EXCL lock). Returns the running record."""
        recs = []
        for p in (self.root / "jobs").glob("*.json"):
            try:
                r = json.loads(p.read_text())
            except Exception:
                continue
            if r.get("status") == QUEUED:
                recs.append(r)
        recs.sort(key=lambda r: r.get("created_at", 0.0))
        for rec in recs:
            lock = self._job_path(rec["job_id"]).with_suffix(".lock")
            try:
                fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
            except FileExistsError:
                continue
            cur = self._read(rec["job_id"])
            if cur is None or cur.get("status") != QUEUED:
                self._unlock(rec["job_id"])
                continue
            cur["status"] = RUNNING
            cur["claimed_at"] = time.time()
            cur["progress_stage"] = "claimed"
            cur["progress_fraction"] = max(
                float(cur.get("progress_fraction") or 0.0), 0.01
            )
            cur["progress_updated_at"] = time.time()
            self._write(cur)
            return cur
        return None

    def progress(self, job_id: str, *, stage: str, fraction: float) -> None:
        rec = self._read(job_id)
        if rec is None or rec.get("status") == CANCELLED:
            return
        rec["progress_stage"] = str(stage)
        rec["progress_fraction"] = max(0.0, min(float(fraction), 1.0))
        rec["progress_updated_at"] = time.time()
        self._write(rec)

    def complete(
        self,
        job_id: str,
        *,
        artifact_ref: dict,
        structure_status: str = "ok",
        peak_rss_mb: float | None = None,
        vcpu_ms: int | None = None,
    ) -> None:
        rec = self._read(job_id)
        if rec is None:
            return
        if rec.get("status") == CANCELLED:
            self._unlock(job_id)
            return
        rec.update(
            status=DONE,
            artifact_ref=artifact_ref,
            structure_status=structure_status,
            finished_at=time.time(),
            progress_stage="complete",
            progress_fraction=1.0,
            progress_updated_at=time.time(),
            peak_rss_mb=peak_rss_mb,
            vcpu_ms=vcpu_ms,
        )
        self._write(rec)
        self._unlock(job_id)

    def fail(self, job_id: str, error: str) -> None:
        rec = self._read(job_id)
        if rec is None:
            return
        if rec.get("status") == CANCELLED:
            self._unlock(job_id)
            return
        rec.update(status=ERROR, error=str(error), finished_at=time.time())
        self._write(rec)
        self._unlock(job_id)

    def cancel(self, job_id: str, reason: str = "cancel_requested") -> None:
        rec = self._read(job_id)
        if rec is None:
            return
        rec.update(
            status=CANCELLED,
            cancel_reason=reason,
            cancelled_at=time.time(),
            finished_at=time.time(),
        )
        self._write(rec)
        self._unlock(job_id)

    def status(self, job_id: str) -> Optional[dict]:
        return self._read(job_id)


def get_job_queue() -> FsJobQueue:
    """Local job queue rooted under ``JOB_QUEUE_DIR`` (default: ``<OBJECT_STORE_DIR>/_queue``)."""
    default = os.path.join(
        os.environ.get("OBJECT_STORE_DIR", "./.localstore"), "_queue"
    )
    return FsJobQueue(os.environ.get("JOB_QUEUE_DIR", default))
