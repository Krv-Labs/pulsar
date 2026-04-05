from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import time
import uuid


_CACHE_DIR = Path(tempfile.gettempdir()) / "pulsar_mcp"
_DATASETS_PATH = _CACHE_DIR / "datasets.json"
_RUNS_DIR = _CACHE_DIR / "runs"


@dataclass
class DatasetRecord:
    dataset_id: str
    path: str
    name: str
    size_bytes: int
    mtime_ns: int
    ingested_at: float


@dataclass
class RunRecord:
    run_id: str
    dataset_id: str | None
    config_yaml: str
    metrics: dict
    resolved_threshold: float
    graph_summary: dict
    created_at: float


class MCPRegistry:
    """Small persistent registry for MCP dataset and run handles."""

    def __init__(self) -> None:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _RUNS_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> Path:
        return _CACHE_DIR

    def register_dataset(self, path: str) -> DatasetRecord:
        resolved = Path(path).expanduser().resolve(strict=True)
        stat = resolved.stat()
        digest = hashlib.sha256(
            f"{resolved}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
        ).hexdigest()[:12]
        record = DatasetRecord(
            dataset_id=f"ds_{digest}",
            path=str(resolved),
            name=resolved.name,
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            ingested_at=time.time(),
        )
        datasets = self._load_datasets()
        datasets[record.dataset_id] = asdict(record)
        self._write_json(_DATASETS_PATH, datasets)
        return record

    def get_dataset(self, dataset_id: str) -> DatasetRecord | None:
        raw = self._load_datasets().get(dataset_id)
        if raw is None:
            return None
        return DatasetRecord(**raw)

    def save_run(
        self,
        *,
        dataset_id: str | None,
        config_yaml: str,
        metrics: dict,
        resolved_threshold: float,
        graph_summary: dict,
    ) -> RunRecord:
        record = RunRecord(
            run_id=f"run_{uuid.uuid4().hex[:12]}",
            dataset_id=dataset_id,
            config_yaml=config_yaml,
            metrics=metrics,
            resolved_threshold=resolved_threshold,
            graph_summary=graph_summary,
            created_at=time.time(),
        )
        self._write_json(self._run_path(record.run_id), asdict(record))
        return record

    def get_run(self, run_id: str) -> RunRecord | None:
        path = self._run_path(run_id)
        if not path.exists():
            return None
        return RunRecord(**json.loads(path.read_text()))

    def _load_datasets(self) -> dict[str, dict]:
        if not _DATASETS_PATH.exists():
            return {}
        return json.loads(_DATASETS_PATH.read_text())

    def _run_path(self, run_id: str) -> Path:
        return _RUNS_DIR / f"{run_id}.json"

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
