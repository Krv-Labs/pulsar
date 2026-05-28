from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid


_CACHE_DIR = Path(tempfile.gettempdir()) / "pulsar_mcp"
_DATASETS_PATH = _CACHE_DIR / "datasets.json"
_DATASET_FILES_DIR = _CACHE_DIR / "datasets"
_UPLOADS_DIR = _CACHE_DIR / "uploads"
_RUNS_DIR = _CACHE_DIR / "runs"
_LOCK_PATH = _CACHE_DIR / ".registry.lock"
_PROCESS_LOCK = threading.RLock()
_UTF8_BOM = b"\xef\xbb\xbf"


def _strip_utf8_bom_in_place(path: Path) -> None:
    with path.open("rb") as handle:
        prefix = handle.read(len(_UTF8_BOM))
    if prefix != _UTF8_BOM:
        return
    contents = path.read_bytes()
    path.write_bytes(contents[len(_UTF8_BOM) :])


@dataclass
class DatasetRecord:
    dataset_id: str
    path: str
    name: str
    size_bytes: int
    mtime_ns: int
    ingested_at: float
    source: str = "path"


@dataclass
class RunRecord:
    run_id: str
    dataset_id: str | None
    config_yaml: str
    metrics: dict
    resolved_construction_threshold: float
    graph_summary: dict
    created_at: float


STALE_RUN_RECORD_MESSAGE = (
    "Run record uses removed field 'resolved_threshold'. "
    "Rerun the sweep to create a record with "
    "'resolved_construction_threshold'."
)


@dataclass
class UploadRecord:
    upload_id: str
    filename: str
    media_type: str
    created_at: float
    bytes_received: int


class MCPRegistry:
    """Small persistent registry for MCP dataset and run handles."""

    def __init__(self) -> None:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _DATASET_FILES_DIR.mkdir(parents=True, exist_ok=True)
        _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        _RUNS_DIR.mkdir(parents=True, exist_ok=True)
        _LOCK_PATH.touch(exist_ok=True)

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
        with self._locked_registry():
            self._store_dataset_record_unlocked(record)
        return record

    def register_dataset_from_file(
        self,
        filename: str,
        path: Path,
        *,
        source: str,
    ) -> DatasetRecord:
        """Register a dataset from an existing file using streaming move (zero RAM)."""
        with self._locked_registry():
            return self._register_dataset_from_file_unlocked(
                filename,
                path,
                source=source,
            )

    def begin_upload(self, filename: str, media_type: str = "text/csv") -> UploadRecord:
        upload_id = f"upload_{uuid.uuid4().hex[:12]}"
        safe_name = Path(filename).name or "uploaded.csv"
        record = UploadRecord(
            upload_id=upload_id,
            filename=safe_name,
            media_type=media_type,
            created_at=time.time(),
            bytes_received=0,
        )
        with self._locked_registry():
            self._staging_path(upload_id).write_bytes(b"")
            self._write_json(self._upload_meta_path(upload_id), asdict(record))
        return record

    def append_upload_chunk(self, upload_id: str, chunk: bytes) -> UploadRecord | None:
        with self._locked_registry():
            record = self.get_upload(upload_id)
            if record is None:
                return None
            with self._staging_path(upload_id).open("ab") as handle:
                handle.write(chunk)
            record.bytes_received += len(chunk)
            self._write_json(self._upload_meta_path(upload_id), asdict(record))
            return record

    def finalize_upload(self, upload_id: str) -> DatasetRecord | None:
        with self._locked_registry():
            record = self.get_upload(upload_id)
            if record is None:
                return None
            staging_path = self._staging_path(upload_id)
            dataset = self._register_dataset_from_file_unlocked(
                record.filename,
                staging_path,
                source="upload",
            )
            # register_dataset_from_file already moved the staging file
            self._upload_meta_path(upload_id).unlink(missing_ok=True)
            return dataset

    def get_upload(self, upload_id: str) -> UploadRecord | None:
        path = self._upload_meta_path(upload_id)
        if not path.exists():
            return None
        return UploadRecord(**json.loads(path.read_text()))

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
        resolved_construction_threshold: float,
        graph_summary: dict,
    ) -> RunRecord:
        record = RunRecord(
            run_id=f"run_{uuid.uuid4().hex[:12]}",
            dataset_id=dataset_id,
            config_yaml=config_yaml,
            metrics=metrics,
            resolved_construction_threshold=resolved_construction_threshold,
            graph_summary=graph_summary,
            created_at=time.time(),
        )
        with self._locked_registry():
            self._write_json(self._run_path(record.run_id), asdict(record))
        return record

    def get_run(self, run_id: str) -> RunRecord | None:
        path = self._run_path(run_id)
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        if "resolved_threshold" in raw:
            raise ValueError(STALE_RUN_RECORD_MESSAGE)
        return RunRecord(**raw)

    def _load_datasets(self) -> dict[str, dict]:
        with self._locked_registry():
            return self._load_datasets_unlocked()

    def _load_datasets_unlocked(self) -> dict[str, dict]:
        if not _DATASETS_PATH.exists():
            return {}
        return json.loads(_DATASETS_PATH.read_text())

    def _run_path(self, run_id: str) -> Path:
        return _RUNS_DIR / f"{run_id}.json"

    def _upload_meta_path(self, upload_id: str) -> Path:
        return _UPLOADS_DIR / f"{upload_id}.json"

    def _staging_path(self, upload_id: str) -> Path:
        return _UPLOADS_DIR / f"{upload_id}.bin"

    def _store_dataset_record_unlocked(self, record: DatasetRecord) -> None:
        datasets = self._load_datasets_unlocked()
        datasets[record.dataset_id] = asdict(record)
        self._write_json(_DATASETS_PATH, datasets)

    def _register_dataset_from_file_unlocked(
        self,
        filename: str,
        path: Path,
        *,
        source: str,
    ) -> DatasetRecord:
        import shutil

        safe_name = Path(filename).name or "uploaded.csv"

        _strip_utf8_bom_in_place(path)

        sha256_hash = hashlib.sha256(source.encode("utf-8") + b"\x00")
        with path.open("rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)

        dataset_id = f"ds_{sha256_hash.hexdigest()[:12]}"
        stored_path = _DATASET_FILES_DIR / f"{dataset_id}_{safe_name}"
        shutil.move(str(path), str(stored_path))

        stat = stored_path.stat()
        record = DatasetRecord(
            dataset_id=dataset_id,
            path=str(stored_path),
            name=safe_name,
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            ingested_at=time.time(),
            source=source,
        )
        self._store_dataset_record_unlocked(record)
        return record

    @contextmanager
    def _locked_registry(self):
        with _PROCESS_LOCK:
            with _LOCK_PATH.open("a+b") as handle:
                self._acquire_file_lock(handle)
                try:
                    yield
                finally:
                    self._release_file_lock(handle)

    @staticmethod
    def _acquire_file_lock(handle) -> None:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            return
        except ImportError:
            pass

        try:
            import msvcrt

            handle.seek(0, os.SEEK_SET)
            if handle.tell() == 0 and handle.read(1) == b"":
                handle.write(b"0")
                handle.flush()
                handle.seek(0, os.SEEK_SET)
            else:
                handle.seek(0, os.SEEK_SET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        except ImportError:
            return

    @staticmethod
    def _release_file_lock(handle) -> None:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return
        except ImportError:
            pass

        try:
            import msvcrt

            handle.seek(0, os.SEEK_SET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except ImportError:
            return

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        serialized = json.dumps(payload, indent=2, sort_keys=True)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                dir=path.parent,
                delete=False,
                encoding="utf-8",
            ) as tmp:
                tmp.write(serialized)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)


registry = MCPRegistry()
