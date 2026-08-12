"""Persistent dataset registry for EHRData handles."""

from __future__ import annotations

import json
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

_CACHE_DIR = Path(tempfile.gettempdir()) / "ehrapy_mcp"
_DATASETS_PATH = _CACHE_DIR / "datasets.json"
_LOCK_PATH = _CACHE_DIR / ".registry.lock"
_PROCESS_LOCK = threading.RLock()


@dataclass
class DatasetRecord:
    """Metadata for a cached EHRData handle."""

    edata_id: str
    cache_path: str
    name: str
    format: str
    size_bytes: int
    mtime_ns: int
    ingested_at: float
    source_path: str | None = None
    n_obs: int | None = None
    n_vars: int | None = None
    parent_id: str | None = None


class MCPRegistry:
    """Process-wide registry of cached EHRData handles."""

    def __init__(self) -> None:
        """Create cache directories."""
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (_CACHE_DIR / "edata").mkdir(parents=True, exist_ok=True)
        (_CACHE_DIR / "plots").mkdir(parents=True, exist_ok=True)

    def cache_dir(self) -> Path:
        """Return the MCP cache root."""
        return _CACHE_DIR

    def plots_dir(self) -> Path:
        """Return the plot artifact directory."""
        return _CACHE_DIR / "plots"

    def store_record(self, record: DatasetRecord) -> DatasetRecord:
        """Insert or update a dataset record."""
        with self._locked_registry():
            datasets = self._load_datasets_unlocked()
            datasets[record.edata_id] = asdict(record)
            self._write_json(_DATASETS_PATH, datasets)
        return record

    def get_dataset(self, edata_id: str) -> DatasetRecord | None:
        """Return a dataset record by handle, if present."""
        payload = self._load_datasets().get(edata_id)
        if payload is None:
            return None
        return DatasetRecord(**payload)

    def list_datasets(self) -> list[DatasetRecord]:
        """Return all registered dataset records."""
        return [DatasetRecord(**v) for v in self._load_datasets().values()]

    def _load_datasets(self) -> dict[str, dict]:
        with self._locked_registry():
            return self._load_datasets_unlocked()

    def _load_datasets_unlocked(self) -> dict[str, dict]:
        if not _DATASETS_PATH.is_file():
            return {}
        return json.loads(_DATASETS_PATH.read_text(encoding="utf-8"))

    @contextmanager
    def _locked_registry(self):
        with _PROCESS_LOCK:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with _LOCK_PATH.open("a+", encoding="utf-8") as handle:
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
        except ImportError:
            pass

    @staticmethod
    def _release_file_lock(handle) -> None:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except ImportError:
            pass

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)


registry = MCPRegistry()
