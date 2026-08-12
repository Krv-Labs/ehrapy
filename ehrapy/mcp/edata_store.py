"""Load and persist EHRData objects by edata_id."""

from __future__ import annotations

import dataclasses
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from ehrdata.io import read_h5ed, write_h5ed

from ehrapy.mcp.registry import DatasetRecord, registry

if TYPE_CHECKING:
    from ehrdata import EHRData


def _cache_path(edata_id: str) -> Path:
    return registry.cache_dir() / "edata" / f"{edata_id}.h5ed"


def save_edata(
    edata: EHRData,
    *,
    name: str,
    source_path: str | None = None,
    fmt: str = "h5ed",
    edata_id: str | None = None,
    parent_id: str | None = None,
) -> DatasetRecord:
    """Persist EHRData to the MCP cache and register a handle."""
    edata_id = edata_id or str(uuid.uuid4())
    path = _cache_path(edata_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_h5ed(edata, path)
    stat = path.stat()
    record = DatasetRecord(
        edata_id=edata_id,
        cache_path=str(path),
        source_path=source_path,
        name=name,
        format=fmt,
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        ingested_at=time.time(),
        n_obs=edata.n_obs,
        n_vars=edata.n_vars,
        parent_id=parent_id,
    )
    registry.store_record(record)
    return record


def load_edata(edata_id: str) -> EHRData:
    """Load cached EHRData for an ``edata_id``."""
    record = registry.get_dataset(edata_id)
    if record is None:
        raise KeyError(edata_id)
    cache = Path(record.cache_path)
    if not cache.is_file():
        raise FileNotFoundError(record.cache_path)
    return read_h5ed(cache)


def fork_edata(edata_id: str, *, name: str | None = None) -> DatasetRecord:
    """Copy cached EHRData to a new handle."""
    edata = load_edata(edata_id)
    parent = registry.get_dataset(edata_id)
    label = name or f"{parent.name if parent else edata_id}-fork"
    return save_edata(edata, name=label, parent_id=edata_id)


def persist_edata(edata_id: str, edata: EHRData) -> DatasetRecord:
    """Overwrite the cached EHRData for an existing handle."""
    record = registry.get_dataset(edata_id)
    if record is None:
        return save_edata(edata, name=f"edata-{edata_id[:8]}")
    write_h5ed(edata, record.cache_path)
    stat = Path(record.cache_path).stat()
    updated = dataclasses.replace(
        record,
        n_obs=edata.n_obs,
        n_vars=edata.n_vars,
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        ingested_at=time.time(),
    )
    registry.store_record(updated)
    return updated
