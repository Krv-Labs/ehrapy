"""Regression tests for `_locked_registry` error handling.

The outer ``except OSError: yield`` used to catch errors from the body
itself, which then yielded a second time and surfaced as
``RuntimeError: generator didn't stop after throw()`` -- destroying the
real I/O error. Only lock setup may be tolerated; body errors must
propagate.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

import pytest

from ehrapy.mcp.registry import DatasetRecord, MCPRegistry


@pytest.fixture(autouse=True)
def _restore_global_dataset_dirs():
    """Undo the process-wide demo-path retargeting MCPRegistry performs on construction."""
    import ehrdata.core.constants as ed_const
    import ehrdata.dt.datasets as ed_datasets
    import scanpy as sc

    import ehrapy as ep

    before = (
        ep.settings.datasetdir,
        sc.settings.datasetdir,
        ed_const.DEFAULT_DATA_PATH,
        ed_datasets.DEFAULT_DATA_PATH,
    )
    yield
    ep.settings.datasetdir, sc.settings.datasetdir = before[0], before[1]
    ed_const.DEFAULT_DATA_PATH = before[2]
    ed_datasets.DEFAULT_DATA_PATH = before[3]


def _record(edata_id: str, cache_dir: Path) -> DatasetRecord:
    return DatasetRecord(
        edata_id=edata_id,
        cache_path=str(cache_dir / "edata" / f"{edata_id}.h5ad"),
        name=f"dataset-{edata_id}",
        format="h5ad",
        size_bytes=123,
        mtime_ns=456,
        ingested_at=789.0,
    )


def test_unopenable_lock_file_still_runs_body(tmp_path: Path) -> None:
    """TEST A: a lock file that cannot be opened must not block the body."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / ".registry.lock").mkdir()  # opening a directory as a file raises OSError
    reg = MCPRegistry(cache_dir=cache_dir)

    ran = False
    with reg._locked_registry():
        ran = True
    assert ran


def test_oserror_from_body_propagates_untouched(tmp_path: Path) -> None:
    """TEST B: an OSError inside the body is the caller's error, not a lock failure."""
    reg = MCPRegistry(cache_dir=tmp_path / "cache")

    with pytest.raises(OSError, match="disk full") as excinfo:
        with reg._locked_registry():
            raise OSError("disk full")
    assert excinfo.type is OSError


def test_runtimeerror_never_replaces_body_error(tmp_path: Path) -> None:
    reg = MCPRegistry(cache_dir=tmp_path / "cache")

    try:
        with reg._locked_registry():
            raise OSError("write failed")
    except RuntimeError as err:
        pytest.fail(f"body OSError was masked as RuntimeError: {err}")
    except OSError:
        pass


def test_file_lock_released_when_body_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    original_acquire = MCPRegistry._acquire_file_lock
    original_release = MCPRegistry._release_file_lock

    def acquire(handle) -> None:
        calls.append("acquire")
        original_acquire(handle)

    def release(handle) -> None:
        calls.append("release")
        original_release(handle)

    monkeypatch.setattr(MCPRegistry, "_acquire_file_lock", staticmethod(acquire))
    monkeypatch.setattr(MCPRegistry, "_release_file_lock", staticmethod(release))

    reg = MCPRegistry(cache_dir=tmp_path / "cache")
    with pytest.raises(OSError, match="boom"):
        with reg._locked_registry():
            raise OSError("boom")
    assert calls == ["acquire", "release"]


def test_store_record_round_trip(tmp_path: Path) -> None:
    reg = MCPRegistry(cache_dir=tmp_path / "cache")
    rec = _record("abc123", tmp_path / "cache")
    reg.store_record(rec)
    assert reg.get_dataset("abc123") == rec
    assert [d.edata_id for d in reg.list_datasets()] == ["abc123"]
