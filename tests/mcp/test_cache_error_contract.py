"""Contract: a CACHE_DIR_UNAVAILABLE from the cache surfaces identically everywhere.

Every tool path that writes to the cache (ingest, fork, dispatched mutations)
must translate CacheDirUnavailableError into error_code="CACHE_DIR_UNAVAILABLE"
with a usable agent_action, and unrelated exceptions must keep their original
error codes.
"""

from __future__ import annotations

import asyncio
from pathlib import Path  # noqa: TC003

import pytest
from fastmcp import Client

import ehrapy.mcp.registry as registry_module
from ehrapy.mcp.server import mcp


def _run(coro):
    return asyncio.run(coro)


def _lock_cache_and_break_probe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Lock confinement and make every cache probe fail.

    The registry singleton keeps the real cache path it resolved at import;
    patching the probe makes ``require_usable()`` raise the real
    CacheDirUnavailableError, exactly as a vanished pinned directory would.
    """
    monkeypatch.setenv("EHRAPY_MCP_CACHE_DIR", str(tmp_path / "pinned-cache"))
    monkeypatch.delenv("EHRAPY_MCP_ALLOWED_ROOTS", raising=False)
    monkeypatch.delenv("EHRAPY_MCP_READ_ONLY", raising=False)
    monkeypatch.setattr(registry_module, "_probe_writable", lambda path: False)


def _ingest_csv(csv: Path) -> str:
    async def _call():
        async with Client(mcp) as client:
            return await client.call_tool("ingest_dataset", {"file_path": str(csv)})

    res = _run(_call())
    assert res.structured_content["status"] == "ok", res.structured_content
    return res.structured_content["edata_id"]


@pytest.fixture
def cohort(tmp_path: Path) -> Path:
    csv = tmp_path / "cohort.csv"
    csv.write_text("subject_id,age,sex\n1,60,F\n2,55,M\n3,71,F\n", encoding="utf-8")
    return csv


def _assert_cache_dir_unavailable(struct: dict) -> None:
    assert struct["status"] == "error"
    assert struct["error_code"] == "CACHE_DIR_UNAVAILABLE"
    assert struct.get("agent_action"), "agent_action must survive the tool boundary"


def test_ingest_surfaces_cache_dir_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cohort: Path) -> None:
    _ingest_csv(cohort)  # baseline: unlocked ingest works
    _lock_cache_and_break_probe(monkeypatch, tmp_path)

    async def _test():
        async with Client(mcp) as client:
            other = tmp_path / "cohort2.csv"
            other.write_text("subject_id,age\n4,66\n5,80\n", encoding="utf-8")
            res = await client.call_tool("ingest_dataset", {"file_path": str(other)})
            return res.structured_content

    _assert_cache_dir_unavailable(_run(_test()))


def test_ingest_unrelated_error_keeps_original_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cohort: Path
) -> None:
    import ehrdata.io as ed_io

    def _boom(*args, **kwargs):
        raise RuntimeError("parser exploded")

    monkeypatch.setattr(ed_io, "read_csv", _boom)

    async def _test():
        async with Client(mcp) as client:
            res = await client.call_tool("ingest_dataset", {"file_path": str(cohort)})
            return res.structured_content

    struct = _run(_test())
    assert struct["status"] == "error"
    assert struct["error_code"] == "INGEST_ERROR"


def test_fork_surfaces_cache_dir_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cohort: Path) -> None:
    edata_id = _ingest_csv(cohort)
    _lock_cache_and_break_probe(monkeypatch, tmp_path)

    async def _test():
        async with Client(mcp) as client:
            res = await client.call_tool("fork_edata_handle", {"edata_id": edata_id})
            return res.structured_content

    _assert_cache_dir_unavailable(_run(_test()))


def test_fork_unrelated_error_keeps_original_code(monkeypatch: pytest.MonkeyPatch, cohort: Path) -> None:
    import ehrapy.mcp.tools.dispatch_tools as dispatch_tools

    edata_id = _ingest_csv(cohort)

    def _boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(dispatch_tools, "fork_edata", _boom)

    async def _test():
        async with Client(mcp) as client:
            res = await client.call_tool("fork_edata_handle", {"edata_id": edata_id})
            return res.structured_content

    struct = _run(_test())
    assert struct["status"] == "error"
    assert struct["error_code"] == "FORK_ERROR"


def test_dispatch_surfaces_cache_dir_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cohort: Path) -> None:
    edata_id = _ingest_csv(cohort)
    _lock_cache_and_break_probe(monkeypatch, tmp_path)

    import ehrapy as ep

    def _noop(edata, *args, **kwargs):
        return None

    monkeypatch.setattr(ep.preprocessing, "qc_metrics", _noop)

    async def _test():
        async with Client(mcp) as client:
            res = await client.call_tool(
                "run_preprocessing",
                {"function": "qc_metrics", "edata_id": edata_id, "params": {"qc_vars": []}},
            )
            return res.structured_content

    _assert_cache_dir_unavailable(_run(_test()))


def test_dispatch_unrelated_error_keeps_original_code(monkeypatch: pytest.MonkeyPatch, cohort: Path) -> None:
    edata_id = _ingest_csv(cohort)

    import ehrapy as ep

    def _boom(*args, **kwargs):
        raise ValueError("bad cohort")

    monkeypatch.setattr(ep.preprocessing, "qc_metrics", _boom)

    async def _test():
        async with Client(mcp) as client:
            res = await client.call_tool(
                "run_preprocessing",
                {"function": "qc_metrics", "edata_id": edata_id, "params": {"qc_vars": []}},
            )
            return res.structured_content

    struct = _run(_test())
    assert struct["status"] == "error"
    assert struct["error_code"] == "INVALID_VALUE"
