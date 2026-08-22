from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from fastmcp import Client

from ehrapy.mcp.server import mcp
from ehrapy.mcp.tools import ALL_TOOLS_LIST

if TYPE_CHECKING:
    from pathlib import Path

    pass


def _run(coro):
    return asyncio.run(coro)


def test_registered_tools_count_and_unique_names() -> None:
    assert len(ALL_TOOLS_LIST) == 14
    names = [fn.__name__ for fn in ALL_TOOLS_LIST]
    assert len(names) == len(set(names))


def test_get_workflow_guide() -> None:
    async def _test():
        async with Client(mcp) as client:
            res = await client.call_tool("get_workflow_guide", {})
            assert res.structured_content["status"] == "ok"
            assert "run_preprocessing" in res.content[0].text

    _run(_test())


def test_get_runtime_context() -> None:
    async def _test():
        async with Client(mcp) as client:
            res = await client.call_tool("get_runtime_context", {})
            assert res.structured_content["status"] == "ok"
            assert "ehrapy_version" in res.structured_content
            assert "cache_dir" in res.structured_content

    _run(_test())


def test_list_and_help() -> None:
    async def _test():
        async with Client(mcp) as client:
            res = await client.call_tool("list_ehrapy_functions", {"namespace": "preprocessing"})
            assert res.structured_content["status"] == "ok"
            assert res.structured_content["count"] >= 30

            help_res = await client.call_tool("get_function_help", {"namespace": "get", "function": "obs_df"})
            assert help_res.structured_content["status"] == "ok"
            assert help_res.structured_content["function"] == "obs_df"

    _run(_test())


def test_unknown_function_returns_error_envelope() -> None:
    async def _test():
        async with Client(mcp) as client:
            res = await client.call_tool("run_preprocessing", {"function": "not_a_real_function"})
            assert res.structured_content["status"] == "error"
            assert res.structured_content["error_code"] == "UNKNOWN_FUNCTION"

    _run(_test())


def test_ingest_and_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "patients.csv"
    path.write_text("patient_id,age,sex\np1,45,F\np2,62,M\n", encoding="utf-8")

    async def _test():
        async with Client(mcp) as client:
            ingest_res = await client.call_tool("ingest_dataset", {"file_path": str(path), "index_col": "patient_id"})
            assert ingest_res.structured_content["status"] == "ok"
            edata_id = ingest_res.structured_content["edata_id"]

            snap_res = await client.call_tool("get_edata_snapshot", {"edata_id": edata_id})
            assert snap_res.structured_content["status"] == "ok"
            assert snap_res.structured_content["n_obs"] == 2
            assert snap_res.structured_content["n_vars"] == 2

    _run(_test())


def test_demo_preprocessing_and_get() -> None:
    async def _test():
        async with Client(mcp) as client:
            demo_res = await client.call_tool("load_demo_dataset", {"dataset": "mimic_2"})
            assert demo_res.structured_content["status"] == "ok"
            edata_id = demo_res.structured_content["edata_id"]

            qc_res = await client.call_tool(
                "run_preprocessing",
                {"function": "qc_metrics", "edata_id": edata_id, "params": {"qc_vars": []}},
            )
            assert qc_res.structured_content["status"] == "ok"

            get_res = await client.call_tool(
                "run_get",
                {"function": "obs_df", "edata_id": edata_id, "params": {"keys": ["age"]}},
            )
            assert get_res.structured_content["status"] == "ok"
            assert get_res.structured_content["edata_id"] == edata_id

    _run(_test())
