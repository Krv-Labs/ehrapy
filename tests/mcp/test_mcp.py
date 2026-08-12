from __future__ import annotations

import asyncio
import json

import pytest

fastmcp = pytest.importorskip("fastmcp")

from typing import TYPE_CHECKING

from ehrapy.mcp.tools import ALL_TOOLS_LIST
from ehrapy.mcp.tools.dispatch_tools import (
    get_edata_snapshot,
    get_function_help,
    list_ehrapy_functions,
    load_demo_dataset,
    run_get,
    run_preprocessing,
)
from ehrapy.mcp.tools.ingestion import ingest_dataset
from ehrapy.mcp.tools.meta import get_package_info, get_workflow_guide

if TYPE_CHECKING:
    from pathlib import Path


def _run(coro):
    return asyncio.run(coro)


def test_registered_tools_have_unique_names() -> None:
    names = [fn.__name__ for fn in ALL_TOOLS_LIST]
    assert len(names) == len(set(names))


def test_get_workflow_guide() -> None:
    guide = _run(get_workflow_guide())
    assert "run_preprocessing" in guide


def test_get_package_info() -> None:
    payload = json.loads(_run(get_package_info()))
    assert payload["total_functions"] > 100
    assert "preprocessing" in payload["dispatch_namespaces"]


def test_list_and_help() -> None:
    catalog = json.loads(_run(list_ehrapy_functions()))
    assert catalog["preprocessing"]["count"] >= 30
    help_payload = json.loads(_run(get_function_help("get", "obs_df")))
    assert help_payload["function"] == "obs_df"


def test_unknown_function_returns_error_envelope() -> None:
    payload = json.loads(_run(run_preprocessing("not_a_real_function")))
    assert payload["status"] == "error"
    assert payload["error_code"] == "FUNCTION_UNKNOWN"


def test_ingest_and_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "patients.csv"
    path.write_text("patient_id,age,sex\np1,45,F\np2,62,M\n", encoding="utf-8")
    ingest_payload = json.loads(_run(ingest_dataset(str(path))))
    edata_id = ingest_payload["edata_id"]
    summary = json.loads(_run(get_edata_snapshot(edata_id=edata_id)))
    assert summary["status"] == "ok"
    assert summary["n_obs"] >= 1


def test_demo_preprocessing_and_get() -> None:
    loaded = json.loads(_run(load_demo_dataset("mimic_2")))
    edata_id = loaded["edata_id"]
    qc = json.loads(_run(run_preprocessing("qc_metrics", edata_id=edata_id, params={"qc_vars": []})))
    assert qc["status"] == "ok"
    result = json.loads(_run(run_get("obs_df", edata_id=edata_id, params={"keys": ["age"]})))
    assert result["status"] == "ok"
    assert result["result"]["type"] == "dataframe"
