from __future__ import annotations

import asyncio
import json

import pytest

fastmcp = pytest.importorskip("fastmcp")

from typing import TYPE_CHECKING

from ehrapy.mcp.server import mcp
from ehrapy.mcp.session import get_session
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
from ehrapy.mcp.tools.inspection import summarize_edata
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


def test_snapshot_includes_feature_and_obs_names() -> None:
    loaded = json.loads(_run(load_demo_dataset("mimic_2")))
    edata_id = loaded["edata_id"]
    snap = json.loads(_run(get_edata_snapshot(edata_id=edata_id)))
    assert snap["status"] == "ok"
    assert "var_names" in snap
    assert "obs_names" in snap
    assert len(snap["var_names"]) > 0
    assert "aline_flg" in snap["var_names"]
    assert None not in snap["layers"]


def test_run_get_without_keys_returns_annotation_table() -> None:
    loaded = json.loads(_run(load_demo_dataset("mimic_2")))
    edata_id = loaded["edata_id"]
    obs_res = json.loads(_run(run_get("obs_df", edata_id=edata_id)))
    assert obs_res["status"] == "ok"
    assert obs_res["result"]["type"] == "dataframe"
    assert len(obs_res["result"]["columns"]) >= 1
    assert obs_res["result"]["columns"][0] == "index"

    var_res = json.loads(_run(run_get("var_df", edata_id=edata_id)))
    assert var_res["status"] == "ok"
    assert var_res["result"]["type"] == "dataframe"
    assert "index" in var_res["result"]["columns"]
    assert len(var_res["result"]["data"]) == 46


def test_catalog_and_help_polish() -> None:
    help_var = json.loads(_run(get_function_help("get", "var_df")))
    assert "variables/features" in help_var["docstring"]
    for p in help_var["parameters"]:
        if "annotation" in p:
            assert "<class" not in p["annotation"]
            assert "collections.abc" not in p["annotation"]

    unknown_dt = json.loads(_run(load_demo_dataset("not_a_cohort")))
    assert unknown_dt["status"] == "error"
    assert unknown_dt["error_code"] == "DATASET_UNKNOWN"

    summary_alias = json.loads(_run(summarize_edata()))
    assert summary_alias["status"] == "ok" or summary_alias["error_code"] == "EDATA_ID_MISSING"


def test_agnostic_fastmcp_folds_top_level_kwargs() -> None:
    loaded = json.loads(_run(load_demo_dataset("mimic_2")))
    edata_id = loaded["edata_id"]
    # Call with top-level kwargs instead of inside `params`
    result_raw = _run(
        mcp.call_tool(
            "run_analysis",
            {
                "function": "kaplan_meier",
                "edata_id": edata_id,
                "duration_col": "icu_los_day",
                "event_col": "hosp_exp_flg",
            },
        )
    )
    # result_raw can be a CallToolResult object or list of contents
    content_text = result_raw.content[0].text if hasattr(result_raw, "content") else str(result_raw)
    result = json.loads(content_text)
    assert result["status"] == "ok"
    assert result["function"] == "kaplan_meier"


def test_session_thread_safety() -> None:
    session = get_session()
    session.edata_id = "test-id"
    assert session.edata_id == "test-id"

    import concurrent.futures

    def set_id(i: int):
        s = get_session()
        s.edata_id = f"id-{i}"
        return s.edata_id

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(set_id, i) for i in range(20)]
        results = [f.result() for f in futures]
    assert all(r.startswith("id-") for r in results)

