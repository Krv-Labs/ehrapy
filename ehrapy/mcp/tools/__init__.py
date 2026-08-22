from __future__ import annotations

from ehrapy.mcp.tools.dispatch_tools import (
    export_edata,
    fork_edata_handle,
    get_edata_snapshot,
    get_function_help,
    list_ehrapy_functions,
    load_demo_dataset,
    run_analysis,
    run_get,
    run_io,
    run_plot,
    run_preprocessing,
)
from ehrapy.mcp.tools.ingestion import ingest_dataset
from ehrapy.mcp.tools.inspection import summarize_edata
from ehrapy.mcp.tools.meta import get_package_info, get_runtime_context, get_workflow_guide

ALL_TOOLS_LIST = [
    # Meta
    get_workflow_guide,
    get_runtime_context,
    get_package_info,
    list_ehrapy_functions,
    get_function_help,
    # Data handles
    ingest_dataset,
    load_demo_dataset,
    fork_edata_handle,
    export_edata,
    get_edata_snapshot,
    summarize_edata,
    # Dispatch (full ehrapy API surface)
    run_preprocessing,
    run_analysis,
    run_get,
    run_plot,
    run_io,
]

__all__ = [
    "ALL_TOOLS_LIST",
    "get_workflow_guide",
    "get_runtime_context",
    "get_package_info",
    "list_ehrapy_functions",
    "get_function_help",
    "ingest_dataset",
    "load_demo_dataset",
    "fork_edata_handle",
    "export_edata",
    "get_edata_snapshot",
    "summarize_edata",
    "run_preprocessing",
    "run_analysis",
    "run_get",
    "run_plot",
    "run_io",
]
