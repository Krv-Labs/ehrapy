from __future__ import annotations

import json
from pathlib import Path

from fastmcp import Context  # noqa: TC002

import ehrapy as ep
from ehrapy.mcp.catalog import catalog_summary
from ehrapy.mcp.prompts import WORKFLOW_PROMPT
from ehrapy.mcp.registry import registry
from ehrapy.mcp.session import get_session


async def get_workflow_guide(ctx: Context = None) -> str:
    """Recommended ehrapy MCP workflow and namespace map. Call once at session start."""
    return WORKFLOW_PROMPT


async def get_runtime_context(ctx: Context = None) -> str:
    """MCP host runtime: cwd, cache_dir, session edata_id, registered datasets."""
    session = get_session(ctx)
    datasets = [
        {"edata_id": r.edata_id, "name": r.name, "n_obs": r.n_obs, "n_vars": r.n_vars} for r in registry.list_datasets()
    ]
    payload = {
        "status": "ok",
        "cwd": str(Path.cwd()),
        "cache_dir": str(registry.cache_dir()),
        "plots_dir": str(registry.plots_dir()),
        "latest_edata_id": session.edata_id,
        "registered_datasets": datasets[-20:],
        "ehrapy_version": ep.__version__,
    }
    return json.dumps(payload, indent=2)


async def get_package_info(ctx: Context = None) -> str:
    """Ehrapy version and dispatch namespace summary."""
    summary = catalog_summary()
    totals = {ns: spec["count"] for ns, spec in summary.items()}
    payload = {
        "status": "ok",
        "ehrapy_version": ep.__version__,
        "dispatch_namespaces": summary,
        "total_functions": sum(totals.values()),
    }
    return json.dumps(payload, indent=2)
