from __future__ import annotations

from fastmcp import Context  # noqa: TC002

from ehrapy.mcp.tools.dispatch_tools import get_edata_snapshot


async def summarize_edata(
    edata_id: str | None = None,
    ctx: Context = None,
) -> str:
    """Return obs/var column names, shape, layers, and uns keys for an edata_id (alias of get_edata_snapshot)."""
    return await get_edata_snapshot(edata_id=edata_id, ctx=ctx)

