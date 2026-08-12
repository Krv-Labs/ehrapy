"""FastMCP server for ehrapy."""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

_KNOWN_ORCHESTRATION_KEYS = frozenset({"wait_for_previous"})


class AgnosticFastMCP(FastMCP):
    """Strip non-schema params injected by some MCP clients."""

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ):
        """Strip client-injected arguments that are not in the tool schema."""
        if arguments:
            tool = await self.get_tool(name, version=kwargs.get("version"))
            if tool is not None:
                valid_keys = set(tool.parameters.get("properties", {}).keys())
                unexpected = set(arguments.keys()) - valid_keys
                if unexpected:
                    unrecognized = unexpected - _KNOWN_ORCHESTRATION_KEYS
                    if unrecognized:
                        logger.warning(
                            "Stripped unrecognized parameters from tool %s: %s",
                            name,
                            sorted(unrecognized),
                        )
                arguments = {k: v for k, v in arguments.items() if k in valid_keys}
        return await super().call_tool(name, arguments, *args, **kwargs)


mcp = AgnosticFastMCP(
    "ehrapy",
    instructions=(
        "Full ehrapy EHR analysis API via namespace dispatch. "
        "Start with get_workflow_guide. Discover functions via list_ehrapy_functions. "
        "Handles: edata_id (cached EHRData). "
        "Dispatch: run_preprocessing (ep.pp), run_analysis (ep.tl), "
        "run_get, run_plot, run_io, load_demo_dataset."
    ),
)

from ehrapy.mcp.tools import ALL_TOOLS_LIST

for tool_fn in ALL_TOOLS_LIST:
    mcp.tool()(tool_fn)


def main() -> None:
    """Start the ehrapy FastMCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
