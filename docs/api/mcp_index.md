# MCP

Optional [FastMCP](https://github.com/jlowin/fastmcp) server that exposes the ehrapy API to agents.
Install with `pip install ehrapy[mcp]` and start with `ehrapy-mcp`.

Agents discover functions through `list_ehrapy_functions` / `get_function_help` and call
`ep.pp`, `ep.tl`, `ep.get`, `ep.pl`, `ehrdata.io`, and `ehrdata.dt` via namespace dispatch tools.

```{eval-rst}
.. module:: ehrapy.mcp
    :no-index:
```

```{eval-rst}
.. autosummary::
    :toctree: mcp
    :nosignatures:

    mcp.main
    mcp.catalog_summary
    mcp.list_namespaces
    mcp.list_functions
```
