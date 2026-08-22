# MCP Server

The `ehrapy` Model Context Protocol (MCP) server enables AI agents (Claude Desktop, Cursor, Gemini CLI, Goose, etc.) to perform exploratory electronic health record (EHR) data analysis safely and efficiently.

Install with:

```console
pip install ehrapy[mcp]
```

Start with:

```console
ehrapy-mcp
```

## Tool Catalog (14 Tools)

| Tool Name | Namespace / Kind | Annotations | Description |
| --- | --- | --- | --- |
| `get_workflow_guide` | Meta | `readOnlyHint=True` | Return structured guide for end-to-end EHR analysis workflows |
| `get_runtime_context` | Meta | `readOnlyHint=True` | Return runtime status, active dataset handle, and loaded namespaces |
| `list_ehrapy_functions` | Meta | `readOnlyHint=True` | List available functions in a namespace or across all namespaces |
| `get_function_help` | Meta | `readOnlyHint=True` | Get parameter table, docstrings, choices, and synthetic example calls |
| `ingest_dataset` | Ingestion | `readOnlyHint=False` | Read flat/wide/long CSV, TSV, or h5ad/h5ed files into a managed EHRData handle |
| `export_edata` | Ingestion | `readOnlyHint=False` | Export an EHRData dataset to CSV, TSV, or h5ed on disk |
| `get_edata_snapshot` | Inspection | `readOnlyHint=True` | Get lightweight overview (obs/var counts, columns, layers, uns keys) |
| `fork_edata_handle` | Session | `readOnlyHint=False` | Fork an existing dataset handle for branching analysis |
| `load_demo_dataset` | Demo | `readOnlyHint=False` | Load standard demo datasets (e.g., `mimic_2`) |
| `run_preprocessing` | Dispatch | `readOnlyHint=False` | Dispatch to `ehrapy.preprocessing` (`ep.pp`) functions |
| `run_analysis` | Dispatch | `readOnlyHint=False` | Dispatch to `ehrapy.tools` (`ep.tl`) analysis algorithms |
| `run_get` | Dispatch | `readOnlyHint=True` | Query observations, variables, or differential results (`ep.get`) |
| `run_plot` | Dispatch | `readOnlyHint=True` | Render quality control, clustering, survival, or embedding figures |
| `run_io` | Dispatch | `readOnlyHint=False` | Read and write datasets with `ehrdata.io` |

## Workflows & Prompts

The MCP server provides 3 pre-built MCP Prompts for guiding agent workflows:

- `ehrapy-explore`: Complete exploratory data analysis and quality control workflow.
- `ehrapy-clustering`: Full unsupervised clustering, dimensionality reduction, and biomarker ranking.
- `ehrapy-survival`: Kaplan-Meier and Cox proportional hazards survival analysis.

## 3-Tier Result Serialization

Tool outputs are returned as dual-channel `ToolResult` objects containing both human/LLM-readable Markdown `content` and typed `structured_content`:

1. **Tier 1 (Pass-through):** Small tables (<= 2,500 chars) are rendered as complete Markdown tables.
2. **Tier 2 (Column Profile):** Larger tables are summarized as column profiles (ranked by missingness, variance, and relevance).
3. **Tier 3 (Row Samples):** Passing `response_format="detailed"` includes head, tail, and deterministic random samples.
4. **Plots & Visualizations:** Matplotlib figures are automatically rendered to PNG and returned as multimodal image blocks alongside summary text.

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
