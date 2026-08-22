from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import Context

from ehrapy.mcp.catalog import catalog_summary, list_functions, list_namespaces
from ehrapy.mcp.dispatch import dispatch_json, help_json
from ehrapy.mcp.edata_store import fork_edata, load_edata
from ehrapy.mcp.errors import mcp_error, unknown_handle_error
from ehrapy.mcp.session import get_session


async def list_ehrapy_functions(
    namespace: str | None = None,
    ctx: Context = None,
) -> str:
    """List dispatchable ehrapy functions. Omit namespace for full catalog."""
    try:
        if namespace is None:
            return json.dumps(catalog_summary(), indent=2)
        return json.dumps(
            {"namespace": namespace, "functions": list_functions(namespace)},
            indent=2,
        )
    except KeyError:
        return mcp_error(
            "list_ehrapy_functions",
            f"Unknown namespace '{namespace}'.",
            error_code="NAMESPACE_UNKNOWN",
            agent_action=f"Use one of: {', '.join(list_namespaces())}",
        )
    except Exception as exc:  # noqa: BLE001
        return mcp_error("list_ehrapy_functions", str(exc))


async def get_function_help(
    namespace: str,
    function: str,
    ctx: Context = None,
) -> str:
    """Return signature and docstring for a namespaced ehrapy function."""
    try:
        return help_json(namespace, function)
    except KeyError as exc:
        return mcp_error(
            "get_function_help",
            str(exc),
            error_code="FUNCTION_UNKNOWN",
            agent_action="Call list_ehrapy_functions to discover valid names.",
        )
    except Exception as exc:  # noqa: BLE001
        return mcp_error("get_function_help", str(exc))


async def _run_namespace(
    tool: str,
    namespace: str,
    function: str,
    edata_id: str | None = None,
    params: dict[str, Any] | None = None,
    in_place: bool = True,
    ctx: Context = None,
) -> str:
    try:
        return dispatch_json(
            namespace,
            function,
            edata_id=edata_id,
            params=params,
            ctx=ctx,
            in_place=in_place,
        )
    except KeyError as exc:
        return mcp_error(
            tool,
            str(exc),
            error_code="FUNCTION_UNKNOWN",
            agent_action="Call list_ehrapy_functions or get_function_help.",
        )
    except ValueError as exc:
        return mcp_error(tool, str(exc), error_code="INVALID_INPUT")
    except Exception as exc:  # noqa: BLE001
        return mcp_error(tool, str(exc))


async def run_preprocessing(
    function: str,
    edata_id: str | None = None,
    params: dict[str, Any] | None = None,
    in_place: bool = True,
    ctx: Context = None,
) -> str:
    """Run an ep.preprocessing (ep.pp.*) function. Mutates EHRData in place by default."""
    return await _run_namespace("run_preprocessing", "preprocessing", function, edata_id, params, in_place, ctx)


async def run_analysis(
    function: str,
    edata_id: str | None = None,
    params: dict[str, Any] | None = None,
    in_place: bool = True,
    ctx: Context = None,
) -> str:
    """Run an ep.tools (ep.tl.*) function: survival, causal, embedding, clustering, etc."""
    return await _run_namespace("run_analysis", "tools", function, edata_id, params, in_place, ctx)


async def run_get(
    function: str,
    edata_id: str | None = None,
    params: dict[str, Any] | None = None,
    ctx: Context = None,
) -> str:
    """Run an ep.get.* accessor (obs_df, var_df, rank_features_groups_df). Read-only."""
    return await _run_namespace("run_get", "get", function, edata_id, params, True, ctx)


async def run_plot(
    function: str,
    edata_id: str | None = None,
    params: dict[str, Any] | None = None,
    ctx: Context = None,
) -> str:
    """Run an ep.plot.* visualization. Saves PNG to cache and returns path."""
    return await _run_namespace("run_plot", "plot", function, edata_id, params, True, ctx)


async def run_io(
    function: str,
    edata_id: str | None = None,
    params: dict[str, Any] | None = None,
    ctx: Context = None,
) -> str:
    """Run ehrdata.io.* (read_csv, write_h5ed, from_pandas, ...). Loads return new edata_id."""
    return await _run_namespace("run_io", "io", function, edata_id, params, True, ctx)


async def load_demo_dataset(
    dataset: str,
    params: dict[str, Any] | None = None,
    ctx: Context = None,
) -> str:
    """Load a built-in ehrdata.dt demo cohort (mimic_2, physionet2012, ...)."""
    available_datasets = list_functions("dt")
    if dataset not in available_datasets:
        return mcp_error(
            "load_demo_dataset",
            f"Unknown dataset '{dataset}' in namespace 'dt'.",
            error_code="DATASET_UNKNOWN",
            agent_action=f"Use one of: {', '.join(available_datasets)}",
        )
    return await _run_namespace("load_demo_dataset", "dt", dataset, None, params, True, ctx)


async def fork_edata_handle(
    edata_id: str | None = None,
    name: str | None = None,
    ctx: Context = None,
) -> str:
    """Copy an edata_id to a new handle before destructive edits."""
    session = get_session(ctx)
    handle = edata_id or session.edata_id
    if not handle:
        return mcp_error(
            "fork_edata_handle",
            "No edata_id provided.",
            error_code="EDATA_ID_MISSING",
        )
    try:
        record = fork_edata(handle, name=name)
        session.edata_id = record.edata_id
        return json.dumps(dataclasses.asdict(record), indent=2)
    except KeyError:
        return unknown_handle_error("fork_edata_handle", "edata_id", handle)
    except Exception as exc:  # noqa: BLE001
        return mcp_error("fork_edata_handle", str(exc))


async def export_edata(
    path: str,
    edata_id: str | None = None,
    format: str = "h5ed",
    ctx: Context = None,
) -> str:
    """Write cached EHRData to a host-visible path (h5ed or csv via to_pandas)."""
    session = get_session(ctx)
    handle = edata_id or session.edata_id
    if not handle:
        return mcp_error("export_edata", "No edata_id provided.", error_code="EDATA_ID_MISSING")
    try:
        if format == "h5ed":
            return await run_io("write_h5ed", handle, {"filename": path}, ctx)
        if format == "csv":
            from ehrdata.io import to_pandas

            edata = load_edata(handle)
            df = to_pandas(edata)
            df.to_csv(path, index=False)
            return json.dumps({"status": "ok", "path": path, "format": "csv"}, indent=2)
        return mcp_error(
            "export_edata",
            f"Unsupported format '{format}'.",
            error_code="FORMAT_UNSUPPORTED",
            agent_action="Use format='h5ed' or 'csv'.",
        )
    except Exception as exc:  # noqa: BLE001
        return mcp_error("export_edata", str(exc))


async def get_edata_snapshot(
    edata_id: str | None = None,
    ctx: Context = None,
) -> str:
    """Return obs/var column names, shape, layers, and uns keys for an edata_id."""
    session = get_session(ctx)
    handle = edata_id or session.edata_id
    if not handle:
        return mcp_error("get_edata_snapshot", "No edata_id provided.", error_code="EDATA_ID_MISSING")
    try:
        edata = load_edata(handle)
        layers = [str(k) for k in getattr(edata, "layers", {}).keys() if k is not None]
        payload = {
            "status": "ok",
            "edata_id": handle,
            "n_obs": edata.n_obs,
            "n_vars": edata.n_vars,
            "shape": list(edata.shape),
            "obs_names": [str(x) for x in edata.obs_names[:100]],
            "var_names": [str(x) for x in edata.var_names[:100]],
            "obs_columns": [str(x) for x in edata.obs.columns[:100]],
            "var_columns": [str(x) for x in edata.var.columns[:100]],
            "layers": layers,
            "obsm_keys": [str(k) for k in getattr(edata, "obsm", {}).keys() if k is not None],
            "uns_keys": [str(k) for k in getattr(edata, "uns", {}).keys() if k is not None],
        }
        return json.dumps(payload, indent=2)
    except KeyError:
        return unknown_handle_error("get_edata_snapshot", "edata_id", handle)
    except Exception as exc:  # noqa: BLE001
        return mcp_error("get_edata_snapshot", str(exc))
