"""Dispatch engine for ehrapy/ehrdata namespaces with dual-channel output."""

from __future__ import annotations

import asyncio
import collections
import inspect
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastmcp import Context  # noqa: TC002
from fastmcp.tools.tool import ToolResult
from fastmcp.utilities.types import Image

from ehrapy.mcp.catalog import get_callable, get_namespace_kind
from ehrapy.mcp.edata_store import load_edata, persist_edata, save_edata
from ehrapy.mcp.errors import (
    classify_exception_error,
    mcp_error,
    path_access_error,
    unknown_handle_error,
)
from ehrapy.mcp.policy import PathNotAllowedError, ReadOnlyModeError, check_path_allowed
from ehrapy.mcp.registry import registry
from ehrapy.mcp.serialization import serialize_result
from ehrapy.mcp.session import get_session
from ehrapy.mcp.steering import get_suggested_next

if TYPE_CHECKING:
    from collections.abc import Callable

SLOW_FUNCTIONS = frozenset(
    {
        "knn_impute",
        "miss_forest_impute",
        "neighbors",
        "umap",
        "tsne",
        "leiden",
        "iptw",
        "aipw",
        "g_computation",
        "t_learner",
        "read_csv",
    }
)


_FITTER_CACHE_CAPACITY = 8
_FITTER_CACHE: collections.OrderedDict[tuple[str, str], Any] = collections.OrderedDict()

# Plot functions whose first parameter is a fitted model/result rather than the EHRData
# object, mapped to the run_analysis function that produces it. Ported from origin fix #5.
_PLOT_FITTER_SOURCES: dict[str, tuple[str, str]] = {
    # plot function -> (analysis function that fits it, parameter name it binds to)
    "kaplan_meier": ("kaplan_meier", "kmfs"),
    "love_plot": ("covariate_balance", "balance"),
    "propensity_overlap": ("positivity_check", "positivity"),
}


def cache_fitter(edata_id: str, function: str, value: Any) -> None:
    """Record a fitted model so a downstream plot call can bind it (bounded LRU)."""
    key = (edata_id, function)
    if key in _FITTER_CACHE:
        _FITTER_CACHE.move_to_end(key)
    _FITTER_CACHE[key] = value
    while len(_FITTER_CACHE) > _FITTER_CACHE_CAPACITY:
        _FITTER_CACHE.popitem(last=False)


def get_fitter(edata_id: str, function: str) -> Any | None:
    """Return a cached fitter for a handle, if one was recorded."""
    key = (edata_id, function)
    if key not in _FITTER_CACHE:
        return None
    _FITTER_CACHE.move_to_end(key)
    return _FITTER_CACHE[key]


def clear_fitter_cache() -> None:
    """Clear the fitter cache (useful for testing)."""
    _FITTER_CACHE.clear()


def _build_call_params(
    fn: Callable[..., Any],
    edata: Any,
    params: dict[str, Any],
    *,
    edata_id: str | None = None,
    function: str | None = None,
    kind: str | None = None,
) -> dict[str, Any]:
    """Build the kwargs for a dispatched call, binding edata or a fitted model as needed.

    Most ehrapy functions take the EHRData object first. A few plot functions instead take a
    fitted model produced by an earlier run_analysis call (ported from origin fix #5); for
    those, bind the cached fitter -- or the copy persisted in ``edata.uns`` -- rather than
    handing them an EHRData they cannot use.
    """
    sig = inspect.signature(fn)
    param_names = list(sig.parameters.keys())
    if not param_names:
        return dict(params)

    first_param = param_names[0]
    if first_param in params:
        return dict(params)

    if kind == "plot" and function in _PLOT_FITTER_SOURCES:
        source_fn, bind_name = _PLOT_FITTER_SOURCES[function]
        if bind_name in sig.parameters:
            fitted = get_fitter(edata_id or "", source_fn)
            if fitted is None and hasattr(edata, "uns"):
                fitted = edata.uns.get(source_fn)
            if fitted is None:
                raise ValueError(
                    f"No fitted result available for plot '{function}'. "
                    f"Run run_analysis(function='{source_fn}', ...) on this handle first."
                )
            if bind_name == "kmfs" and not isinstance(fitted, (list, tuple)):
                fitted = [fitted]
            return {bind_name: fitted, **params}

    return {first_param: edata, **params}


def _inject_edata(fn: Callable[..., Any], edata: Any, params: dict[str, Any], **kwargs: Any) -> Any:
    """Call ``fn`` with edata (or a fitted model) bound to its first parameter."""
    return fn(**_build_call_params(fn, edata, params, **kwargs))


async def run_dispatch(
    namespace: str,
    function: str,
    *,
    edata_id: str | None = None,
    params: dict[str, Any] | None = None,
    response_format: Literal["concise", "detailed"] = "concise",
    ctx: Context | None = None,
) -> ToolResult:
    """Execute a function from an ehrapy namespace and return a dual-channel ToolResult."""
    params = dict(params or {})
    tool_name = f"run_{namespace}" if namespace != "demo" else "load_demo_dataset"
    session = get_session(ctx)

    # Progress reporting for slow functions
    if ctx is not None and function in SLOW_FUNCTIONS:
        try:
            await ctx.info(f"Starting {namespace}.{function}...")
            await ctx.report_progress(progress=0, total=100)
        except Exception:  # noqa: BLE001
            pass

    # Resolve callable
    try:
        fn = get_callable(namespace, function)
        kind = get_namespace_kind(namespace)
    except KeyError as exc:
        return mcp_error(
            tool_name,
            str(exc),
            error_code="UNKNOWN_FUNCTION",
            agent_action=f"Call list_ehrapy_functions(namespace='{namespace}') to inspect valid function names.",
        )

    # 1. Demo datasets
    if kind == "demo":
        try:
            edata = fn(**params)
            record = save_edata(edata, name=function, fmt="demo")
            session.set_latest_edata_id(record.edata_id)

            suggestions = get_suggested_next(namespace, function)
            struct: dict[str, Any] = {
                "status": "ok",
                "edata_id": record.edata_id,
                "name": record.name,
                "n_obs": edata.n_obs,
                "n_vars": edata.n_vars,
                "namespace": namespace,
                "function": function,
            }
            if suggestions:
                struct["suggested_next"] = [s["call"] for s in suggestions]

            md_lines = [
                f"### Loaded demo dataset `{function}`",
                f"- **Handle (edata_id):** `{record.edata_id}`",
                f"- **Observations:** {edata.n_obs}",
                f"- **Variables:** {edata.n_vars}",
            ]
            if suggestions:
                md_lines.extend(["", "**Suggested next:**"])
                for s in suggestions:
                    md_lines.append(f"- `{s['call']}` — {s['reason']}")

            if ctx is not None and function in SLOW_FUNCTIONS:
                try:
                    await ctx.report_progress(progress=100, total=100)
                except Exception:  # noqa: BLE001
                    pass

            return ToolResult(structured_content=struct, content="\n".join(md_lines))
        except Exception as exc:  # noqa: BLE001
            return mcp_error(
                tool_name,
                f"Failed to load demo dataset '{function}': {exc}",
                error_code="DEMO_LOAD_ERROR",
            )

    # 2. IO read functions (standalone ingestion)
    if kind == "io" and function.startswith("read_"):
        try:
            file_arg = params.get("filename") or params.get("path") or params.get("file_path")
            if file_arg:
                checked_path = check_path_allowed(file_arg, for_write=False)
                # Update param with resolved path string
                for k in ("filename", "path", "file_path"):
                    if k in params:
                        params[k] = str(checked_path)

            edata = fn(**params)
            name_stem = Path(file_arg).name if file_arg else function
            record = save_edata(edata, name=name_stem, source_path=str(file_arg) if file_arg else None)
            session.set_latest_edata_id(record.edata_id)

            suggestions = get_suggested_next(namespace, function)
            struct = {
                "status": "ok",
                "edata_id": record.edata_id,
                "name": record.name,
                "n_obs": edata.n_obs,
                "n_vars": edata.n_vars,
                "namespace": namespace,
                "function": function,
            }
            if suggestions:
                struct["suggested_next"] = [s["call"] for s in suggestions]

            md_lines = [
                f"### Loaded dataset from `{file_arg or function}`",
                f"- **Handle (edata_id):** `{record.edata_id}`",
                f"- **Observations:** {edata.n_obs}",
                f"- **Variables:** {edata.n_vars}",
            ]
            if suggestions:
                md_lines.extend(["", "**Suggested next:**"])
                for s in suggestions:
                    md_lines.append(f"- `{s['call']}` — {s['reason']}")

            return ToolResult(structured_content=struct, content="\n".join(md_lines))
        except PathNotAllowedError as exc:
            return mcp_error(
                tool_name,
                str(exc),
                error_code=exc.error_code,
                agent_action=exc.agent_action,
            )
        except ReadOnlyModeError as exc:
            return mcp_error(
                tool_name,
                str(exc),
                error_code=exc.error_code,
                agent_action=exc.agent_action,
            )
        except FileNotFoundError:
            return path_access_error(tool_name, str(params.get("filename") or params.get("path") or ""))
        except Exception as exc:  # noqa: BLE001
            return classify_exception_error(
                tool_name, exc, namespace=namespace, function=function, fallback_code="IO_READ_ERROR"
            )

    # For all other operations, resolve the active dataset handle
    used_latest = False
    handle = edata_id
    if handle is None:
        handle = session.get_latest_edata_id()
        used_latest = True

    if handle is None:
        return mcp_error(
            tool_name,
            "No dataset handle provided and no active dataset in session.",
            error_code="NO_ACTIVE_DATASET",
            agent_action="Load a dataset with load_demo_dataset(dataset='mimic_2') or ingest_dataset(file_path=...).",
        )

    # Load dataset
    try:
        edata = load_edata(handle)
    except (KeyError, FileNotFoundError):
        return unknown_handle_error(tool_name, "edata_id", handle)
    except Exception as exc:  # noqa: BLE001
        return mcp_error(
            tool_name,
            f"Failed to load dataset '{handle}': {exc}",
            error_code="DATASET_LOAD_ERROR",
        )

    # 3. Read-Only get namespace (T1)
    if kind == "get":
        try:
            res = _inject_edata(fn, edata, params)
            meta, content_payload = serialize_result(
                res,
                response_format=response_format,
                params=params,
                function=function,
                edata=edata,
            )
            suggestions = get_suggested_next(namespace, function)

            status_val = meta.get("status", "ok")
            struct = {
                "status": status_val,
                "edata_id": handle,
                "namespace": namespace,
                "function": function,
                **{k: v for k, v in meta.items() if k != "status"},
            }
            if used_latest:
                struct["used_latest"] = True
            if suggestions:
                struct["suggested_next"] = [s["call"] for s in suggestions]

            md_content = (
                content_payload if isinstance(content_payload, str) else "\n".join(str(c) for c in content_payload)
            )
            if suggestions:
                sug_lines = ["\n\n**Suggested next:**"]
                for s in suggestions:
                    sug_lines.append(f"- `{s['call']}` — {s['reason']}")
                md_content += "\n".join(sug_lines)

            return ToolResult(structured_content=struct, content=md_content)
        except Exception as exc:  # noqa: BLE001
            return classify_exception_error(
                tool_name, exc, namespace=namespace, function=function, fallback_code="GET_ERROR"
            )

    # 4. Plot namespace (T10)
    if kind == "plot":
        try:
            # Inject show=False if function accepts show
            sig = inspect.signature(fn)
            if "show" in sig.parameters and "show" not in params:
                params["show"] = False
            # Prefer an explicit figure handle over scraping plt.gcf() (origin fix #4).
            if "return_fig" in sig.parameters and "return_fig" not in params:
                params["return_fig"] = True

            plt.close("all")
            res = _inject_edata(fn, edata, params, edata_id=handle, function=function, kind="plot")
            if res is None and plt.get_fignums():
                res = plt.gcf()

            meta, content_payload = serialize_result(
                res,
                plots_dir=registry.plots_dir(),
                params=params,
                function=function,
                edata=edata,
            )
            plt.close("all")

            suggestions = get_suggested_next(namespace, function)
            rendered = meta.get("type") == "figure"
            struct = {
                "status": "ok" if rendered else "ok_no_figure",
                "edata_id": handle,
                "namespace": namespace,
                "function": function,
                **meta,
            }
            if not rendered:
                # A plot call that produced no image previously reported a bare "ok",
                # leaving the agent to assume a figure existed.
                struct["agent_action"] = (
                    f"'{function}' returned {meta.get('type', 'no renderable figure')} rather than a figure. "
                    "Check that the required inputs were fitted first, or call "
                    f"get_function_help(namespace='{namespace}', function='{function}')."
                )

            if used_latest:
                struct["used_latest"] = True
            if suggestions:
                struct["suggested_next"] = [s["call"] for s in suggestions]

            # Build ToolResult content blocks
            content_blocks: list[Any] = []
            if isinstance(content_payload, list):
                for item in content_payload:
                    content_blocks.append(item)
            else:
                content_blocks.append(content_payload)

            if suggestions:
                sug_lines = ["\n\n**Suggested next:**"]
                for s in suggestions:
                    sug_lines.append(f"- `{s['call']}` — {s['reason']}")
                content_blocks.append("\n".join(sug_lines))

            return ToolResult(structured_content=struct, content=content_blocks)
        except Exception as exc:  # noqa: BLE001
            plt.close("all")
            return classify_exception_error(
                tool_name, exc, namespace=namespace, function=function, fallback_code="PLOT_ERROR"
            )

    # 5. IO write / export / to_pandas functions
    if kind == "io":
        try:
            if function == "to_pandas":
                df = fn(edata, **params)
                meta, content_payload = serialize_result(
                    df,
                    response_format=response_format,
                    params=params,
                    function=function,
                    edata=edata,
                )
                struct = {
                    "status": "ok",
                    "edata_id": handle,
                    "namespace": namespace,
                    "function": function,
                    **meta,
                }
                if used_latest:
                    struct["used_latest"] = True
                return ToolResult(
                    structured_content=struct,
                    content=(content_payload if isinstance(content_payload, str) else str(content_payload)),
                )

            # write functions
            target_path = params.get("filename") or params.get("path") or params.get("file_path")
            if target_path:
                checked_path = check_path_allowed(target_path, for_write=True, operation=function)
                for k in ("filename", "path", "file_path"):
                    if k in params:
                        params[k] = str(checked_path)

            fn(edata, **params)
            struct = {
                "status": "ok",
                "edata_id": handle,
                "namespace": namespace,
                "function": function,
                "target_path": str(target_path) if target_path else None,
            }
            if used_latest:
                struct["used_latest"] = True
            md = f"Dataset `{handle}` successfully exported to `{target_path}`."
            return ToolResult(structured_content=struct, content=md)
        except PathNotAllowedError as exc:
            return mcp_error(
                tool_name,
                str(exc),
                error_code=exc.error_code,
                agent_action=exc.agent_action,
            )
        except ReadOnlyModeError as exc:
            return mcp_error(
                tool_name,
                str(exc),
                error_code=exc.error_code,
                agent_action=exc.agent_action,
            )
        except Exception as exc:  # noqa: BLE001
            return classify_exception_error(
                tool_name, exc, namespace=namespace, function=function, fallback_code="IO_ERROR"
            )

    # 6. Mutating edata namespaces (preprocessing & analysis)
    if kind == "edata":
        try:
            res = _inject_edata(fn, edata, params, edata_id=handle, function=function, kind=kind)
            if res is not None and function in {src for src, _ in _PLOT_FITTER_SOURCES.values()}:
                cache_fitter(handle, function, res)
            if hasattr(res, "n_obs") and hasattr(res, "n_vars"):
                # Function returned a modified copy
                edata = res

            # Op-log provenance tracking (T19)
            import json

            ops = edata.uns.setdefault("ehrapy_mcp_ops", [])
            ops.append(
                json.dumps(
                    {
                        "namespace": namespace,
                        "function": function,
                        "params": {
                            str(k): (v if isinstance(v, (int, float, str, bool, list)) else str(v))
                            for k, v in params.items()
                        },
                        "timestamp": time.time(),
                    }
                )
            )

            # Persist dataset write-through
            persist_edata(handle, edata)
            session.set_latest_edata_id(handle)

            meta, content_payload = serialize_result(
                res,
                response_format=response_format,
                params=params,
                function=function,
                edata=edata,
            )
            suggestions = get_suggested_next(namespace, function)

            struct = {
                "status": "ok",
                "edata_id": handle,
                "n_obs": edata.n_obs,
                "n_vars": edata.n_vars,
                "namespace": namespace,
                "function": function,
                **meta,
            }
            if used_latest:
                struct["used_latest"] = True
            if suggestions:
                struct["suggested_next"] = [s["call"] for s in suggestions]

            md_content = (
                content_payload if isinstance(content_payload, str) else "\n".join(str(c) for c in content_payload)
            )
            if suggestions:
                sug_lines = ["\n\n**Suggested next:**"]
                for s in suggestions:
                    sug_lines.append(f"- `{s['call']}` — {s['reason']}")
                md_content += "\n".join(sug_lines)

            if ctx is not None and function in SLOW_FUNCTIONS:
                try:
                    await ctx.report_progress(progress=100, total=100)
                except Exception:  # noqa: BLE001
                    pass

            return ToolResult(structured_content=struct, content=md_content)
        except TypeError as exc:
            return mcp_error(
                tool_name,
                f"Invalid parameter for {namespace}.{function}: {exc}",
                error_code="INVALID_PARAMETER",
                agent_action=f"Call get_function_help(namespace='{namespace}', function='{function}') to inspect valid parameters.",
            )
        except ValueError as exc:
            return mcp_error(
                tool_name,
                f"Value error in {namespace}.{function}: {exc}",
                error_code="INVALID_VALUE",
                agent_action=f"Inspect data or parameters for {namespace}.{function}.",
            )
        except Exception as exc:  # noqa: BLE001
            return classify_exception_error(tool_name, exc, namespace=namespace, function=function)

    return mcp_error(tool_name, f"Unhandled namespace kind '{kind}'", error_code="UNHANDLED_KIND")
