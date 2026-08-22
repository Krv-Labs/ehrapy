"""Dispatch ehrapy / ehrdata functions from MCP tools."""

from __future__ import annotations

import inspect
import json
from typing import Any

import matplotlib

matplotlib.use("Agg")

from ehrdata import EHRData

from ehrapy.mcp.catalog import function_help, get_callable, get_namespace_kind
from ehrapy.mcp.edata_store import load_edata, persist_edata, save_edata
from ehrapy.mcp.registry import registry
from ehrapy.mcp.serialization import serialize_result
from ehrapy.mcp.session import get_session


def _coerce_params(fn: Any, params: dict[str, Any] | None) -> dict[str, Any]:
    if not params:
        return {}
    sig = inspect.signature(fn)
    coerced: dict[str, Any] = {}
    for key, value in params.items():
        if key not in sig.parameters:
            coerced[key] = value
            continue
        ann = sig.parameters[key].annotation
        if ann is inspect.Parameter.empty:
            coerced[key] = value
            continue
        # JSON may deliver lists where tuples are expected
        if ann in (tuple, tuple[Any, ...]) or "tuple" in str(ann).lower():
            if isinstance(value, list):
                coerced[key] = tuple(value)
                continue
        coerced[key] = value
    return coerced


def _first_param_name(fn: Any) -> str | None:
    params = list(inspect.signature(fn).parameters.values())
    if not params:
        return None
    return params[0].name


def _inject_edata(fn: Any, edata: EHRData, params: dict[str, Any]) -> dict[str, Any]:
    first = _first_param_name(fn)
    if first and first not in params:
        params = {first: edata, **params}
    return params


def _require_edata_id(edata_id: str | None, session: Any, message: str) -> str:
    handle = edata_id or session.edata_id
    if not handle:
        raise ValueError(message)
    return handle


def _ok(
    namespace: str,
    function: str,
    result: Any,
    *,
    edata_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "ok",
        "namespace": namespace,
        "function": function,
        "result": result,
    }
    if edata_id is not None:
        payload["edata_id"] = edata_id
    return payload


def _dispatch_edata_or_plot(
    fn: Any,
    namespace: str,
    function: str,
    kind: str,
    *,
    edata_id: str | None,
    params: dict[str, Any],
    session: Any,
    plots_dir: Any,
    in_place: bool,
) -> dict[str, Any]:
    edata_id = _require_edata_id(edata_id, session, "edata_id is required for this operation")
    edata = load_edata(edata_id)
    if namespace == "get" and function == "obs_df" and not params.get("keys") and not params.get("obsm_keys"):
        result = edata.obs.reset_index()
    elif namespace == "get" and function == "var_df" and not params.get("keys") and not params.get("varm_keys"):
        result = edata.var.reset_index()
    else:
        call_params = _inject_edata(fn, edata, params)
        result = fn(**call_params) if call_params else fn(edata)
    if isinstance(result, EHRData):
        record = save_edata(result, name=f"{function}-result", parent_id=edata_id)
        session.edata_id = record.edata_id
        return _ok(
            namespace,
            function,
            serialize_result(result, plots_dir=plots_dir),
            edata_id=record.edata_id,
        )
    if in_place and kind == "edata":
        persist_edata(edata_id, edata)
    session.edata_id = edata_id
    return _ok(
        namespace,
        function,
        serialize_result(result, plots_dir=plots_dir if kind == "plot" else None),
        edata_id=edata_id,
    )


def _dispatch_io(
    fn: Any,
    namespace: str,
    function: str,
    *,
    edata_id: str | None,
    params: dict[str, Any],
    session: Any,
    plots_dir: Any,
) -> dict[str, Any]:
    first = _first_param_name(fn)
    if first == "edata" or "edata" in params:
        edata_id = _require_edata_id(edata_id, session, "edata_id is required for write/export io operations")
        edata = load_edata(edata_id)
        result = fn(**_inject_edata(fn, edata, params))
        return _ok(
            namespace,
            function,
            serialize_result(result, plots_dir=plots_dir),
            edata_id=edata_id,
        )
    result = fn(**params)
    if isinstance(result, EHRData):
        record = save_edata(result, name=f"{function}-{result.n_obs}x{result.n_vars}")
        session.edata_id = record.edata_id
        return _ok(
            namespace,
            function,
            {"type": "EHRData", "n_obs": result.n_obs, "n_vars": result.n_vars},
            edata_id=record.edata_id,
        )
    return _ok(namespace, function, serialize_result(result))


def _dispatch_demo(
    fn: Any,
    namespace: str,
    function: str,
    *,
    params: dict[str, Any],
    session: Any,
) -> dict[str, Any]:
    result = fn(**params)
    if not isinstance(result, EHRData):
        raise TypeError(f"Demo loader {function} did not return EHRData")
    record = save_edata(result, name=function, fmt="demo")
    session.edata_id = record.edata_id
    return _ok(
        namespace,
        function,
        {"type": "EHRData", "n_obs": result.n_obs, "n_vars": result.n_vars},
        edata_id=record.edata_id,
    )


def run_dispatch(
    namespace: str,
    function: str,
    *,
    edata_id: str | None = None,
    params: dict[str, Any] | None = None,
    ctx: Any = None,
    in_place: bool = True,
) -> dict[str, Any]:
    """Invoke a namespaced ehrapy function and serialize the result."""
    fn = get_callable(namespace, function)
    kind = get_namespace_kind(namespace)
    params = _coerce_params(fn, params)
    plots_dir = registry.plots_dir()
    session = get_session(ctx)
    if kind in {"edata", "plot"}:
        return _dispatch_edata_or_plot(
            fn,
            namespace,
            function,
            kind,
            edata_id=edata_id,
            params=params,
            session=session,
            plots_dir=plots_dir,
            in_place=in_place,
        )
    if kind == "io":
        return _dispatch_io(
            fn,
            namespace,
            function,
            edata_id=edata_id,
            params=params,
            session=session,
            plots_dir=plots_dir,
        )
    if kind == "demo":
        return _dispatch_demo(fn, namespace, function, params=params, session=session)
    raise ValueError(f"Unsupported namespace kind '{kind}'")


def dispatch_json(
    namespace: str,
    function: str,
    *,
    edata_id: str | None = None,
    params: dict[str, Any] | None = None,
    ctx: Any = None,
    in_place: bool = True,
) -> str:
    """JSON-encode :func:`run_dispatch`."""
    payload = run_dispatch(
        namespace,
        function,
        edata_id=edata_id,
        params=params,
        ctx=ctx,
        in_place=in_place,
    )
    return json.dumps(payload, indent=2, default=str)


def help_json(namespace: str, function: str) -> str:
    """JSON-encode :func:`function_help`."""
    return json.dumps(function_help(namespace, function), indent=2)
