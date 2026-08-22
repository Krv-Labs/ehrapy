"""Dispatch ehrapy / ehrdata functions from MCP tools."""

from __future__ import annotations

import inspect
import json
from collections.abc import Sequence
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
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
            if key == "store" and "filename" in sig.parameters:
                coerced["filename"] = value
                continue
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


_FITTER_CACHE: dict[tuple[str, str], Any] = {}


def _first_param_name(fn: Any) -> str | None:
    params = list(inspect.signature(fn).parameters.values())
    if not params:
        return None
    return params[0].name


def _inject_edata(
    fn: Any,
    edata: EHRData,
    params: dict[str, Any],
    *,
    edata_id: str | None = None,
    function: str | None = None,
    kind: str | None = None,
) -> dict[str, Any]:
    first = _first_param_name(fn)
    if not first:
        return params
    if first in params:
        return params

    if first in {"edata", "adata", "data", "data_or_subcohorts"}:
        return {first: edata, **params}

    if kind == "plot":
        if function == "kaplan_meier" or first == "kmfs":
            fitter = _FITTER_CACHE.get((edata_id or "", "kaplan_meier"))
            if fitter is not None:
                params["kmfs"] = fitter if isinstance(fitter, (list, tuple, Sequence)) else [fitter]
                return params
            if "duration_col" in params:
                dur = params.pop("duration_col")
                evt = params.pop("event_col", None)
                import ehrapy as ep

                kmf = ep.tools.kaplan_meier(edata, dur, evt)
                if edata_id:
                    _FITTER_CACHE[(edata_id, "kaplan_meier")] = kmf
                params["kmfs"] = [kmf]
                return params
            raise ValueError(
                "No fitted KaplanMeierFitter found. Run run_analysis('kaplan_meier', ...) first "
                "or provide duration_col and event_col."
            )
        if function == "love_plot" or first == "balance":
            bal = _FITTER_CACHE.get((edata_id or "", "covariate_balance")) or edata.uns.get("covariate_balance")
            if bal is not None:
                params["balance"] = bal
                return params
            raise ValueError("No covariate balance DataFrame found. Run run_analysis('covariate_balance', ...) first.")
        if function == "propensity_overlap" or first == "positivity":
            pos = _FITTER_CACHE.get((edata_id or "", "positivity_check")) or edata.uns.get("positivity_check")
            if pos is not None:
                params["positivity"] = pos
                return params
            raise ValueError("No positivity check result found. Run run_analysis('positivity_check', ...) first.")

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
        if kind == "plot":
            sig = inspect.signature(fn)
            if "return_fig" in sig.parameters and "return_fig" not in params:
                params["return_fig"] = True
            if "show" in sig.parameters and "show" not in params:
                params["show"] = False

        call_params = _inject_edata(fn, edata, params, edata_id=edata_id, function=function, kind=kind)
        result = fn(**call_params) if call_params else fn(edata)

        if kind == "plot" and result is None and plt.get_fignums():
            result = plt.gcf()
    if isinstance(result, EHRData):
        record = save_edata(result, name=f"{function}-result", parent_id=edata_id)
        session.edata_id = record.edata_id
        return _ok(
            namespace,
            function,
            serialize_result(result, plots_dir=plots_dir, stem=function),
            edata_id=record.edata_id,
        )

    if result is not None:
        _FITTER_CACHE[(edata_id, function)] = result

    if in_place and kind == "edata":
        persist_edata(edata_id, edata)
    session.edata_id = edata_id

    serialized = serialize_result(result, plots_dir=plots_dir if kind == "plot" else None, stem=function)
    if kind == "plot":
        plt.close("all")

    return _ok(
        namespace,
        function,
        serialized,
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
