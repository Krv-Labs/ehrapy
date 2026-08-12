"""Catalog of dispatchable ehrapy / ehrdata functions."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import ehrdata.dt as ed_dt
import ehrdata.io as ed_io

import ehrapy as ep

if TYPE_CHECKING:
    from collections.abc import Callable

# scanpy-derived tools not listed in ep.tools.__all__
_EXTRA_TOOLS = ("leiden", "dendrogram", "dpt", "paga", "ingest")

# classes exported in __all__ but not invokable as functions
_SKIP_TOOLS = frozenset({"CausalEstimate", "CohortTracker", "Literal"})

# plot module imports many names; skip non-functions
_SKIP_PLOT = frozenset({"Colormaps", "LinearSegmentedColormap", "hv"})


def _plot_functions() -> list[str]:
    names: list[str] = []
    for name in dir(ep.plot):
        if name.startswith("_") or name in _SKIP_PLOT:
            continue
        obj = getattr(ep.plot, name)
        if inspect.isfunction(obj):
            names.append(name)
    return sorted(names)


def _tools_functions() -> list[str]:
    names = [n for n in ep.tools.__all__ if n not in _SKIP_TOOLS]
    for extra in _EXTRA_TOOLS:
        if extra not in names and hasattr(ep.tools, extra):
            names.append(extra)
    return sorted(names)


NAMESPACES: dict[str, dict[str, Any]] = {
    "preprocessing": {
        "module": ep.preprocessing,
        "functions": list(ep.preprocessing.__all__),
        "kind": "edata",
        "description": "Quality control, encoding, imputation, normalization, filtering (ep.pp.*)",
    },
    "tools": {
        "module": ep.tools,
        "functions": _tools_functions(),
        "kind": "edata",
        "description": "Analysis: survival, causal, embedding, clustering, feature ranking (ep.tl.*)",
    },
    "get": {
        "module": ep.get,
        "functions": list(ep.get.__all__),
        "kind": "edata",
        "description": "Read obs/var tables and ranked feature results (ep.get.*)",
    },
    "plot": {
        "module": ep.plot,
        "functions": _plot_functions(),
        "kind": "plot",
        "description": "Visualization; saves PNG artifacts (ep.pl.*)",
    },
    "io": {
        "module": ed_io,
        "functions": [
            "read_csv",
            "read_h5ad",
            "read_h5ed",
            "read_zarr",
            "from_pandas",
            "write_h5ad",
            "write_h5ed",
            "write_zarr",
            "to_pandas",
        ],
        "kind": "io",
        "description": "Load/save EHRData from files (ehrdata.io.*)",
    },
    "dt": {
        "module": ed_dt,
        "functions": list(ed_dt.__all__),
        "kind": "demo",
        "description": "Built-in demo cohorts (ehrdata.dt.*)",
    },
}


def list_namespaces() -> list[str]:
    """Return sorted dispatch namespace names."""
    return sorted(NAMESPACES)


def list_functions(namespace: str) -> list[str]:
    """Return function names in a dispatch namespace."""
    spec = NAMESPACES.get(namespace)
    if spec is None:
        raise KeyError(namespace)
    return list(spec["functions"])


def get_callable(namespace: str, function: str) -> Callable[..., Any]:
    """Resolve a namespaced ehrapy/ehrdata callable."""
    spec = NAMESPACES.get(namespace)
    if spec is None:
        raise KeyError(f"Unknown namespace '{namespace}'")
    if function not in spec["functions"]:
        raise KeyError(f"Unknown function '{function}' in namespace '{namespace}'")
    return getattr(spec["module"], function)


def get_namespace_kind(namespace: str) -> str:
    """Return the dispatch kind for a namespace."""
    return NAMESPACES[namespace]["kind"]


def function_help(namespace: str, function: str) -> dict[str, Any]:
    """Return signature metadata and a short docstring."""
    fn = get_callable(namespace, function)
    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn) or ""
    params = []
    for name, param in sig.parameters.items():
        entry: dict[str, Any] = {"name": name}
        if param.default is not inspect.Parameter.empty:
            entry["default"] = repr(param.default)
        if param.annotation is not inspect.Parameter.empty:
            entry["annotation"] = str(param.annotation)
        params.append(entry)
    return {
        "namespace": namespace,
        "function": function,
        "kind": get_namespace_kind(namespace),
        "parameters": params,
        "docstring": doc.split("\n\n")[0][:500],
    }


def catalog_summary() -> dict[str, Any]:
    """Summarize every dispatch namespace and its functions."""
    return {
        ns: {
            "kind": spec["kind"],
            "count": len(spec["functions"]),
            "description": spec["description"],
            "functions": spec["functions"],
        }
        for ns, spec in NAMESPACES.items()
    }
