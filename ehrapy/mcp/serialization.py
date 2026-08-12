"""Serialize ehrapy return values to JSON-friendly payloads."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from ehrdata import EHRData

if TYPE_CHECKING:
    from pathlib import Path

_MAX_ROWS = 200
_MAX_COLS = 50
_SCALAR_TYPES = (str, int, float, bool, np.integer, np.floating, np.bool_)


def _truncate_df(df: pd.DataFrame) -> dict[str, Any]:
    truncated = df.shape[0] > _MAX_ROWS or df.shape[1] > _MAX_COLS
    view = df.iloc[:_MAX_ROWS, :_MAX_COLS]
    payload: dict[str, Any] = {
        "type": "dataframe",
        "shape": list(df.shape),
        "columns": list(view.columns),
        "data": view.to_dict(orient="records"),
    }
    if truncated:
        payload["truncated"] = True
        payload["note"] = f"Showing first {_MAX_ROWS} rows and {_MAX_COLS} columns."
    return payload


def _is_scalar(obj: Any) -> bool:
    return obj is None or isinstance(obj, _SCALAR_TYPES)


def _serialize_scalar(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return obj


def _serialize_tabular(obj: Any) -> dict[str, Any]:
    if isinstance(obj, EHRData):
        return {"type": "EHRData", "n_obs": obj.n_obs, "n_vars": obj.n_vars}
    if isinstance(obj, pd.DataFrame):
        return _truncate_df(obj)
    if isinstance(obj, pd.Series):
        return {"type": "series", "name": obj.name, "data": obj.head(_MAX_ROWS).to_dict()}
    return {"type": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}


def _unique_plot_path(plots_dir: Path, stem: str) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / f"{stem}.png"
    counter = 0
    while path.exists():
        counter += 1
        path = plots_dir / f"{stem}_{counter}.png"
    return path


def _save_figure(fig: Any, plots_dir: Path, stem: str) -> dict[str, Any]:
    path = _unique_plot_path(plots_dir, stem)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    return {"type": "figure", "path": str(path), "media_type": "image/png"}


def _try_save_holoviews(obj: Any, plots_dir: Path) -> dict[str, Any] | None:
    try:
        import holoviews as hv

        if isinstance(obj, hv.core.dimension.Dimensioned):
            path = _unique_plot_path(plots_dir, "holoviews")
            hv.save(obj, str(path), fmt="png")
            return {"type": "figure", "path": str(path), "media_type": "image/png"}
    except Exception:  # noqa: BLE001
        return None
    return None


def _try_save_figure(obj: Any, plots_dir: Path | None) -> dict[str, Any] | None:
    if plots_dir is None:
        return None
    fig = getattr(obj, "figure", None)
    if fig is not None:
        return _save_figure(fig, plots_dir, stem=type(obj).__name__)
    if type(obj).__module__.startswith("matplotlib"):
        return _save_figure(obj, plots_dir, stem="matplotlib")
    return _try_save_holoviews(obj, plots_dir)


def _serialize_visual_or_repr(obj: Any, plots_dir: Path | None) -> Any:
    saved = _try_save_figure(obj, plots_dir)
    if saved is not None:
        return saved
    summary = getattr(obj, "summary", None)
    if callable(summary):
        try:
            return {"type": type(obj).__name__, "summary": str(summary())[:4000]}
        except Exception:  # noqa: BLE001
            pass
    return {"type": type(obj).__name__, "repr": repr(obj)[:2000]}


def _serialize_object(obj: Any, *, plots_dir: Path | None = None) -> Any:
    if _is_scalar(obj):
        return _serialize_scalar(obj)
    if isinstance(obj, (EHRData, pd.DataFrame, pd.Series, np.ndarray)):
        return _serialize_tabular(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return {"type": type(obj).__name__, **asdict(obj)}
    if isinstance(obj, dict):
        return {str(k): _serialize_object(v, plots_dir=plots_dir) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_object(v, plots_dir=plots_dir) for v in obj]
    return _serialize_visual_or_repr(obj, plots_dir)


def serialize_result(result: Any, *, plots_dir: Path | None = None) -> Any:
    """Convert an ehrapy return value into a JSON-friendly payload."""
    return _serialize_object(result, plots_dir=plots_dir)
