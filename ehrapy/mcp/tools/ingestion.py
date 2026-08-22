from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING

from ehrdata.io import read_csv

from ehrapy.mcp.edata_store import save_edata
from ehrapy.mcp.errors import mcp_error, path_access_error
from ehrapy.mcp.session import get_session

if TYPE_CHECKING:
    from fastmcp import Context


def _infer_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".tsv", ".txt"}:
        return "tsv"
    return suffix.lstrip(".") or "unknown"


async def ingest_dataset(path: str, ctx: Context = None) -> str:
    """Load a host-visible CSV/TSV into EHRData; returns edata_id."""
    source = Path(path).expanduser()
    if not source.is_file():
        return path_access_error(
            "ingest_dataset",
            str(source),
            missing_action="Provide a host-visible absolute dataset path.",
        )

    fmt = _infer_format(source)
    if fmt not in {"csv", "tsv"}:
        return mcp_error(
            "ingest_dataset",
            f"Unsupported format '{fmt}'. Use run_io for h5ed/zarr.",
            error_code="FORMAT_UNSUPPORTED",
        )

    try:
        edata = read_csv(str(source))
        record = save_edata(
            edata,
            name=source.name,
            source_path=str(source.resolve()),
            fmt=fmt,
        )
        session = get_session(ctx)
        session.edata_id = record.edata_id
        return json.dumps(dataclasses.asdict(record), indent=2)
    except PermissionError:
        return mcp_error(
            "ingest_dataset",
            "Dataset path exists but is not readable.",
            error_code="FILE_PERMISSION_DENIED",
        )
    except Exception as exc:  # noqa: BLE001
        return mcp_error("ingest_dataset", str(exc))
