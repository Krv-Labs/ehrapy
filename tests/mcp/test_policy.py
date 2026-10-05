"""Tests for the MCP filesystem policy's platform-specific allowed roots."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ehrapy.mcp.policy import PathNotAllowedError, ReadOnlyModeError, check_path_allowed, get_allowed_roots


@pytest.mark.parametrize(
    ("separator", "roots"),
    [
        (":", ("/data/twins", "/data/ehrapy-cache")),
        (";", (r"C:\Users\clin\twins", r"D:\ehrapy-cache")),
    ],
)
def test_allowed_roots_use_platform_separator(
    separator: str, roots: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Windows drive's colon is part of its root, not a root separator."""
    monkeypatch.setattr(os, "pathsep", separator)
    monkeypatch.setenv("EHRAPY_MCP_ALLOWED_ROOTS", separator.join(roots))

    assert get_allowed_roots() == [Path(root).expanduser().resolve() for root in roots]


def test_multiple_allowed_roots_remain_confined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both configured roots load while files outside them remain refused."""
    roots = (tmp_path / "twins", tmp_path / "ehrapy-cache")
    monkeypatch.setenv("EHRAPY_MCP_ALLOWED_ROOTS", os.pathsep.join(str(root) for root in roots))
    for root in roots:
        root.mkdir()
        twin = root / "cohort.csv"
        twin.write_text("token,dose\nTOKEN-1,25\n")
        assert check_path_allowed(twin) == twin.resolve()

    raw = tmp_path / "raw.csv"
    raw.write_text("mrn,dose\nRAW-1,25\n")
    with pytest.raises(PathNotAllowedError):
        check_path_allowed(raw)


def test_allowed_roots_do_not_override_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An allowed twin remains readable while exports remain refused."""
    monkeypatch.setenv("EHRAPY_MCP_ALLOWED_ROOTS", str(tmp_path))
    monkeypatch.setenv("EHRAPY_MCP_READ_ONLY", "1")
    twin = tmp_path / "cohort.csv"
    twin.write_text("token,dose\nTOKEN-1,25\n")
    assert check_path_allowed(twin) == twin.resolve()
    with pytest.raises(ReadOnlyModeError):
        check_path_allowed(twin, for_write=True)
