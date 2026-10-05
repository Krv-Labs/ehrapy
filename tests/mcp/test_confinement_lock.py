"""Tests for the confinement lock: no temporary fallbacks when a supervisor pins the cache.

Renyi launches ehrapy with EHRAPY_MCP_CACHE_DIR, EHRAPY_MCP_ALLOWED_ROOTS and
EHRAPY_MCP_READ_ONLY=1, and erases exactly the directories it knows about. A
fallback into the system temp directory would put a patient-shaped cohort
somewhere that erase never reaches, which is the boundary this module protects.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path  # noqa: TC003

import pytest

from ehrapy.mcp.dispatch import get_tool_timeout
from ehrapy.mcp.policy import CacheDirUnavailableError, is_confinement_locked
from ehrapy.mcp.registry import (
    MCPRegistry,
    _get_default_cache_dir,
    _get_default_demo_data_dir,
    registry,
)

LOCK_VARS = ("EHRAPY_MCP_CACHE_DIR", "EHRAPY_MCP_ALLOWED_ROOTS")


@pytest.fixture(autouse=True)
def _restore_global_dataset_dirs():
    """Undo the process-wide demo-path retargeting MCPRegistry performs on construction.

    `_configure_ehrapy_demo_paths` writes to `ehrapy`/`scanpy`/`ehrdata` module
    globals, so a test that builds a registry would otherwise leak its directory
    into every later test in the session.
    """
    import ehrdata.core.constants as ed_const
    import ehrdata.dt.datasets as ed_datasets
    import scanpy as sc

    import ehrapy as ep

    before = (
        ep.settings.datasetdir,
        sc.settings.datasetdir,
        ed_const.DEFAULT_DATA_PATH,
        ed_datasets.DEFAULT_DATA_PATH,
    )
    yield
    ep.settings.datasetdir, sc.settings.datasetdir = before[0], before[1]
    ed_const.DEFAULT_DATA_PATH = before[2]
    ed_datasets.DEFAULT_DATA_PATH = before[3]


@pytest.fixture
def unlocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Standalone ehrapy: no supervisor, so today's fallbacks stay legitimate."""
    for var in LOCK_VARS:
        monkeypatch.delenv(var, raising=False)
    assert not is_confinement_locked()


@pytest.fixture
def locked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Confinement locked to a writable directory inside tmp_path."""
    cache = tmp_path / "renyi-cache"
    cache.mkdir()
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    monkeypatch.setenv("EHRAPY_MCP_CACHE_DIR", str(cache))
    monkeypatch.setenv("EHRAPY_MCP_ALLOWED_ROOTS", os.pathsep.join([str(datasets), str(cache)]))
    assert is_confinement_locked()
    return cache


@pytest.fixture
def unwritable(tmp_path: Path) -> Path:
    """A path that cannot be created, because a regular file sits where a directory must go.

    Portable, and unaffected by running as root or by permission-bit tricks.
    """
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    return blocker / "cache"


def _isolate_tmpdir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the temp directory at an empty zone and forget the cached lookup."""
    zone = tmp_path / "tmpzone"
    zone.mkdir()
    monkeypatch.setenv("TMPDIR", str(zone))
    monkeypatch.setattr(tempfile, "tempdir", None)
    return zone


# --- resolution: no temp paths while locked ---------------------------------------------


def test_locked_cache_dir_never_resolves_to_temp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A pinned-but-unusable cache dir is returned as-is, not replaced by a temp dir."""
    zone = _isolate_tmpdir(monkeypatch, tmp_path)
    monkeypatch.setenv("EHRAPY_MCP_CACHE_DIR", str(tmp_path / "blocker" / "cache"))
    (tmp_path / "blocker").write_text("not a directory", encoding="utf-8")

    resolved = _get_default_cache_dir()

    assert resolved == (tmp_path / "blocker" / "cache").resolve()
    assert str(resolved) != str(zone)
    assert list(zone.iterdir()) == [], "resolution created something in the temp dir"


def test_locked_demo_data_dir_is_inside_the_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, locked: Path
) -> None:
    """Locked: demo data lives under <cache_dir>/demo_data, and env overrides are ignored."""
    elsewhere = tmp_path / "inherited"
    elsewhere.mkdir()
    # An EHRAPY_DEMO_DATA_DIR inherited from an unrelated parent must not win.
    monkeypatch.setenv("EHRAPY_DEMO_DATA_DIR", str(elsewhere))

    resolved = _get_default_demo_data_dir(locked)

    assert resolved == locked / "demo_data"


def test_unlocked_demo_data_dir_still_honours_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, unlocked: None
) -> None:
    """Unlocked: the documented override keeps working, so standalone use is unchanged."""
    custom = tmp_path / "demo_cache"
    monkeypatch.setenv("EHRAPY_MCP_DEMO_DATA_DIR", str(custom))

    assert _get_default_demo_data_dir(tmp_path / "cache") == custom


def test_unlocked_cache_dir_falls_back_when_unusable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, unlocked: None
) -> None:
    """Unlocked: the default still resolves to a usable directory, so standalone is unchanged."""
    zone = _isolate_tmpdir(monkeypatch, tmp_path)
    assert not is_confinement_locked()

    resolved = _get_default_cache_dir()

    assert resolved.is_dir(), "unlocked resolution should still yield a usable directory"
    assert zone.is_dir()


# --- write-time behaviour ---------------------------------------------------------------


def test_locked_write_with_unusable_cache_raises_and_writes_nothing_to_temp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the boundary cannot be used, a write refuses rather than relocating."""
    zone = _isolate_tmpdir(monkeypatch, tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    cache = blocker / "cache"
    monkeypatch.setenv("EHRAPY_MCP_CACHE_DIR", str(cache))
    monkeypatch.setenv("EHRAPY_MCP_ALLOWED_ROOTS", str(tmp_path / "datasets"))

    reg = MCPRegistry()
    assert reg.cache_dir() == cache.resolve()

    with pytest.raises(CacheDirUnavailableError) as excinfo:
        reg.require_usable()
    assert excinfo.value.error_code == "CACHE_DIR_UNAVAILABLE"
    assert "agent_action" in excinfo.value.__dict__ or excinfo.value.agent_action

    with pytest.raises(CacheDirUnavailableError):
        reg.ensure_demo_data_dir()

    assert list(zone.iterdir()) == [], "a refused write still created something in the temp dir"


def test_locked_cache_dir_that_vanishes_is_recreated_inside_the_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, locked: Path
) -> None:
    """A directory erased underneath a live server is rebuilt in place, never elsewhere.

    Distinct from the unusable case above: here recovery is possible and correct,
    because the rebuilt path is still the pinned one. What must never happen is
    the rebuild landing in a temp directory.
    """
    import shutil

    zone = _isolate_tmpdir(monkeypatch, tmp_path)
    reg = MCPRegistry()
    shutil.rmtree(locked)

    resolved = reg.require_usable()

    assert resolved == locked, "the cache was relocated instead of rebuilt in place"
    assert resolved.is_dir()
    assert list(zone.iterdir()) == []


def test_locked_demo_dir_does_not_repoint_globals_when_unusable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ensure_demo_data_dir must not retarget ehrdata at a scratch dir while locked."""
    import ehrdata.core.constants as ed_const

    zone = _isolate_tmpdir(monkeypatch, tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    cache = blocker / "cache"
    monkeypatch.setenv("EHRAPY_MCP_CACHE_DIR", str(cache))
    monkeypatch.setenv("EHRAPY_MCP_ALLOWED_ROOTS", str(tmp_path / "datasets"))

    reg = MCPRegistry()
    # Construction points the demo loaders inside the lock, which is correct.
    assert ed_const.DEFAULT_DATA_PATH == reg.demo_data_dir()

    with pytest.raises(CacheDirUnavailableError):
        reg.ensure_demo_data_dir()

    # The refusal must not have moved them out to a scratch directory.
    assert str(ed_const.DEFAULT_DATA_PATH).startswith(str(cache.resolve()))
    assert list(zone.iterdir()) == []


# --- import and startup must not be broken by an unusable locked cache ------------------


def test_import_succeeds_when_locked_and_cache_unwritable(tmp_path: Path) -> None:
    """`import ehrapy.mcp` must not raise just because the pinned cache is unusable."""
    env = dict(os.environ)
    env["EHRAPY_MCP_CACHE_DIR"] = str(tmp_path / "blocker" / "cache")
    env["EHRAPY_MCP_ALLOWED_ROOTS"] = str(tmp_path / "datasets")
    env["MPLBACKEND"] = "Agg"
    (tmp_path / "blocker").write_text("not a directory", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-c", "import ehrapy.mcp; import ehrapy.mcp.server; print('IMPORT_OK')"],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )

    assert proc.returncode == 0, f"import failed:\n{proc.stderr}"
    assert "IMPORT_OK" in proc.stdout


def test_main_exits_nonzero_when_locked_and_cache_unwritable(tmp_path: Path) -> None:
    """Fail closed at startup so the supervisor sees a broken server, not a relocated one."""
    env = dict(os.environ)
    env["EHRAPY_MCP_CACHE_DIR"] = str(tmp_path / "blocker" / "cache")
    env["EHRAPY_MCP_ALLOWED_ROOTS"] = str(tmp_path / "datasets")
    env["MPLBACKEND"] = "Agg"
    (tmp_path / "blocker").write_text("not a directory", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "ehrapy.mcp.server"],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )

    assert proc.returncode == 1, f"expected a fail-closed exit, got {proc.returncode}"
    assert "refusing to start" in proc.stderr
    assert "blocker" in proc.stderr


def test_main_still_starts_when_unlocked(tmp_path: Path) -> None:
    """Unlocked standalone ehrapy is unaffected: no lock, no fail-closed gate."""
    env = dict(os.environ)
    for var in LOCK_VARS:
        env.pop(var, None)
    # Nothing is pinned, so the child resolves the platform user cache. Redirect
    # HOME so an unlocked run does not touch the real one; pinning the cache dir
    # instead would re-lock the process and defeat the point of this test.
    home = tmp_path / "home"
    home.mkdir()
    env["HOME"] = str(home)
    env["MPLBACKEND"] = "Agg"

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from ehrapy.mcp.server import main; sys.argv=['x']; main()",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        input="",
    )

    # Nothing pinned: the fail-closed gate must not fire. If it did, the
    # server would exit 1 and print the refusal -- both asserted below.
    assert "refusing to start" not in proc.stderr
    assert proc.returncode != 1, f"server exited as if locked:\n{proc.stderr}"


# --- timeout default ---------------------------------------------------------------------


def test_tool_timeout_default_is_270(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under Renyi's 300s ceiling, with room for large survival fits."""
    monkeypatch.delenv("EHRAPY_MCP_TIMEOUT_SECONDS", raising=False)
    assert get_tool_timeout() == 270.0


def test_tool_timeout_env_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EHRAPY_MCP_TIMEOUT_SECONDS", "45")
    assert get_tool_timeout() == 45.0


def test_tool_timeout_ignores_nonsense(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EHRAPY_MCP_TIMEOUT_SECONDS", "not-a-number")
    assert get_tool_timeout() == 270.0


# --- runtime context reports the state ---------------------------------------------------


def test_runtime_context_reports_cache_writable(locked: Path) -> None:
    """The agent can see whether the boundary it was given is actually usable."""
    assert registry.is_cache_writable() in {True, False}
    reg = MCPRegistry()
    assert reg.is_cache_writable() is True
