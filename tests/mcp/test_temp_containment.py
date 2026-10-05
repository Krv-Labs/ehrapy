"""Tests for temp-directory containment: third-party tempfile use stays inside the pinned cache.

Renyi launches the server with EHRAPY_MCP_CACHE_DIR (a directory it erases),
EHRAPY_MCP_ALLOWED_ROOTS and EHRAPY_MCP_READ_ONLY=1. ehradata unpacks compressed
demo cohorts (physionet2012, physionet2019) with ``tempfile.mkdtemp()`` no matter
which output path it is given, and pooch, matplotlib and the stdlib reach for the
system temp directory the same way -- none of it raises on the success path, so
the process temp directory itself is the boundary. These tests pin that down,
including the unlocked no-op so standalone ehrapy keeps its operator's temp dir.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from ehrapy.mcp.policy import is_confinement_locked
from ehrapy.mcp.registry import _contain_tempdir, registry

LOCK_VARS = ("EHRAPY_MCP_CACHE_DIR", "EHRAPY_MCP_ALLOWED_ROOTS")


@pytest.fixture(autouse=True)
def _restore_temp_globals():
    """Undo the process-global temp retargeting so nothing leaks into other test files.

    ``_contain_tempdir`` writes ``os.environ["TMPDIR"]`` and ``tempfile.tempdir``
    from inside the function under test, so monkeypatch cannot undo those writes.
    Leaking either would silently redirect every later test's temp files.
    """
    saved_tmpdir = os.environ.get("TMPDIR")
    saved_tempdir = tempfile.tempdir
    yield
    if saved_tmpdir is None:
        os.environ.pop("TMPDIR", None)
    else:
        os.environ["TMPDIR"] = saved_tmpdir
    tempfile.tempdir = saved_tempdir


@pytest.fixture
def locked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Confinement pinned to a writable directory inside tmp_path, as Renyi does it."""
    cache = tmp_path / "renyi-cache"
    cache.mkdir()
    monkeypatch.setenv("EHRAPY_MCP_CACHE_DIR", str(cache))
    monkeypatch.setenv("EHRAPY_MCP_ALLOWED_ROOTS", str(cache))
    assert is_confinement_locked()
    return cache


@pytest.fixture
def unlocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Standalone ehrapy: no supervisor, so today's temp behaviour stays legitimate."""
    for var in LOCK_VARS:
        monkeypatch.delenv(var, raising=False)
    assert not is_confinement_locked()


def _isolate_tmpdir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the temp directory at an empty zone and forget the cached lookup."""
    zone = tmp_path / "tmpzone"
    zone.mkdir()
    monkeypatch.setenv("TMPDIR", str(zone))
    monkeypatch.setattr(tempfile, "tempdir", None)
    return zone


# --- containment while locked -------------------------------------------------------------


def test_locked_containment_keeps_every_tempfile_call_inside_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, locked: Path
) -> None:
    """gettempdir, mkdtemp and TemporaryDirectory must all resolve under <cache_dir>/tmp."""
    zone = _isolate_tmpdir(monkeypatch, tmp_path)

    target = _contain_tempdir(locked)

    assert target == locked / "tmp"
    assert Path(tempfile.gettempdir()) == target

    made = Path(tempfile.mkdtemp())
    try:
        with tempfile.TemporaryDirectory() as scratch:
            assert made.is_relative_to(target)
            assert Path(scratch).is_relative_to(target)
            assert Path(scratch).is_relative_to(locked)
    finally:
        shutil.rmtree(made, ignore_errors=True)
    assert list(zone.iterdir()) == [], "temp files still escaped to the old temp directory"


def test_locked_containment_overrides_inherited_operator_tmpdir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, locked: Path
) -> None:
    """An operator TMPDIR cannot pull temp files out of the pinned boundary.

    The pinned cache wins because it is the directory the supervisor knows about
    and erases; an inherited TMPDIR may be an arbitrary leftover from a parent
    process, and a temp file the supervisor cannot find is a leak it does not
    know it has.
    """
    outside = tmp_path / "operator-tmp"
    outside.mkdir()
    monkeypatch.setenv("TMPDIR", str(outside))
    monkeypatch.setattr(tempfile, "tempdir", None)
    assert Path(tempfile.gettempdir()) == outside, "baseline: the operator TMPDIR was live"

    target = _contain_tempdir(locked)

    assert os.environ["TMPDIR"] == str(target)
    assert Path(tempfile.gettempdir()) == target
    for _ in range(3):
        tempfile.mkdtemp()
    with tempfile.NamedTemporaryFile():
        pass
    assert list(outside.iterdir()) == [], "an inherited operator TMPDIR still received temp files"


def test_locked_containment_target_is_owner_only(monkeypatch: pytest.MonkeyPatch, locked: Path) -> None:
    """The contained directory exists with mode 0o700, and a looser one is tightened."""
    assert not (locked / "tmp").exists()

    target = _contain_tempdir(locked)

    assert target.is_dir()
    assert target.stat().st_mode & 0o777 == 0o700

    # Idempotent re-runs repair permissions too, instead of trusting an existing dir.
    target.chmod(0o755)
    _contain_tempdir(locked)
    assert target.stat().st_mode & 0o777 == 0o700


def test_containment_is_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, locked: Path) -> None:
    """Calling twice must not raise, and must not move the resolved path."""
    _isolate_tmpdir(monkeypatch, tmp_path)

    first = _contain_tempdir(locked)
    resolved = tempfile.gettempdir()

    second = _contain_tempdir(locked)

    assert second == first
    assert tempfile.gettempdir() == resolved
    assert Path(tempfile.mkdtemp()).is_relative_to(first)


# --- containment must not touch anything while unlocked ----------------------------------


def test_unlocked_containment_is_a_noop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, unlocked: None) -> None:
    """Unlocked standalone ehrapy: TMPDIR and tempfile resolution are left alone."""
    zone = _isolate_tmpdir(monkeypatch, tmp_path)
    cache = tmp_path / "cache"

    assert _contain_tempdir(cache) is None

    assert os.environ["TMPDIR"] == str(zone)
    assert Path(tempfile.gettempdir()) == zone
    assert not (cache / "tmp").exists(), "containment must not create anything while unlocked"
    assert Path(tempfile.mkdtemp()).is_relative_to(zone)


# --- main(): placement relative to the fail-closed gate -----------------------------------


def test_main_pins_tempdir_before_mcp_run(tmp_path: Path) -> None:
    """main() contains temp before mcp.run, so a started server is never uncontained."""
    cache = tmp_path / "renyi-cache"
    cache.mkdir()
    outside = tmp_path / "operator-tmp"
    outside.mkdir()
    env = dict(os.environ)
    env["EHRAPY_MCP_CACHE_DIR"] = str(cache)
    env["EHRAPY_MCP_ALLOWED_ROOTS"] = str(cache)
    env["TMPDIR"] = str(outside)
    env["MPLBACKEND"] = "Agg"
    script = (
        "import sys, tempfile;"
        "from ehrapy.mcp import server;"
        "server.mcp.run = lambda **kw: print('TEMPDIR', tempfile.gettempdir());"
        "sys.argv = ['ehrapy-mcp', '--no-purge'];"
        "server.main()"
    )

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )

    assert proc.returncode == 0, f"main failed:\n{proc.stderr}"
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("TEMPDIR "))
    assert Path(line.split(" ", 1)[1]).is_relative_to(cache.resolve()), "mcp.run saw an uncontained temp dir"
    assert list(outside.iterdir()) == [], "temp files escaped to the operator's TMPDIR"


def test_main_does_not_mutate_temp_when_the_gate_refuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A server that refuses to start must leave the temp directory untouched."""
    from ehrapy.mcp import server as server_module

    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    cache = blocker / "cache"
    monkeypatch.setenv("EHRAPY_MCP_CACHE_DIR", str(cache))
    monkeypatch.setenv("EHRAPY_MCP_ALLOWED_ROOTS", str(tmp_path / "datasets"))
    # The gate probes the singleton registry, so pin its cache at the blocker path
    # the way an import under Renyi's environment would have resolved it.
    monkeypatch.setattr(registry, "_cache_dir_path", cache)

    class _StubMCP:
        def __init__(self) -> None:
            self.run_calls: list[dict] = []

        def run(self, **kwargs) -> None:
            self.run_calls.append(kwargs)

    stub = _StubMCP()
    monkeypatch.setattr(server_module, "mcp", stub)
    monkeypatch.setattr(sys, "argv", ["ehrapy-mcp", "--no-purge"])
    zone = _isolate_tmpdir(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        server_module.main()

    assert excinfo.value.code == 1
    assert stub.run_calls == [], "the gate did not stop the server"
    assert os.environ["TMPDIR"] == str(zone), "temp was mutated before the gate refused"
    assert Path(tempfile.gettempdir()) == zone
