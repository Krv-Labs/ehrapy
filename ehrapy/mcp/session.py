"""Per-client MCP session state."""

from __future__ import annotations

import threading
from typing import Any

_SESSION_LOCK = threading.RLock()


class _EHRapySession:
    """Thread-safe session container for per-client EHRData handle state."""

    def __init__(self, edata_id: str | None = None) -> None:
        self._edata_id = edata_id
        self._lock = threading.RLock()

    @property
    def edata_id(self) -> str | None:
        with self._lock:
            return self._edata_id

    @edata_id.setter
    def edata_id(self, value: str | None) -> None:
        with self._lock:
            self._edata_id = value


_SESSIONS: dict[str, _EHRapySession] = {}


def _session_key(ctx: Any) -> str:
    if ctx is None:
        return "default"
    try:
        client_id = getattr(ctx, "client_id", None)
        if client_id:
            return str(client_id)
    except (RuntimeError, AttributeError):
        pass
    try:
        session_id = getattr(ctx, "session_id", None)
        if session_id:
            return str(session_id)
    except (RuntimeError, AttributeError):
        pass
    try:
        request_id = getattr(ctx, "request_id", None)
        if request_id:
            return str(request_id)
    except (RuntimeError, AttributeError):
        pass
    return "default"


def get_session(ctx: Any = None) -> _EHRapySession:
    """Return the per-client session, creating one if needed."""
    key = _session_key(ctx)
    with _SESSION_LOCK:
        if key not in _SESSIONS:
            _SESSIONS[key] = _EHRapySession()
        return _SESSIONS[key]

