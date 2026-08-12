"""Per-client MCP session state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class _EHRapySession:
    edata_id: str | None = None


_SESSIONS: dict[str, _EHRapySession] = {}


def _session_key(ctx: Any) -> str:
    if ctx is None:
        return "default"
    client_id = getattr(ctx, "client_id", None)
    if client_id:
        return str(client_id)
    session_id = getattr(ctx, "session_id", None)
    if session_id:
        return str(session_id)
    return "default"


def get_session(ctx: Any = None) -> _EHRapySession:
    """Return the per-client session, creating one if needed."""
    key = _session_key(ctx)
    if key not in _SESSIONS:
        _SESSIONS[key] = _EHRapySession()
    return _SESSIONS[key]
