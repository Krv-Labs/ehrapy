"""Per-client MCP session state."""

from __future__ import annotations

import threading
from typing import Any

from mcp_types.version import HANDSHAKE_PROTOCOL_VERSIONS

_SESSIONS_LOCK = threading.RLock()


class _EHRapySession:
    """Thread-safe per-client container for the active EHRData handle."""

    def __init__(self, edata_id: str | None = None) -> None:
        self._edata_id = edata_id
        self._lock = threading.RLock()

    @property
    def edata_id(self) -> str | None:
        """Return the active edata_id for this session."""
        with self._lock:
            return self._edata_id

    @edata_id.setter
    def edata_id(self, value: str | None) -> None:
        with self._lock:
            self._edata_id = value

    def get_latest_edata_id(self) -> str | None:
        """Return the active edata_id for this session."""
        return self.edata_id

    def set_latest_edata_id(self, edata_id: str | None) -> None:
        """Set the active edata_id for this session."""
        self.edata_id = edata_id


_SESSIONS: dict[str, _EHRapySession] = {}


def _probe(ctx: Any, attr: str) -> Any:
    """Read an attribute defensively: FastMCP raises outside an active request."""
    try:
        return getattr(ctx, attr, None)
    except (RuntimeError, AttributeError):
        return None


def _session_key(ctx: Any) -> str:
    """Derive a stable per-client key from a FastMCP Context.

    Not every attribute a Context exposes is an identity, and one of them used to
    be mistaken for one.

    ``client_id`` is the only cross-era-stable identity FastMCP offers: it is
    whatever the client puts in ``_meta.client_id``, and it reads the same on both
    protocol eras. A supervisor may send it, so it is consulted first.

    ``session_id`` is **not** a per-client key on the modern ``2026-07-28``
    protocol. MCP SDK v2 constructs a fresh ``Connection`` for every request on
    that era (``mcp/server/runner.py``), so FastMCP finds no cached value, mints a
    new ``uuid4``, and throws it away with the connection
    (``fastmcp/server/context.py``). Its own docstring only promises "a generated
    ID for other transports". It is trustworthy only where it is a real
    negotiated identity: a stateful HTTP transport speaking a handshake-era
    protocol version, where it is the client's ``mcp-session-id``. Everywhere else
    it changes on every call, which silently loses the active dataset handle.

    stdio serves exactly one client per process -- one stdin/stdout pair, one
    connection -- so a process-global key is correct there by construction, on
    either era. The in-process transport used by the test suite is treated the
    same way, because it is single-tenant too.

    Each attribute is probed defensively, and a server-wide fallback is
    preferable to propagating an error into every tool call.
    """
    if ctx is None:
        return "default"

    client_id = _probe(ctx, "client_id")
    if client_id:
        return f"client:{client_id}"

    transport = _probe(ctx, "transport")
    protocol = _probe(_probe(ctx, "request_context"), "protocol_version")
    if transport not in ("stdio", None) and protocol in HANDSHAKE_PROTOCOL_VERSIONS:
        session_id = _probe(ctx, "session_id")
        if session_id:
            return f"session:{session_id}"

    return "default"


def get_session(ctx: Any = None) -> _EHRapySession:
    """Return the per-client session, creating one if needed."""
    key = _session_key(ctx)
    with _SESSIONS_LOCK:
        if key not in _SESSIONS:
            _SESSIONS[key] = _EHRapySession()
        return _SESSIONS[key]


def reset_sessions() -> None:
    """Reset all session states (useful for testing)."""
    with _SESSIONS_LOCK:
        _SESSIONS.clear()
