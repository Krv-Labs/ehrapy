"""Tests for how the per-client session key is derived.

`ctx.session_id` looks like a client identity and is not one. On the modern
`2026-07-28` protocol, MCP SDK v2 builds a fresh `Connection` per request, so
FastMCP mints a new `uuid4` for `session_id` on every tool call and discards it
with the connection. Keying sessions on it silently loses the active dataset
handle from the second call onward.

These tests pin the reasoning rather than the implementation, so the probe order
cannot be "simplified" back into the bug.
"""

from __future__ import annotations

from typing import Any

import pytest

from ehrapy.mcp.session import _session_key, get_session, reset_sessions

MODERN = "2026-07-28"
HANDSHAKE = "2025-11-25"


class FakeRequestContext:
    def __init__(self, protocol_version: str | None) -> None:
        self.protocol_version = protocol_version


class FakeContext:
    """Stands in for a FastMCP Context, including the era-dependent instability."""

    def __init__(
        self,
        *,
        transport: str | None,
        protocol_version: str | None,
        session_id: str | None = None,
        client_id: str | None = None,
    ) -> None:
        self.transport = transport
        self.request_context = FakeRequestContext(protocol_version)
        self._session_id = session_id
        self.client_id = client_id

    @property
    def session_id(self) -> str:
        # Mirrors FastMCP: a fresh uuid4 per call unless something cached it.
        if self._session_id is None:
            self._session_id = f"uuid-{id(self):x}-next"
        return self._session_id


def test_modern_era_session_id_is_not_an_identity() -> None:
    """The core bug: on 2026-07-28, consecutive calls get different session ids."""
    first = FakeContext(transport="stdio", protocol_version=MODERN)
    second = FakeContext(transport="stdio", protocol_version=MODERN)

    assert first.session_id != second.session_id, "premise: session_id must be unstable here"


@pytest.mark.parametrize("transport", ["stdio", None])
def test_stdio_and_in_process_fall_back_to_a_process_key(transport: str | None) -> None:
    """One client per process means one key, whichever era we are on.

    A changing `session_id` must not change the key: for stdio the transport
    carries a single client, so the id FastMCP mints is noise.
    """
    keys = {
        _session_key(FakeContext(transport=transport, protocol_version=era))
        for era in (MODERN, HANDSHAKE)
        for _ in range(3)
    }
    assert keys == {"default"}, f"expected one process-global key, got {keys}"


def test_stateful_handshake_http_may_use_the_negotiated_session_id() -> None:
    """Where `session_id` really is an identity, it is still used."""
    a = FakeContext(transport="streamable-http", protocol_version=HANDSHAKE, session_id="sid-a")
    b = FakeContext(transport="streamable-http", protocol_version=HANDSHAKE, session_id="sid-b")

    assert _session_key(a) == "session:sid-a"
    assert _session_key(b) == "session:sid-b"
    assert _session_key(a) != _session_key(b)


def test_modern_http_does_not_trust_a_per_request_session_id() -> None:
    """Stateful HTTP still mints per call on the modern era, so it is not trusted."""
    keys = {_session_key(FakeContext(transport="streamable-http", protocol_version=MODERN)) for _ in range(3)}
    assert keys == {"default"}


def test_unknown_protocol_version_does_not_trust_session_id() -> None:
    """An unrecognised era falls back rather than guessing."""
    ctx = FakeContext(transport="streamable-http", protocol_version="1999-01-01", session_id="sid")
    assert _session_key(ctx) == "default"


def test_client_id_wins_and_isolates_clients() -> None:
    """`client_id` is stable on both eras, so it outranks everything else."""
    a = FakeContext(transport="streamable-http", protocol_version=MODERN, client_id="renyi-1", session_id="x")
    b = FakeContext(transport="streamable-http", protocol_version=MODERN, client_id="renyi-2", session_id="y")

    assert _session_key(a) == "client:renyi-1"
    assert _session_key(b) == "client:renyi-2"


def test_active_handle_survives_repeated_calls_on_the_modern_era() -> None:
    """The regression itself: an ingested dataset stays the active one.

    Without the era check this fails on the second call with NO_ACTIVE_DATASET.
    """
    reset_sessions()
    try:
        # Set the handle once, as ingest_dataset does...
        get_session(FakeContext(transport="stdio", protocol_version=MODERN)).set_latest_edata_id("handle-abc")
        # ...then every later call must find it, without being told again.
        for _ in range(4):
            session = get_session(FakeContext(transport="stdio", protocol_version=MODERN))
            assert session.get_latest_edata_id() == "handle-abc"
    finally:
        reset_sessions()


def test_missing_context_attributes_fall_back() -> None:
    """Probing must never raise, whatever the context looks like."""

    class Hostile:
        @property
        def client_id(self) -> Any:
            raise RuntimeError("outside request context")

        @property
        def transport(self) -> Any:
            raise RuntimeError("outside request context")

        @property
        def session_id(self) -> Any:
            raise RuntimeError("outside request context")

    assert _session_key(Hostile()) == "default"
    assert _session_key(None) == "default"
