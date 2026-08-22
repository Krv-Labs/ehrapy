"""Structured MCP error envelopes."""

from __future__ import annotations

import json
from typing import Any

_SANDBOX_PATH_PREFIXES = (
    "/home/claude",
    "/mnt/",
    "/workspace/",
    "/tmp/claude",
    "/System/Volumes/Data/home/claude",
)


def mcp_error(
    tool: str,
    reason: str,
    *,
    error_code: str | None = None,
    agent_action: str | None = None,
    details: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> str:
    """Return a JSON error envelope for MCP tools."""
    return json.dumps(
        {
            "status": "error",
            "tool": tool,
            "reason": reason,
            "error_code": error_code,
            "agent_action": agent_action,
            "details": details or {},
            "metrics": metrics or {},
        },
        indent=2,
    )


def classify_path(path: str) -> dict[str, Any]:
    """Flag paths that look like agent-sandbox locations."""
    matched_prefix = next(
        (prefix for prefix in _SANDBOX_PATH_PREFIXES if path.startswith(prefix)),
        None,
    )
    return {
        "attempted_path": path,
        "looks_like_sandbox_path": matched_prefix is not None,
        "matched_prefix": matched_prefix,
    }


def path_access_error(
    tool: str,
    path: str,
    *,
    missing_code: str = "FILE_NOT_FOUND",
    missing_reason: str = "Path does not exist on the MCP host filesystem.",
    missing_action: str = ("Provide a host-visible absolute path, or call get_runtime_context first."),
    sandbox_action: str | None = None,
) -> str:
    """Return a path-access error envelope."""
    path_context = classify_path(path)
    if path_context["looks_like_sandbox_path"]:
        return mcp_error(
            tool,
            "Path is not visible to the MCP server host filesystem.",
            error_code="HOST_PATH_NOT_VISIBLE",
            agent_action=sandbox_action
            or ("Ask the user for a host-visible absolute path, or call get_runtime_context first."),
            details={"path_context": path_context},
        )

    return mcp_error(
        tool,
        missing_reason,
        error_code=missing_code,
        agent_action=missing_action,
        details={"path_context": path_context},
    )


def unknown_handle_error(tool: str, handle_name: str, handle_value: str) -> str:
    """Return an unknown-handle error envelope."""
    return mcp_error(
        tool,
        f"Unknown {handle_name} '{handle_value}'.",
        error_code=f"{handle_name.upper()}_UNKNOWN",
        agent_action=(f"Create or retrieve a valid {handle_name} before retrying this tool."),
        details={handle_name: handle_value},
    )


def classify_exception_error(
    tool: str,
    exc: Exception,
    *,
    namespace: str | None = None,
    function: str | None = None,
) -> str:
    """Classify an exception into a structured MCP error envelope with a non-null error_code."""
    exc_type = type(exc).__name__
    exc_msg = str(exc)

    if isinstance(exc, TypeError):
        action = (
            f"Call get_function_help('{namespace or 'dispatch'}', '{function}') to inspect expected parameters."
            if function
            else "Inspect tool parameters."
        )
        return mcp_error(
            tool,
            exc_msg,
            error_code="INVALID_INPUT",
            agent_action=action,
            details={"exception_type": exc_type, "function": function, "namespace": namespace},
        )

    if isinstance(exc, (ImportError, ModuleNotFoundError)) or "Install with" in exc_msg or "requires" in exc_msg:
        return mcp_error(
            tool,
            exc_msg,
            error_code="DEPENDENCY_MISSING",
            agent_action="Install the required optional dependency or extra package.",
            details={"exception_type": exc_type},
        )

    if isinstance(exc, KeyError):
        return mcp_error(
            tool,
            exc_msg,
            error_code="KEY_NOT_FOUND" if "Unknown" not in exc_msg else "FUNCTION_UNKNOWN",
            agent_action="Check the provided key, column name, or function name.",
            details={"exception_type": exc_type},
        )

    if isinstance(exc, FileNotFoundError):
        return mcp_error(
            tool,
            exc_msg,
            error_code="FILE_NOT_FOUND",
            agent_action="Verify the file path exists on the host filesystem.",
            details={"exception_type": exc_type},
        )

    if isinstance(exc, ValueError):
        return mcp_error(
            tool,
            exc_msg,
            error_code="INVALID_INPUT",
            agent_action="Check argument types and values.",
            details={"exception_type": exc_type},
        )

    return mcp_error(
        tool,
        exc_msg,
        error_code="INTERNAL",
        agent_action="Check server logs or retry with valid inputs.",
        details={"exception_type": exc_type},
    )
