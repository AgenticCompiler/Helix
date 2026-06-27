from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any

from triton_agent.run_eval_mcp_server import (
    RUN_EVAL_MCP_SERVER_NAME,
    RunningHttpMCPServer,
    start_http_server,
)


@dataclass
class _ManagedMcpScopeState:
    server: RunningHttpMCPServer | None = None
    ref_count: int = 0


_scope_lock = threading.RLock()
_active_scope: _ManagedMcpScopeState | None = None


def managed_mcp_server_names_for_request(
    staged_skill_names: tuple[str, ...] | None,
    *,
    enable_mcp: bool,
) -> tuple[str, ...] | None:
    if not enable_mcp or staged_skill_names is None:
        return None
    if "ascend-npu-run-eval" not in staged_skill_names:
        return None
    return (RUN_EVAL_MCP_SERVER_NAME,)


@contextmanager
def managed_mcp_scope() -> Iterator[None]:
    global _active_scope
    created = False
    with _scope_lock:
        if _active_scope is None:
            _active_scope = _ManagedMcpScopeState(server=None, ref_count=0)
            created = True
        _active_scope.ref_count += 1
    try:
        yield
    finally:
        state_to_close: _ManagedMcpScopeState | None = None
        with _scope_lock:
            assert _active_scope is not None
            _active_scope.ref_count -= 1
            if _active_scope.ref_count == 0 and created:
                state_to_close = _active_scope
                _active_scope = None
        if state_to_close is not None and state_to_close.server is not None:
            state_to_close.server.close()


def resolve_managed_mcp_servers(
    *,
    workdir: Path,
    server_names: tuple[str, ...] | None,
) -> dict[str, dict[str, Any]]:
    if not server_names:
        return {}

    with _ensure_scope():
        state = ensure_managed_mcp_server()
        assert state.server is not None
        resolved: dict[str, dict[str, Any]] = {}
        for server_name in server_names:
            if server_name == RUN_EVAL_MCP_SERVER_NAME:
                resolved[server_name] = {
                    "transport": "http",
                    "url": state.server.url_for_workspace(workdir),
                }
                continue
            raise ValueError(f"Unsupported managed MCP server: {server_name}")
        return resolved


def current_managed_mcp_scope() -> _ManagedMcpScopeState:
    with _scope_lock:
        if _active_scope is None:
            raise RuntimeError("Managed MCP scope is not active.")
        return _active_scope


def ensure_managed_mcp_server() -> _ManagedMcpScopeState:
    with _scope_lock:
        state = current_managed_mcp_scope()
        if state.server is None:
            state.server = start_http_server()
        return state


@contextmanager
def _ensure_scope() -> Iterator[None]:
    with _scope_lock:
        active = _active_scope is not None
    if active:
        yield
        return
    with managed_mcp_scope():
        yield
