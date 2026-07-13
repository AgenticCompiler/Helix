from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, TextIO

from helix.backends.base import AgentRunner
from helix.backends.hook_common import cleanup_hook_stage, describe_cleanup, describe_prepare
from helix.backends.opencode_hooks import prepare_opencode_hooks
from helix.eval.mcp import resolve_managed_mcp_servers
from helix.models import AgentRequest
from helix.terminal.verbose import emit_verbose_lines


_OPENCODE_CONFIG_PATH = Path(".opencode") / "opencode.json"


def _opencode_workspace_config(
    managed_servers: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    permission = {"task": {"general": "deny", "explore": "deny"}}
    config: dict[str, object] = {
        "$schema": "https://opencode.ai/config.json",
        "agent": {
            "build": {"mode": "primary", "permission": permission},
            "plan": {"mode": "primary", "permission": permission},
        },
    }
    if managed_servers:
        mcp: dict[str, object] = {}
        for name, server in managed_servers.items():
            mcp[name] = {
                "type": "remote",
                "url": server["url"],
            }
        config["mcp"] = mcp
    return config


class OpenCodeRunner(AgentRunner):
    def __init__(self, executable: str = "opencode", stall_timeout_seconds: int = 900) -> None:
        super().__init__(executable, stall_timeout_seconds)

    def supports_mcp_servers(self) -> bool:
        return True

    def build_command(self, request: AgentRequest) -> list[str]:
        if request.interact:
            command = [
                self.executable,
                str(request.workdir),
                "--prompt",
                request.prompt,
            ]
            if not request.enable_agent_hooks and not request.log_tools:
                command.insert(2, "--pure")
            return command

        command = [
            self.executable,
            "run",
            "--dir",
            str(request.workdir),
            "--dangerously-skip-permissions",
            "--thinking",
            request.prompt,
        ]
        if not request.enable_agent_hooks and not request.log_tools:
            command.insert(5, "--pure")
        return command

    @contextmanager
    def _prepare_run_context(
        self,
        request: AgentRequest,
        stderr: Optional[TextIO] = None,
    ) -> Iterator[None]:
        hook_state = None
        if request.enable_agent_hooks or request.log_tools:
            hook_state = prepare_opencode_hooks(
                _hooks_root(),
                request.workdir,
                self._hook_options(request),
                extra_allowed_read_roots=self._extra_allowed_read_roots(request),
            )
            if request.verbose:
                emit_verbose_lines(stderr or sys.stderr, "hooks", describe_prepare(hook_state))

        config_path = request.workdir / _OPENCODE_CONFIG_PATH
        if config_path.exists() or config_path.is_symlink():
            warning_stream = stderr or sys.stderr
            print(
                f"Warning: Existing OpenCode workspace config detected; skipping staged config: {config_path}",
                file=warning_stream,
            )
            yield
            return

        managed_servers = resolve_managed_mcp_servers(
            workdir=request.workdir,
            server_names=request.mcp_servers,
        )
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(_opencode_workspace_config(managed_servers), indent=2) + "\n",
            encoding="utf-8",
        )
        try:
            yield
        finally:
            if config_path.exists() or config_path.is_symlink():
                config_path.unlink()
            if hook_state is not None:
                if request.verbose:
                    emit_verbose_lines(stderr or sys.stderr, "hooks", describe_cleanup(hook_state))
                cleanup_warnings = cleanup_hook_stage(hook_state)
                if cleanup_warnings:
                    emit_verbose_lines(stderr or sys.stderr, "hooks", cleanup_warnings)


def _hooks_root() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "hooks"
