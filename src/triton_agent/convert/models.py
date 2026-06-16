from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConvertOptions:
    interact: bool
    verbose: bool
    stream_output: bool
    force_overwrite: bool
    agent_name: str
    remote: str | None
    remote_workdir: str | None
    output: str | None
    test_mode: str | None
    prompt: str | None = None
    log_tools: bool = False
    enable_mcp: bool = False
