from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationOptions:
    interact: bool
    verbose: bool
    stream_output: bool
    force_overwrite: bool
    agent_name: str
    remote: str | None
    remote_workdir: str | None
    min_rounds: int | None
    continue_optimize: bool
    output: str | None
    test_mode: str | None
    bench_mode: str | None
    npu_devices: str | None = None
    workers_per_npu: str | None = None
    prompt: str | None = None
    log_tools: bool = False
    enable_mcp: bool = False
