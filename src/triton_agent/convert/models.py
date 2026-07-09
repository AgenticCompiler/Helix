from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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
    npu_devices: str | None = None
    workers_per_npu: str | None = None
    language: Literal["triton", "tilelang"] = "triton"
    prompt: str | None = None
    log_tools: bool = False
    enable_mcp: bool = False
