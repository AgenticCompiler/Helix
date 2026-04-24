from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConvertOptions:
    interact: bool
    verbose: bool
    show_output: bool
    force_overwrite: bool
    agent_name: str
    remote: str | None
    remote_workdir: str | None
    output: str | None
    test_mode: str | None
    prompt: str | None = None
