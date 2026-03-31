from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class CommandKind(str, Enum):
    GEN_TEST = "gen-test"
    RUN_TEST = "run-test"
    GEN_BENCH = "gen-bench"
    RUN_BENCH = "run-bench"
    OPTIMIZE = "optimize"


COMMAND_TO_SKILL = {
    CommandKind.GEN_TEST: "test-gen",
    CommandKind.RUN_TEST: "test-run",
    CommandKind.GEN_BENCH: "bench-gen",
    CommandKind.RUN_BENCH: "bench-run",
    CommandKind.OPTIMIZE: "optimize",
}


@dataclass
class AgentRequest:
    command_kind: CommandKind
    input_path: Path
    output_path: Optional[Path]
    test_mode: Optional[str]
    bench_mode: Optional[str]
    interact: bool
    verbose: bool
    show_output: bool
    force_overwrite: bool
    agent_name: str
    skill_name: str
    prompt: str
    workdir: Path


@dataclass
class AgentResult:
    return_code: int
    stdout: str
    stderr: str
    stalled: bool = False
    session_id: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.return_code == 0 and not self.stalled
