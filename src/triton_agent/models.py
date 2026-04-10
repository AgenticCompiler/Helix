from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Optional


class CommandKind(str, Enum):
    GEN_EVAL = "gen-eval"
    GEN_EVAL_BATCH = "gen-eval-batch"
    GEN_TEST = "gen-test"
    RUN_TEST = "run-test"
    GEN_BENCH = "gen-bench"
    RUN_BENCH = "run-bench"
    COMPARE_RESULT = "compare-result"
    COMPARE_PERF = "compare-perf"
    OPTIMIZE_STATUS = "optimize-status"
    OPTIMIZE = "optimize"
    OPTIMIZE_BATCH = "optimize-batch"


COMMAND_TO_SKILL = {
    CommandKind.GEN_EVAL: "eval-gen",
    CommandKind.GEN_EVAL_BATCH: "",
    CommandKind.GEN_TEST: "test-gen",
    CommandKind.RUN_TEST: "",
    CommandKind.GEN_BENCH: "bench-gen",
    CommandKind.RUN_BENCH: "",
    CommandKind.COMPARE_RESULT: "",
    CommandKind.COMPARE_PERF: "",
    CommandKind.OPTIMIZE_STATUS: "",
    CommandKind.OPTIMIZE: "optimize",
    CommandKind.OPTIMIZE_BATCH: "",
}


@dataclass
class AgentRequest:
    command_kind: CommandKind
    input_path: Path
    operator_path: Optional[Path]
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
    min_rounds: Optional[int] = None
    continue_optimize: bool = False
    require_analysis: bool = False
    no_agent_session: bool = False
    staged_skill_names: tuple[str, ...] | None = None
    optimize_role: str | None = None
    round_brief_path: Optional[Path] = None
    supervisor_report_path: Optional[Path] = None

    def with_prompt(self, prompt: str) -> "AgentRequest":
        return replace(self, prompt=prompt)


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
