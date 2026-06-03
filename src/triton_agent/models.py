from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Literal, Optional


class CommandKind(str, Enum):
    GEN_EVAL = "gen-eval"
    GEN_EVAL_BATCH = "gen-eval-batch"
    CONVERT = "convert"
    CONVERT_BATCH = "convert-batch"
    GEN_TEST = "gen-test"
    RUN_TEST = "run-test"
    GEN_BENCH = "gen-bench"
    RUN_BENCH = "run-bench"
    COMPARE_RESULT = "compare-result"
    COMPARE_PERF = "compare-perf"
    VERIFY = "verify"
    VERIFY_BATCH = "verify-batch"
    STATUS = "status"
    LOG_CHECK = "log-check"
    LOG_CHECK_BATCH = "log-check-batch"
    TRACE_ANALYZE = "trace-analyze"
    OPTIMIZE = "optimize"
    OPTIMIZE_BATCH = "optimize-batch"
    UPLOAD_OPTIMIZE = "upload-optimize"
    REPORT = "report"
    REPORT_BATCH = "report-batch"
    CLEAN = "clean"


COMMAND_TO_SKILL = {
    CommandKind.GEN_EVAL: "triton-npu-gen-eval-suite",
    CommandKind.GEN_EVAL_BATCH: "",
    CommandKind.CONVERT: "triton-npu-convert-pytorch-operator",
    CommandKind.CONVERT_BATCH: "",
    CommandKind.GEN_TEST: "triton-npu-gen-test",
    CommandKind.RUN_TEST: "",
    CommandKind.GEN_BENCH: "triton-npu-gen-bench",
    CommandKind.RUN_BENCH: "",
    CommandKind.COMPARE_RESULT: "",
    CommandKind.COMPARE_PERF: "",
    CommandKind.VERIFY: "",
    CommandKind.VERIFY_BATCH: "",
    CommandKind.STATUS: "",
    CommandKind.LOG_CHECK: "",
    CommandKind.LOG_CHECK_BATCH: "",
    CommandKind.TRACE_ANALYZE: "",
    CommandKind.OPTIMIZE: "triton-npu-optimize",
    CommandKind.OPTIMIZE_BATCH: "",
    CommandKind.UPLOAD_OPTIMIZE: "",
    CommandKind.REPORT: "",
    CommandKind.REPORT_BATCH: "",
    CommandKind.CLEAN: "",
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
    remote: Optional[str] = None
    remote_workdir: Optional[str] = None
    extra_env: dict[str, str] | None = None
    min_rounds: Optional[int] = None
    continue_optimize: bool = False
    no_agent_session: bool = False
    round_mode: Literal["continuous", "checked", "supervised"] = "continuous"
    staged_skill_names: tuple[str, ...] | None = None
    staged_skill_sources: dict[str, str] | None = None
    optimize_role: str | None = None
    supervisor_report_path: Optional[Path] = None
    target_chip: Literal["A3", "A5"] = "A5"
    optimize_target: Literal["kernel", "operator"] = "kernel"
    compiler_source_analysis: Literal["off", "auto"] = "off"
    compiler_source_path: Optional[Path] = None
    compiler_source_commit: Optional[str] = None
    enable_subagent: bool = False
    enable_agent_hooks: bool = False
    log_tools: bool = False
    show_output_label: str = ""
    run_id: str = ""

    def with_prompt(self, prompt: str) -> "AgentRequest":
        return replace(self, prompt=prompt)


@dataclass
class AgentResult:
    return_code: int
    stdout: str
    stderr: str
    stalled: bool = False
    session_id: Optional[str] = None
    retryable_failure: bool = False

    @property
    def succeeded(self) -> bool:
        return self.return_code == 0 and not self.stalled
