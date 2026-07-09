from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Callable, Literal, Optional


class CommandKind(str, Enum):
    GEN_EVAL = "gen-eval"
    GEN_EVAL_BATCH = "gen-eval-batch"
    CONVERT = "convert"
    CONVERT_BATCH = "convert-batch"
    GEN_TEST = "gen-test"
    RUN_TEST = "run-test"
    GEN_BENCH = "gen-bench"
    RUN_BENCH = "run-bench"
    PROBE_BENCH = "probe-bench"
    RUN_SIMULATOR = "run-simulator"
    COMPARE_RESULT = "compare-result"
    COMPARE_PERF = "compare-perf"
    VERIFY = "verify"
    VERIFY_BATCH = "verify-batch"
    STATUS = "status"
    LOG_CHECK = "log-check"
    LOG_CHECK_BATCH = "log-check-batch"
    TRACE_ANALYZE = "trace-analyze"
    RUN_EVAL_MCP_SERVER = "run-eval-mcp-server"
    OPTIMIZE = "optimize"
    OPTIMIZE_BATCH = "optimize-batch"
    UPLOAD_OPTIMIZE = "upload-optimize"
    REPORT = "report"
    REPORT_BATCH = "report-batch"
    CLEAN = "clean"
    DISTILL = "distill"


def command_to_skill(command_kind: CommandKind, language: str = "triton") -> str:
    return {
        CommandKind.GEN_EVAL: "ascend-npu-gen-eval-suite",
        CommandKind.CONVERT: f"{language}-npu-convert-pytorch-operator",
        CommandKind.GEN_TEST: "ascend-npu-gen-test",
        CommandKind.GEN_BENCH: "ascend-npu-gen-bench",
        CommandKind.OPTIMIZE: f"{language}-npu-optimize",
    }.get(command_kind, "")



ProgressProbe = Callable[[], Optional[float]]


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
    stream_output: bool
    force_overwrite: bool
    agent_name: str
    skill_name: str
    prompt: str
    workdir: Path
    remote: Optional[str] = None
    remote_workdir: Optional[str] = None
    npu_devices: Optional[str] = None
    workers_per_npu: Optional[str] = None
    extra_env: dict[str, str] | None = None
    min_rounds: Optional[int] = None
    min_speedup: Optional[float] = None
    continue_optimize: bool = False
    no_agent_session: bool = False
    round_mode: Literal["checked", "supervised"] = "checked"
    round_batch_size: int = 5
    current_round: int = 1
    final_round: int = 1
    user_prompt: Optional[str] = None
    staged_skill_names: tuple[str, ...] | None = None
    staged_skill_sources: dict[str, str] | None = None
    supervisor_report_path: Optional[Path] = None
    language: Literal["triton", "tilelang"] = "triton"
    target_chip: Literal["A3", "A5"] = "A5"
    optimize_target: Literal["kernel", "operator"] = "kernel"
    compiler_source_analysis: Literal["off", "auto"] = "off"
    compiler_source_path: Optional[Path] = None
    compiler_source_commit: Optional[str] = None
    enable_subagent: bool = False
    enable_agent_hooks: bool = False
    log_tools: bool = False
    enable_mcp: bool = False
    mcp_servers: tuple[str, ...] | None = None
    show_output_label: str = ""
    run_id: str = ""
    disable_backend_retry: bool = False
    progress_probe: ProgressProbe | None = None

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
