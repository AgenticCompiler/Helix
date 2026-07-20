from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

@dataclass(frozen=True)
class OptimizeCheckResult:
    kind: Literal["baseline", "round"]
    status: Literal["pass", "fail"]
    issues: tuple[str, ...]
    summary: str
    next_option: str | None = None


@dataclass(frozen=True)
class BaselineState:
    baseline_kind: str
    source_operator: str
    baseline_operator: str
    test_file: str
    test_mode: str
    bench_file: str
    bench_mode: str
    perf_artifact: str
    correctness_status: str
    benchmark_status: str
    baseline_established: bool
    preparation_notes: str | None = None
    baseline_repairs_summary: str | None = None


@dataclass(frozen=True)
class BaselineArtifactsInspection:
    baseline_dir: Path
    state_path: Path | None
    perf_path: Path | None
    operator_path: Path | None
    issues: tuple[str, ...]


@dataclass(frozen=True)
class RoundState:
    round_name: str
    parent_round: str
    hypothesis: str
    evidence_sources: tuple[str, ...]
    correctness_status: str
    benchmark_status: str
    perf_artifact: str | None
    comparison_target_path: str | None
    effective_metric_source: str | None
    summary_path: str
    opt_note_updated: bool
    analysis_skipped_reason: str | None = None
    profile_dir: str | None = None
    ir_dir: str | None = None
    perf_analysis_path: str | None = None


@dataclass(frozen=True)
class RoundArtifactsInspection:
    round_dir: Path
    operator_path: Path | None
    attempts_path: Path | None
    summary_path: Path | None
    perf_path: Path | None
    perf_analysis_path: Path | None
    round_state_path: Path | None
    issues: tuple[str, ...]


@dataclass(frozen=True)
class OptimizeRunOptions:
    agent_name: str
    interact: bool
    verbose: bool
    stream_output: bool
    remote: str | None
    remote_workdir: str | None
    min_rounds: int
    resume_mode: str
    reset_optimize: bool
    no_agent_session: bool
    round_mode: Literal["checked", "supervised"]
    output: str | None
    test_mode: str | None
    bench_mode: str | None
    prompt: str | None
    system_prompt: str | None = None
    npu_devices: str | None = None
    workers_per_npu: str | None = None
    min_speedup: float | None = None
    post_optimize_command: str | None = None
    language: Literal["triton", "tilelang"] = "triton"
    round_batch_size: int = 5
    target_chip: Literal["A3", "A5"] = "A5"
    optimize_target: Literal["kernel", "operator"] = "kernel"
    optimize_knowledge: Literal["v1", "v2", "v3"] = "v1"
    compiler_source_analysis: Literal["off", "auto"] = "off"
    enable_cann_ext_api: bool = False
    enable_subagent: bool = False
    enable_agent_hooks: bool = False
    upload_enabled: bool = True
    report: bool = False
    log_tools: bool = False
    enable_mcp: bool = False


@dataclass(frozen=True)
class BatchOptimizeWorkspace:
    workspace: Path
    operator_file: Path


@dataclass(frozen=True)
class BatchOptimizeResult:
    workspace: Path
    status: Literal["ok", "failed", "skipped"]
    message: str

    @property
    def succeeded(self) -> bool:
        return self.status == "ok"


class BaselinePreflightState(str, Enum):
    READY = "ready"
    NEEDS_PREPARE = "needs-prepare"
    NEEDS_REPAIR = "needs-repair"


@dataclass(frozen=True)
class BaselinePreflightResult:
    state: BaselinePreflightState
    issues: tuple[str, ...]
