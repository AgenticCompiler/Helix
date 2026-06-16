from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from triton_agent.optimize.skill_contract import (
    optimize_submit_baseline_module,
    optimize_submit_round_module,
)


_OPTIMIZE_BASELINE_MODULE = optimize_submit_baseline_module()
_OPTIMIZE_ROUND_MODULE = optimize_submit_round_module()

BaselineState = _OPTIMIZE_BASELINE_MODULE.BaselineState  # type: ignore[reportUnknownVariableType]
BaselineArtifactsInspection = _OPTIMIZE_BASELINE_MODULE.BaselineArtifactsInspection  # type: ignore[reportUnknownVariableType]
RoundState = _OPTIMIZE_ROUND_MODULE.RoundState  # type: ignore[reportUnknownVariableType]
RoundArtifactsInspection = _OPTIMIZE_ROUND_MODULE.RoundArtifactsInspection  # type: ignore[reportUnknownVariableType]
OptimizeCheckResult = _OPTIMIZE_ROUND_MODULE.OptimizeCheckResult  # type: ignore[reportUnknownVariableType]


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


@dataclass(frozen=True)
class OptimizeStatusRound:
    round_name: str
    effective_metric_source: str
    avg_improvement: float
    geomean_speedup: float
    mean_latency: float


@dataclass(frozen=True)
class OptimizeStatusWorkspace:
    workspace: Path
    state: str
    avg_improvement: float | None
    geomean_speedup: float | None
    best_round: str | None
    logged_best: str | None
    warnings: tuple[str, ...]
    latest_verify_state: Path | None = None
    verified: bool = False
    verified_geomean_speedup: float | None = None

class BaselinePreflightState(str, Enum):
    READY = "ready"
    NEEDS_PREPARE = "needs-prepare"
    NEEDS_REPAIR = "needs-repair"


@dataclass(frozen=True)
class BaselinePreflightResult:
    state: BaselinePreflightState
    issues: tuple[str, ...]
