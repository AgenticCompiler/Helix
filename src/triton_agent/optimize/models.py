from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from triton_agent.optimize.skill_contract import optimize_check_module


_OPTIMIZE_CHECK_MODULE = optimize_check_module()

BaselineState = _OPTIMIZE_CHECK_MODULE.BaselineState  # type: ignore[reportUnknownVariableType]
BaselineArtifactsInspection = _OPTIMIZE_CHECK_MODULE.BaselineArtifactsInspection  # type: ignore[reportUnknownVariableType]
RoundState = _OPTIMIZE_CHECK_MODULE.RoundState  # type: ignore[reportUnknownVariableType]
RoundArtifactsInspection = _OPTIMIZE_CHECK_MODULE.RoundArtifactsInspection  # type: ignore[reportUnknownVariableType]
OptimizeCheckResult = _OPTIMIZE_CHECK_MODULE.OptimizeCheckResult  # type: ignore[reportUnknownVariableType]


@dataclass(frozen=True)
class OptimizeRunOptions:
    agent_name: str
    interact: bool
    verbose: bool
    show_output: bool
    remote: str | None
    remote_workdir: str | None
    min_rounds: int | None
    resume_mode: str
    reset_optimize: bool
    no_agent_session: bool
    supervise: Literal["on", "off"]
    output: str | None
    test_mode: str | None
    bench_mode: str | None
    prompt: str | None
    target_chip: Literal["A3", "A5"] = "A5"
    optimize_knowledge: Literal["v1", "v2"] = "v1"
    compiler_source_analysis: Literal["off", "auto"] = "off"
    enable_cann_ext_api: bool = False


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
    avg_improvement: float
    geomean_speedup: float
    total_speedup: float
    mean_latency: float


@dataclass(frozen=True)
class OptimizeStatusWorkspace:
    workspace: Path
    state: str
    baseline_mean: float | None
    best_mean: float | None
    avg_improvement: float | None
    geomean_speedup: float | None
    total_speedup: float | None
    best_round: str | None
    logged_best: str | None
    warnings: tuple[str, ...]
    latest_verify_state: Path | None = None
    verified: bool = False
    verified_geomean_speedup: float | None = None
    verified_total_speedup: float | None = None


class GateDecision(str, Enum):
    PASS_CONTINUE = "pass-continue"
    PASS_STOP = "pass-stop"
    REVISE_METADATA = "revise-metadata"
    REVISE_REQUIRED = "revise-required"
    HARD_FAIL = "hard-fail"


@dataclass(frozen=True)
class GateResult:
    decision: GateDecision
    blocking_issues: tuple[str, ...]
    auto_repairs_applied: tuple[str, ...] = ()
    next_parent_round: str | None = None
    next_hypothesis: str | None = None
    required_evidence_for_next_round: tuple[str, ...] = ()
