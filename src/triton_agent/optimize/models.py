from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal


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
    require_analysis: bool
    no_agent_session: bool
    supervise: Literal["on", "off"]
    output: str | None
    test_mode: str | None
    bench_mode: str | None
    prompt: str | None


@dataclass(frozen=True)
class BatchOptimizeWorkspace:
    workspace: Path
    operator_file: Path


@dataclass(frozen=True)
class BatchOptimizeResult:
    workspace: Path
    succeeded: bool
    message: str


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
    perf_artifact: str
    canonical_baseline: str
    comparison_target: str
    perf_summary_source: str
    summary_path: str
    opt_note_updated: bool
    next_recommendation: str
    analysis_skipped_reason: str | None = None
    profile_dir: str | None = None
    ir_dir: str | None = None
    validated_candidate: bool | None = None


@dataclass(frozen=True)
class RoundArtifactsInspection:
    round_dir: Path
    operator_path: Path | None
    attempts_path: Path | None
    summary_path: Path | None
    perf_path: Path | None
    round_state_path: Path | None
    issues: tuple[str, ...]


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
