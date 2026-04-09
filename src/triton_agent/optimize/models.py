from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
    no_agent_session: bool
    output: str | None
    test_mode: str | None
    bench_mode: str | None


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
    score: float
    mean_latency: float


@dataclass(frozen=True)
class OptimizeStatusWorkspace:
    workspace: Path
    state: str
    baseline_mean: float | None
    best_mean: float | None
    avg_improvement: float | None
    best_round: str | None
    logged_best: str | None
    warnings: tuple[str, ...]
