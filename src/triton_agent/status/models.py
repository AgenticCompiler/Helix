from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
    rounds: tuple[OptimizeStatusRound, ...] = ()
