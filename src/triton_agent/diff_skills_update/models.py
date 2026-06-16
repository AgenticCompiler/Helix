from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from triton_agent.models import AgentResult


Status = Literal["aligned", "not_aligned", "failed", "skipped"]


@dataclass(frozen=True)
class DiffSkillsUpdateConfig:
    input_root: Path
    skills_dir: Path
    agent_name: str
    max_iterations: int
    concurrency: int
    show_output: bool
    verbose: bool
    force: bool
    skip_existing: bool
    promote_converged_skills: bool


@dataclass(frozen=True)
class OperatorPair:
    operator_dir: Path
    baseline_path: Path
    expected_path: Path

    @property
    def stem(self) -> str:
        return self.baseline_path.stem


@dataclass(frozen=True)
class SkipRecord:
    operator_dir: Path
    reason: str
    opt_path: Path | None = None


@dataclass(frozen=True)
class DiscoveryResult:
    pairs: tuple[OperatorPair, ...]
    skips: tuple[SkipRecord, ...]


@dataclass
class DiffAgentOutput:
    matched_patterns: list[str] = field(default_factory=list)
    updated_patterns: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class IterationReport:
    iteration: int
    status: Status
    candidate_path: Path
    simulate_return_code: int
    analysis_return_code: int
    analysis_summary: str
    updated_patterns: list[str] = field(default_factory=list)


@dataclass
class PairRunResult:
    pair: OperatorPair
    status: Status
    matched_patterns: list[str]
    updated_patterns: list[str]
    iterations: list[IterationReport]
    report_path: Path
    message: str = ""

    @property
    def succeeded(self) -> bool:
        return self.status == "aligned"


@dataclass(frozen=True)
class AgentCallResult:
    result: AgentResult
    output_json: dict[str, object]
