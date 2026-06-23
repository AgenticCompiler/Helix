from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from triton_agent.models import AgentResult


Status = Literal["aligned", "not_aligned", "failed", "skipped"]
DiffSkillsUpdateSource = Literal["code-diff", "optimize-process", "git-repo"]


def _empty_str_list() -> list[str]:
    """default_factory compatible with Python 3.9 where ``list[str]`` is not callable."""
    return []


@dataclass(frozen=True)
class DiffSkillsUpdateConfig:
    input_root: Path
    skills_dir: Path
    update_skills_dir: Path
    source: DiffSkillsUpdateSource
    agent_name: str
    language: str
    max_iterations: int
    concurrency: int
    stream_output: bool
    verbose: bool
    force: bool
    skip_existing: bool
    promote_converged_skills: bool
    base_revision: str = ""  # empty → auto-detect from origin/HEAD


@dataclass(frozen=True)
class OperatorPair:
    operator_dir: Path
    baseline_path: Path
    expected_path: Path
    learned_lessons_path: Path | None = None
    opt_note_path: Path | None = None
    context_paths: tuple[Path, ...] = ()
    source_kind: Literal["operator-pair", "optimize-process"] = "operator-pair"

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
    matched_patterns: list[str] = field(default_factory=_empty_str_list)
    updated_patterns: list[str] = field(default_factory=_empty_str_list)
    summary: str = ""


@dataclass
class IterationReport:
    iteration: int
    status: Status
    candidate_path: Path
    simulate_return_code: int
    analysis_return_code: int
    analysis_summary: str
    updated_patterns: list[str] = field(default_factory=_empty_str_list)


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
