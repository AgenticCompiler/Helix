"""Stage taxonomy, dependency graph, and per-round gating for the optimize orchestrator.

The stage taxonomy, hard dependency edges, and issue→stage routing live in the
machine-readable contract ``skills/triton/triton-npu-optimize/references/stages.json``
(single source of truth, adjustable without code changes). This module loads that
contract and exposes the graph operations the orchestrator needs.

Gate model (respects the optimize skill's "no forward-looking plans" rule): the
orchestrator never commits to a multi-stage schedule. Each round it computes the
set of stages whose prerequisites are already *addressed* or *skipped*, and only
those stages are runnable this round. If a desired stage is not yet runnable, the
gate redirects to its first unmet prerequisite.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Iterable

from triton_agent.paths import skills_root


class Stage(str, Enum):
    """Optimization stages, ordered by decreasing semantic scope (per cluster.md)."""

    BOUNDARY = "boundary"
    PARALLEL = "parallel"
    MEMORY_ACCESS = "memory_access"
    ALGORITHMIC = "algorithmic"
    PIPELINE = "pipeline"
    COMPILE_HINTS = "compile_hints"
    PARAMETERIZATION = "parameterization"


_CONTRACT_PATH = (
    skills_root()
    / "triton"
    / "triton-npu-optimize"
    / "references"
    / "stages.json"
)


@dataclass(frozen=True)
class StageDescriptor:
    id: Stage
    name: str
    priority_level: int
    description: str
    patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateResult:
    """Outcome of gating a desired stage against the current history.

    When ``allowed`` is True, ``stage`` is runnable this round and ``redirect_to``
    is None. When False, ``stage`` is the stage that was desired and
    ``redirect_to`` is the first unmet prerequisite the round should address
    instead.
    """

    stage: Stage
    allowed: bool
    redirect_to: Stage | None
    reason: str

    @staticmethod
    def allow(stage: Stage) -> "GateResult":
        return GateResult(stage=stage, allowed=True, redirect_to=None, reason="")

    @staticmethod
    def block(stage: Stage, redirect_to: Stage, reason: str) -> "GateResult":
        return GateResult(
            stage=stage, allowed=False, redirect_to=redirect_to, reason=reason
        )


@dataclass(frozen=True)
class StageGraph:
    """Loaded stage contract: descriptors, hard dependency edges."""

    stages: tuple[StageDescriptor, ...]
    dependencies: tuple[tuple[Stage, Stage], ...]

    @property
    def stage_ids(self) -> tuple[Stage, ...]:
        return tuple(descriptor.id for descriptor in self.stages)

    def descriptor(self, stage: Stage) -> StageDescriptor:
        for descriptor in self.stages:
            if descriptor.id == stage:
                return descriptor
        raise KeyError(f"unknown stage: {stage!r}")

    def prereqs(self, stage: Stage) -> tuple[Stage, ...]:
        """Direct prerequisites of ``stage`` (stages that must precede it)."""
        return tuple(before for before, after in self.dependencies if after == stage)

    def dependents(self, stage: Stage) -> tuple[Stage, ...]:
        """Stages that directly depend on ``stage`` (stages that must follow it)."""
        return tuple(after for before, after in self.dependencies if before == stage)

    def allowable_stages(
        self,
        addressed: Iterable[Stage],
        skipped: Iterable[Stage],
    ) -> tuple[Stage, ...]:
        """Stages whose prerequisites are all in ``addressed ∪ skipped``.

        A stage is runnable this round iff every direct prerequisite has either
        been addressed in a prior round or has been skipped (no detected issues).
        Stages with no prerequisites (only ``boundary``) are always runnable.
        """
        resolved = set(addressed) | set(skipped)
        runnable: list[Stage] = []
        for stage in self.stage_ids:
            if all(prereq in resolved for prereq in self.prereqs(stage)):
                runnable.append(stage)
        return tuple(runnable)

    def blocked_stages(
        self,
        addressed: Iterable[Stage],
        skipped: Iterable[Stage],
    ) -> tuple[tuple[Stage, Stage], ...]:
        """Stages that are not runnable, paired with their first unmet prerequisite."""
        resolved = set(addressed) | set(skipped)
        blocked: list[tuple[Stage, Stage]] = []
        for stage in self.stage_ids:
            unmet = [p for p in self.prereqs(stage) if p not in resolved]
            if unmet:
                blocked.append((stage, unmet[0]))
        return tuple(blocked)

    def gate(
        self,
        desired: Stage,
        addressed: Iterable[Stage],
        skipped: Iterable[Stage],
    ) -> GateResult:
        """Gate a desired stage against the current history.

        Returns Allow if all prerequisites are resolved; otherwise Block with the
        first unmet prerequisite as the redirect target.
        """
        resolved = set(addressed) | set(skipped)
        unmet = [p for p in self.prereqs(desired) if p not in resolved]
        if not unmet:
            return GateResult.allow(desired)
        redirect = unmet[0]
        return GateResult.block(
            desired,
            redirect_to=redirect,
            reason=(
                f"stage {desired.value} requires prerequisite {redirect.value} "
                f"to be addressed or skipped first"
            ),
        )

    def has_cycle(self) -> bool:
        """Detect cycles in the dependency graph (contract self-check)."""
        visited: set[Stage] = set()
        stack: set[Stage] = set()

        def visit(node: Stage) -> bool:
            if node in stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            stack.add(node)
            for dependent in self.dependents(node):
                if visit(dependent):
                    return True
            stack.remove(node)
            return False

        return any(visit(stage) for stage in self.stage_ids)


def _parse_stage(value: str) -> Stage:
    try:
        return Stage(value)
    except ValueError as exc:
        raise ValueError(f"unknown stage id in stages.json: {value!r}") from exc


def load_stage_graph(path=_CONTRACT_PATH) -> StageGraph:
    """Load and validate the stages.json contract into a StageGraph."""
    data = json.loads(path.read_text(encoding="utf-8"))

    descriptors: list[StageDescriptor] = []
    seen_ids: set[Stage] = set()
    for raw in data.get("stages", []):
        stage = _parse_stage(str(raw["id"]))
        if stage in seen_ids:
            raise ValueError(f"duplicate stage id in stages.json: {stage!r}")
        seen_ids.add(stage)
        descriptors.append(
            StageDescriptor(
                id=stage,
                name=str(raw.get("name", stage.value)),
                priority_level=int(raw.get("priority_level", 0)),
                description=str(raw.get("description", "")),
                patterns=tuple(str(p) for p in raw.get("patterns", [])),
            )
        )
    if not descriptors:
        raise ValueError("stages.json defines no stages")

    known = seen_ids
    deps: list[tuple[Stage, Stage]] = []
    for edge in data.get("dependencies", []):
        before = _parse_stage(str(edge["before"]))
        after = _parse_stage(str(edge["after"]))
        if before not in known or after not in known:
            raise ValueError(
                f"dependency edge references unknown stage: {edge!r}"
            )
        if before == after:
            raise ValueError(f"self-dependency in stages.json: {before!r}")
        deps.append((before, after))

    return StageGraph(
        stages=tuple(descriptors),
        dependencies=tuple(deps),
    )


@lru_cache(maxsize=None)
def default_stage_graph() -> StageGraph:
    """Cached default StageGraph loaded from the canonical contract path."""
    return load_stage_graph(_CONTRACT_PATH)
