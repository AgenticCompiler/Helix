from __future__ import annotations

from dataclasses import dataclass

from triton_agent.models import CommandKind


@dataclass(frozen=True)
class StageRule:
    directives: tuple[str, ...]
    skill_sources: dict[str, str] | None = None


STAGE_RULES: dict[CommandKind, StageRule] = {
    CommandKind.GEN_EVAL: StageRule(
        directives=(
            "+triton-npu-gen-eval-suite",
            "+triton-npu-gen-test",
            "+triton-npu-gen-bench",
            "+triton-npu-run-eval",
            "+triton-npu-repair-guide",
        ),
    ),
    CommandKind.GEN_TEST: StageRule(
        directives=(
            "+triton-npu-gen-test",
            "+triton-npu-run-eval",
            "+triton-npu-repair-guide",
        )
    ),
    CommandKind.GEN_BENCH: StageRule(
        directives=(
            "+triton-npu-gen-bench",
            "+triton-npu-run-eval",
            "+triton-npu-repair-guide",
        ),
    ),
    CommandKind.CONVERT: StageRule(
        directives=(
            "+triton-npu-convert-pytorch-operator",
            "+triton-npu-gen-test",
            "+triton-npu-run-eval",
            "+triton-npu-repair-guide",
        ),
    ),
    CommandKind.LOG_CHECK: StageRule(
        directives=(
            "+triton-npu-optimize-knowledge",
            "+triton-npu-optimize-check",
        ),
    ),
    CommandKind.LOG_CHECK_BATCH: StageRule(
        directives=(
            "+triton-npu-optimize-knowledge",
            "+triton-npu-optimize-check",
        ),
    ),
    CommandKind.PATTERN_VALIDATION_LOOP: StageRule(
        directives=(
            "+triton-npu-pattern-validation-loop",
            "+triton-npu-optimize-knowledge",
            "+triton-npu-optimize",
            "+triton-npu-optimize-check",
        ),
    ),
    CommandKind.REPORT: StageRule(
        directives=(
            "+triton-npu-report",
        ),
    ),
    CommandKind.OPTIMIZE: StageRule(
        directives=(
            "+triton-npu-optimize",
            "+triton-npu-optimize-knowledge",
            "+triton-npu-prepare-optimize-baseline",
            "+triton-npu-gen-test",
            "+triton-npu-gen-bench",
            "+triton-npu-run-eval",
            "+triton-npu-optimize-check",
            "+triton-npu-profile-operator",
            "+triton-npu-analyze-round-performance",
            "+triton-npu-analyze-ir",
            "+triton-npu-analyze-compiler-source",
            "+triton-npu-repair-guide",
        ),
    ),
}


def resolve_staged_skills(
    command_kind: CommandKind,
    *,
    optimize_knowledge: str | None = None,
    optimize_target: str = "kernel",
    enable_cann_ext_api: bool = False,
) -> tuple[tuple[str, ...] | None, dict[str, str] | None]:
    rule = STAGE_RULES.get(command_kind)
    if rule is None:
        return None, None

    staged_skill_names = _apply_stage_directives(rule.directives)
    if (
        command_kind == CommandKind.OPTIMIZE
        and staged_skill_names is not None
        and optimize_target == "operator"
    ):
        staged_skill_names = staged_skill_names + ("torch-npu-optimize-knowledge",)
    if command_kind == CommandKind.OPTIMIZE and staged_skill_names is not None and enable_cann_ext_api:
        staged_skill_names = staged_skill_names + ("triton-npu-cann-ext-api-patterns",)

    staged_skill_sources = _resolve_skill_sources(
        command_kind,
        staged_skill_names,
        optimize_knowledge=optimize_knowledge,
    )
    return staged_skill_names, staged_skill_sources


def _apply_stage_directives(directives: tuple[str, ...]) -> tuple[str, ...] | None:
    selected: list[str] = []
    full_copy = False
    for directive in directives:
        if directive == "*":
            full_copy = True
            selected.clear()
            continue
        if len(directive) < 2:
            raise ValueError(f"Invalid stage directive: {directive!r}")
        operator = directive[0]
        skill_name = directive[1:]
        if not skill_name:
            raise ValueError(f"Invalid stage directive: {directive!r}")
        if operator == "+":
            if skill_name not in selected:
                selected.append(skill_name)
            continue
        if operator == "-":
            selected = [name for name in selected if name != skill_name]
            continue
        raise ValueError(f"Invalid stage directive: {directive!r}")

    if full_copy:
        return None
    return tuple(selected)


def _resolve_skill_sources(
    command_kind: CommandKind,
    staged_skill_names: tuple[str, ...] | None,
    *,
    optimize_knowledge: str | None = None,
) -> dict[str, str] | None:
    if staged_skill_names is None or "triton-npu-optimize-knowledge" not in staged_skill_names:
        return None
    if command_kind not in {CommandKind.OPTIMIZE, CommandKind.PATTERN_VALIDATION_LOOP}:
        return None
    if optimize_knowledge == "v2":
        return {"triton-npu-optimize-knowledge": "triton-npu-optimize-knowledge-v2"}
    if optimize_knowledge == "v3":
        return {"triton-npu-optimize-knowledge": "triton-npu-optimize-knowledge-v3"}
    return None
