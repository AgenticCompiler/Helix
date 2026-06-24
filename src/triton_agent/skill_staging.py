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
            "+ascend-npu-gen-eval-suite",
            "+ascend-npu-gen-test",
            "+ascend-npu-gen-bench",
            "+ascend-npu-run-eval",
            "+triton-npu-repair-guide",
        ),
    ),
    CommandKind.GEN_TEST: StageRule(
        directives=(
            "+ascend-npu-gen-test",
            "+ascend-npu-run-eval",
            "+triton-npu-repair-guide",
        )
    ),
    CommandKind.GEN_BENCH: StageRule(
        directives=(
            "+ascend-npu-gen-bench",
            "+ascend-npu-run-eval",
            "+triton-npu-repair-guide",
        ),
    ),
    CommandKind.CONVERT: StageRule(
        directives=(
            "+triton-npu-convert-pytorch-operator",
            "+ascend-npu-gen-test",
            "+ascend-npu-run-eval",
            "+triton-npu-repair-guide",
        ),
    ),
    CommandKind.LOG_CHECK: StageRule(
        directives=(
            "+triton-npu-optimize-knowledge",
            "+ascend-npu-optimize-submit-baseline",
            "+ascend-npu-optimize-submit-round",
        ),
    ),
    CommandKind.LOG_CHECK_BATCH: StageRule(
        directives=(
            "+triton-npu-optimize-knowledge",
            "+ascend-npu-optimize-submit-baseline",
            "+ascend-npu-optimize-submit-round",
        ),
    ),
    CommandKind.REPORT: StageRule(
        directives=(
            "+ascend-npu-report",
        ),
    ),
    CommandKind.OPTIMIZE: StageRule(
        directives=(
            "+triton-npu-optimize",
            "+triton-npu-optimize-knowledge",
            "+ascend-npu-prepare-optimize-baseline",
            "+ascend-npu-gen-test",
            "+ascend-npu-gen-bench",
            "+ascend-npu-run-eval",
            "+ascend-npu-optimize-submit-baseline",
            "+ascend-npu-optimize-submit-round",
            "+ascend-npu-optimize-start-round",
            "+ascend-npu-profile-operator",
            "+ascend-npu-analyze-round-performance",
            "+ascend-npu-analyze-ir",
            "+triton-npu-analyze-compiler-source",
            "+triton-npu-repair-guide",
        ),
    ),
    CommandKind.DIFF_SKILLS_UPDATE: StageRule(
        directives=(
            "+triton-npu-optimize-knowledge",
        ),
    ),
    CommandKind.TRACE_ANALYZE: StageRule(
        directives=(
            "+triton-npu-optimize-knowledge",
        ),
    ),
}


def resolve_staged_skills(
    command_kind: CommandKind,
    *,
    optimize_knowledge: str | None = None,
    optimize_target: str = "kernel",
    enable_cann_ext_api: bool = False,
    enable_mcp: bool = False,
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
        enable_mcp=enable_mcp,
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
    enable_mcp: bool = False,
) -> dict[str, str] | None:
    sources: dict[str, str] = {}
    if enable_mcp and staged_skill_names is not None and "ascend-npu-run-eval" in staged_skill_names:
        sources["ascend-npu-run-eval"] = "ascend-npu-run-eval-mcp"
    if command_kind == CommandKind.OPTIMIZE and staged_skill_names is not None:
        if "triton-npu-optimize-knowledge" not in staged_skill_names:
            return sources or None
        if optimize_knowledge == "v2":
            sources["triton-npu-optimize-knowledge"] = "triton-npu-optimize-knowledge-v2"
        if optimize_knowledge == "v3":
            sources["triton-npu-optimize-knowledge"] = "triton-npu-optimize-knowledge-v3"
    return sources or None
