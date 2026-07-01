from __future__ import annotations

from dataclasses import dataclass

from triton_agent.models import CommandKind


@dataclass(frozen=True)
class StageRule:
    directives: tuple[str, ...]
    skill_sources: dict[str, str] | None = None


# Skill names use {language} as a placeholder that is resolved at runtime.
#   +{language}-npu-xxx  → language-based (triton-npu-xxx / tilelang-npu-xxx)
#   +npu-xxx             → common (shared across languages)
STAGE_RULES: dict[CommandKind, StageRule] = {
    CommandKind.GEN_EVAL: StageRule(
        directives=(
            "+ascend-npu-gen-eval-suite",
            "+ascend-npu-gen-test",
            "+ascend-npu-gen-bench",
            "+ascend-npu-run-eval",
            "+{language}-npu-repair-guide",
        ),
    ),
    CommandKind.GEN_TEST: StageRule(
        directives=(
            "+ascend-npu-gen-test",
            "+ascend-npu-run-eval",
            "+{language}-npu-repair-guide",
        )
    ),
    CommandKind.GEN_BENCH: StageRule(
        directives=(
            "+ascend-npu-gen-bench",
            "+ascend-npu-run-eval",
            "+{language}-npu-repair-guide",
        ),
    ),
    CommandKind.CONVERT: StageRule(
        directives=(
            "+{language}-npu-convert-pytorch-operator",
            "+ascend-npu-gen-test",
            "+ascend-npu-run-eval",
            "+{language}-npu-repair-guide",
        ),
    ),
    CommandKind.LOG_CHECK: StageRule(
        directives=(
            "+{language}-npu-optimize-knowledge",
            "+ascend-npu-optimize-state",
        ),
    ),
    CommandKind.LOG_CHECK_BATCH: StageRule(
        directives=(
            "+{language}-npu-optimize-knowledge",
            "+ascend-npu-optimize-state",
        ),
    ),
    CommandKind.REPORT: StageRule(
        directives=("+ascend-npu-report",),
    ),
    CommandKind.OPTIMIZE: StageRule(
        directives=(
            "+{language}-npu-optimize",
            "+{language}-npu-optimize-knowledge",
            "+ascend-npu-prepare-optimize-baseline",
            "+ascend-npu-gen-test",
            "+ascend-npu-gen-bench",
            "+ascend-npu-run-eval",
            "+ascend-npu-optimize-state",
            "+ascend-npu-profile-operator",
            "+ascend-npu-analyze-round-performance",
            "+{language}-npu-analyze-ir",
            "+{language}-npu-analyze-compiler-source",
            "+{language}-npu-repair-guide",
        ),
    ),
    CommandKind.DIFF_SKILLS_UPDATE: StageRule(
        directives=("+{language}-npu-optimize-knowledge",),
    ),
}


def resolve_staged_skills(
    command_kind: CommandKind,
    *,
    language: str = "triton",
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
        staged_skill_names = staged_skill_names + (f"{language}-npu-cann-ext-api-patterns",)

    if staged_skill_names is not None:
        staged_skill_names = tuple(
            name.replace("{language}", language) for name in staged_skill_names
        )

    # Conditional: only tilelang has an api-reference skill; staging a
    # {language}-templated directive would stage non-existent triton-npu-api-reference.
    if (
        staged_skill_names is not None
        and command_kind in (CommandKind.CONVERT, CommandKind.OPTIMIZE)
        and language == "tilelang"
    ):
        staged_skill_names = staged_skill_names + (f"{language}-npu-api-reference",)

    staged_skill_sources = _resolve_skill_sources(
        command_kind,
        staged_skill_names,
        language=language,
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
    language: str = "triton",
    optimize_knowledge: str | None = None,
    enable_mcp: bool = False,
) -> dict[str, str] | None:
    sources: dict[str, str] = {}

    run_eval_name = "ascend-npu-run-eval"
    if enable_mcp and staged_skill_names is not None and run_eval_name in staged_skill_names:
        sources[run_eval_name] = f"{run_eval_name}-mcp"

    knowledge_name = f"{language}-npu-optimize-knowledge"
    if command_kind == CommandKind.OPTIMIZE and staged_skill_names is not None and knowledge_name in staged_skill_names:
        if optimize_knowledge == "v2":
            sources[knowledge_name] = knowledge_name + "-v2"
        if optimize_knowledge == "v3":
            sources[knowledge_name] = knowledge_name + "-v3"
    return sources or None
