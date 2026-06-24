from __future__ import annotations

from pathlib import Path
from triton_agent.subagents import RenderedSubagent, SubagentDefinition


PERF_DIAGNOSIS_SUBAGENT_ID = "triton-agent-perf-diagnosis-advisor"

_COMMON_SKILL_NAMES: tuple[str, ...] = (
    "triton-npu-optimize-knowledge",
    "ascend-npu-run-eval",
    "ascend-npu-profile-operator",
    "ascend-npu-analyze-ir",
)


def perf_diagnosis_subagent_definition(
    *,
    optimize_target: str,
    enable_cann_ext_api: bool,
) -> SubagentDefinition:
    return SubagentDefinition(
        id=PERF_DIAGNOSIS_SUBAGENT_ID,
        supported_backends=("codex", "claude", "opencode"),
        render=lambda backend: _render_perf_diagnosis_subagent(
            backend=backend,
            optimize_target=optimize_target,
            enable_cann_ext_api=enable_cann_ext_api,
        ),
    )


def optimize_subagent_recommendation_lines() -> list[str]:
    return [
        "A diagnosis subagent named `triton-agent-perf-diagnosis-advisor` is available in this workspace.",
        "Use it proactively when the bottleneck hypothesis is still unclear before deeper optimize edits.",
        "That subagent is diagnosis-only: it may read existing harnesses and evidence, may collect fresh benchmark/profile/IR artifacts, and must not perform optimization work.",
    ]


def _render_perf_diagnosis_subagent(
    *,
    backend: str,
    optimize_target: str,
    enable_cann_ext_api: bool,
) -> RenderedSubagent:
    prompt = _render_common_prompt(
        optimize_target=optimize_target,
        enable_cann_ext_api=enable_cann_ext_api,
    )
    if backend == "codex":
        return RenderedSubagent(
            relative_path=Path(".codex") / "agents" / f"{PERF_DIAGNOSIS_SUBAGENT_ID}.toml",
            content=_render_codex_agent(prompt),
        )
    if backend == "claude":
        return RenderedSubagent(
            relative_path=Path(".claude") / "agents" / f"{PERF_DIAGNOSIS_SUBAGENT_ID}.md",
            content=_render_claude_agent(
                prompt=prompt,
                optimize_target=optimize_target,
                enable_cann_ext_api=enable_cann_ext_api,
            ),
        )
    if backend == "opencode":
        return RenderedSubagent(
            relative_path=Path(".opencode") / "agents" / f"{PERF_DIAGNOSIS_SUBAGENT_ID}.md",
            content=_render_opencode_agent(
                prompt=prompt,
                optimize_target=optimize_target,
                enable_cann_ext_api=enable_cann_ext_api,
            ),
        )
    raise ValueError(f"Unsupported subagent backend: {backend}")


def _render_common_prompt(
    *,
    optimize_target: str,
    enable_cann_ext_api: bool,
) -> str:
    lines = [
        f"You are `{PERF_DIAGNOSIS_SUBAGENT_ID}`, a diagnosis-only helper for Triton Ascend NPU optimize sessions.",
        (
            "Focus on end-to-end operator latency diagnosis."
            if optimize_target == "operator"
            else "Focus on Triton kernel-path performance diagnosis."
        ),
        "Use the staged skill tree in this workspace as your source of truth.",
        "Start with skill `triton-npu-optimize-knowledge` and read its `SKILL.md`.",
        "Read its `pattern_index.md` before detailed pattern cards.",
        "Use its `symptom_index.md` when profile or IR evidence needs symptom routing.",
        (
            "Use skill `ascend-npu-run-eval`, `ascend-npu-profile-operator`, and "
            "`ascend-npu-analyze-ir` for documented evidence-collection entrypoints."
        ),
        "You may inspect existing operator files, generated test and benchmark harnesses, previous perf artifacts, profiler outputs, and archived IR.",
        "You may collect fresh benchmark, profiler, or IR evidence when diagnosis needs new facts.",
        "If you use Bash, use it only for read-only inspection or documented evidence-collection entrypoints.",
        "Do not read staged skill implementation files under the skills' `scripts/` directories just to understand workflow behavior.",
        "You must not perform optimization work.",
        "Do not edit the operator implementation, optimized candidates, generated harnesses, or optimize round artifacts.",
        "Do not write or apply patches.",
        "Do not create `subagent-advice.md` or any other coordination file; return your diagnosis directly in your reply.",
        "Summarize the likely bottleneck, the evidence you used or collected, the candidate patterns that fit, and concrete next optimization directions for the parent agent to evaluate.",
    ]
    if optimize_target == "operator":
        lines.insert(
            6,
            (
                "Also use skill `torch-npu-optimize-knowledge` and its `pattern_index.md` "
                "for operator-level Torch NPU pattern guidance."
            ),
        )
    if enable_cann_ext_api:
        lines.insert(
            7,
            (
                "Also use skill `triton-npu-cann-ext-api-patterns` and its `index.md` "
                "when the kernel structure may match CANN extension API patterns."
            ),
        )
    return "\n".join(lines)


def _render_codex_agent(prompt: str) -> str:
    return (
        f'name = "{PERF_DIAGNOSIS_SUBAGENT_ID}"\n'
        'description = "Diagnosis-only performance advisor for Triton optimize sessions. '
        'Use proactively when the bottleneck hypothesis is unclear before deeper optimize edits."\n'
        'developer_instructions = """\n'
        f"{prompt}\n"
        '"""\n'
    )


def _render_claude_agent(
    *,
    prompt: str,
    optimize_target: str,
    enable_cann_ext_api: bool,
) -> str:
    lines = [
        "---",
        f"name: {PERF_DIAGNOSIS_SUBAGENT_ID}",
        "description: Diagnosis-only performance advisor for Triton optimize sessions. Use proactively when the bottleneck hypothesis is unclear before deeper optimize edits.",
        "tools:",
        "  - Read",
        "  - Grep",
        "  - Glob",
        "  - Bash",
        "  - Skill",
        "skills:",
    ]
    lines.extend(
        f"  - {name}"
        for name in _claude_preloaded_skill_names(
            optimize_target=optimize_target,
            enable_cann_ext_api=enable_cann_ext_api,
        )
    )
    lines.extend(["---", prompt])
    return "\n".join(lines) + "\n"


def _render_opencode_agent(
    *,
    prompt: str,
    optimize_target: str,
    enable_cann_ext_api: bool,
) -> str:
    permission_lines = [
        "  edit: deny",
        "  task:",
        '    "*": deny',
        "  skill:",
        '    "*": deny',
    ]
    for skill_name in _opencode_allowed_skill_names(
        optimize_target=optimize_target,
        enable_cann_ext_api=enable_cann_ext_api,
    ):
        permission_lines.append(f'    "{skill_name}": allow')
    permission_lines.extend(
        [
            "  bash:",
            '    "*": deny',
            '    "python3 .opencode/skills/ascend-npu-run-eval/scripts/run-command.py run-bench*": allow',
            '    "python3 .opencode/skills/ascend-npu-run-eval/scripts/run-command.py profile-bench*": allow',
            '    "python3 .opencode/skills/ascend-npu-analyze-ir/scripts/capture_ir.py*": allow',
            '    "python3 .opencode/skills/ascend-npu-analyze-ir/scripts/inspect_ir.py*": allow',
            "  webfetch: deny",
            "  websearch: deny",
        ]
    )
    permission_block = "\n".join(permission_lines)
    lines = [
        "---",
        "description: Diagnosis-only performance advisor for Triton optimize sessions. Use proactively when the bottleneck hypothesis is unclear before deeper optimize edits.",
        "mode: subagent",
        "permission:",
        permission_block,
        "---",
        prompt,
    ]
    return "\n".join(lines) + "\n"


def _claude_preloaded_skill_names(
    *,
    optimize_target: str,
    enable_cann_ext_api: bool,
) -> tuple[str, ...]:
    names: list[str] = list(_COMMON_SKILL_NAMES)
    if optimize_target == "operator":
        names.append("torch-npu-optimize-knowledge")
    if enable_cann_ext_api:
        names.append("triton-npu-cann-ext-api-patterns")
    return tuple(names)


def _opencode_allowed_skill_names(
    *,
    optimize_target: str,
    enable_cann_ext_api: bool,
) -> tuple[str, ...]:
    return _claude_preloaded_skill_names(
        optimize_target=optimize_target,
        enable_cann_ext_api=enable_cann_ext_api,
    )
