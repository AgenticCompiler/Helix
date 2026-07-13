#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helix.models import CommandKind
from helix.paths import application_root
from helix.skills.catalog import resolve_skill_source_dir
from helix.skills.selection import resolve_staged_skills


_PLUGIN_NAME = "triton-optimizer"
_PLUGIN_VERSION = "0.1.0"
_OPTIMIZE_AGENT_NAME = "helix-optimizer"
_CONVERT_AGENT_NAME = "helix-convert"
_PLUGIN_TEST_MODE = "differential"
_PLUGIN_BENCH_MODE = "torch-npu-profiler"


@dataclass(frozen=True)
class ClaudeOptimizePluginAssets:
    text_files: dict[str, str]
    skill_names: tuple[str, ...]
    skill_sources: dict[str, str] | None
    optimize_skill_names: tuple[str, ...]
    convert_skill_names: tuple[str, ...]


def build_claude_optimize_plugin_assets(
    *,
    language: str = "triton",
    enable_cann_ext_api: bool = False,
) -> ClaudeOptimizePluginAssets:
    optimize_skill_names, optimize_skill_sources = resolve_staged_skills(
        CommandKind.OPTIMIZE,
        language=language,
        enable_cann_ext_api=enable_cann_ext_api,
    )
    if optimize_skill_names is None:
        raise RuntimeError("Optimize plugin packaging requires an explicit optimize skill list.")
    optimize_skill_names = _select_plugin_optimize_skill_names(optimize_skill_names)
    convert_skill_names, convert_skill_sources = resolve_staged_skills(
        CommandKind.CONVERT,
        language="triton",
    )
    if convert_skill_names is None:
        raise RuntimeError("Optimize plugin packaging requires an explicit convert skill list.")

    skill_names = _deduplicate_skill_names(optimize_skill_names + convert_skill_names)
    skill_sources = _merge_skill_sources(optimize_skill_sources, convert_skill_sources)
    optimize_agent_text = _render_claude_optimize_agent(skill_names=optimize_skill_names)
    convert_agent_text = _render_claude_convert_agent(skill_names=convert_skill_names)
    return ClaudeOptimizePluginAssets(
        text_files={
            "agents/helix-optimizer.md": optimize_agent_text,
            "agents/helix-convert.md": convert_agent_text,
            "README.md": _render_plugin_readme(),
        },
        skill_names=skill_names,
        skill_sources=skill_sources,
        optimize_skill_names=optimize_skill_names,
        convert_skill_names=convert_skill_names,
    )


def build_claude_optimize_plugin(
    output_dir: Path,
    *,
    language: str = "triton",
    enable_cann_ext_api: bool = False,
) -> Path:
    root = output_dir.resolve()
    assets = build_claude_optimize_plugin_assets(
        language=language,
        enable_cann_ext_api=enable_cann_ext_api,
    )

    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    _write_plugin_manifest(root / ".claude-plugin" / "plugin.json")
    _write_text_files(root, assets.text_files)
    _copy_hook_assets(root / "hooks")
    _copy_selected_skills(
        root / "skills",
        skill_names=assets.skill_names,
        skill_sources=assets.skill_sources,
    )
    return root


def _render_claude_optimize_agent(*, skill_names: tuple[str, ...]) -> str:
    primary_skill_name = skill_names[0]
    baseline_skill_name = "ascend-npu-prepare-optimize-baseline"
    state_skill_name = "ascend-npu-optimize-state"
    lines = [
        "---",
        f"name: {_OPTIMIZE_AGENT_NAME}",
        "description: Use this agent for Helix optimize sessions with bundled optimize skills and plugin-managed workflow state.",
        "model: inherit",
        "tools:",
        "  - Read",
        "  - Grep",
        "  - Glob",
        "  - Bash",
        "  - Edit",
        "  - MultiEdit",
        "  - Write",
        "  - Skill",
        "skills:",
    ]
    lines.extend(f"  - {skill_name}" for skill_name in skill_names)
    lines.extend(
        [
            "---",
            "# Optimize Agent",
            "",
            f"Use `{primary_skill_name}` as the primary workflow skill.",
            "Treat the bundled optimize skills as the workflow source of truth, and let them pull in sibling skills when needed.",
            "The plugin manages the temporary `.helix/` runtime directory for this agent. Do not create, edit, or depend on it manually.",
            "",
            "## Fixed Optimize Modes",
            "",
            f"- Use test-mode: `{_PLUGIN_TEST_MODE}`.",
            f"- Use bench-mode: `{_PLUGIN_BENCH_MODE}`.",
            "- Apply these modes when generating or reusing harnesses, running correctness and benchmark validation, and writing `baseline/state.json`.",
            "- If a resumable baseline already exists, it must record matching modes before reuse.",
            "",
            "## Critical Workflow Rules",
            "",
            f"- Use `{baseline_skill_name}` to establish or repair `baseline/` before the first round when needed.",
            f"- Use `{state_skill_name}` `submit-baseline` to accept the initial baseline.",
            f"- Use `{state_skill_name}` `start-round` immediately before beginning a new `opt-round-N/`.",
            f"- Use `{state_skill_name}` `set-current-round-state` if the active round's strategy or evidence depth changes mid-round.",
            f"- Use `{state_skill_name}` `submit-round` after each complete round before stopping or opening the next round.",
            "- Keep exactly one optimize round active at a time.",
            "- When SessionStart provides compiler source path and commit, treat that checkout as read-only evidence and use compiler source only as the deepest escalation.",
            "",
            "## Stable Optimize Guidance",
            "",
            "- Read files cautiously. Do not read unrelated files speculatively or just in case.",
            "- Prefer the smallest source that can unblock the next decision.",
            "- Follow the user's instructions strictly.",
            "- Use the staged workspace skills as the workflow source of truth.",
            "- Invocation-specific behavior comes from the user prompt, SessionStart context, workflow state, and existing round artifacts.",
            "- Treat `baseline/` as the canonical optimize baseline.",
            "- Use `compare-perf` as the authoritative source for round performance summaries.",
            "",
            "## Analysis Ladder",
            "",
            "- Choose the analysis level for each round before editing code.",
            "- Record the round's primary analysis level separately from its supporting evidence.",
            "- Escalate analysis in this order: pattern triage, profiling diagnosis, IR attribution, compiler-source escalation.",
            "- Use pattern triage only to decide whether a strong pattern-backed hypothesis already exists.",
            "- Use the staged `triton-npu-optimize-knowledge` skill for generic pattern and symptom references.",
            "- When pattern triage is used, record candidate patterns, the selected pattern if one is chosen, and why that pattern looks plausible in `opt-round-N/attempts.md`.",
            "- When a named pattern guides the round, record the final selected pattern direction in `opt-round-N/summary.md`.",
            "- Read the staged `triton-npu-optimize-knowledge` skill's generated `references/pattern_index.md` before detailed pattern references.",
            "- Inspect the operator file directly when code structure is still unclear at pattern triage.",
            "- Use profiling diagnosis as the default deeper entrypoint when pattern triage is not enough.",
            "- Use the staged `triton-npu-optimize-knowledge` skill's symptom cards to narrow pattern candidates after structured profiler or IR evidence exists.",
            "- Use IR attribution only after profiler-backed symptoms need explanation.",
            "- Use compiler-source escalation only when profiler and IR evidence have already narrowed the issue.",
            "- When starting from a deeper level, cite the reused evidence path and explain why the shallower level is already established or insufficient.",
            "- Do not begin with blind tiling or launch-parameter search.",
            "",
            "## High-Priority Generic Pattern Reminders",
            "",
            "- `a5-force-simt-only-discrete-access`: Launch discrete-memory-access Triton kernels on A5 with `force_simt_only=True`, then retune `num_warps` and grid decomposition.",
            "- `autotune`: Use Triton-Ascend autotune as the default way to search split sizes, tile sizes, and selected compile options when the kernel structure is already reasonable and the main open question is parameter choice.",
            "- `grid-flatten-and-ub-buffering`: Flatten logical work items onto physical cores and batch small row-wise memory transfers into wider UB stores to reduce launch overhead and improve per-core work density.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_claude_convert_agent(*, skill_names: tuple[str, ...]) -> str:
    primary_skill_name = skill_names[0]
    test_skill_name = "ascend-npu-gen-test"
    run_eval_skill_name = "ascend-npu-run-eval"
    repair_skill_name = "triton-npu-repair-guide"
    lines = [
        "---",
        f"name: {_CONVERT_AGENT_NAME}",
        "description: Use this agent for Helix convert sessions with bundled Triton convert skills.",
        "model: inherit",
        "tools:",
        "  - Read",
        "  - Grep",
        "  - Glob",
        "  - Bash",
        "  - Edit",
        "  - MultiEdit",
        "  - Write",
        "  - Skill",
        "skills:",
    ]
    lines.extend(f"  - {skill_name}" for skill_name in skill_names)
    lines.extend(
        [
            "---",
            "# Convert Agent",
            "",
            f"Use `{primary_skill_name}` as the primary workflow skill.",
            "Treat the bundled Triton convert skills as the workflow source of truth, and let them pull in sibling skills when needed.",
            "",
            "## Critical Workflow Rules",
            "",
            "- Treat the original input operator file as immutable source material.",
            "- Do not modify or overwrite the original input operator file.",
            "- Write the converted operator only to the requested output path.",
            "- Preserve the trailing input-helper block from the source file in the converted output.",
            "- Keep the converted artifact PyTorch-facing while backing computation with a real Triton Ascend NPU kernel path.",
            f"- Use `{test_skill_name}` when no suitable reusable test exists.",
            f"- Use `{run_eval_skill_name}` to execute validation.",
            f"- Use `{repair_skill_name}` for Triton compile, JIT, launch, or kernel-structure repair.",
            "- Finish only after validation passes or a clear environment blocker prevents further progress.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_plugin_readme() -> str:
    return (
        "# Triton Optimizer Plugin\n\n"
        "This plugin packages the Claude optimize workflow and Triton convert workflow for Helix.\n\n"
        "It includes one optimize agent with plugin-managed workflow state automation "
        "and first-session compiler source provisioning, plus one Triton convert agent "
        "with the minimum Triton convert skill set.\n\n"
        "## Usage\n\n"
        "### Start Claude with the optimize agent\n\n"
        "Run:\n\n"
        "`claude --agent triton-optimizer:helix-optimizer`\n\n"
        "Then ask:\n\n"
        "`Please optimize @your_triton_operator.py in the current directory. Stop after "
        "reaching Xx speedup over the baseline or after X rounds.`\n"
    )


def _write_plugin_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "$schema": "https://anthropic.com/claude-code/plugin.schema.json",
                "name": _PLUGIN_NAME,
                "version": _PLUGIN_VERSION,
                "description": "Claude Triton optimizer workflow plugin for Helix",
                "author": {
                    "name": "helix",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_text_files(root: Path, text_files: dict[str, str]) -> None:
    for relative_path, content in text_files.items():
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")


def _copy_hook_assets(destination_root: Path) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    source_root = application_root() / "hooks"
    plugin_root = source_root / "claude_plugin"
    for source_path in plugin_root.iterdir():
        if source_path.is_dir():
            shutil.copytree(source_path, destination_root / source_path.name)
        else:
            shutil.copy2(source_path, destination_root / source_path.name)
    shutil.copytree(application_root() / "src" / "hook_runtime", destination_root / "hook_runtime")


def _copy_selected_skills(
    destination_root: Path,
    *,
    skill_names: tuple[str, ...],
    skill_sources: dict[str, str] | None,
) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    for staged_name in skill_names:
        source_name = skill_sources.get(staged_name, staged_name) if skill_sources else staged_name
        source_dir = resolve_skill_source_dir(source_name)
        shutil.copytree(source_dir, destination_root / staged_name, symlinks=False)


def _deduplicate_skill_names(skill_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(skill_names))


def _select_plugin_optimize_skill_names(skill_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        skill_name
        for skill_name in skill_names
        if skill_name != "torch-npu-optimize-knowledge"
    )


def _merge_skill_sources(*skill_source_maps: dict[str, str] | None) -> dict[str, str] | None:
    merged: dict[str, str] = {}
    for skill_source_map in skill_source_maps:
        if skill_source_map:
            merged.update(skill_source_map)
    return merged or None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="dist/triton-optimizer",
        help="Directory where the Claude Triton optimizer plugin should be generated.",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    built_dir = build_claude_optimize_plugin(output_dir)
    print(built_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
