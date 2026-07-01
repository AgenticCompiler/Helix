#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.models import CommandKind
from triton_agent.resources import application_root
from triton_agent.skill_catalog import resolve_skill_source_dir
from triton_agent.skill_staging import resolve_staged_skills


_PLUGIN_NAME = "triton-agent-optimize"
_PLUGIN_VERSION = "0.1.0"


@dataclass(frozen=True)
class ClaudeOptimizePluginAssets:
    text_files: dict[str, str]
    skill_names: tuple[str, ...]
    skill_sources: dict[str, str] | None


def build_claude_optimize_plugin_assets(
    *,
    language: str = "triton",
    optimize_target: str = "kernel",
    enable_cann_ext_api: bool = False,
    enable_subagent: bool = False,
) -> ClaudeOptimizePluginAssets:
    skill_names, skill_sources = resolve_staged_skills(
        CommandKind.OPTIMIZE,
        language=language,
        optimize_target=optimize_target,
        enable_cann_ext_api=enable_cann_ext_api,
    )
    if skill_names is None:
        raise RuntimeError("Optimize plugin packaging requires an explicit optimize skill list.")

    agent_text = _render_claude_optimize_agent(skill_names=skill_names)
    return ClaudeOptimizePluginAssets(
        text_files={
            "agents/triton-agent-optimize.md": agent_text,
            "README.md": _render_plugin_readme(),
        },
        skill_names=skill_names,
        skill_sources=skill_sources,
    )


def build_claude_optimize_plugin(
    output_dir: Path,
    *,
    language: str = "triton",
    optimize_target: str = "kernel",
    enable_cann_ext_api: bool = False,
    enable_subagent: bool = False,
) -> Path:
    root = output_dir.resolve()
    assets = build_claude_optimize_plugin_assets(
        language=language,
        optimize_target=optimize_target,
        enable_cann_ext_api=enable_cann_ext_api,
        enable_subagent=enable_subagent,
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
        f"name: {_PLUGIN_NAME}",
        "description: Use this agent for Triton Agent optimize sessions with bundled optimize skills and plugin-managed workflow state.",
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
            "The plugin manages the temporary `.triton-agent/` runtime directory for this agent. Do not create, edit, or depend on it manually.",
            "",
            "## Critical Workflow Rules",
            "",
            f"- Use `{baseline_skill_name}` to establish or repair `baseline/` before the first round when needed.",
            f"- Use `{state_skill_name}` `submit-baseline` to accept the initial baseline.",
            f"- Use `{state_skill_name}` `start-round` immediately before beginning a new `opt-round-N/`.",
            f"- Use `{state_skill_name}` `set-current-round-state` if the active round's strategy or evidence depth changes mid-round.",
            f"- Use `{state_skill_name}` `submit-round` after each complete round before stopping or opening the next round.",
            "- Keep exactly one optimize round active at a time.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_plugin_readme() -> str:
    return (
        "# Triton Agent Optimize Plugin\n\n"
        "This plugin packages the Claude optimize workflow for Triton Agent.\n\n"
        "It only supports the optimize workflow and includes one optimize agent, "
        "plugin-managed workflow state automation, and the minimum optimize skill set.\n"
    )


def _write_plugin_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "$schema": "https://anthropic.com/claude-code/plugin.schema.json",
                "name": _PLUGIN_NAME,
                "version": _PLUGIN_VERSION,
                "description": "Claude optimize workflow plugin for Triton Agent",
                "author": {
                    "name": "triton-agent",
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="dist/triton-agent-optimize",
        help="Directory where the Claude optimize plugin should be generated.",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    built_dir = build_claude_optimize_plugin(output_dir)
    print(built_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
