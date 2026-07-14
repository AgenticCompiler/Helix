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
    return ClaudeOptimizePluginAssets(
        text_files={
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


def _render_plugin_readme() -> str:
    return (
        "# Triton Optimizer Plugin\n\n"
        "This plugin packages the Claude optimize workflow and Triton convert workflow for Helix.\n\n"
        "It includes optimize and Triton convert skills with plugin-managed workflow state "
        "automation and first-session compiler source provisioning.\n\n"
        "## Usage\n\n"
        "Start Claude in the target workspace and ask it to optimize or convert the target operator. "
        "The plugin makes the bundled skills available directly.\n"
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
