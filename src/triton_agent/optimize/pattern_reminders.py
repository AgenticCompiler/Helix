from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

from triton_agent.skills.catalog import resolve_skill_source_dir
from triton_agent.skills.loader import load_skill_script_module


def _optimize_knowledge_skill_name(language: str) -> str:
    return f"{language}-npu-optimize-knowledge"


def resolve_generic_optimize_knowledge_skill_name(
    staged_skill_names: tuple[str, ...] | None,
    staged_skill_sources: dict[str, str] | None,
    *,
    language: str = "triton",
) -> str | None:
    optimize_knowledge_skill = _optimize_knowledge_skill_name(language)
    if (
        staged_skill_names is not None
        and optimize_knowledge_skill not in staged_skill_names
    ):
        return None

    skill_name = optimize_knowledge_skill
    if staged_skill_sources is not None:
        skill_name = staged_skill_sources.get(optimize_knowledge_skill, optimize_knowledge_skill)

    skill_path = resolve_skill_source_dir(skill_name)
    if not skill_path.exists():
        raise FileNotFoundError(f"Selected optimize knowledge skill does not exist: {skill_path}")
    return skill_name


def build_high_priority_pattern_reminder_lines(skill_name: str) -> list[str]:
    patterns_dir = resolve_skill_source_dir(skill_name) / "references" / "patterns"
    if not patterns_dir.exists():
        raise FileNotFoundError(f"Pattern directory does not exist: {patterns_dir}")

    module = load_skill_script_module(skill_name, "pattern_catalog")
    builder = getattr(module, "build_high_priority_reminder_lines", None)
    if not callable(builder):
        raise AttributeError(
            f"Skill {skill_name} pattern catalog does not expose build_high_priority_reminder_lines"
        )
    typed_builder = cast(Callable[[Path], list[str]], builder)
    reminder_lines = typed_builder(patterns_dir)
    return [str(line) for line in reminder_lines]
