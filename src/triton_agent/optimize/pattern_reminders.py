from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

from triton_agent.resources import skills_root
from triton_agent.skill_loader import load_skill_script_module


DEFAULT_GENERIC_OPTIMIZE_KNOWLEDGE_SKILL = "triton-npu-optimize-knowledge"


def resolve_generic_optimize_knowledge_skill_name(
    staged_skill_names: tuple[str, ...] | None,
    staged_skill_sources: dict[str, str] | None,
) -> str | None:
    if (
        staged_skill_names is not None
        and DEFAULT_GENERIC_OPTIMIZE_KNOWLEDGE_SKILL not in staged_skill_names
    ):
        return None

    skill_name = DEFAULT_GENERIC_OPTIMIZE_KNOWLEDGE_SKILL
    if staged_skill_sources is not None:
        skill_name = staged_skill_sources.get(
            DEFAULT_GENERIC_OPTIMIZE_KNOWLEDGE_SKILL,
            DEFAULT_GENERIC_OPTIMIZE_KNOWLEDGE_SKILL,
        )

    skill_path = skills_root() / skill_name
    if not skill_path.exists():
        raise FileNotFoundError(f"Selected optimize knowledge skill does not exist: {skill_path}")
    return skill_name


def build_high_priority_pattern_reminder_lines(skill_name: str) -> list[str]:
    patterns_dir = skills_root() / skill_name / "references" / "patterns"
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
