from __future__ import annotations

from pathlib import Path

from triton_agent.skills import staged_skill_dir

OPTIMIZE_KNOWLEDGE_SKILL_NAME = "triton-npu-optimize-knowledge"


def workspace_staged_skill_path(workdir: Path, agent_name: str, skill_name: str) -> Path:
    return workdir.resolve() / staged_skill_dir(agent_name) / skill_name


def build_skills_source_overrides(
    workdir: Path,
    agent_name: str,
    skills_source_dir: Path | None,
    skill_names: tuple[str, ...] | None,
) -> dict[str, Path] | None:
    """Map staged skill names to persistent repo-local skill trees for overwrite copy."""
    if skills_source_dir is None or not skill_names:
        return None

    source_root = skills_source_dir.expanduser().resolve()
    if not source_root.is_dir():
        raise ValueError(f"--skills-source-dir is not a directory: {source_root}")

    overrides: dict[str, Path] = {}
    for skill_name in skill_names:
        candidate = source_root / skill_name
        if not candidate.is_dir():
            continue
        dest = workspace_staged_skill_path(workdir, agent_name, skill_name)
        if candidate.resolve() == dest.resolve():
            continue
        overrides[skill_name] = candidate

    return overrides or None
