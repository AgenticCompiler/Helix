from __future__ import annotations

import shutil
from pathlib import Path

from triton_agent.resources import skills_root
from triton_agent.skills_source_dir import OPTIMIZE_KNOWLEDGE_SKILL_NAME

DEFAULT_SKILLS_DIR_NAME = "pattern-validation-skills"


def resolve_repo_skills_workdir(repo_root: Path, skills_dir: str) -> Path:
    path = Path(skills_dir).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def resolve_install_knowledge_source(install_root: Path, optimize_knowledge: str) -> Path:
    if optimize_knowledge == "v2":
        return install_root / "triton-npu-optimize-knowledge-v2"
    if optimize_knowledge == "v3":
        return install_root / "triton-npu-optimize-knowledge-v3"
    return install_root / OPTIMIZE_KNOWLEDGE_SKILL_NAME


def seed_pattern_validation_skills_dir(
    repo_root: Path,
    skills_dir: str,
    *,
    optimize_knowledge: str = "v1",
    install_root: Path | None = None,
) -> Path:
    """Create persistent loop skills workdir; seed optimize-knowledge once from install bundle."""
    workdir = resolve_repo_skills_workdir(repo_root, skills_dir)
    workdir.mkdir(parents=True, exist_ok=True)

    install = (install_root or skills_root()).resolve()
    destination = workdir / OPTIMIZE_KNOWLEDGE_SKILL_NAME
    patterns_dir = destination / "references" / "patterns"
    if patterns_dir.is_dir():
        return workdir

    source = resolve_install_knowledge_source(install, optimize_knowledge)
    if not source.is_dir():
        raise ValueError(
            f"Cannot seed {destination.as_posix()}: install skill not found at {source.as_posix()}",
        )
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, symlinks=False)
    return workdir
