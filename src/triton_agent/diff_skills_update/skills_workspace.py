from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from triton_agent.resources import skills_root


KNOWLEDGE_SKILL_NAME = "triton-npu-optimize-knowledge"


def ensure_skills_workspace(skills_dir: Path) -> Path:
    skills_dir.mkdir(parents=True, exist_ok=True)
    knowledge_dir = skills_dir / KNOWLEDGE_SKILL_NAME
    if knowledge_dir.exists():
        if not knowledge_dir.is_dir():
            raise ValueError(f"Skills workspace entry is not a directory: {knowledge_dir}")
        return knowledge_dir

    bundled = skills_root() / KNOWLEDGE_SKILL_NAME
    if not bundled.is_dir():
        raise ValueError(f"Bundled knowledge skill does not exist: {bundled}")
    shutil.copytree(bundled, knowledge_dir, symlinks=False)
    return knowledge_dir


def regenerate_pattern_index(knowledge_dir: Path) -> None:
    script = knowledge_dir / "scripts" / "build_pattern_index.py"
    patterns_dir = knowledge_dir / "references" / "patterns"
    output = knowledge_dir / "references" / "pattern_index.md"
    if not script.exists():
        raise ValueError(f"Pattern index builder does not exist: {script}")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--patterns-dir",
            str(patterns_dir),
            "--output",
            str(output),
        ],
        cwd=knowledge_dir,
        check=True,
    )


def promote_converged_knowledge_workspace(source_knowledge_dir: Path) -> Path:
    if not source_knowledge_dir.is_dir():
        raise ValueError(f"Converged knowledge skill does not exist: {source_knowledge_dir}")
    destination = skills_root() / KNOWLEDGE_SKILL_NAME
    if source_knowledge_dir.resolve() != destination.resolve():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _remove_existing_path(destination)
        shutil.copytree(source_knowledge_dir, destination, symlinks=False)
    regenerate_pattern_index(destination)
    return destination


def _remove_existing_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    if path.exists():
        path.unlink()
