from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class SkillLinkSet:
    created_paths: List[Path]


class SkillLinkManager:
    def __init__(self, skills_root: Path) -> None:
        self.skills_root = skills_root.resolve()

    def _iter_skill_dirs(self) -> Iterable[Path]:
        for entry in sorted(self.skills_root.iterdir()):
            if entry.is_dir():
                yield entry

    def prepare_codex_skills(self, workdir: Path) -> SkillLinkSet:
        codex_dir = workdir / ".codex"
        target = codex_dir / "skills"
        created: List[Path] = []
        codex_dir.mkdir(parents=True, exist_ok=True)

        if not target.exists():
            shutil.copytree(self.skills_root, target, symlinks=False)
            created.append(target)
            return SkillLinkSet(created)

        if target.is_symlink():
            raise RuntimeError(f"Existing Codex skills path must not be a symlink: {target}")

        if not target.is_dir():
            raise RuntimeError(f"Existing Codex skills path is not a directory: {target}")

        for skill_dir in self._iter_skill_dirs():
            staged_path = target / skill_dir.name
            if staged_path.exists():
                if staged_path.is_symlink():
                    raise RuntimeError(f"Skill path already exists as a symlink: {staged_path}")
                continue
            shutil.copytree(skill_dir, staged_path, symlinks=False)
            created.append(staged_path)

        return SkillLinkSet(created)

    def prepare_opencode_skills(self, workdir: Path) -> SkillLinkSet:
        opencode_dir = workdir / ".opencode" / "skills"
        created: List[Path] = []
        opencode_dir.mkdir(parents=True, exist_ok=True)

        # OpenCode expects one skill directory per entry under `.opencode/skills`.
        for skill_dir in self._iter_skill_dirs():
            staged_path = opencode_dir / skill_dir.name
            if staged_path.exists():
                if staged_path.is_symlink():
                    raise RuntimeError(f"Skill path already exists as a symlink: {staged_path}")
                continue
            shutil.copytree(skill_dir, staged_path, symlinks=False)
            created.append(staged_path)

        return SkillLinkSet(created)

    def prepare_skills(self, backend: str, workdir: Path) -> SkillLinkSet:
        if backend == "codex":
            return self.prepare_codex_skills(workdir)
        if backend == "opencode":
            return self.prepare_opencode_skills(workdir)
        raise RuntimeError(f"Unsupported skill backend: {backend}")

    def cleanup(self, link_set: SkillLinkSet) -> list[str]:
        warnings = []
        for path in reversed(link_set.created_paths):
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
            except OSError as exc:
                warnings.append(f"Failed to remove skill copy {path}: {exc}")
        return warnings

    def describe_prepare(self, link_set: SkillLinkSet) -> list[str]:
        if not link_set.created_paths:
            return ["No new skill copies were created."]
        return [f"created skill copy {path}" for path in link_set.created_paths]

    def describe_cleanup(self, link_set: SkillLinkSet) -> list[str]:
        if not link_set.created_paths:
            return ["No skill copies needed cleanup."]
        return [f"removed skill copy {path}" for path in reversed(link_set.created_paths)]
