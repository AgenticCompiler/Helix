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

    def _iter_selected_skill_dirs(self, skill_names: tuple[str, ...] | None) -> Iterable[Path]:
        if skill_names is None:
            yield from self._iter_skill_dirs()
            return

        seen: set[str] = set()
        for skill_name in skill_names:
            if skill_name in seen:
                continue
            seen.add(skill_name)
            skill_dir = self.skills_root / skill_name
            if not skill_dir.exists() or not skill_dir.is_dir():
                raise RuntimeError(f"Requested skill does not exist: {skill_dir}")
            yield skill_dir

    def prepare_codex_skills(
        self,
        workdir: Path,
        skill_names: tuple[str, ...] | None = None,
    ) -> SkillLinkSet:
        codex_dir = workdir / ".codex"
        target = codex_dir / "skills"
        created: List[Path] = []
        codex_dir.mkdir(parents=True, exist_ok=True)

        if not target.exists() and skill_names is None:
            shutil.copytree(self.skills_root, target, symlinks=False)
            created.append(target)
            return SkillLinkSet(created)

        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)
            for skill_dir in self._iter_selected_skill_dirs(skill_names):
                shutil.copytree(skill_dir, target / skill_dir.name, symlinks=False)
            created.append(target)
            return SkillLinkSet(created)

        if target.is_symlink():
            raise RuntimeError(f"Existing Codex skills path must not be a symlink: {target}")

        if not target.is_dir():
            raise RuntimeError(f"Existing Codex skills path is not a directory: {target}")

        for skill_dir in self._iter_selected_skill_dirs(skill_names):
            staged_path = target / skill_dir.name
            if staged_path.exists():
                if staged_path.is_symlink():
                    raise RuntimeError(f"Skill path already exists as a symlink: {staged_path}")
                continue
            shutil.copytree(skill_dir, staged_path, symlinks=False)
            created.append(staged_path)

        return SkillLinkSet(created)

    def prepare_opencode_skills(
        self,
        workdir: Path,
        skill_names: tuple[str, ...] | None = None,
    ) -> SkillLinkSet:
        opencode_dir = workdir / ".opencode" / "skills"
        created: List[Path] = []
        opencode_dir.mkdir(parents=True, exist_ok=True)

        # OpenCode expects one skill directory per entry under `.opencode/skills`.
        for skill_dir in self._iter_selected_skill_dirs(skill_names):
            staged_path = opencode_dir / skill_dir.name
            if staged_path.exists():
                if staged_path.is_symlink():
                    raise RuntimeError(f"Skill path already exists as a symlink: {staged_path}")
                continue
            shutil.copytree(skill_dir, staged_path, symlinks=False)
            created.append(staged_path)

        return SkillLinkSet(created)

    def prepare_pi_skills(
        self,
        workdir: Path,
        skill_names: tuple[str, ...] | None = None,
    ) -> SkillLinkSet:
        pi_dir = workdir / ".pi"
        target = pi_dir / "skills"
        created: List[Path] = []
        pi_dir.mkdir(parents=True, exist_ok=True)

        if not target.exists() and skill_names is None:
            shutil.copytree(self.skills_root, target, symlinks=False)
            created.append(target)
            return SkillLinkSet(created)

        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)
            for skill_dir in self._iter_selected_skill_dirs(skill_names):
                shutil.copytree(skill_dir, target / skill_dir.name, symlinks=False)
            created.append(target)
            return SkillLinkSet(created)

        if target.is_symlink():
            raise RuntimeError(f"Existing Pi skills path must not be a symlink: {target}")

        if not target.is_dir():
            raise RuntimeError(f"Existing Pi skills path is not a directory: {target}")

        for skill_dir in self._iter_selected_skill_dirs(skill_names):
            staged_path = target / skill_dir.name
            if staged_path.exists():
                if staged_path.is_symlink():
                    raise RuntimeError(f"Skill path already exists as a symlink: {staged_path}")
                continue
            shutil.copytree(skill_dir, staged_path, symlinks=False)
            created.append(staged_path)

        return SkillLinkSet(created)

    def prepare_claude_skills(
        self,
        workdir: Path,
        skill_names: tuple[str, ...] | None = None,
    ) -> SkillLinkSet:
        claude_dir = workdir / ".claude"
        target = claude_dir / "skills"
        created: List[Path] = []
        claude_dir.mkdir(parents=True, exist_ok=True)

        if not target.exists() and skill_names is None:
            shutil.copytree(self.skills_root, target, symlinks=False)
            created.append(target)
            return SkillLinkSet(created)

        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)
            for skill_dir in self._iter_selected_skill_dirs(skill_names):
                shutil.copytree(skill_dir, target / skill_dir.name, symlinks=False)
            created.append(target)
            return SkillLinkSet(created)

        if target.is_symlink():
            raise RuntimeError(f"Existing Claude skills path must not be a symlink: {target}")

        if not target.is_dir():
            raise RuntimeError(f"Existing Claude skills path is not a directory: {target}")

        for skill_dir in self._iter_selected_skill_dirs(skill_names):
            staged_path = target / skill_dir.name
            if staged_path.exists():
                if staged_path.is_symlink():
                    raise RuntimeError(f"Skill path already exists as a symlink: {staged_path}")
                continue
            shutil.copytree(skill_dir, staged_path, symlinks=False)
            created.append(staged_path)

        return SkillLinkSet(created)

    def prepare_skills(
        self,
        backend: str,
        workdir: Path,
        skill_names: tuple[str, ...] | None = None,
    ) -> SkillLinkSet:
        if backend == "codex":
            return self.prepare_codex_skills(workdir, skill_names=skill_names)
        if backend == "opencode":
            return self.prepare_opencode_skills(workdir, skill_names=skill_names)
        if backend == "pi":
            return self.prepare_pi_skills(workdir, skill_names=skill_names)
        if backend == "claude":
            return self.prepare_claude_skills(workdir, skill_names=skill_names)
        raise RuntimeError(f"Unsupported skill backend: {backend}")

    def cleanup(self, link_set: SkillLinkSet) -> list[str]:
        warnings: list[str] = []
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
