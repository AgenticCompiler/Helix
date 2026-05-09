from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class _SkillBackendConfig:
    target_parts: tuple[str, ...]
    copy_root_when_missing: bool


_SKILL_BACKEND_CONFIGS: dict[str, _SkillBackendConfig] = {
    "codex": _SkillBackendConfig(target_parts=(".codex", "skills"), copy_root_when_missing=True),
    "opencode": _SkillBackendConfig(
        target_parts=(".opencode", "skills"),
        copy_root_when_missing=False,
    ),
    "pi": _SkillBackendConfig(target_parts=(".pi", "skills"), copy_root_when_missing=True),
    "claude": _SkillBackendConfig(target_parts=(".claude", "skills"), copy_root_when_missing=True),
    "openhands": _SkillBackendConfig(
        target_parts=(".openhands", "skills"),
        copy_root_when_missing=True,
    ),
    "traecli": _SkillBackendConfig(
        target_parts=(".traecli", "skills"),
        copy_root_when_missing=True,
    ),
}


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

    def _iter_selected_skill_dirs(
        self,
        skill_names: tuple[str, ...] | None,
        skill_sources: dict[str, str] | None = None,
    ) -> Iterable[tuple[str, Path]]:
        if skill_names is None:
            for entry in self._iter_skill_dirs():
                yield entry.name, entry
            return

        seen: set[str] = set()
        for skill_name in skill_names:
            if skill_name in seen:
                continue
            seen.add(skill_name)
            source_name = skill_sources.get(skill_name, skill_name) if skill_sources else skill_name
            skill_dir = self.skills_root / source_name
            if not skill_dir.exists() or not skill_dir.is_dir():
                raise RuntimeError(f"Requested skill does not exist: {skill_dir}")
            yield skill_name, skill_dir

    def _target_path(self, workdir: Path, backend: str) -> Path:
        config = _SKILL_BACKEND_CONFIGS.get(backend)
        if config is None:
            raise RuntimeError(f"Unsupported skill backend: {backend}")
        return workdir.joinpath(*config.target_parts)

    def _prepare_target_dir(self, target: Path) -> None:
        if target.exists():
            if target.is_symlink():
                raise RuntimeError(f"Existing skills path must not be a symlink: {target}")
            if not target.is_dir():
                raise RuntimeError(f"Existing skills path is not a directory: {target}")
            return
        target.mkdir(parents=True, exist_ok=True)

    def _copy_selected_skill_dirs(
        self,
        target: Path,
        skill_names: tuple[str, ...] | None,
        skill_sources: dict[str, str] | None = None,
    ) -> list[Path]:
        created: list[Path] = []
        for staged_name, skill_dir in self._iter_selected_skill_dirs(skill_names, skill_sources):
            staged_path = target / staged_name
            if staged_path.exists():
                if staged_path.is_symlink():
                    raise RuntimeError(f"Skill path already exists as a symlink: {staged_path}")
                continue
            shutil.copytree(skill_dir, staged_path, symlinks=False)
            created.append(staged_path)
        return created

    def prepare_skills(
        self,
        backend: str,
        workdir: Path,
        skill_names: tuple[str, ...] | None = None,
        skill_sources: dict[str, str] | None = None,
    ) -> SkillLinkSet:
        config = _SKILL_BACKEND_CONFIGS.get(backend)
        if config is None:
            raise RuntimeError(f"Unsupported skill backend: {backend}")

        target = workdir.joinpath(*config.target_parts)
        created: list[Path] = []

        if config.copy_root_when_missing and not target.exists() and skill_names is None:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.skills_root, target, symlinks=False)
            created.append(target)
            return SkillLinkSet(created)

        self._prepare_target_dir(target)

        if config.copy_root_when_missing and not any(target.iterdir()) and skill_names is not None:
            created.extend(self._copy_selected_skill_dirs(target, skill_names, skill_sources))
            if created:
                return SkillLinkSet([target])
            return SkillLinkSet([])

        created.extend(self._copy_selected_skill_dirs(target, skill_names, skill_sources))
        return SkillLinkSet(created)

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
