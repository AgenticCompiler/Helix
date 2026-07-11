from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

from helix.skills.catalog import (
    get_skill_catalog_entry,
    list_catalog_skill_names,
)


@dataclass(frozen=True)
class _SkillBackendConfig:
    target_parts: tuple[str, ...]
    copy_root_when_missing: bool
    backend_root: str


_SKILL_BACKEND_CONFIGS: dict[str, _SkillBackendConfig] = {
    "codex": _SkillBackendConfig(
        target_parts=(".codex", "skills"),
        copy_root_when_missing=True,
        backend_root=".codex",
    ),
    "opencode": _SkillBackendConfig(
        target_parts=(".opencode", "skills"),
        copy_root_when_missing=False,
        backend_root=".opencode",
    ),
    "pi": _SkillBackendConfig(
        target_parts=(".pi", "skills"),
        copy_root_when_missing=True,
        backend_root=".pi",
    ),
    "claude": _SkillBackendConfig(
        target_parts=(".claude", "skills"),
        copy_root_when_missing=True,
        backend_root=".claude",
    ),
    "openhands": _SkillBackendConfig(
        target_parts=(".openhands", "skills"),
        copy_root_when_missing=True,
        backend_root=".openhands",
    ),
    "traecli": _SkillBackendConfig(
        target_parts=(".traecli", "skills"),
        copy_root_when_missing=True,
        backend_root=".traecli",
    ),
}


def staged_skill_dir(backend: str) -> Path:
    """Return the workspace-relative directory where skills are staged for a given backend."""
    config = _SKILL_BACKEND_CONFIGS[backend]
    return Path(config.target_parts[0]) / config.target_parts[1]


@dataclass
class SkillLinkSet:
    created_paths: List[Path]
    refreshed_paths: List[Path] = field(default_factory=lambda: [])
    temporary_git_dir: Path | None = None


def _reset_temporary_git_repo_enabled() -> bool:
    value = os.environ.get("HELIX_RESET_GIT_REPO")
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


class SkillLinkManager:
    def __init__(self, skills_root: Path) -> None:
        self.skills_root = skills_root.resolve()

    def _iter_skill_dirs(self) -> Iterable[tuple[str, Path]]:
        for skill_name in list_catalog_skill_names():
            entry = get_skill_catalog_entry(skill_name)
            path = self.skills_root / entry.source_group / skill_name
            if path.exists() and path.is_dir():
                yield skill_name, path

    def _iter_selected_skill_dirs(
        self,
        skill_names: tuple[str, ...] | None,
        skill_sources: dict[str, str] | None = None,
    ) -> Iterable[tuple[str, Path]]:
        if skill_names is None:
            for name, path in self._iter_skill_dirs():
                yield name, path
            return

        seen: set[str] = set()
        for skill_name in skill_names:
            if skill_name in seen:
                continue
            seen.add(skill_name)
            source_name = skill_sources.get(skill_name, skill_name) if skill_sources else skill_name
            try:
                entry = get_skill_catalog_entry(source_name)
            except KeyError:
                raise RuntimeError(f"Requested skill does not exist: {source_name}")
            skill_dir = self.skills_root / entry.source_group / source_name
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

    def _ensure_local_git_boundary(self, workdir: Path) -> Path | None:
        git_path = workdir / ".git"
        if git_path.exists():
            return None
        if shutil.which("git") is None:
            return None
        try:
            subprocess.run(
                ["git", "init", "-q"],
                cwd=workdir,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or str(exc)
            raise RuntimeError(f"Failed to initialize temporary git repo in {workdir}: {detail}") from exc
        if not git_path.exists():
            raise RuntimeError(f"Failed to initialize temporary git repo in {workdir}: .git not created")
        return git_path

    def _remove_path(self, path: Path) -> None:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
            return
        if path.exists():
            path.unlink()

    def _copy_selected_skill_dirs(
        self,
        target: Path,
        skill_names: tuple[str, ...] | None,
        skill_sources: dict[str, str] | None = None,
    ) -> tuple[list[Path], list[Path]]:
        created: list[Path] = []
        refreshed: list[Path] = []
        for staged_name, skill_dir in self._iter_selected_skill_dirs(skill_names, skill_sources):
            staged_path = target / staged_name
            if staged_path.exists():
                if staged_path.is_symlink():
                    raise RuntimeError(f"Skill path already exists as a symlink: {staged_path}")
                if not staged_path.is_dir():
                    raise RuntimeError(f"Skill path already exists and is not a directory: {staged_path}")
                self._remove_path(staged_path)
                refreshed.append(staged_path)
            shutil.copytree(skill_dir, staged_path, symlinks=False)
            created.append(staged_path)
        return created, refreshed

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

        temporary_git_dir = self._ensure_local_git_boundary(workdir)
        try:
            target = workdir.joinpath(*config.target_parts)
            backend_root_path = workdir / config.backend_root
            root_pre_existed = backend_root_path.exists()
            created: list[Path] = []

            if config.copy_root_when_missing and not target.exists() and skill_names is None:
                target.parent.mkdir(parents=True, exist_ok=True)
                for name, skill_dir in self._iter_skill_dirs():
                    staged_path = target / name
                    if not staged_path.exists():
                        shutil.copytree(skill_dir, staged_path, symlinks=False)
                created.append(target)
                if not root_pre_existed:
                    created.insert(0, backend_root_path)
                return SkillLinkSet(created, temporary_git_dir=temporary_git_dir)

            self._prepare_target_dir(target)
            refreshed: list[Path] = []

            if config.copy_root_when_missing and not any(target.iterdir()) and skill_names is not None:
                created_paths, refreshed = self._copy_selected_skill_dirs(target, skill_names, skill_sources)
                created.extend(created_paths)
                if created:
                    result = [target]
                    if not root_pre_existed:
                        result.insert(0, backend_root_path)
                    return SkillLinkSet(result, refreshed_paths=refreshed, temporary_git_dir=temporary_git_dir)
                return SkillLinkSet([], refreshed_paths=refreshed, temporary_git_dir=temporary_git_dir)

            created_paths, refreshed = self._copy_selected_skill_dirs(target, skill_names, skill_sources)
            created.extend(created_paths)
            if not root_pre_existed:
                created.insert(0, backend_root_path)
            return SkillLinkSet(created, refreshed_paths=refreshed, temporary_git_dir=temporary_git_dir)
        except Exception:
            if temporary_git_dir is not None:
                self._remove_path(temporary_git_dir)
            raise

    def cleanup(self, link_set: SkillLinkSet) -> list[str]:
        warnings: list[str] = []
        for path in reversed(link_set.created_paths):
            try:
                self._remove_path(path)
            except OSError as exc:
                warnings.append(f"Failed to remove skill copy {path}: {exc}")
        if link_set.temporary_git_dir is not None and _reset_temporary_git_repo_enabled():
            try:
                self._remove_path(link_set.temporary_git_dir)
            except OSError as exc:
                warnings.append(f"Failed to remove temporary git repo {link_set.temporary_git_dir}: {exc}")
        return warnings

    def describe_prepare(self, link_set: SkillLinkSet) -> list[str]:
        messages: list[str] = []
        refreshed_paths = set(link_set.refreshed_paths)
        if link_set.temporary_git_dir is not None:
            messages.append(f"created temporary git repo {link_set.temporary_git_dir}")
        for path in link_set.created_paths:
            if path in refreshed_paths:
                messages.append(f"refreshed skill copy {path}")
                continue
            messages.append(f"created skill copy {path}")
        if not messages:
            return ["No new skill copies were created."]
        return messages

    def describe_cleanup(self, link_set: SkillLinkSet) -> list[str]:
        messages = [f"removed skill copy {path}" for path in reversed(link_set.created_paths)]
        if link_set.temporary_git_dir is not None:
            if _reset_temporary_git_repo_enabled():
                messages.append(f"removed temporary git repo {link_set.temporary_git_dir}")
            else:
                messages.append(f"preserved temporary git repo {link_set.temporary_git_dir}")
        if not messages:
            return ["No skill copies needed cleanup."]
        return messages
