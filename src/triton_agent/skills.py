from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from triton_agent.verbose import format_symlink


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

        # When the workspace has no Codex skills directory yet, link the whole source
        # tree so the agent sees the repository skills as-is.
        if not target.exists():
            target.symlink_to(self.skills_root, target_is_directory=True)
            created.append(target)
            return SkillLinkSet(created)

        # Reusing an existing root symlink keeps repeated runs idempotent and avoids
        # trying to add per-skill links inside the linked source tree itself.
        if target.is_symlink() and target.resolve() == self.skills_root:
            return SkillLinkSet(created)

        if not target.is_dir():
            raise RuntimeError(f"Existing Codex skills path is not a directory: {target}")

        # If `.codex/skills` already exists, only add missing per-skill links so we do
        # not disturb any pre-existing workspace-local content.
        for skill_dir in self._iter_skill_dirs():
            link_path = target / skill_dir.name
            if link_path.exists():
                if link_path.is_symlink() and link_path.resolve() == skill_dir.resolve():
                    continue
                raise RuntimeError(f"Skill path already exists and cannot be replaced: {link_path}")
            link_path.symlink_to(skill_dir, target_is_directory=True)
            created.append(link_path)

        return SkillLinkSet(created)

    def prepare_opencode_skills(self, workdir: Path) -> SkillLinkSet:
        opencode_dir = workdir / ".opencode" / "skills"
        created: List[Path] = []
        opencode_dir.mkdir(parents=True, exist_ok=True)

        # OpenCode expects one skill directory per entry under `.opencode/skills`.
        for skill_dir in self._iter_skill_dirs():
            link_path = opencode_dir / skill_dir.name
            if link_path.exists():
                if link_path.is_symlink() and link_path.resolve() == skill_dir.resolve():
                    continue
                raise RuntimeError(f"Skill path already exists and cannot be replaced: {link_path}")
            link_path.symlink_to(skill_dir, target_is_directory=True)
            created.append(link_path)

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
                if path.is_symlink():
                    path.unlink()
            except OSError as exc:
                warnings.append(f"Failed to remove symlink {path}: {exc}")
        return warnings

    def describe_prepare(self, link_set: SkillLinkSet) -> list[str]:
        if not link_set.created_paths:
            return ["No new skill links were created."]
        return [f"created skill link {format_symlink(path)}" for path in link_set.created_paths]

    def describe_cleanup(self, link_set: SkillLinkSet) -> list[str]:
        if not link_set.created_paths:
            return ["No skill links needed cleanup."]
        return [f"removed skill link {format_symlink(path)}" for path in reversed(link_set.created_paths)]
