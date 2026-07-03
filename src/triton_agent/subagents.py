from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class RenderedSubagent:
    relative_path: Path
    content: str


@dataclass(frozen=True)
class SubagentDefinition:
    id: str
    supported_backends: tuple[str, ...]
    render: Callable[[str], RenderedSubagent]


@dataclass
class SubagentStageSet:
    created_paths: list[Path]


class SubagentManager:
    def prepare(
        self,
        backend: str,
        workdir: Path,
        definitions: tuple[SubagentDefinition, ...],
    ) -> SubagentStageSet:
        created_paths: list[Path] = []
        try:
            for definition in definitions:
                if backend not in definition.supported_backends:
                    continue
                rendered = definition.render(backend)
                full_path = workdir / rendered.relative_path
                created_paths.extend(self._prepare_parent_dirs(full_path.parent))
                if full_path.exists() or full_path.is_symlink():
                    raise RuntimeError(f"Existing subagent file must not be overwritten: {full_path}")
                full_path.write_text(rendered.content, encoding="utf-8")
                created_paths.append(full_path)
            return SubagentStageSet(created_paths)
        except Exception:
            self.cleanup(SubagentStageSet(created_paths))
            raise

    def cleanup(self, stage_set: SubagentStageSet) -> list[str]:
        warnings: list[str] = []
        for path in reversed(stage_set.created_paths):
            try:
                if path.is_dir() and not path.is_symlink():
                    path.rmdir()
                elif path.exists() or path.is_symlink():
                    path.unlink()
            except OSError as exc:
                warnings.append(f"Failed to remove staged subagent path {path}: {exc}")
        return warnings

    def _prepare_parent_dirs(self, target_dir: Path) -> list[Path]:
        created: list[Path] = []
        missing: list[Path] = []
        current = target_dir
        while not current.exists():
            missing.append(current)
            if current.parent == current:
                break
            current = current.parent
        for directory in reversed(missing):
            directory.mkdir()
            created.append(directory)
        return created
