"""Resolve and stage fixed-worker Python dependencies for remote workspaces."""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable
from pathlib import Path


StageFile = Callable[[Path, str], None]


def resolve_remote_python_bundle(entry_scripts: Iterable[Path]) -> list[Path]:
    """Return fixed-worker scripts and their local static-import closure."""
    entries = [entry.resolve() for entry in entry_scripts]
    if not entries:
        return []
    scripts_root = Path(__file__).resolve().parent
    pending = list(entries)
    resolved: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in resolved:
            continue
        if not _is_local_script(path, scripts_root):
            raise ValueError(f"Bundle entry must be a Python script under {scripts_root}: {path}")
        resolved.add(path)
        pending.extend(_local_import_paths(path, scripts_root))
    return sorted(resolved, key=lambda path: path.relative_to(scripts_root).as_posix())


def stage_remote_python_bundle(
    entry_scripts: Iterable[Path],
    remote_workspace: str,
    stage_file: StageFile,
) -> list[Path]:
    """Stage a fixed-worker bundle through a caller-provided transport."""
    scripts_root = Path(__file__).resolve().parent
    bundle = resolve_remote_python_bundle(entry_scripts)
    for source in bundle:
        relative = source.relative_to(scripts_root).as_posix()
        stage_file(source, f"{remote_workspace}/{relative}")
    return bundle


def _is_local_script(path: Path, scripts_root: Path) -> bool:
    try:
        path.relative_to(scripts_root)
    except ValueError:
        return False
    return path.is_file() and path.suffix == ".py"


def _local_import_paths(path: Path, scripts_root: Path) -> list[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    dependencies: set[Path] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                dependency = _resolve_absolute_module(alias.name, scripts_root)
                if dependency is not None:
                    dependencies.add(dependency)
        elif isinstance(node, ast.ImportFrom):
            dependencies.update(_resolve_from_import(node, path, scripts_root))
    return sorted(dependencies, key=lambda dependency: dependency.as_posix())


def _resolve_from_import(node: ast.ImportFrom, path: Path, scripts_root: Path) -> set[Path]:
    base = path.parent
    if node.level:
        for _ in range(node.level - 1):
            base = base.parent
        if node.module is None:
            dependencies: set[Path] = set()
            for alias in node.names:
                dependency = _resolve_module_at(base / alias.name)
                if dependency is not None:
                    dependencies.add(dependency)
            return dependencies
        candidate = _resolve_module_at(base / Path(*node.module.split(".")))
        return {candidate} if candidate is not None else set()
    if node.module is None:
        return set()
    candidate = _resolve_absolute_module(node.module, scripts_root)
    return {candidate} if candidate is not None else set()


def _resolve_absolute_module(module_name: str, scripts_root: Path) -> Path | None:
    return _resolve_module_at(scripts_root / Path(*module_name.split(".")))


def _resolve_module_at(base: Path) -> Path | None:
    module_file = base.with_suffix(".py")
    if module_file.is_file():
        return module_file.resolve()
    package_init = base / "__init__.py"
    if package_init.is_file():
        return package_init.resolve()
    return None
