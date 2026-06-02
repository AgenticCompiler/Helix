from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from triton_agent.optimize.naming import resolve_batch_optimize_operator_file

_WORKSPACE_FILE_ARTIFACT_NAMES = (
    "opt-note.md",
    "learned_lessons.md",
    "report.md",
    "log_check_result.json",
    "log_check_result.md",
    "pattern_analysis.json",
    "pattern_analysis.md",
    "extra-info.json",
)
_WORKSPACE_DIR_ARTIFACT_NAMES = (
    "baseline",
    "opt-verify",
    ".triton-agent",
    "triton-agent-logs",
)
_BATCH_FILE_ARTIFACT_NAMES = (
    "optimize-batch-status.json",
    "log_check_summary.md",
    "log_check_summary.json",
    "report-batch-state.json",
    "report-batch.md",
)
_CASE_PREFIXES = ("test_", "differential_test_", "bench_")


@dataclass(frozen=True)
class CleanupResult:
    target: Path
    removed: tuple[Path, ...]
    missing: tuple[Path, ...]


def is_cleanable_workspace(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        resolve_batch_optimize_operator_file(path)
        return True
    except ValueError:
        pass

    if any(path.glob("triton_*.py")):
        return True
    if any(path.glob("opt_*.py")):
        return True
    if any(path.glob("PROF_*")):
        return True
    if any(path.glob("*_result.pt")):
        return True
    if any(path.glob("*_perf.txt")):
        return True
    if any(path.glob("opt-round-*")):
        return True
    if any((path / name).exists() for name in (*_WORKSPACE_FILE_ARTIFACT_NAMES, *_WORKSPACE_DIR_ARTIFACT_NAMES)):
        return True
    return False


def discover_clean_workspaces(root: Path) -> list[Path]:
    if is_cleanable_workspace(root):
        return [root]
    return sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))


def clean_workspace(workspace: Path, *, deep: bool) -> CleanupResult:
    operator_path = _resolve_original_operator(workspace)
    targets = _workspace_cleanup_targets(workspace, operator_path=operator_path, deep=deep)
    return _remove_paths(workspace, targets)


def clean_batch_root_artifacts(root: Path) -> CleanupResult:
    targets = [root / name for name in _BATCH_FILE_ARTIFACT_NAMES]
    return _remove_paths(root, targets)


def _resolve_original_operator(workspace: Path) -> Path | None:
    try:
        return resolve_batch_optimize_operator_file(workspace)
    except ValueError:
        return None


def _workspace_cleanup_targets(
    workspace: Path,
    *,
    operator_path: Path | None,
    deep: bool,
) -> list[Path]:
    targets: list[Path] = []
    targets.extend(workspace / name for name in _WORKSPACE_FILE_ARTIFACT_NAMES)
    targets.extend(workspace / name for name in _WORKSPACE_DIR_ARTIFACT_NAMES)
    targets.extend(sorted(path for path in workspace.glob("*_result.pt")))
    targets.extend(sorted(path for path in workspace.glob("*_perf.txt")))
    targets.extend(sorted(path for path in workspace.glob("opt-round-*")))
    targets.extend(sorted(path for path in workspace.glob("PROF_*")))
    targets.extend(sorted(path for path in workspace.glob("triton_*.py")))
    targets.extend(sorted(path for path in workspace.glob("opt_*.py")))

    if operator_path is not None:
        if deep:
            targets.append(workspace / f"test_{operator_path.stem}.py")
            targets.append(workspace / f"differential_test_{operator_path.stem}.py")
            targets.append(workspace / f"bench_{operator_path.stem}.py")
    elif deep:
        for prefix in _CASE_PREFIXES:
            targets.extend(sorted(path for path in workspace.glob(f"{prefix}*.py")))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in targets:
        resolved = path.resolve(strict=False)
        if operator_path is not None and resolved == operator_path.resolve(strict=False):
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _remove_paths(target: Path, paths: list[Path]) -> CleanupResult:
    removed: list[Path] = []
    missing: list[Path] = []
    for path in paths:
        if path.is_symlink():
            path.unlink()
            removed.append(path)
            continue
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(path)
            continue
        if path.exists():
            path.unlink()
            removed.append(path)
            continue
        missing.append(path)
    return CleanupResult(target=target, removed=tuple(removed), missing=tuple(missing))
