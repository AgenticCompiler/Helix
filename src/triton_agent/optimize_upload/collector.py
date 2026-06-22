from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, cast

from triton_agent.optimize.naming import resolve_batch_optimize_operator_file
from triton_agent.optimize.skill_contract import optimize_submit_round_module
from triton_agent.optimize_upload.models import CollectedUpload

_OPTIMIZE_ROUND = optimize_submit_round_module()


_EXCLUDED_DIRS = frozenset({
    "ir", "opt-verify", "ASCEND_PROFILER_OUTPUT",
    "mindstudio_profiler_output", "extra-info", "__pycache__",
})
_EXCLUDED_EXTENSIONS = frozenset({".pt", ".tar", ".tar.gz", ".tgz", ".zip"})

_ROOT_WHITELIST_GLOBS = ("*.py", "opt-note.md", "learned_lessons.md", "report.md")
_OPTIMIZE_LOG_PATTERN = "**/show-output*.log"


def _path_matches_excluded_dir(path: Path, workspace: Path) -> bool:
    try:
        rel = path.relative_to(workspace)
    except ValueError:
        return False
    for part in rel.parts:
        if part in _EXCLUDED_DIRS:
            return True
        if part.startswith("PROF_"):
            return True
    return False


def _has_excluded_extension(path: Path) -> bool:
    name = path.name
    for ext in _EXCLUDED_EXTENSIONS:
        if name.endswith(ext):
            return True
    return False


def _resolve_baseline_artifacts(workspace: Path) -> list[Path]:
    """Resolve baseline operator and perf files from baseline/state.json.

    The optimize contract stores path fields relative to the directory that
    contains baseline/state.json. Fall back to workspace-relative lookup for
    compatibility with older outputs.
    """
    baseline_dir = workspace / "baseline"
    if not baseline_dir.is_dir():
        return []
    state_file = baseline_dir / "state.json"
    if not state_file.exists():
        return []

    files: list[Path] = [state_file]
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return files

    if isinstance(state, dict):
        state_dict = cast("dict[str, object]", state)
        for key in ("baseline_operator", "perf_artifact"):
            val = state_dict.get(key)
            if val and isinstance(val, str):
                p = _resolve_state_path(baseline_dir, workspace, val)
                if p.exists():
                    files.append(p)
    return files


def _resolve_state_path(state_dir: Path, workspace: Path, relative_path: str) -> Path:
    declared = Path(relative_path)
    state_relative = (state_dir / declared).resolve()
    if state_relative.exists():
        return state_relative
    return (workspace / declared).resolve()


def _resolve_round_artifacts(workspace: Path, round_dir: Path) -> list[Path]:
    files: list[Path] = []
    # Round operator via the contract resolver so we don't accidentally
    # pick up bench_kernel.py, test_kernel.py, etc.
    op = _OPTIMIZE_ROUND.resolve_round_operator_file(round_dir)
    if op is not None and op.exists():
        files.append(op)
    # Round notes, state, and perf analysis.
    for name in ("attempts.md", "summary.md", "round-state.json", "perf-analysis.md", "compiler-analysis.md"):
        p = round_dir / name
        if p.exists():
            files.append(p)
    # Round perf artifact via the contract resolver.
    perf = _OPTIMIZE_ROUND.resolve_round_perf_file(round_dir)
    if perf is not None and perf.exists():
        files.append(perf)
    # Round operator .py fallback: include so the upload has something.
    for py_file in sorted(round_dir.glob("*.py")):
        if py_file not in files:
            files.append(py_file)
    return files


def collect_workspace_upload_files(workspace: Path) -> CollectedUpload:
    if not workspace.exists():
        raise ValueError(f"Workspace path does not exist: {workspace}")
    if not workspace.is_dir():
        raise ValueError(f"Workspace path is not a directory: {workspace}")

    has_baseline = (workspace / "baseline").is_dir()
    has_round = any(p.is_dir() and p.name.startswith("opt-round-") for p in workspace.iterdir())
    has_opt_note = (workspace / "opt-note.md").is_file()
    if not has_baseline or not has_round or not has_opt_note:
        raise ValueError(
            f"Workspace must have baseline/, at least one opt-round-* directory, "
            f"and an opt-note.md before uploading: {workspace}"
        )

    included: list[Path] = []
    excluded_entries: list[tuple[str, str]] = []

    # Resolve the source operator via the contract-aware resolver so we do
    # not accidentally point at bench_kernel.py or test_kernel.py.
    source_operator: Optional[Path] = None
    try:
        source_operator = resolve_batch_optimize_operator_file(workspace)
    except ValueError:
        pass

    root_py_files = sorted(workspace.glob("*.py"))
    root_md_files = (
        sorted(workspace.glob("opt-note.md"))
        + sorted(workspace.glob("learned_lessons.md"))
        + sorted(workspace.glob("report.md"))
    )
    for f in root_py_files + root_md_files:
        if f.is_file():
            included.append(f)

    included.extend(_resolve_baseline_artifacts(workspace))

    for child in sorted(workspace.iterdir()):
        if child.is_dir() and child.name.startswith("opt-round-"):
            included.extend(_resolve_round_artifacts(workspace, child))

    logs_dir = workspace / "triton-agent-logs"
    if logs_dir.is_dir():
        for f in sorted(logs_dir.glob(_OPTIMIZE_LOG_PATTERN)):
            included.append(f)

    filtered: list[Path] = []
    for f in included:
        try:
            rel = f.relative_to(workspace)
        except ValueError:
            continue
        rel_str = str(rel)

        if _path_matches_excluded_dir(f, workspace):
            excluded_entries.append((rel_str, "under excluded directory"))
            continue
        if _has_excluded_extension(f):
            excluded_entries.append((rel_str, "excluded file extension"))
            continue
        if rel_str.endswith("agent-sessions.jsonl"):
            excluded_entries.append((rel_str, "excluded file type"))
            continue
        if "/agent-session-" in f"/{rel_str}" and rel_str.endswith(".json"):
            excluded_entries.append((rel_str, "excluded file type"))
            continue

        filtered.append(f)

    seen: set[Path] = set()
    unique_files: list[Path] = []
    for f in filtered:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    return CollectedUpload(
        workspace=workspace,
        operator_file=source_operator,
        included_files=tuple(unique_files),
        excluded_entries=tuple(excluded_entries),
    )
