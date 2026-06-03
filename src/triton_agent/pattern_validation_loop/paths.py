from __future__ import annotations

from pathlib import Path

DEFAULT_SYNTHESIS_FILE = "PERF_PATTERN_SYNTHESIS.md"
DEFAULT_BATCH_DIR = "pattern-validation-batch"
DEFAULT_STATE_FILE = ".triton-agent/pattern-validation-loop-state.json"


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()
