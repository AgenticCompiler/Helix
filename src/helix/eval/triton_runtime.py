from __future__ import annotations

import errno
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


TRITON_CACHE_DIR_ENV = "TRITON_CACHE_DIR"
TRITON_ALWAYS_COMPILE_ENV = "TRITON_ALWAYS_COMPILE"
REMOTE_TRITON_CACHE_ENV = "HELIX_REMOTE_TRITON_CACHE"
_CACHE_PARENT_NAME = ".helix-triton-cache"


@dataclass(frozen=True)
class TritonRuntimeSession:
    cache_dir: Path
    cache_parent: Path
    created_parent: bool


def prepare_triton_runtime_session(workdir: Path, run_id: str) -> TritonRuntimeSession:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("Triton cache run id must be one path component.")
    workspace = workdir.resolve()
    cache_parent = workspace / _CACHE_PARENT_NAME
    created_parent = False
    try:
        cache_parent.mkdir()
        created_parent = True
    except FileExistsError:
        if not cache_parent.is_dir():
            raise RuntimeError(f"Triton cache parent is not a directory: {cache_parent}")

    cache_dir = cache_parent / run_id
    try:
        cache_dir.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"Triton cache directory already exists: {cache_dir}") from exc
    return TritonRuntimeSession(
        cache_dir=cache_dir,
        cache_parent=cache_parent,
        created_parent=created_parent,
    )


def triton_runtime_env(
    session: TritonRuntimeSession,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    merged = dict(extra_env or {})
    merged.update(
        {
            TRITON_CACHE_DIR_ENV: str(session.cache_dir),
            TRITON_ALWAYS_COMPILE_ENV: "1",
            REMOTE_TRITON_CACHE_ENV: "1",
        }
    )
    return merged


def triton_runtime_prompt(session: TritonRuntimeSession) -> str:
    return "\n".join(
        [
            "Triton compilation is already forced with `TRITON_ALWAYS_COMPILE=1`.",
            f"Your isolated Triton cache directory is: `{session.cache_dir}`.",
            "Never read, write, or delete `~/.triton` or another session's cache directory.",
            "You may clear only your assigned cache directory, and only while no evaluation subprocess is running.",
        ]
    )


def cleanup_triton_runtime_session(session: TritonRuntimeSession) -> list[str]:
    warnings: list[str] = []
    try:
        if session.cache_dir.exists():
            shutil.rmtree(session.cache_dir)
    except OSError as exc:
        warnings.append(f"Failed to remove isolated Triton cache {session.cache_dir}: {exc}")
        return warnings

    if session.created_parent:
        try:
            session.cache_parent.rmdir()
        except OSError as exc:
            if exc.errno != errno.ENOTEMPTY:
                warnings.append(
                    f"Failed to remove isolated Triton cache parent {session.cache_parent}: {exc}"
                )
    return warnings
