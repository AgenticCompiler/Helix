from __future__ import annotations

import functools
import json
import subprocess
from pathlib import Path


def _find_git_root(start: Path) -> Path | None:
    """Walk up from *start* looking for a ``.git`` directory or worktree file.

    Returns the repository root, or ``None`` when no Git metadata is found.
    """
    current = start.resolve()
    while True:
        git_path = current / ".git"
        if git_path.exists():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _resolve_source_checkout_commit() -> str | None:
    """Return the full 40-character ``HEAD`` commit from a source checkout.

    Returns ``None`` when the running code is not backed by a Git checkout
    or the commit cannot be resolved.
    """
    package_dir = Path(__file__).resolve().parent
    repo_root = _find_git_root(package_dir)
    if repo_root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit if commit else None


def _load_embedded_commit() -> str | None:
    """Read ``git_commit`` from the embedded ``_build_meta.json`` payload.

    Returns ``None`` when the file is missing, unreadable, or does not
    contain a valid ``git_commit`` string.
    """
    meta_path = Path(__file__).with_name("_build_meta.json")
    try:
        raw = meta_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    commit = data.get("git_commit")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if not isinstance(commit, str) or not commit:
        return None
    return commit


def _is_installed_package() -> bool:
    """Return True when the package appears to be installed, not a source checkout."""
    parts = Path(__file__).resolve().parts
    return "site-packages" in parts or "dist-packages" in parts


@functools.lru_cache(maxsize=1)
def get_build_commit() -> str | None:
    """Return the full 40-character build commit, or ``None``."""
    if not _is_installed_package():
        source = _resolve_source_checkout_commit()
        if source is not None:
            return source
    return _load_embedded_commit()


def get_build_info_display() -> str:
    """Return the help-display commit value (12‑char short SHA or ``"unknown"``)."""
    commit = get_build_commit()
    if commit is None:
        return "unknown"
    return commit[:12]
