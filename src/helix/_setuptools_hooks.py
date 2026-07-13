from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist


_PACKAGE_NAME = "helix"


def _resolve_build_commit() -> str:
    env_commit = os.environ.get("HELIX_BUILD_GIT_COMMIT", "").strip()
    if env_commit:
        return env_commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    commit = result.stdout.strip()
    return commit if commit else "unknown"


def _write_meta(dest_dir: Path, *, commit: str | None = None) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if commit is None:
        commit = _resolve_build_commit()
    payload = {"git_commit": commit}
    (dest_dir / "_build_meta.json").write_text(json.dumps(payload), encoding="utf-8")


def _reset_build_package_dir(build_lib: str) -> Path:
    dest_dir = Path(build_lib) / _PACKAGE_NAME
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    return dest_dir


class BuildPyWithMeta(_build_py):
    def run(self) -> None:
        dest_dir: Path | None = None
        commit: str | None = None
        if self.build_lib:
            commit = _resolve_build_commit()
            if commit == "unknown":
                commit = _try_source_meta_commit()
            dest_dir = _reset_build_package_dir(self.build_lib)
        super().run()
        if dest_dir is not None:
            _write_meta(dest_dir, commit=commit)


def _try_source_meta_commit() -> str:
    source_meta = Path(__file__).resolve().parent / "_build_meta.json"
    try:
        raw = source_meta.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return "unknown"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return "unknown"
    if not isinstance(data, dict):
        return "unknown"
    commit_raw = data.get("git_commit")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if isinstance(commit_raw, str) and commit_raw:
        return commit_raw
    return "unknown"


class SdistWithMeta(_sdist):
    def make_distribution(self) -> None:
        source_dir = Path(__file__).resolve().parent
        source_meta = source_dir / "_build_meta.json"
        _write_meta(source_dir)
        repo_root = Path(__file__).resolve().parents[2]
        rel_meta = str(source_meta.relative_to(repo_root))
        if rel_meta not in self.filelist.files:
            self.filelist.files.append(rel_meta)
        try:
            super().make_distribution()
        finally:
            if source_meta.exists():
                source_meta.unlink()
