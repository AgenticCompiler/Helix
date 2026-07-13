from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping

from env_registry import HELIX_REMOTE, HELIX_REMOTE_WORKDIR


def remote_target_env_name() -> str:
    return HELIX_REMOTE


def remote_workdir_env_name() -> str:
    return HELIX_REMOTE_WORKDIR


def build_remote_execution_env(
    remote: str | None,
    remote_workdir: str | None,
) -> dict[str, str]:
    resolved_remote = _normalize_value(remote)
    if resolved_remote is None:
        return {}
    env = {HELIX_REMOTE: resolved_remote}
    resolved_workdir = _normalize_value(remote_workdir)
    if resolved_workdir is not None:
        env[HELIX_REMOTE_WORKDIR] = resolved_workdir
    return env


def resolve_remote_execution(
    explicit_remote: str | None,
    explicit_remote_workdir: str | None,
    environ: Mapping[str, str] | None = None,
) -> tuple[str | None, str | None]:
    source = os.environ if environ is None else environ
    remote = _normalize_value(explicit_remote) or _normalize_value(source.get(HELIX_REMOTE))
    if remote is None:
        return None, None
    remote_workdir = _normalize_value(explicit_remote_workdir) or _normalize_value(
        source.get(HELIX_REMOTE_WORKDIR)
    )
    return remote, remote_workdir


def apply_remote_execution_env(
    explicit_remote: str | None,
    explicit_remote_workdir: str | None,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    target = os.environ if environ is None else environ
    remote = _normalize_value(explicit_remote)
    if remote is None:
        target.pop(HELIX_REMOTE, None)
        target.pop(HELIX_REMOTE_WORKDIR, None)
        return
    target[HELIX_REMOTE] = remote
    remote_workdir = _normalize_value(explicit_remote_workdir)
    if remote_workdir is None:
        target.pop(HELIX_REMOTE_WORKDIR, None)
        return
    target[HELIX_REMOTE_WORKDIR] = remote_workdir


def _normalize_value(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.strip()
    return value or None
