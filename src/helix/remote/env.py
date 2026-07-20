from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from helix.skill_bridges import run_eval_remote


def remote_target_env_name() -> str:
    return run_eval_remote.remote_target_env_name()


def remote_workdir_env_name() -> str:
    return run_eval_remote.remote_workdir_env_name()


def build_remote_execution_env(
    remote: str | None,
    remote_workdir: str | None,
) -> dict[str, str]:
    return run_eval_remote.build_remote_execution_env(remote, remote_workdir)


def merge_remote_execution_env(
    extra_env: Mapping[str, str] | None,
    remote: str | None,
    remote_workdir: str | None,
) -> dict[str, str] | None:
    merged = dict(extra_env or {})
    merged.update(build_remote_execution_env(remote, remote_workdir))
    return merged or None


def resolve_remote_execution(
    explicit_remote: str | None,
    explicit_remote_workdir: str | None,
    environ: Mapping[str, str] | None = None,
) -> tuple[str | None, str | None]:
    return run_eval_remote.resolve_remote_execution(
        explicit_remote,
        explicit_remote_workdir,
        environ,
    )


def apply_remote_execution_env(
    explicit_remote: str | None,
    explicit_remote_workdir: str | None,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    run_eval_remote.apply_remote_execution_env(
        explicit_remote,
        explicit_remote_workdir,
        environ,
    )
