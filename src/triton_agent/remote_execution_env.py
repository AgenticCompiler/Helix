from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Protocol, cast

from triton_agent.skill_loader import load_operator_eval_script_module


class RemoteExecutionEnvModule(Protocol):
    def remote_target_env_name(self) -> str: ...

    def remote_workdir_env_name(self) -> str: ...

    def build_remote_execution_env(
        self,
        remote: str | None,
        remote_workdir: str | None,
    ) -> dict[str, str]: ...

    def resolve_remote_execution(
        self,
        explicit_remote: str | None,
        explicit_remote_workdir: str | None,
        environ: Mapping[str, str] | None = None,
    ) -> tuple[str | None, str | None]: ...

    def apply_remote_execution_env(
        self,
        explicit_remote: str | None,
        explicit_remote_workdir: str | None,
        environ: MutableMapping[str, str] | None = None,
    ) -> None: ...


def _load_remote_execution_env() -> RemoteExecutionEnvModule:
    return cast(RemoteExecutionEnvModule, load_operator_eval_script_module("remote_execution_env"))


def remote_target_env_name() -> str:
    return _load_remote_execution_env().remote_target_env_name()


def remote_workdir_env_name() -> str:
    return _load_remote_execution_env().remote_workdir_env_name()


def build_remote_execution_env(
    remote: str | None,
    remote_workdir: str | None,
) -> dict[str, str]:
    return _load_remote_execution_env().build_remote_execution_env(remote, remote_workdir)


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
    return _load_remote_execution_env().resolve_remote_execution(
        explicit_remote,
        explicit_remote_workdir,
        environ,
    )


def apply_remote_execution_env(
    explicit_remote: str | None,
    explicit_remote_workdir: str | None,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    _load_remote_execution_env().apply_remote_execution_env(
        explicit_remote,
        explicit_remote_workdir,
        environ,
    )
