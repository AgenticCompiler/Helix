"""Typed bridge for run-eval remote environment and target parsing."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from functools import lru_cache
from typing import Protocol, TypedDict, cast

from helix.skills.loader import load_operator_eval_script_module


class RemoteSpec(TypedDict):
    user_host: str
    port: int | None


class RemoteExecutionEnvApi(Protocol):
    def remote_target_env_name(self) -> str: ...
    def remote_workdir_env_name(self) -> str: ...
    def build_remote_execution_env(self, remote: str | None, remote_workdir: str | None) -> dict[str, str]: ...
    def resolve_remote_execution(self, explicit_remote: str | None, explicit_remote_workdir: str | None, environ: Mapping[str, str] | None = None) -> tuple[str | None, str | None]: ...
    def apply_remote_execution_env(self, explicit_remote: str | None, explicit_remote_workdir: str | None, environ: MutableMapping[str, str] | None = None) -> None: ...


class RunRuntimeApi(Protocol):
    def parse_remote_spec(self, remote: str) -> RemoteSpec: ...


@lru_cache(maxsize=1)
def _env_api() -> RemoteExecutionEnvApi:
    return cast(RemoteExecutionEnvApi, load_operator_eval_script_module("remote_execution_env_api"))


@lru_cache(maxsize=1)
def _runtime_api() -> RunRuntimeApi:
    return cast(RunRuntimeApi, load_operator_eval_script_module("run_runtime_api"))


def remote_target_env_name() -> str:
    return str(_env_api().remote_target_env_name())


def remote_workdir_env_name() -> str:
    return str(_env_api().remote_workdir_env_name())


def build_remote_execution_env(remote: str | None, remote_workdir: str | None) -> dict[str, str]:
    return _env_api().build_remote_execution_env(remote, remote_workdir)


def resolve_remote_execution(explicit_remote: str | None, explicit_remote_workdir: str | None, environ: Mapping[str, str] | None = None) -> tuple[str | None, str | None]:
    return _env_api().resolve_remote_execution(explicit_remote, explicit_remote_workdir, environ)


def apply_remote_execution_env(explicit_remote: str | None, explicit_remote_workdir: str | None, environ: MutableMapping[str, str] | None = None) -> None:
    _env_api().apply_remote_execution_env(explicit_remote, explicit_remote_workdir, environ)


def parse_remote_spec(remote: str) -> RemoteSpec:
    return _runtime_api().parse_remote_spec(remote)
