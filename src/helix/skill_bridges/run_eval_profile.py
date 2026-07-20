"""Typed bridge for the profile-bench skill facade."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Protocol, TextIO, cast

from helix.skills.loader import load_operator_eval_script_module


class RunProfileApi(Protocol):
    def run_local_profile_bench(
        self,
        bench_file: Path,
        operator_file: Path,
        case_id: str | None = None,
        kernel_name: str | None = None,
    ) -> tuple[dict[str, object], Path | None]: ...
    def run_remote_profile_bench(
        self,
        bench_file: Path,
        operator_file: Path,
        remote: str,
        remote_workdir: str | None,
        case_id: str | None = None,
        kernel_name: str | None = None,
        keep_remote_workdir: bool = False,
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> tuple[dict[str, object], Path | None, str]: ...


@lru_cache(maxsize=1)
def _api() -> RunProfileApi:
    return cast(RunProfileApi, load_operator_eval_script_module("run_profile_api"))


def run_local_profile_bench(
    bench_file: Path,
    operator_file: Path,
    case_id: str | None = None,
    kernel_name: str | None = None,
) -> tuple[dict[str, object], Path | None]:
    return _api().run_local_profile_bench(
        bench_file,
        operator_file,
        case_id=case_id,
        kernel_name=kernel_name,
    )


def run_remote_profile_bench(
    bench_file: Path,
    operator_file: Path,
    remote: str,
    remote_workdir: str | None,
    case_id: str | None = None,
    kernel_name: str | None = None,
    *,
    keep_remote_workdir: bool = False,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[dict[str, object], Path | None, str]:
    return _api().run_remote_profile_bench(
        bench_file,
        operator_file,
        remote,
        remote_workdir,
        case_id=case_id,
        kernel_name=kernel_name,
        keep_remote_workdir=keep_remote_workdir,
        verbose=verbose,
        stderr=stderr,
    )
