"""Typed bridge for the run-test skill facade."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Protocol, TextIO, cast

from helix.skills.loader import load_operator_eval_script_module


class RunTestApi(Protocol):
    def parse_test_metadata(self, test_file: Path) -> dict[str, str]: ...
    def run_local_test(
        self,
        test_file: Path,
        operator_file: Path,
        test_mode: str,
        *,
        case_id: str | None = None,
        accuracy_mode: str | None = None,
        extra_env: Mapping[str, str] | None = None,
        verbose: bool = False,
    ) -> tuple[dict[str, object], Path | None]: ...
    def run_remote_test(
        self,
        test_file: Path,
        operator_file: Path,
        test_mode: str,
        remote: str,
        remote_workdir: str | None,
        *,
        case_id: str | None = None,
        accuracy_mode: str | None = None,
        extra_env: Mapping[str, str] | None = None,
        keep_remote_workdir: bool = False,
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> tuple[dict[str, object], Path | None, str]: ...
    def run_local_test_case_payload(
        self,
        test_file: Path,
        operator_file: Path,
        *,
        case_id: str,
        accuracy_mode: str | None = None,
        verbose: bool = False,
    ) -> tuple[dict[str, object], object | None]: ...
    def run_remote_test_case_payload(
        self,
        test_file: Path,
        operator_file: Path,
        remote: str,
        remote_workdir: str | None,
        *,
        case_id: str,
        accuracy_mode: str | None = None,
        keep_remote_workdir: bool = False,
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> tuple[dict[str, object], object | None, str]: ...
    def run_remote_differential_comparison(
        self,
        test_file: Path,
        ref_operator_file: Path,
        operator_file: Path,
        remote: str,
        remote_workdir: str | None,
        *,
        case_id: str | None = None,
        accuracy_mode: str | None = None,
        extra_env: Mapping[str, str] | None = None,
        keep_remote_workdir: bool = False,
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> tuple[dict[str, object], str]: ...


@lru_cache(maxsize=1)
def _api() -> RunTestApi:
    return cast(RunTestApi, load_operator_eval_script_module("run_test_api"))


def parse_test_metadata(test_file: Path) -> dict[str, str]:
    return _api().parse_test_metadata(test_file)


def run_local_test(
    test_file: Path,
    operator_file: Path,
    test_mode: str,
    *,
    case_id: str | None = None,
    accuracy_mode: str | None = None,
    extra_env: Mapping[str, str] | None = None,
    verbose: bool = False,
) -> tuple[dict[str, object], Path | None]:
    return _api().run_local_test(
        test_file,
        operator_file,
        test_mode,
        case_id=case_id,
        accuracy_mode=accuracy_mode,
        extra_env=extra_env,
        verbose=verbose,
    )


def run_remote_test(
    test_file: Path,
    operator_file: Path,
    test_mode: str,
    remote: str,
    remote_workdir: str | None,
    *,
    case_id: str | None = None,
    accuracy_mode: str | None = None,
    extra_env: Mapping[str, str] | None = None,
    keep_remote_workdir: bool = False,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[dict[str, object], Path | None, str]:
    return _api().run_remote_test(
        test_file,
        operator_file,
        test_mode,
        remote,
        remote_workdir,
        case_id=case_id,
        accuracy_mode=accuracy_mode,
        extra_env=extra_env,
        keep_remote_workdir=keep_remote_workdir,
        verbose=verbose,
        stderr=stderr,
    )


def run_local_test_case_payload(
    test_file: Path,
    operator_file: Path,
    *,
    case_id: str,
    accuracy_mode: str | None = None,
    verbose: bool = False,
) -> tuple[dict[str, object], object | None]:
    return _api().run_local_test_case_payload(
        test_file,
        operator_file,
        case_id=case_id,
        accuracy_mode=accuracy_mode,
        verbose=verbose,
    )


def run_remote_test_case_payload(
    test_file: Path,
    operator_file: Path,
    remote: str,
    remote_workdir: str | None,
    *,
    case_id: str,
    accuracy_mode: str | None = None,
    keep_remote_workdir: bool = False,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[dict[str, object], object | None, str]:
    return _api().run_remote_test_case_payload(
        test_file,
        operator_file,
        remote,
        remote_workdir,
        case_id=case_id,
        accuracy_mode=accuracy_mode,
        keep_remote_workdir=keep_remote_workdir,
        verbose=verbose,
        stderr=stderr,
    )


def run_remote_differential_comparison(
    test_file: Path,
    ref_operator_file: Path,
    operator_file: Path,
    remote: str,
    remote_workdir: str | None,
    *,
    case_id: str | None = None,
    accuracy_mode: str | None = None,
    extra_env: Mapping[str, str] | None = None,
    keep_remote_workdir: bool = False,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[dict[str, object], str]:
    return _api().run_remote_differential_comparison(
        test_file,
        ref_operator_file,
        operator_file,
        remote,
        remote_workdir,
        case_id=case_id,
        accuracy_mode=accuracy_mode,
        extra_env=extra_env,
        keep_remote_workdir=keep_remote_workdir,
        verbose=verbose,
        stderr=stderr,
    )
