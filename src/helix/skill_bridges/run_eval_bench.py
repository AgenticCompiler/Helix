"""Typed bridge for the run-bench skill facade."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Protocol, TextIO, cast

from helix.skills.loader import load_operator_eval_script_module


class RunBenchApi(Protocol):
    def parse_bench_metadata(self, bench_file: Path) -> dict[str, str]: ...
    def resolve_bench_kernel_names(
        self, bench_file: Path, operator_file: Path
    ) -> tuple[str, ...]: ...
    def normalize_bench_mode(self, raw_mode: str) -> str: ...
    def run_local_bench(
        self,
        bench_file: Path,
        operator_file: Path,
        bench_mode: str,
        npu_devices: str | None = None,
        verbose: bool = False,
        output: str | None = None,
    ) -> tuple[dict[str, object], Path | None]: ...
    def run_remote_bench(
        self,
        bench_file: Path,
        operator_file: Path,
        bench_mode: str,
        remote: str,
        remote_workdir: str | None,
        npu_devices: str | None = None,
        keep_remote_workdir: bool = False,
        verbose: bool = False,
        stderr: TextIO | None = None,
        output: str | None = None,
    ) -> tuple[dict[str, object], Path | None, str]: ...


@lru_cache(maxsize=1)
def _api() -> RunBenchApi:
    return cast(RunBenchApi, load_operator_eval_script_module("run_bench_api"))


def parse_bench_metadata(bench_file: Path) -> dict[str, str]:
    return _api().parse_bench_metadata(bench_file)


def resolve_bench_kernel_names(
    bench_file: Path, operator_file: Path
) -> tuple[str, ...]:
    return tuple(_api().resolve_bench_kernel_names(bench_file, operator_file))


def normalize_bench_mode(raw_mode: str) -> str:
    return str(_api().normalize_bench_mode(raw_mode))


def run_local_bench(
    bench_file: Path,
    operator_file: Path,
    bench_mode: str,
    npu_devices: str | None = None,
    *,
    verbose: bool = False,
    output: str | None = None,
) -> tuple[dict[str, object], Path | None]:
    return _api().run_local_bench(
        bench_file,
        operator_file,
        bench_mode,
        npu_devices,
        verbose=verbose,
        output=output,
    )


def run_remote_bench(
    bench_file: Path,
    operator_file: Path,
    bench_mode: str,
    remote: str,
    remote_workdir: str | None,
    npu_devices: str | None = None,
    *,
    keep_remote_workdir: bool = False,
    verbose: bool = False,
    stderr: TextIO | None = None,
    output: str | None = None,
) -> tuple[dict[str, object], Path | None, str]:
    return _api().run_remote_bench(
        bench_file,
        operator_file,
        bench_mode,
        remote,
        remote_workdir,
        npu_devices,
        keep_remote_workdir=keep_remote_workdir,
        verbose=verbose,
        stderr=stderr,
        output=output,
    )
