"""Typed bridge for the probe-bench skill facade."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Protocol, TextIO, cast

from helix.skills.loader import load_operator_eval_script_module


class RunProbeApi(Protocol):
    def run_local_probe_bench(
        self,
        bench_file: Path,
        operator_file: Path,
        baseline_operator_file: Path,
        bench_mode: str,
        *,
        metric_source: str = "auto",
        npu_devices: str | None = None,
        verbose: bool = False,
    ) -> object: ...
    def run_remote_probe_bench(
        self,
        bench_file: Path,
        operator_file: Path,
        baseline_operator_file: Path,
        bench_mode: str,
        remote: str,
        remote_workdir: str | None,
        *,
        metric_source: str = "auto",
        npu_devices: str | None = None,
        keep_remote_workdir: bool = False,
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> object: ...


@lru_cache(maxsize=1)
def _api() -> RunProbeApi:
    return cast(RunProbeApi, load_operator_eval_script_module("run_probe_api"))


def run_local_probe_bench(
    bench_file: Path,
    operator_file: Path,
    baseline_operator_file: Path,
    bench_mode: str,
    *,
    metric_source: str = "auto",
    npu_devices: str | None = None,
    verbose: bool = False,
) -> object:
    return _api().run_local_probe_bench(
        bench_file,
        operator_file,
        baseline_operator_file,
        bench_mode,
        metric_source=metric_source,
        npu_devices=npu_devices,
        verbose=verbose,
    )


def run_remote_probe_bench(
    bench_file: Path,
    operator_file: Path,
    baseline_operator_file: Path,
    bench_mode: str,
    remote: str,
    remote_workdir: str | None,
    *,
    metric_source: str = "auto",
    npu_devices: str | None = None,
    keep_remote_workdir: bool = False,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> object:
    return _api().run_remote_probe_bench(
        bench_file,
        operator_file,
        baseline_operator_file,
        bench_mode,
        remote,
        remote_workdir,
        metric_source=metric_source,
        npu_devices=npu_devices,
        keep_remote_workdir=keep_remote_workdir,
        verbose=verbose,
        stderr=stderr,
    )
