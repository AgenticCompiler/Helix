"""Remote API for probe benchmark orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import TextIO

from run_bench_api import normalize_bench_mode
from run_bench_remote_api import run_remote_bench_with_limits
from run_probe_execution import ProbeBenchResult, run_probe_bench


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
) -> ProbeBenchResult:
    normalized_mode = normalize_bench_mode(bench_mode)

    def measure(target_operator: Path, warmup_cap: int, repeats_cap: int):
        payload, perf_path, workspace = run_remote_bench_with_limits(
            bench_file,
            target_operator,
            normalized_mode,
            remote,
            remote_workdir,
            warmup_cap=warmup_cap,
            repeats_cap=repeats_cap,
            npu_devices=npu_devices,
            keep_remote_workdir=keep_remote_workdir,
            verbose=verbose,
            stderr=stderr,
            output=None,
        )
        return dict(payload), perf_path, workspace

    return run_probe_bench(
        bench_file,
        operator_file,
        baseline_operator_file,
        normalized_mode,
        measure,
        metric_source=metric_source,
        npu_devices=npu_devices,
        verbose=verbose,
        remote=remote,
        remote_workdir=remote_workdir,
    )
