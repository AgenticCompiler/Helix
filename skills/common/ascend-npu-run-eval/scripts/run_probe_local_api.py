"""Local API for probe benchmark orchestration."""

from __future__ import annotations

from pathlib import Path

from run_bench_api import normalize_bench_mode
from run_bench_local_api import run_local_bench_with_limits
from run_probe_execution import ProbeBenchResult, run_probe_bench


def run_local_probe_bench(
    bench_file: Path,
    operator_file: Path,
    baseline_operator_file: Path,
    bench_mode: str,
    *,
    metric_source: str = "auto",
    npu_devices: str | None = None,
    verbose: bool = False,
) -> ProbeBenchResult:
    normalized_mode = normalize_bench_mode(bench_mode)

    def measure(target_operator: Path, warmup_cap: int, repeats_cap: int):
        payload, perf_path = run_local_bench_with_limits(
            bench_file,
            target_operator,
            normalized_mode,
            warmup_cap=warmup_cap,
            repeats_cap=repeats_cap,
            npu_devices=npu_devices,
            verbose=verbose,
            output=None,
        )
        return dict(payload), perf_path, None

    return run_probe_bench(
        bench_file,
        operator_file,
        baseline_operator_file,
        normalized_mode,
        measure,
        metric_source=metric_source,
        npu_devices=npu_devices,
        verbose=verbose,
        remote=None,
        remote_workdir=None,
    )
