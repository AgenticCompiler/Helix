"""Remote workspace API for canonical benchmark and probe execution."""

from __future__ import annotations

from pathlib import Path
from typing import TextIO

from debug_device import maybe_print_visible_devices
from remote_python_bundle import stage_remote_python_bundle
from result_payload import ResultPayload
from run_bench_modes import execute_remote_bench_workspace, stage_remote_bench_input_files
from run_runtime import (
    RemoteSpec,
    cleanup_remote_workspace,
    copy_file_to_remote,
    create_remote_workspace,
)


def run_remote_bench(
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
) -> tuple[ResultPayload, Path | None, str]:
    return run_remote_bench_with_limits(
        bench_file,
        operator_file,
        bench_mode,
        remote,
        remote_workdir,
        npu_devices=npu_devices,
        keep_remote_workdir=keep_remote_workdir,
        verbose=verbose,
        stderr=stderr,
        output=output,
        warmup_cap=None,
        repeats_cap=None,
    )


def run_remote_bench_with_limits(
    bench_file: Path,
    operator_file: Path,
    bench_mode: str,
    remote: str,
    remote_workdir: str | None,
    *,
    warmup_cap: int | None,
    repeats_cap: int | None,
    npu_devices: str | None = None,
    keep_remote_workdir: bool = False,
    verbose: bool = False,
    stderr: TextIO | None = None,
    output: str | None = None,
) -> tuple[ResultPayload, Path | None, str]:
    if (warmup_cap is None) != (repeats_cap is None):
        raise ValueError("Both execution limits are required together")
    maybe_print_visible_devices()
    spec, workspace = create_remote_workspace(remote, remote_workdir, verbose=verbose, stderr=stderr)
    try:
        _stage_remote_bench_inputs(spec, workspace, bench_file, operator_file, verbose=verbose, stderr=stderr)
        return execute_remote_bench_workspace(
            bench_file,
            operator_file,
            bench_mode,
            spec,
            workspace,
            npu_devices=npu_devices,
            verbose=verbose,
            stderr=stderr,
            output=output,
            execution_limits=(warmup_cap, repeats_cap) if warmup_cap is not None and repeats_cap is not None else None,
        )
    finally:
        if not keep_remote_workdir:
            cleanup_remote_workspace(spec, workspace, verbose=verbose, stderr=stderr)


def _stage_remote_bench_inputs(
    spec: RemoteSpec,
    workspace: str,
    bench_file: Path,
    operator_file: Path,
    *,
    verbose: bool,
    stderr: TextIO | None,
) -> None:
    _stage_remote_python_bundle(
        spec,
        workspace,
        verbose=verbose,
        stderr=stderr,
    )
    copy_file_to_remote(spec, bench_file, f"{workspace}/{bench_file.name}", verbose=verbose, stderr=stderr)
    bench_cases_file = bench_file.with_suffix(".json")
    if bench_cases_file.exists():
        copy_file_to_remote(
            spec,
            bench_cases_file,
            f"{workspace}/{bench_cases_file.name}",
            verbose=verbose,
            stderr=stderr,
        )
    copy_file_to_remote(
        spec,
        operator_file,
        f"{workspace}/{operator_file.name}",
        verbose=verbose,
        stderr=stderr,
    )
    stage_remote_bench_input_files(
        spec,
        workspace,
        bench_file,
        operator_file,
        verbose=verbose,
        stderr=stderr,
    )


def _stage_remote_python_bundle(
    spec: RemoteSpec,
    workspace: str,
    *,
    verbose: bool,
    stderr: TextIO | None,
) -> None:
    stage_remote_python_bundle(
        [Path(__file__).resolve().with_name("run_bench_remote_worker.py")],
        workspace,
        lambda source, target: copy_file_to_remote(
            spec, source, target, verbose=verbose, stderr=stderr
        ),
    )
