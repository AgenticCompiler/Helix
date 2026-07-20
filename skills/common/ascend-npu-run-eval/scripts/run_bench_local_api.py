"""Parent-process API for isolated local benchmark execution."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional, cast

from debug_device import maybe_print_visible_devices
from result_payload import ResultPayload, make_result
from run_runtime import (
    eval_timeout_seconds,
    local_python_executable,
    result_succeeded,
    run_buffered_process,
    run_streaming_process,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def run_local_bench(
    bench_file: Path,
    operator_file: Path,
    bench_mode: str,
    npu_devices: str | None = None,
    verbose: bool = False,
    output: str | None = None,
) -> tuple[ResultPayload, Path | None]:
    return run_local_bench_with_limits(
        bench_file,
        operator_file,
        bench_mode,
        npu_devices=npu_devices,
        verbose=verbose,
        output=output,
        warmup_cap=None,
        repeats_cap=None,
    )


def run_local_bench_with_limits(
    bench_file: Path,
    operator_file: Path,
    bench_mode: str,
    *,
    warmup_cap: int | None,
    repeats_cap: int | None,
    npu_devices: str | None = None,
    verbose: bool = False,
    output: str | None = None,
) -> tuple[ResultPayload, Path | None]:
    if (warmup_cap is None) != (repeats_cap is None):
        raise ValueError("Both execution limits are required together")
    return _run_local_bench_worker(
        bench_file,
        operator_file,
        bench_mode,
        npu_devices=npu_devices,
        verbose=verbose,
        output=output,
        warmup_cap=warmup_cap,
        repeats_cap=repeats_cap,
    )


def _run_local_bench_worker(
    bench_file: Path,
    operator_file: Path,
    bench_mode: str,
    *,
    npu_devices: str | None,
    verbose: bool,
    output: str | None,
    warmup_cap: int | None,
    repeats_cap: int | None,
) -> tuple[ResultPayload, Path | None]:
    maybe_print_visible_devices()
    with tempfile.TemporaryDirectory() as tmp:
        result_file = Path(tmp) / "local-bench-result.json"
        command = [
            local_python_executable(),
            str(SCRIPT_DIR / "run_bench_local_worker.py"),
            "--bench-file",
            str(bench_file.resolve()),
            "--operator-file",
            str(operator_file.resolve()),
            "--bench-mode",
            bench_mode,
            "--result-file",
            str(result_file),
        ]
        if npu_devices is not None:
            command.extend(["--npu-devices", npu_devices])
        if verbose:
            command.append("--verbose")
        if output is not None:
            command.extend(["--output", output])
        if warmup_cap is not None and repeats_cap is not None:
            command.extend(["--warmup-cap", str(warmup_cap), "--repeats-cap", str(repeats_cap)])
        run_process = run_streaming_process if verbose else run_buffered_process
        worker_result = run_process(
            command,
            str(bench_file.resolve().parent),
            stall_timeout_seconds=0,
            timeout_seconds=eval_timeout_seconds(),
        )
        if not result_succeeded(worker_result):
            return worker_result, None
        if not result_file.exists():
            return (
                make_result(
                    return_code=1,
                    stdout=str(worker_result["stdout"]),
                    stderr=str(worker_result["stderr"])
                    + f"Local benchmark worker did not write result payload: {result_file}",
                    stalled=bool(worker_result["stalled"]),
                    session_id=worker_result["session_id"],
                ),
                None,
            )
        return _read_payload(result_file)


def _read_payload(result_file: Path) -> tuple[ResultPayload, Path | None]:
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    raw_result = payload["result"]
    result = make_result(
        return_code=int(raw_result["return_code"]),
        stdout=str(raw_result["stdout"]),
        stderr=str(raw_result["stderr"]),
        stalled=bool(raw_result["stalled"]),
        session_id=cast(Optional[str], raw_result["session_id"]),
    )
    raw_perf_path = payload.get("perf_path")
    perf_path = None if raw_perf_path is None else Path(str(raw_perf_path)).expanduser().resolve()
    return result, perf_path
