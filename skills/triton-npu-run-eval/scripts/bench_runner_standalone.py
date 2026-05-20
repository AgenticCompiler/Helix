from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO, cast

from npu_affinity import NpuDevicePool, affinity_env_for_device
from perf_artifacts import (
    PerfCaseRecord,
    PerfMetrics,
    perf_output_path,
    render_perf_case_records,
    write_perf_lines,
)
from run_runtime import RemoteSpec, ResultPayload, make_result, result_succeeded


def run_remote_bench_standalone(
    deps: Any,
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[ResultPayload, Path | None, str]:
    for support_path in deps._standalone_runtime_support_paths():
        deps.copy_file_to_remote(
            spec,
            support_path,
            f"{remote_workspace}/{support_path.name}",
            verbose=verbose,
            stderr=stderr,
        )
    perf_path = perf_output_path(operator_file)
    with deps._stream_target_for_verbosity(False) as quiet_stdout:
        result = deps.run_remote_command_streaming(
            spec,
            remote_workspace,
            [
                "python3",
                "-c",
                _build_remote_standalone_run_all_script(),
                bench_file.name,
                operator_file.name,
                perf_path.name,
            ],
            stdout=quiet_stdout,
            verbose=verbose,
            stderr=stderr,
            stall_timeout_seconds=deps._bench_timeout(),
        )
    copied_perf_path: Path | None = None
    try:
        deps.copy_file_from_remote(
            spec,
            f"{remote_workspace}/{perf_path.name}",
            perf_path,
            verbose=verbose,
            stderr=stderr,
        )
        copied_perf_path = perf_path
    except RuntimeError:
        if result_succeeded(result):
            raise
    return result, copied_perf_path, remote_workspace


def run_local_standalone_bench(
    deps: Any,
    bench_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
) -> tuple[ResultPayload, Path]:
    runtime = deps._load_standalone_runtime_module()
    return runtime.run_local_standalone_bench(bench_file, operator_file, verbose=verbose)


def run_local_bench_standalone_parallel(
    deps: Any,
    bench_file: Path,
    operator_file: Path,
    devices: tuple[str, ...],
    *,
    verbose: bool = False,
) -> tuple[ResultPayload, Path]:
    runtime = deps._load_standalone_runtime_module()
    cases, _resolution = runtime.load_standalone_bench_cases(bench_file, operator_file)
    case_ids = [case.case_id for case in cases]
    pool = NpuDevicePool(devices)

    def _worker(case_id: str) -> PerfCaseRecord:
        case_workspace, cleanup = _create_local_standalone_case_workspace(deps, bench_file, operator_file, case_id)
        try:
            with pool.acquire() as device:
                return _run_local_standalone_case_in_subprocess(
                    deps,
                    case_workspace / bench_file.name,
                    case_workspace / operator_file.name,
                    case_id,
                    device,
                )
        finally:
            cleanup()

    case_records = deps._run_parallel_case_workers(case_ids, min(len(case_ids), len(devices)), _worker)
    deps._sort_case_records(case_records, case_ids)
    perf_path = _write_standalone_perf(operator_file, case_records)
    return _build_standalone_result(case_records), perf_path


def run_remote_bench_standalone_parallel(
    deps: Any,
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    devices: tuple[str, ...],
    *,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[ResultPayload, Path, str]:
    runtime = deps._load_standalone_runtime_module()
    cases, _resolution = runtime.load_standalone_bench_cases(bench_file, operator_file)
    case_ids = [case.case_id for case in cases]
    pool = NpuDevicePool(devices)

    def _worker(case_id: str) -> PerfCaseRecord:
        case_workspace = f"{remote_workspace}/case-{case_id}"
        deps.run_remote_command_buffered(
            spec,
            remote_workspace,
            ["mkdir", "-p", case_workspace],
            verbose=verbose,
            stderr=stderr,
        )
        _stage_remote_standalone_case_workspace(
            deps,
            spec,
            bench_file,
            operator_file,
            case_workspace,
            verbose=verbose,
            stderr=stderr,
        )
        try:
            with pool.acquire() as device:
                return _run_remote_standalone_case(
                    deps,
                    spec,
                    case_workspace,
                    bench_file.name,
                    operator_file.name,
                    case_id,
                    device,
                    verbose=verbose,
                    stderr=stderr,
                )
        finally:
            deps.run_remote_command_buffered(
                spec,
                remote_workspace,
                ["rm", "-rf", case_workspace],
                verbose=verbose,
                stderr=stderr,
            )

    case_records = deps._run_parallel_case_workers(case_ids, min(len(case_ids), len(devices)), _worker)
    deps._sort_case_records(case_records, case_ids)
    perf_path = _write_standalone_perf(operator_file, case_records)
    return _build_standalone_result(case_records), perf_path, remote_workspace


def _build_remote_standalone_run_all_script() -> str:
    return (
        "import pathlib, shutil, sys; "
        "import standalone_bench_runtime as runtime; "
        "bench_file = pathlib.Path(sys.argv[1]); "
        "operator_file = pathlib.Path(sys.argv[2]); "
        "target_path = pathlib.Path(sys.argv[3]); "
        "result, perf_path = runtime.run_local_standalone_bench(bench_file, operator_file); "
        "target_path.parent.mkdir(parents=True, exist_ok=True); "
        "shutil.copyfile(perf_path, target_path) if perf_path != target_path else None; "
        "raise SystemExit(int(result['return_code']))"
    )


def _build_standalone_run_one_case_script() -> str:
    return (
        "import json, pathlib, sys; "
        "import standalone_bench_runtime as runtime; "
        "bench_file = pathlib.Path(sys.argv[1]); "
        "operator_file = pathlib.Path(sys.argv[2]); "
        "case_id = sys.argv[3]; "
        "record = runtime.run_one_standalone_case_record(bench_file, operator_file, case_id); "
        "payload = {"
        "'case_label': record.case_label, "
        "'kernel_names': record.kernel_names, "
        "'kernel_source': record.kernel_source, "
        "'metrics': record.metrics, "
        "'error_message': record.error_message, "
        "'elapsed_seconds': record.elapsed_seconds"
        "}; "
        "print(json.dumps(payload, separators=(',', ':')))"
    )


def _create_local_standalone_case_workspace(
    deps: Any,
    bench_file: Path,
    operator_file: Path,
    case_id: str,
) -> tuple[Path, Any]:
    return deps._create_local_case_workspace(
        prefix=f"triton-agent-standalone-case-{case_id}-",
        input_paths=deps._bench_case_input_paths(
            bench_file,
            operator_file,
            support_paths=deps._standalone_runtime_support_paths(),
        ),
    )


def _run_local_standalone_case_in_subprocess(
    deps: Any,
    bench_file: Path,
    operator_file: Path,
    case_id: str,
    device: str,
) -> PerfCaseRecord:
    result = deps.run_buffered_process(
        [
            deps.local_python_executable(),
            "-c",
            _build_standalone_run_one_case_script(),
            bench_file.name,
            operator_file.name,
            case_id,
        ],
        str(bench_file.parent),
        stall_timeout_seconds=deps._bench_timeout(),
        extra_env=affinity_env_for_device(device),
    )
    return _parse_standalone_case_result_payload(
        result,
        case_id=case_id,
        fallback_kernel_source="metadata",
    )


def _stage_remote_standalone_case_workspace(
    deps: Any,
    spec: RemoteSpec,
    bench_file: Path,
    operator_file: Path,
    case_workspace: str,
    *,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> None:
    deps._stage_remote_case_workspace(
        spec,
        case_workspace,
        deps._bench_case_input_paths(
            bench_file,
            operator_file,
            support_paths=deps._standalone_runtime_support_paths(),
        ),
        verbose=verbose,
        stderr=stderr,
    )


def _run_remote_standalone_case(
    deps: Any,
    spec: RemoteSpec,
    case_workspace: str,
    bench_filename: str,
    operator_filename: str,
    case_id: str,
    device: str,
    *,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> PerfCaseRecord:
    result = deps.run_remote_command_streaming(
        spec,
        case_workspace,
        [
            "python3",
            "-c",
            _build_standalone_run_one_case_script(),
            bench_filename,
            operator_filename,
            case_id,
        ],
        verbose=verbose,
        stderr=stderr,
        extra_env=affinity_env_for_device(device),
        stall_timeout_seconds=deps._bench_timeout(),
    )
    return _parse_standalone_case_result_payload(
        result,
        case_id=case_id,
        fallback_kernel_source="metadata",
    )


def _parse_standalone_case_result_payload(
    result: ResultPayload,
    *,
    case_id: str,
    fallback_kernel_source: str,
) -> PerfCaseRecord:
    if not result_succeeded(result):
        return PerfCaseRecord(
            case_label=case_id,
            kernel_names=[],
            kernel_source=fallback_kernel_source,
            error_message=_format_standalone_command_failure(result),
            elapsed_seconds=None,
        )
    stdout_text = str(result["stdout"]).strip()
    if not stdout_text:
        return PerfCaseRecord(
            case_label=case_id,
            kernel_names=[],
            kernel_source=fallback_kernel_source,
            error_message="standalone worker produced no JSON payload",
            elapsed_seconds=None,
        )
    try:
        payload = stdout_text.splitlines()[-1].strip()
        parsed = json.loads(payload)
    except (IndexError, json.JSONDecodeError) as exc:
        return PerfCaseRecord(
            case_label=case_id,
            kernel_names=[],
            kernel_source=fallback_kernel_source,
            error_message=f"failed to parse standalone worker payload: {exc}",
            elapsed_seconds=None,
        )
    metrics_payload = parsed["metrics"]
    return PerfCaseRecord(
        case_label=str(parsed["case_label"]),
        kernel_names=[str(name) for name in parsed["kernel_names"]],
        kernel_source=str(parsed["kernel_source"]),
        metrics=None if metrics_payload is None else cast(PerfMetrics, metrics_payload),
        error_message=None if parsed["error_message"] is None else str(parsed["error_message"]),
        elapsed_seconds=None if parsed["elapsed_seconds"] is None else float(parsed["elapsed_seconds"]),
    )


def _format_standalone_command_failure(result: ResultPayload) -> str:
    details = str(result["stderr"]).strip() or str(result["stdout"]).strip()
    prefix = f"standalone command failed with return code {int(result['return_code'])}"
    return f"{prefix}: {details}" if details else prefix


def _write_standalone_perf(operator_file: Path, case_records: list[PerfCaseRecord]) -> Path:
    return write_perf_lines(
        perf_output_path(operator_file),
        render_perf_case_records(
            case_records,
            latency_prefix="latency",
            raw_prefix="raw-op-statistic",
            resolved_kernels_prefix="resolved-kernels",
            kernel_source_prefix="kernel-source",
            latency_error_prefix="latency-error",
            missing_kernel_match_error="no resolved kernels matched profiler operator details",
        ),
    )


def _build_standalone_result(case_records: list[PerfCaseRecord]) -> ResultPayload:
    errors = [f"{record.case_label}: {record.error_message}" for record in case_records if record.error_message is not None]
    return make_result(
        return_code=1 if errors else 0,
        stdout="",
        stderr="\n".join(errors),
    )
