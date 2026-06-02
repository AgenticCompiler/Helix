from __future__ import annotations
# pyright: reportPrivateUsage=false

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TextIO, cast

from bench_runner_deps import BenchRunnerDeps
from npu_affinity import NpuDevicePool, affinity_env_for_device
from perf_artifacts import (
    PerfCaseRecord,
    PerfMetrics,
    perf_output_path,
    render_perf_case_records_jsonl,
    write_perf_lines,
)
from run_runtime import RemoteSpec, ResultPayload, make_result, result_succeeded

_LOCAL_BENCH_OUTPUT_DIR_ENV = "TRITON_AGENT_BENCH_OUTPUT_DIR"
_PRESERVED_RUN_DIR_NONE_SENTINEL = "__NONE__"


def run_remote_bench_standalone(
    deps: BenchRunnerDeps,
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
    stderr: TextIO | None = None,
    force_recompile: bool = False,
    output: str | None = None,
) -> tuple[ResultPayload, Path | None, str]:
    for support_path in deps._standalone_runtime_support_paths():
        deps.copy_file_to_remote(
            spec,
            support_path,
            f"{remote_workspace}/{support_path.name}",
            verbose=verbose,
            stderr=stderr,
        )
    perf_path = _resolve_perf_output_path(operator_file, output=output)
    extra_env: dict[str, str] | None = {"TRITON_ALWAYS_COMPILE": "1"} if force_recompile else None
    with deps._stream_target_for_verbosity(verbose) as stream_target:
        result = deps.run_remote_command_streaming(
            spec,
            remote_workspace,
            [
                "python3",
                "-c",
                _build_remote_standalone_run_all_script(verbose=verbose),
                bench_file.name,
                operator_file.name,
                perf_path.name,
            ],
            stdout=stream_target,
            verbose=verbose,
            stderr=stderr,
            stall_timeout_seconds=deps._bench_timeout(),
            extra_env=extra_env,
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
    deps: BenchRunnerDeps,
    bench_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
    force_recompile: bool = False,
    output: str | None = None,
) -> tuple[ResultPayload, Path]:
    runtime = deps._load_standalone_runtime_module()
    return runtime.run_local_standalone_bench(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        bench_file, operator_file, verbose=verbose,  # pyright: ignore[reportUnknownVariableType, reportCallIssue]
        force_recompile=force_recompile,  # pyright: ignore[reportCallIssue]
        output=output,  # pyright: ignore[reportCallIssue]
    )


def run_local_bench_standalone_parallel(
    deps: BenchRunnerDeps,
    bench_file: Path,
    operator_file: Path,
    devices: tuple[str, ...],
    *,
    source_root: Path,
    json_search_root: Path,
    verbose: bool = False,
    force_recompile: bool = False,
    output: str | None = None,
) -> tuple[ResultPayload, Path]:
    runtime = deps._load_standalone_runtime_module()
    cases, _resolution = runtime.load_standalone_bench_cases(bench_file, operator_file)
    case_ids = [case.case_id for case in cases]
    pool = NpuDevicePool(devices)
    preserved_run_dir: Path | None = None
    create_preserved_run_dir = getattr(runtime, "create_local_preserved_profile_run_dir", None)
    if callable(create_preserved_run_dir):
        preserved_run_dir = cast(
            Path | None,
            create_preserved_run_dir(prefix="triton-agent-standalone-bench-"),
        )

    def _worker(case_id: str) -> PerfCaseRecord:
        case_workspace, cleanup = _create_local_standalone_case_workspace(
            deps,
            bench_file,
            operator_file,
            case_id,
            source_root=source_root,
            json_search_root=json_search_root,
            verbose=verbose,
        )
        try:
            with pool.acquire() as device:
                return _run_local_standalone_case_in_subprocess(
                    deps,
                    case_workspace,
                    bench_file,
                    operator_file,
                    case_id,
                    device,
                    preserved_run_dir=preserved_run_dir,
                    source_root=source_root,
                    verbose=verbose,
                    force_recompile=force_recompile,
                )
        finally:
            cleanup()

    case_records = deps._run_parallel_case_workers(case_ids, min(len(case_ids), len(devices)), _worker)
    deps._sort_case_records(case_records, case_ids)
    perf_path = _write_standalone_perf(operator_file, case_records, output=output)
    return _build_standalone_result(case_records), perf_path


def run_remote_bench_standalone_parallel(
    deps: BenchRunnerDeps,
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    devices: tuple[str, ...],
    *,
    source_root: Path,
    json_search_root: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
    force_recompile: bool = False,
    output: str | None = None,
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
        workspace_root = _stage_remote_standalone_case_workspace(
            deps,
            spec,
            bench_file,
            operator_file,
            case_workspace,
            source_root=source_root,
            json_search_root=json_search_root,
            verbose=verbose,
            stderr=stderr,
        )
        try:
            with pool.acquire() as device:
                return _run_remote_standalone_case(
                    deps,
                    spec,
                    workspace_root,
                    bench_file,
                    operator_file,
                    case_id,
                    device,
                    source_root=source_root,
                    verbose=verbose,
                    stderr=stderr,
                    force_recompile=force_recompile,
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
    perf_path = _write_standalone_perf(operator_file, case_records, output=output)
    return _build_standalone_result(case_records), perf_path, remote_workspace


def _build_remote_standalone_run_all_script(*, verbose: bool = False) -> str:
    return (
        "import pathlib, shutil, sys; "
        "import standalone_bench_runtime as runtime; "
        "bench_file = pathlib.Path(sys.argv[1]); "
        "operator_file = pathlib.Path(sys.argv[2]); "
        "target_path = pathlib.Path(sys.argv[3]); "
        f"result, perf_path = runtime.run_local_standalone_bench(bench_file, operator_file, verbose={verbose}); "
        "target_path.parent.mkdir(parents=True, exist_ok=True); "
        "shutil.copyfile(perf_path, target_path) if perf_path != target_path else None; "
        "raise SystemExit(int(result['return_code']))"
    )


def _build_standalone_run_one_case_script(*, verbose: bool = False) -> str:
    return (
        "import json, pathlib, sys; "
        "import standalone_bench_runtime as runtime; "
        "bench_file = pathlib.Path(sys.argv[1]); "
        "operator_file = pathlib.Path(sys.argv[2]); "
        "case_id = sys.argv[3]; "
        "preserved_run_dir_arg = sys.argv[4]; "
        f"preserved_run_dir = None if preserved_run_dir_arg == {_PRESERVED_RUN_DIR_NONE_SENTINEL!r} else pathlib.Path(preserved_run_dir_arg); "
        "record = runtime.run_one_standalone_case_record("
        "bench_file, operator_file, case_id, preserved_run_dir=preserved_run_dir, "
        f"verbose={verbose}"
        "); "
        "payload = {"
        "'case_label': record.case_label, "
        "'kernel_names': record.kernel_names, "
        "'kernel_source': record.kernel_source, "
        "'metrics': record.metrics, "
        "'error_message': record.error_message, "
        "'case_wall_clock_seconds': record.case_wall_clock_seconds"
        "}; "
        "print(json.dumps(payload, separators=(',', ':')))"
    )


def _create_local_standalone_case_workspace(
    deps: BenchRunnerDeps,
    bench_file: Path,
    operator_file: Path,
    case_id: str,
    *,
    source_root: Path,
    json_search_root: Path,
    verbose: bool = False,
) -> tuple[Path, Callable[[], None]]:
    return deps._create_local_case_workspace(
        prefix=f"triton-agent-standalone-case-{case_id}-",
        input_paths=deps._bench_case_input_paths(
            bench_file,
            operator_file,
            json_search_root=json_search_root,
        ),
        flat_input_paths=deps._standalone_runtime_support_paths(),
        source_root=source_root,
        verbose=verbose,
    )


def _run_local_standalone_case_in_subprocess(
    deps: BenchRunnerDeps,
    workspace_root: Path,
    bench_file: Path,
    operator_file: Path,
    case_id: str,
    device: str,
    *,
    preserved_run_dir: Path | None,
    source_root: Path,
    verbose: bool = False,
    force_recompile: bool = False,
) -> PerfCaseRecord:
    extra_env = affinity_env_for_device(device)
    configured_profile_root, _configured_env = deps._resolve_local_bench_profile_output_root()
    if configured_profile_root:
        extra_env[_LOCAL_BENCH_OUTPUT_DIR_ENV] = str(Path(configured_profile_root).expanduser().resolve())
    if force_recompile:
        extra_env["TRITON_ALWAYS_COMPILE"] = "1"
    result = deps.run_buffered_process(
        [
            deps.local_python_executable(),
            "-c",
            _build_standalone_run_one_case_script(verbose=verbose),
            deps._case_workspace_command_path(bench_file, source_root=source_root),
            deps._case_workspace_command_path(operator_file, source_root=source_root),
            case_id,
            (
                _PRESERVED_RUN_DIR_NONE_SENTINEL
                if preserved_run_dir is None
                else preserved_run_dir.resolve().as_posix()
            ),
        ],
        str(workspace_root),
        stall_timeout_seconds=deps._bench_timeout(),
        extra_env=extra_env,
    )
    if verbose and result["stderr"]:
        for line in str(result["stderr"]).strip().splitlines():
            print(f"[profiler] {case_id}: {line}", file=sys.stderr)
    return _parse_standalone_case_result_payload(
        result,
        case_id=case_id,
        fallback_kernel_source="metadata",
    )


def _stage_remote_standalone_case_workspace(
    deps: BenchRunnerDeps,
    spec: RemoteSpec,
    bench_file: Path,
    operator_file: Path,
    case_workspace: str,
    *,
    source_root: Path,
    json_search_root: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> str:
    return deps._stage_remote_case_workspace(
        spec,
        case_workspace,
        deps._bench_case_input_paths(
            bench_file,
            operator_file,
            json_search_root=json_search_root,
        ),
        source_root=source_root,
        flat_input_paths=deps._standalone_runtime_support_paths(),
        verbose=verbose,
        stderr=stderr,
    )


def _run_remote_standalone_case(
    deps: BenchRunnerDeps,
    spec: RemoteSpec,
    case_workspace: str,
    bench_file: Path,
    operator_file: Path,
    case_id: str,
    device: str,
    *,
    source_root: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
    force_recompile: bool = False,
) -> PerfCaseRecord:
    extra_env = affinity_env_for_device(device)
    if force_recompile:
        extra_env["TRITON_ALWAYS_COMPILE"] = "1"
    result = deps.run_remote_command_streaming(
        spec,
        case_workspace,
        [
            "python3",
            "-c",
            _build_standalone_run_one_case_script(verbose=verbose),
            deps._case_workspace_command_path(bench_file, source_root=source_root),
            deps._case_workspace_command_path(operator_file, source_root=source_root),
            case_id,
            _PRESERVED_RUN_DIR_NONE_SENTINEL,
        ],
        verbose=verbose,
        stderr=stderr,
        extra_env=extra_env,
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
            case_wall_clock_seconds=None,
        )
    stdout_text = str(result["stdout"]).strip()
    if not stdout_text:
        return PerfCaseRecord(
            case_label=case_id,
            kernel_names=[],
            kernel_source=fallback_kernel_source,
            error_message="standalone worker produced no JSON payload",
            case_wall_clock_seconds=None,
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
            case_wall_clock_seconds=None,
        )
    metrics_payload = parsed["metrics"]
    return PerfCaseRecord(
        case_label=str(parsed["case_label"]),
        kernel_names=[str(name) for name in parsed["kernel_names"]],
        kernel_source=str(parsed["kernel_source"]),
        metrics=None if metrics_payload is None else cast(PerfMetrics, metrics_payload),
        error_message=None if parsed["error_message"] is None else str(parsed["error_message"]),
        case_wall_clock_seconds=None if parsed["case_wall_clock_seconds"] is None else float(parsed["case_wall_clock_seconds"]),
    )


def _format_standalone_command_failure(result: ResultPayload) -> str:
    details = str(result["stderr"]).strip() or str(result["stdout"]).strip()
    prefix = f"standalone command failed with return code {int(result['return_code'])}"
    return f"{prefix}: {details}" if details else prefix


def _write_standalone_perf(
    operator_file: Path,
    case_records: list[PerfCaseRecord],
    output: str | None = None,
) -> Path:
    return write_perf_lines(
        _resolve_perf_output_path(operator_file, output=output),
        render_perf_case_records_jsonl(
            case_records,
            missing_kernel_match_error="no resolved kernels matched profiler operator details",
        ),
    )


def _resolve_perf_output_path(operator_file: Path, *, output: str | None = None) -> Path:
    if output is not None:
        return Path(output).expanduser().resolve()
    return perf_output_path(operator_file)


def _build_standalone_result(case_records: list[PerfCaseRecord]) -> ResultPayload:
    errors = [f"{record.case_label}: {record.error_message}" for record in case_records if record.error_message is not None]
    return make_result(
        return_code=1 if errors else 0,
        stdout="",
        stderr="\n".join(errors),
    )
