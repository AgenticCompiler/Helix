from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, TextIO, cast

from helix.models import AgentResult
from helix.skill_bridges import (
    run_eval_bench,
    run_eval_probe,
    run_eval_profile,
    run_eval_simulator,
    run_eval_test,
)

_RunSkillPayload = Mapping[str, object]


def _normalize_agent_result(result: AgentResult | _RunSkillPayload) -> AgentResult:
    if isinstance(result, AgentResult):
        return result
    payload = result
    required_keys = ("return_code", "stdout", "stderr")
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        raise ValueError(
            "Run skill result payload is missing required keys: "
            + ", ".join(sorted(missing_keys))
        )
    session_id = payload.get("session_id")
    return AgentResult(
        return_code=int(str(payload["return_code"])),
        stdout=str(payload["stdout"]),
        stderr=str(payload["stderr"]),
        stalled=bool(payload.get("stalled", False)),
        session_id=None if session_id is None else str(session_id),
    )


def run_local_test(
    test_file: Path,
    operator_file: Path,
    test_mode: str,
    *,
    case_id: str | None = None,
    accuracy_mode: str | None = None,
    extra_env: Mapping[str, str] | None = None,
    verbose: bool = False,
) -> tuple[AgentResult, Path | None]:
    result, archived = run_eval_test.run_local_test(
        test_file,
        operator_file,
        test_mode,
        case_id=case_id,
        accuracy_mode=accuracy_mode,
        extra_env=extra_env,
        verbose=verbose,
    )
    return _normalize_agent_result(result), archived


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
) -> tuple[AgentResult, Path | None, str]:
    result, archived, remote_workspace = run_eval_test.run_remote_test(
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
    return _normalize_agent_result(result), archived, remote_workspace


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
) -> tuple[AgentResult, str]:
    result, remote_workspace = run_eval_test.run_remote_differential_comparison(
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
    return _normalize_agent_result(result), remote_workspace


def run_local_test_case_payload(
    test_file: Path,
    operator_file: Path,
    *,
    case_id: str,
    accuracy_mode: str | None = None,
    verbose: bool = False,
) -> tuple[AgentResult, object | None]:
    result, payload = run_eval_test.run_local_test_case_payload(
        test_file,
        operator_file,
        case_id=case_id,
        accuracy_mode=accuracy_mode,
        verbose=verbose,
    )
    return _normalize_agent_result(result), payload


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
) -> tuple[AgentResult, object | None, str]:
    result, payload, remote_workspace = run_eval_test.run_remote_test_case_payload(
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
    return _normalize_agent_result(result), payload, remote_workspace


def parse_test_metadata(test_file: Path) -> dict[str, str]:
    return run_eval_test.parse_test_metadata(test_file)


def resolve_test_mode_from_metadata(test_file: Path) -> str:
    metadata = parse_test_metadata(test_file)
    mode = metadata.get("test-mode")
    if mode not in {"standalone", "differential"}:
        raise ValueError(f"Test metadata is missing required 'test-mode' entry: {test_file}")
    return str(mode)


def run_local_bench(
    bench_file: Path,
    operator_file: Path,
    bench_mode: str,
    npu_devices: str | None = None,
    verbose: bool = False,
    output: str | None = None,
) -> tuple[AgentResult, Path | None]:
    result, perf_path = run_eval_bench.run_local_bench(
        bench_file, operator_file, bench_mode, npu_devices, verbose=verbose,
        output=output,
    )
    return _normalize_agent_result(result), perf_path


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
) -> tuple[AgentResult, Path | None, str]:
    result, perf_path, remote_workspace = run_eval_bench.run_remote_bench(
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
    return _normalize_agent_result(result), perf_path, remote_workspace


def run_local_simulator(
    bench_file: Path,
    operator_file: Path,
    *,
    case_id: str | None = None,
    kernel_name: str | None = None,
) -> AgentResult:
    result = run_eval_simulator.run_local_simulator(
        bench_file,
        operator_file,
        case_id=case_id,
        kernel_name=kernel_name,
    )
    return _normalize_agent_result(result)


def run_local_profile_bench(
    bench_file: Path,
    operator_file: Path,
    case_id: str | None = None,
    kernel_name: str | None = None,
) -> tuple[AgentResult, Path | None]:
    result, profile_dir = run_eval_profile.run_local_profile_bench(
        bench_file, operator_file, case_id=case_id, kernel_name=kernel_name
    )
    return _normalize_agent_result(result), profile_dir


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
) -> tuple[AgentResult, Path | None, str]:
    result, profile_dir, workspace = run_eval_profile.run_remote_profile_bench(
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
    return _normalize_agent_result(result), profile_dir, workspace


def parse_bench_metadata(bench_file: Path) -> dict[str, str]:
    return run_eval_bench.parse_bench_metadata(bench_file)


def resolve_bench_mode_default() -> str:
    return "torch-npu-profiler"


class ProbeBenchResult(Protocol):
    return_code: int
    default_lines: list[str]
    verbose_lines: list[str]
    warnings: list[str]


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
    return cast(ProbeBenchResult, run_eval_probe.run_local_probe_bench(
        bench_file,
        operator_file,
        baseline_operator_file,
        bench_mode,
        metric_source=metric_source,
        npu_devices=npu_devices,
        verbose=verbose,
    ))


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
    return cast(ProbeBenchResult, run_eval_probe.run_remote_probe_bench(
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
    ))
