from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, TextIO, cast

from triton_agent.models import AgentResult
from triton_agent.skill_loader import load_operator_eval_script_module

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


class TestRunnerModule(Protocol):
    def run_local_test(
        self,
        test_file: Path,
        operator_file: Path,
        test_mode: str,
        *,
        verbose: bool = False,
    ) -> tuple[_RunSkillPayload, Path | None]: ...

    def run_remote_test(
        self,
        test_file: Path,
        operator_file: Path,
        test_mode: str,
        remote: str,
        remote_workdir: str | None,
        keep_remote_workdir: bool = False,
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> tuple[_RunSkillPayload, Path | None, str]: ...

    def parse_test_metadata(self, test_file: Path) -> dict[str, str]: ...


class BenchRunnerModule(Protocol):
    def run_local_bench(
        self,
        bench_file: Path,
        operator_file: Path,
        bench_mode: str,
        npu_devices: str | None = None,
        verbose: bool = False,
        output: str | None = None,
    ) -> tuple[_RunSkillPayload, Path | None]: ...

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
    ) -> tuple[_RunSkillPayload, Path | None, str]: ...

    def parse_bench_metadata(self, bench_file: Path) -> dict[str, str]: ...


def _load_test_runner() -> TestRunnerModule:
    return cast(TestRunnerModule, load_operator_eval_script_module("test_runner"))


def _load_bench_runner() -> BenchRunnerModule:
    return cast(BenchRunnerModule, load_operator_eval_script_module("bench_runner"))


def run_local_test(
    test_file: Path,
    operator_file: Path,
    test_mode: str,
    *,
    verbose: bool = False,
) -> tuple[AgentResult, Path | None]:
    result, archived = _load_test_runner().run_local_test(
        test_file,
        operator_file,
        test_mode,
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
    keep_remote_workdir: bool = False,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[AgentResult, Path | None, str]:
    result, archived, remote_workspace = _load_test_runner().run_remote_test(
        test_file,
        operator_file,
        test_mode,
        remote,
        remote_workdir,
        keep_remote_workdir=keep_remote_workdir,
        verbose=verbose,
        stderr=stderr,
    )
    return _normalize_agent_result(result), archived, remote_workspace


def parse_test_metadata(test_file: Path) -> dict[str, str]:
    return _load_test_runner().parse_test_metadata(test_file)


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
    result, perf_path = _load_bench_runner().run_local_bench(
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
    result, perf_path, remote_workspace = _load_bench_runner().run_remote_bench(
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


def parse_bench_metadata(bench_file: Path) -> dict[str, str]:
    return _load_bench_runner().parse_bench_metadata(bench_file)


def resolve_bench_mode_from_metadata(bench_file: Path) -> str:
    metadata = parse_bench_metadata(bench_file)
    mode = metadata.get("bench-mode")
    if mode not in {"standalone", "msprof"}:
        raise ValueError(f"Benchmark metadata is missing required 'bench-mode' entry: {bench_file}")
    return str(mode)
