from __future__ import annotations

import contextlib
import io
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from triton_agent.execution import run_local_bench, run_local_test, run_remote_bench, run_remote_test
from triton_agent.models import AgentResult
from triton_agent.optimize.baseline import load_baseline_state
from triton_agent.optimize.round_contract import inspect_round_artifacts
from triton_agent.optimize.status import inspect_optimize_status_workspace


Phase = Literal["all", "test", "bench"]


@dataclass(frozen=True)
class OptimizeVerifyOptions:
    phase: Phase = "all"
    test_mode: str | None = None
    bench_mode: str | None = None
    remote: str | None = None
    remote_workdir: str | None = None
    keep_remote_workdir: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class OptimizeVerifyTarget:
    workspace: Path
    selected_round: str
    round_dir: Path
    source_operator: Path
    verify_dir: Path
    copied_operator: Path
    source_test_file: Path
    test_file: Path
    test_mode: str
    source_bench_file: Path
    bench_file: Path
    bench_mode: str
    baseline_perf: Path


@dataclass(frozen=True)
class OptimizeVerifyResult:
    return_code: int
    verify_dir: Path
    state_path: Path


def prepare_optimize_verify_target(
    workspace: Path,
    *,
    timestamp_label: str | None = None,
) -> OptimizeVerifyTarget:
    status = inspect_optimize_status_workspace(workspace)
    if status.best_round is None:
        raise ValueError(f"No numeric best round available for workspace: {workspace}")

    round_number = status.best_round.removeprefix("round-")
    round_dir = workspace / f"opt-round-{round_number}"
    if not round_dir.is_dir():
        raise ValueError(f"Best round directory does not exist: {round_dir}")

    baseline_state = load_baseline_state(workspace)
    source_test_file = _resolve_workspace_file(workspace, baseline_state.test_file, label="test_file")
    source_bench_file = _resolve_workspace_file(workspace, baseline_state.bench_file, label="bench_file")
    baseline_perf = _resolve_workspace_file(
        workspace,
        baseline_state.perf_artifact,
        label="perf_artifact",
    )

    round_artifacts = inspect_round_artifacts(round_dir)
    if round_artifacts.operator_path is None:
        raise ValueError(f"Best round is missing round-local operator output: {round_dir}")

    verify_dir = _create_unique_verify_dir(workspace, timestamp_label=timestamp_label)
    copied_operator = verify_dir / round_artifacts.operator_path.name
    shutil.copy2(round_artifacts.operator_path, copied_operator)
    test_file = _copy_verify_input(source_test_file, verify_dir)
    bench_file = _copy_verify_input(source_bench_file, verify_dir)

    return OptimizeVerifyTarget(
        workspace=workspace,
        selected_round=status.best_round,
        round_dir=round_dir,
        source_operator=round_artifacts.operator_path,
        verify_dir=verify_dir,
        copied_operator=copied_operator,
        source_test_file=source_test_file,
        test_file=test_file,
        test_mode=baseline_state.test_mode,
        source_bench_file=source_bench_file,
        bench_file=bench_file,
        bench_mode=baseline_state.bench_mode,
        baseline_perf=baseline_perf,
    )


def run_optimize_verify(
    target: OptimizeVerifyTarget,
    options: OptimizeVerifyOptions,
) -> OptimizeVerifyResult:
    test_entry: dict[str, object] | None = None
    bench_entry: dict[str, object] | None = None
    compare_entry: dict[str, object] | None = None
    archived_result: Path | None = None
    perf_path: Path | None = None
    return_code = 0

    if options.phase in {"all", "test"}:
        test_mode = options.test_mode or target.test_mode
        test_result, archived_result = _run_test(target, options, test_mode)
        _write_result_log(target.verify_dir / "test.log", test_result)
        test_entry = {
            "mode": test_mode,
            "return_code": test_result.return_code,
            "archived_result": _relative_or_none(target.workspace, archived_result),
            "log": _relative_path(target.workspace, target.verify_dir / "test.log"),
        }
        return_code = test_result.return_code
        if return_code != 0:
            state_path = _write_verify_state(
                target,
                test_entry=test_entry,
                bench_entry=bench_entry,
                compare_entry=compare_entry,
            )
            return OptimizeVerifyResult(
                return_code=return_code,
                verify_dir=target.verify_dir,
                state_path=state_path,
            )

    if options.phase in {"all", "bench"}:
        bench_mode = options.bench_mode or target.bench_mode
        bench_result, perf_path = _run_bench(target, options, bench_mode)
        _write_result_log(target.verify_dir / "bench.log", bench_result)
        bench_entry = {
            "mode": bench_mode,
            "return_code": bench_result.return_code,
            "perf_path": _relative_or_none(target.workspace, perf_path),
            "log": _relative_path(target.workspace, target.verify_dir / "bench.log"),
        }
        return_code = bench_result.return_code
        if return_code == 0 and perf_path is not None:
            compare_output = io.StringIO()
            with contextlib.redirect_stdout(compare_output):
                compare_code = compare_perf_files(target.baseline_perf, perf_path)
            compare_log = target.verify_dir / "compare-perf.txt"
            compare_log.write_text(compare_output.getvalue(), encoding="utf-8")
            compare_entry = {
                "return_code": compare_code,
                "output": _relative_path(target.workspace, compare_log),
            }
            return_code = compare_code

    state_path = _write_verify_state(
        target,
        test_entry=test_entry,
        bench_entry=bench_entry,
        compare_entry=compare_entry,
    )
    return OptimizeVerifyResult(
        return_code=return_code,
        verify_dir=target.verify_dir,
        state_path=state_path,
    )


def _resolve_workspace_file(workspace: Path, relative_path: str, *, label: str) -> Path:
    path = workspace / Path(relative_path)
    if not path.is_file():
        raise ValueError(f"Missing {label} path from baseline/state.json: {relative_path}")
    return path


def _create_unique_verify_dir(workspace: Path, *, timestamp_label: str | None = None) -> Path:
    label = timestamp_label or datetime.now().strftime("%Y%m%d-%H%M%S")
    root = workspace / "opt-verify"
    root.mkdir(exist_ok=True)
    candidate = root / f"verify-{label}"
    if not candidate.exists():
        candidate.mkdir()
        return candidate

    suffix = 2
    while True:
        suffixed = root / f"verify-{label}-{suffix}"
        if not suffixed.exists():
            suffixed.mkdir()
            return suffixed
        suffix += 1


def _run_test(
    target: OptimizeVerifyTarget,
    options: OptimizeVerifyOptions,
    test_mode: str,
) -> tuple[AgentResult, Path | None]:
    if options.remote is None:
        return run_local_test(target.test_file, target.copied_operator, test_mode)
    result, archived_result, _remote_workspace = run_remote_test(
        target.test_file,
        target.copied_operator,
        test_mode,
        options.remote,
        options.remote_workdir,
        keep_remote_workdir=options.keep_remote_workdir,
        verbose=options.verbose,
    )
    return result, archived_result


def _run_bench(
    target: OptimizeVerifyTarget,
    options: OptimizeVerifyOptions,
    bench_mode: str,
) -> tuple[AgentResult, Path | None]:
    if options.remote is None:
        return run_local_bench(target.bench_file, target.copied_operator, bench_mode)
    result, perf_path, _remote_workspace = run_remote_bench(
        target.bench_file,
        target.copied_operator,
        bench_mode,
        options.remote,
        options.remote_workdir,
        keep_remote_workdir=options.keep_remote_workdir,
        verbose=options.verbose,
    )
    return result, perf_path


def _write_result_log(path: Path, result: AgentResult) -> None:
    path.write_text(f"{result.stdout}{result.stderr}", encoding="utf-8")


def _write_verify_state(
    target: OptimizeVerifyTarget,
    *,
    test_entry: dict[str, object] | None,
    bench_entry: dict[str, object] | None,
    compare_entry: dict[str, object] | None,
) -> Path:
    state = {
        "selected_round": target.selected_round,
        "round_dir": _relative_path(target.workspace, target.round_dir),
        "source_operator": _relative_path(target.workspace, target.source_operator),
        "copied_operator": _relative_path(target.workspace, target.copied_operator),
        "source_test_file": _relative_path(target.workspace, target.source_test_file),
        "test_file": _relative_path(target.workspace, target.test_file),
        "test_mode": target.test_mode,
        "source_bench_file": _relative_path(target.workspace, target.source_bench_file),
        "bench_file": _relative_path(target.workspace, target.bench_file),
        "bench_mode": target.bench_mode,
        "baseline_perf": _relative_path(target.workspace, target.baseline_perf),
        "test": test_entry,
        "bench": bench_entry,
        "compare_perf": compare_entry,
    }
    state_path = target.verify_dir / "verify-state.json"
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return state_path


def _relative_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _relative_or_none(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    return _relative_path(root, path)


def _copy_verify_input(source: Path, verify_dir: Path) -> Path:
    target = verify_dir / source.name
    if target.exists():
        raise ValueError(f"Verification input filename collision: {source.name}")
    shutil.copy2(source, target)
    return target


def compare_perf_files(baseline_perf: Path, compare_perf: Path) -> int:
    from triton_agent.commands.comparison import compare_perf_files as compare_perf_files_impl

    return compare_perf_files_impl(baseline_perf, compare_perf)
