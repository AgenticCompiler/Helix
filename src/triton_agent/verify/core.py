from __future__ import annotations

import contextlib
import io
import json
import math
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol, cast

from triton_agent.eval.runners import run_local_bench, run_local_test, run_remote_bench, run_remote_test
from triton_agent.models import AgentResult
from triton_agent.optimize.baseline import load_baseline_state
from triton_agent.optimize.pt_cleanup import cleanup_dir_pt_files
from triton_agent.optimize.round_contract import inspect_round_artifacts, load_round_state
from triton_agent.status.core import inspect_optimize_status_workspace
from triton_agent.status.models import OptimizeStatusWorkspace
from triton_agent.skills.loader import load_operator_eval_script_module


Phase = Literal["all", "test", "bench"]
_CONSISTENCY_ABS_TOLERANCE = 0.2


class BenchPerfParserModule(Protocol):
    def parse_perf_file(self, path: Path) -> dict[str, float]: ...

    def parse_required_perf_file(
        self,
        path: Path,
        required_latency_ids: Iterable[str],
    ) -> dict[str, float]: ...

    def parse_perf_file_for_metric_source(
        self,
        path: Path,
        *,
        metric_source: str = "auto",
    ) -> dict[str, float]: ...

    def parse_required_perf_file_for_metric_source(
        self,
        path: Path,
        required_latency_ids: Iterable[str],
        *,
        metric_source: str = "auto",
    ) -> dict[str, float]: ...

    def parse_perf_pair_for_comparison(
        self,
        baseline_perf: Path,
        compare_perf: Path,
        *,
        metric_source: str = "auto",
    ) -> tuple[dict[str, float], dict[str, float], dict[str, str]]: ...


@dataclass(frozen=True)
class VerifyOptions:
    phase: Phase = "all"
    test_mode: str | None = None
    bench_mode: str | None = None
    remote: str | None = None
    remote_workdir: str | None = None
    keep_remote_workdir: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class VerifyTarget:
    workspace: Path
    selected_round: str
    round_dir: Path
    effective_metric_source: str
    source_operator: Path
    source_baseline_operator: Path
    verify_dir: Path
    copied_operator: Path
    baseline_operator: Path
    source_test_file: Path
    test_file: Path
    test_mode: str
    source_bench_file: Path
    bench_file: Path
    bench_mode: str
    baseline_perf: Path
    optimize_status: OptimizeStatusWorkspace


@dataclass(frozen=True)
class VerifyResult:
    return_code: int
    verify_dir: Path
    state_path: Path


def prepare_verify_target(
    workspace: Path,
    *,
    timestamp_label: str | None = None,
) -> VerifyTarget:
    status = inspect_optimize_status_workspace(workspace)
    if status.best_round is None:
        raise ValueError(f"No numeric best round available for workspace: {workspace}")

    round_number = status.best_round.removeprefix("round-")
    round_dir = workspace / f"opt-round-{round_number}"
    if not round_dir.is_dir():
        raise ValueError(f"Best round directory does not exist: {round_dir}")

    baseline_state = load_baseline_state(workspace)
    baseline_state_dir = workspace / "baseline"
    source_baseline_operator = _resolve_state_file(
        baseline_state_dir,
        workspace,
        baseline_state.baseline_operator,
        label="baseline_operator",
    )
    source_test_file = _resolve_state_file(
        baseline_state_dir,
        workspace,
        baseline_state.test_file,
        label="test_file",
    )
    source_bench_file = _resolve_state_file(
        baseline_state_dir,
        workspace,
        baseline_state.bench_file,
        label="bench_file",
    )
    baseline_perf = _resolve_state_file(
        baseline_state_dir,
        workspace,
        baseline_state.perf_artifact,
        label="perf_artifact",
    )

    round_artifacts = inspect_round_artifacts(round_dir)
    if round_artifacts.operator_path is None:
        raise ValueError(f"Best round is missing round-local operator output: {round_dir}")
    round_state = load_round_state(round_dir)
    if round_state.effective_metric_source is None:
        raise ValueError(f"Best round is missing effective_metric_source: {round_dir}")

    verify_dir = _create_unique_verify_dir(workspace, timestamp_label=timestamp_label)
    copied_operator = verify_dir / round_artifacts.operator_path.name
    baseline_operator = verify_dir / f"baseline_{source_baseline_operator.name}"
    shutil.copy2(round_artifacts.operator_path, copied_operator)
    shutil.copy2(source_baseline_operator, baseline_operator)
    test_file = _copy_verify_input(source_test_file, verify_dir)
    bench_file = _copy_verify_input(source_bench_file, verify_dir)

    return VerifyTarget(
        workspace=workspace,
        selected_round=status.best_round,
        round_dir=round_dir,
        effective_metric_source=round_state.effective_metric_source,
        source_operator=round_artifacts.operator_path,
        source_baseline_operator=source_baseline_operator,
        verify_dir=verify_dir,
        copied_operator=copied_operator,
        baseline_operator=baseline_operator,
        source_test_file=source_test_file,
        test_file=test_file,
        test_mode=baseline_state.test_mode,
        source_bench_file=source_bench_file,
        bench_file=bench_file,
        bench_mode=baseline_state.bench_mode,
        baseline_perf=baseline_perf,
        optimize_status=status,
    )


def run_verify(
    target: VerifyTarget,
    options: VerifyOptions,
) -> VerifyResult:
    test_mode = options.test_mode or target.test_mode
    bench_mode = options.bench_mode or target.bench_mode
    test_entry: dict[str, object] | None = None
    baseline_bench_entry: dict[str, object] | None = None
    best_bench_entry: dict[str, object] | None = None
    compare_entry: dict[str, object] | None = None
    archived_result: Path | None = None
    baseline_perf_path: Path | None = None
    best_perf_path: Path | None = None
    return_code = 0

    if options.phase in {"all", "test"}:
        test_result, archived_result = _run_test(target, options, test_mode)
        _write_result_log(target.verify_dir / "test.log", test_result)
        test_entry = {
            "status": _execution_status(test_result.return_code),
            "return_code": test_result.return_code,
            "log": _relative_path(target.workspace, target.verify_dir / "test.log"),
            "result_artifact": _relative_or_none(target.workspace, archived_result),
        }
        return_code = test_result.return_code
        if return_code != 0:
            state_path = _write_verify_state(
                target,
                test_mode=test_mode,
                bench_mode=bench_mode,
                test_entry=test_entry,
                baseline_bench_entry=baseline_bench_entry,
                best_bench_entry=best_bench_entry,
                compare_entry=compare_entry,
                baseline_perf_path=baseline_perf_path,
                best_perf_path=best_perf_path,
            )
            return VerifyResult(
                return_code=return_code,
                verify_dir=target.verify_dir,
                state_path=state_path,
            )

    if options.phase in {"all", "bench"}:
        baseline_bench_result, baseline_perf_path = _run_bench(
            target,
            options,
            bench_mode,
            target.baseline_operator,
        )
        baseline_bench_log = target.verify_dir / "rerun-baseline-bench.log"
        _write_result_log(baseline_bench_log, baseline_bench_result)
        baseline_bench_entry = {
            "status": _execution_status(baseline_bench_result.return_code),
            "return_code": baseline_bench_result.return_code,
            "log": _relative_path(target.workspace, baseline_bench_log),
            "perf_artifact": _relative_or_none(target.workspace, baseline_perf_path),
        }
        return_code = baseline_bench_result.return_code
        if return_code == 0 and baseline_perf_path is not None:
            best_bench_result, best_perf_path = _run_bench(
                target,
                options,
                bench_mode,
                target.copied_operator,
            )
            best_bench_log = target.verify_dir / "rerun-best-bench.log"
            _write_result_log(best_bench_log, best_bench_result)
            best_bench_entry = {
                "status": _execution_status(best_bench_result.return_code),
                "return_code": best_bench_result.return_code,
                "log": _relative_path(target.workspace, best_bench_log),
                "perf_artifact": _relative_or_none(target.workspace, best_perf_path),
            }
            return_code = best_bench_result.return_code
        if return_code == 0 and baseline_perf_path is not None and best_perf_path is not None:
            compare_output = io.StringIO()
            with contextlib.redirect_stdout(compare_output):
                compare_code = compare_perf_files(
                    baseline_perf_path,
                    best_perf_path,
                    metric_source=_metric_source_for_verification(target.effective_metric_source),
                )
            compare_log = target.verify_dir / "compare-perf.txt"
            compare_log.write_text(compare_output.getvalue(), encoding="utf-8")
            compare_entry = {
                "status": _execution_status(compare_code),
                "return_code": compare_code,
                "log": _relative_path(target.workspace, compare_log),
            }
            return_code = compare_code

    state_path = _write_verify_state(
        target,
        test_mode=test_mode,
        bench_mode=bench_mode,
        test_entry=test_entry,
        baseline_bench_entry=baseline_bench_entry,
        best_bench_entry=best_bench_entry,
        compare_entry=compare_entry,
        baseline_perf_path=baseline_perf_path,
        best_perf_path=best_perf_path,
    )
    cleanup_dir_pt_files(target.verify_dir)
    return VerifyResult(
        return_code=return_code,
        verify_dir=target.verify_dir,
        state_path=state_path,
    )


def _resolve_state_file(state_dir: Path, workspace: Path, relative_path: str, *, label: str) -> Path:
    declared = Path(relative_path)
    path = state_dir / declared
    if path.is_file():
        return path
    path = workspace / declared
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
    target: VerifyTarget,
    options: VerifyOptions,
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
    target: VerifyTarget,
    options: VerifyOptions,
    bench_mode: str,
    operator_file: Path,
) -> tuple[AgentResult, Path | None]:
    if options.remote is None:
        return run_local_bench(target.bench_file, operator_file, bench_mode)
    result, perf_path, _remote_workspace = run_remote_bench(
        target.bench_file,
        operator_file,
        bench_mode,
        options.remote,
        options.remote_workdir,
        keep_remote_workdir=options.keep_remote_workdir,
        verbose=options.verbose,
    )
    return result, perf_path


def _write_result_log(path: Path, result: AgentResult) -> None:
    path.write_text(f"{result.stdout}{result.stderr}", encoding="utf-8")


def _execution_status(return_code: int) -> str:
    return "passed" if return_code == 0 else "failed"


def _write_verify_state(
    target: VerifyTarget,
    *,
    test_mode: str,
    bench_mode: str,
    test_entry: dict[str, object] | None,
    baseline_bench_entry: dict[str, object] | None,
    best_bench_entry: dict[str, object] | None,
    compare_entry: dict[str, object] | None,
    baseline_perf_path: Path | None,
    best_perf_path: Path | None,
) -> Path:
    speedup_state = _build_speedup_state(target, baseline_perf_path, best_perf_path)
    state = {
        "selection": _build_selection_state(target),
        "workspace": _build_workspace_state(target),
        "inputs": _build_inputs_state(target, test_mode=test_mode, bench_mode=bench_mode),
        "verify-result": {
            "test": test_entry,
            "rerun_baseline_bench": _with_latency(
                baseline_bench_entry,
                _build_latency_state(baseline_perf_path),
            ),
            "rerun_best_bench": _with_latency(
                best_bench_entry,
                _build_latency_state(best_perf_path),
            ),
            "compare_perf": compare_entry,
            "speedup": speedup_state,
            "consistency": _build_consistency_state(target.optimize_status, speedup_state),
        },
    }
    state_path = target.verify_dir / "verify-state.json"
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return state_path


def _build_selection_state(target: VerifyTarget) -> dict[str, object]:
    return {
        "round_dir": _relative_path(target.workspace, target.round_dir),
        "source_operator": _relative_path(target.workspace, target.source_operator),
        "numeric_best_source": "optimize-status",
        "optimize_status": _build_optimize_status_state(target.optimize_status),
    }


def _build_workspace_state(target: VerifyTarget) -> dict[str, object]:
    return {
        "verify_dir": _relative_path(target.workspace, target.verify_dir),
        "operator": _relative_path(target.workspace, target.copied_operator),
        "baseline_operator": _relative_path(target.workspace, target.baseline_operator),
    }


def _build_inputs_state(
    target: VerifyTarget,
    *,
    test_mode: str,
    bench_mode: str,
) -> dict[str, object]:
    return {
        "test_harness": {
            "source": _relative_path(target.workspace, target.source_test_file),
            "copied": _relative_path(target.workspace, target.test_file),
            "mode": test_mode,
        },
        "bench_harness": {
            "source": _relative_path(target.workspace, target.source_bench_file),
            "copied": _relative_path(target.workspace, target.bench_file),
            "mode": bench_mode,
        },
        "baseline_perf": _relative_path(target.workspace, target.baseline_perf),
    }


def _build_optimize_status_state(status: OptimizeStatusWorkspace) -> dict[str, object]:
    return {
        "state": status.state,
        "avg_improvement": status.avg_improvement,
        "geomean_speedup": status.geomean_speedup,
        "warnings": list(status.warnings),
    }


def _build_speedup_state(
    target: VerifyTarget,
    baseline_perf_path: Path | None,
    best_perf_path: Path | None,
) -> dict[str, object]:
    missing_perf_warnings: list[str] = []
    if baseline_perf_path is None:
        missing_perf_warnings.append("missing rerun baseline perf data")
    if best_perf_path is None:
        missing_perf_warnings.append("missing rerun best perf data")
    if missing_perf_warnings:
        return (
            {
                "avg_improvement": None,
                "geomean_speedup": None,
                "warnings": missing_perf_warnings,
            }
        )

    warnings: list[str] = []
    try:
        parser = _load_bench_perf_parser()
        resolved_baseline_perf_path = cast(Path, baseline_perf_path)
        resolved_best_perf_path = cast(Path, best_perf_path)
        metric_source = _metric_source_for_verification(target.effective_metric_source)
        if metric_source == "auto":
            baseline_values, verify_values, _comparison_modes = parser.parse_perf_pair_for_comparison(
                resolved_baseline_perf_path,
                resolved_best_perf_path,
                metric_source=metric_source,
            )
        else:
            baseline_values = parser.parse_perf_file_for_metric_source(
                resolved_baseline_perf_path,
                metric_source=metric_source,
            )
            verify_values = parser.parse_required_perf_file_for_metric_source(
                resolved_best_perf_path,
                baseline_values,
                metric_source=metric_source,
            )
    except (OSError, ValueError) as exc:
        return (
            {
                "avg_improvement": None,
                "geomean_speedup": None,
                "warnings": [str(exc)],
            }
        )

    latency_ids = sorted(baseline_values)
    improvement_values: list[float] = []
    speedup_values: list[float] = []
    valid_baseline_values: list[float] = []
    valid_verify_values: list[float] = []
    for latency_id in latency_ids:
        baseline_value = baseline_values[latency_id]
        verify_value = verify_values[latency_id]
        if baseline_value <= 0:
            warnings.append(f"baseline latency must be > 0 for {latency_id}")
            continue
        if verify_value <= 0:
            warnings.append(f"best latency must be > 0 for {latency_id}")
            continue
        improvement_values.append((baseline_value - verify_value) / baseline_value)
        speedup_values.append(baseline_value / verify_value)
        valid_baseline_values.append(baseline_value)
        valid_verify_values.append(verify_value)

    if not improvement_values:
        return (
            {
                "avg_improvement": None,
                "geomean_speedup": None,
                "warnings": warnings or ["missing comparable verify perf data"],
            }
        )

    return {
        "avg_improvement": _mean_value(improvement_values),
        "geomean_speedup": _geomean_value(speedup_values),
        "warnings": warnings,
    }


def _build_latency_state(perf_path: Path | None) -> dict[str, float] | None:
    if perf_path is None:
        return None
    try:
        parser = _load_bench_perf_parser()
        return dict(parser.parse_perf_file(perf_path))
    except (OSError, ValueError):
        return None


def _metric_source_for_verification(effective_metric_source: str) -> str:
    if effective_metric_source == "kernel":
        return "kernel"
    if effective_metric_source == "total-op":
        return "total-op"
    return "auto"


def _build_consistency_state(
    optimize_status: OptimizeStatusWorkspace,
    speedup_state: dict[str, object],
) -> dict[str, object]:
    geomean_delta = _metric_delta(
        speedup_state.get("geomean_speedup"),
        optimize_status.geomean_speedup,
    )
    avg_delta = _metric_delta(speedup_state.get("avg_improvement"), optimize_status.avg_improvement)
    decision_deltas = (geomean_delta,)
    warnings_value = speedup_state.get("warnings")
    has_warnings = isinstance(warnings_value, list) and len(cast(list[object], warnings_value)) > 0
    if (
        optimize_status.state == "ok"
        and not has_warnings
        and all(delta is not None for delta in decision_deltas)
    ):
        status = (
            "matched"
            if all(abs(cast(float, delta)) <= _CONSISTENCY_ABS_TOLERANCE + 1e-12 for delta in decision_deltas)
            else "mismatched"
        )
    else:
        status = "incomplete"
    return {
        "status": status,
        "geomean_speedup_delta": geomean_delta,
        "avg_improvement_delta": avg_delta,
    }


def _with_latency(
    bench_entry: dict[str, object] | None,
    latency: dict[str, float] | None,
) -> dict[str, object] | None:
    if bench_entry is None:
        return None
    return {**bench_entry, "latency": latency}


def _metric_delta(actual: object, expected: float | None) -> float | None:
    if actual is None or expected is None:
        return None
    return cast(float, actual) - expected


def _load_bench_perf_parser() -> BenchPerfParserModule:
    return cast(BenchPerfParserModule, load_operator_eval_script_module("perf_artifacts"))


def _mean_value(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items)


def _geomean_value(values: Iterable[float]) -> float:
    items = list(values)
    return math.exp(sum(math.log(item) for item in items) / len(items))


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


def compare_perf_files(
    baseline_perf: Path,
    compare_perf: Path,
    *,
    metric_source: str = "auto",
) -> int:
    from triton_agent.commands.comparison import compare_perf_files as compare_perf_files_impl

    return compare_perf_files_impl(
        baseline_perf,
        compare_perf,
        metric_source=metric_source,
    )
