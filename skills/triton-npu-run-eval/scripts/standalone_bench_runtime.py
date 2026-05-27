from __future__ import annotations

from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
import importlib
import importlib.util
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, cast

from bench_contract import KernelResolution, resolve_bench_kernel_resolution
from perf_artifacts import (
    PerfCaseRecord,
    PerfMetrics,
    perf_output_path,
    render_perf_case_records_jsonl,
    write_perf_lines,
)
from profile_csv_parser import (
    find_latest_op_statistic_csv,
    find_optional_profile_csv,
    parse_kernel_details_csv,
    parse_op_statistic_csv,
    parse_operator_details_csv,
    resolve_perf_metrics,
)
from result_payload import ResultPayload, make_result


WARMUP_DEFAULT = 5
REPEATS_DEFAULT = 50
_MISSING_KERNEL_MATCH_ERROR = "no resolved kernels matched profiler operator details"
_LOCAL_BENCH_OUTPUT_DIR_ENV = "TRITON_AGENT_BENCH_OUTPUT_DIR"

@dataclass(frozen=True)
class StandaloneBenchCase:
    case_id: str
    fn: Callable[[], object]
    warmup: int
    repeats: int

def load_standalone_bench_cases(
    bench_file: Path,
    operator_file: Path,
) -> tuple[list[StandaloneBenchCase], KernelResolution]:
    bench_path = bench_file.resolve()
    operator_path = operator_file.resolve()
    bench_module = _load_module(bench_path, f"standalone_bench_{bench_path.stem}")
    operator_module = _load_module(operator_path, f"standalone_operator_{operator_path.stem}")
    build_operator_api = _require_callable(bench_module, "build_operator_api", bench_path)
    build_cases = _require_callable(bench_module, "build_standalone_bench_cases", bench_path)
    operator_api = build_operator_api(operator_module)
    raw_cases = build_cases(operator_api)
    return _normalize_cases(raw_cases), resolve_bench_kernel_resolution(bench_path, operator_path)


def run_local_standalone_bench(
    bench_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
) -> tuple[ResultPayload, Path]:
    cases, resolution = load_standalone_bench_cases(bench_file, operator_file)
    case_records: list[PerfCaseRecord] = []
    had_failures = False
    stderr_chunks: list[str] = []
    preserved_run_dir = _create_local_preserved_profile_run_dir(prefix="triton-agent-standalone-bench-")

    for case in cases:
        record = _run_standalone_case(case, resolution, preserved_run_dir, bench_file.parent, verbose=verbose)
        if record.error_message is not None:
            had_failures = True
            stderr_chunks.append(f"{case.case_id}: {record.error_message}")
        case_records.append(record)

    perf_path = perf_output_path(operator_file)
    write_perf_lines(
        perf_path,
        render_perf_case_records_jsonl(
            case_records,
            missing_kernel_match_error=_MISSING_KERNEL_MATCH_ERROR,
        ),
    )
    return (
        make_result(
            return_code=1 if had_failures else 0,
            stdout="",
            stderr="\n".join(stderr_chunks),
        ),
        perf_path,
    )


def profile_local_standalone_case(
    bench_file: Path,
    operator_file: Path,
    case_id: str,
) -> ResultPayload:
    cases, resolution = load_standalone_bench_cases(bench_file, operator_file)
    case = _select_case(cases, case_id)
    profile_root = _profile_output_root(bench_file.parent, case.case_id)
    _metrics, error_message = _profile_case_with_profiler(case, resolution, profile_root)
    if error_message is not None:
        return make_result(return_code=1, stdout="", stderr=error_message)
    return make_result(return_code=0, stdout="", stderr="")


def run_one_standalone_case(
    bench_file: Path,
    operator_file: Path,
    case_id: str | None = None,
) -> ResultPayload:
    cases, _ = load_standalone_bench_cases(bench_file, operator_file)
    case = _select_case(cases, case_id)
    case.fn()
    return make_result(return_code=0, stdout="", stderr="")


def run_one_standalone_case_record(
    bench_file: Path,
    operator_file: Path,
    case_id: str,
    *,
    preserved_run_dir: Path | None = None,
    verbose: bool = False,
) -> PerfCaseRecord:
    cases, resolution = load_standalone_bench_cases(bench_file, operator_file)
    case = _select_case(cases, case_id)
    return _run_standalone_case(
        case,
        resolution,
        preserved_run_dir,
        bench_file.parent,
        verbose=verbose,
    )


def runtime_support_paths() -> list[Path]:
    script_dir = Path(__file__).resolve().parent
    return [
        script_dir / "result_payload.py",
        script_dir / "standalone_bench_runtime.py",
        script_dir / "bench_contract.py",
        script_dir / "perf_artifacts.py",
    ]


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(f"{module_name}_{time.time_ns()}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _require_callable(module: object, name: str, bench_file: Path) -> Callable[..., Any]:
    candidate = getattr(module, name, None)
    if not callable(candidate):
        raise ValueError(f"Standalone benchmark module missing required hook '{name}': {bench_file}")
    return candidate


def _normalize_cases(raw_cases: object) -> list[StandaloneBenchCase]:
    if isinstance(raw_cases, (str, bytes)) or isinstance(raw_cases, Mapping) or not isinstance(raw_cases, Iterable):
        raise ValueError("Standalone benchmark hook 'build_standalone_bench_cases' must return an iterable of cases")
    cases: list[StandaloneBenchCase] = []
    seen_case_ids: set[str] = set()
    for raw_case in cast(Iterable[object], raw_cases):
        if not isinstance(raw_case, Mapping):
            raise ValueError("Standalone benchmark cases must be mappings")
        case_map = cast(Mapping[str, object], raw_case)
        case_id = case_map.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Standalone benchmark case is missing required string field 'id'")
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate standalone benchmark case id: {case_id}")
        case_fn = case_map.get("fn")
        if not callable(case_fn):
            raise ValueError(f"Standalone benchmark case '{case_id}' is missing required callable field 'fn'")
        warmup = _normalize_non_negative_int(case_map.get("warmup", WARMUP_DEFAULT), "warmup", case_id)
        repeats = _normalize_positive_int(case_map.get("repeats", REPEATS_DEFAULT), "repeats", case_id)
        seen_case_ids.add(case_id)
        cases.append(
            StandaloneBenchCase(
                case_id=case_id,
                fn=cast(Callable[[], object], case_fn),
                warmup=warmup,
                repeats=repeats,
            )
        )
    if not cases:
        raise ValueError("Standalone benchmark hook 'build_standalone_bench_cases' returned no cases")
    return cases


def _normalize_non_negative_int(value: object, field_name: str, case_id: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"Standalone benchmark case '{case_id}' field '{field_name}' must be an integer")
    if value < 0:
        raise ValueError(f"Standalone benchmark case '{case_id}' field '{field_name}' must be >= 0")
    return value


def _normalize_positive_int(value: object, field_name: str, case_id: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"Standalone benchmark case '{case_id}' field '{field_name}' must be an integer")
    if value <= 0:
        raise ValueError(f"Standalone benchmark case '{case_id}' field '{field_name}' must be > 0")
    return value


def _select_case(cases: list[StandaloneBenchCase], case_id: str | None) -> StandaloneBenchCase:
    if case_id is None:
        return cases[0]
    for case in cases:
        if case.case_id == case_id:
            return case
    available = ", ".join(case.case_id for case in cases)
    raise ValueError(f"Unknown standalone benchmark case id '{case_id}'. Available case ids: {available}")


def _profile_case_with_profiler(
    case: StandaloneBenchCase,
    resolution: KernelResolution,
    profile_root: Path,
    *,
    verbose: bool = False,
) -> tuple[PerfMetrics | None, str | None]:
    try:
        import torch
        torch_npu = cast(Any, importlib.import_module("torch_npu"))
    except ImportError as exc:
        return None, f"Missing profiler dependency: {exc}"

    try:
        profile_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        shutil.rmtree(profile_root, ignore_errors=True)
        profile_root.mkdir(parents=True, exist_ok=False)

    try:
        profiler_api = torch_npu.profiler
        experimental_config = profiler_api._ExperimentalConfig(
            aic_metrics=None,
            profiler_level=profiler_api.ProfilerLevel.Level1,
            l2_cache=False,
            data_simplification=False,
        )

        _suppress_ctx = _suppress_output_streams() if not verbose else nullcontext()
        with _suppress_ctx:
            case.fn()
            _synchronize(torch)

            def _run_once() -> None:
                case.fn()
                _synchronize(torch)

            skip_first = 1 + case.warmup
            total_steps = skip_first + case.repeats

            with profiler_api.profile(
                activities=[
                    profiler_api.ProfilerActivity.NPU,
                    profiler_api.ProfilerActivity.CPU,
                ],
                schedule=profiler_api.schedule(
                    wait=0,
                    warmup=case.warmup,
                    active=case.repeats,
                    repeat=1,
                    skip_first=skip_first,
                ),
                on_trace_ready=profiler_api.tensorboard_trace_handler(str(profile_root)),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config,
            ) as profiler:
                for _ in range(total_steps):
                    _run_once()
                    profiler.step()

        return _read_profiler_metrics(profile_root, case.repeats, resolution.kernel_names), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _run_standalone_case(
    case: StandaloneBenchCase,
    resolution: KernelResolution,
    preserved_run_dir: Path | None,
    cleanup_workdir: Path,
    *,
    verbose: bool = False,
) -> PerfCaseRecord:
    profile_root, temp_dir = _create_local_standalone_profile_dir(case.case_id, preserved_run_dir)
    try:
        t0 = time.monotonic()
        metrics, error_message = _profile_case_with_profiler(
            case,
            resolution,
            profile_root,
            verbose=verbose,
        )
        elapsed = time.monotonic() - t0
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
        _cleanup_local_bench_extra_info(cleanup_workdir)
    if error_message is not None:
        return PerfCaseRecord(
            case_label=case.case_id,
            kernel_names=resolution.kernel_names,
            kernel_source=resolution.kernel_source,
            error_message=error_message,
            case_wall_clock_seconds=elapsed,
        )
    return PerfCaseRecord(
        case_label=case.case_id,
        kernel_names=resolution.kernel_names,
        kernel_source=resolution.kernel_source,
        metrics=metrics,
        case_wall_clock_seconds=elapsed,
    )


@contextmanager
def _suppress_output_streams() -> Iterator[None]:
    with open(os.devnull, "w", encoding="utf-8") as quiet_output:
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            with redirect_stdout(quiet_output), redirect_stderr(quiet_output):
                os.dup2(quiet_output.fileno(), 1)
                os.dup2(quiet_output.fileno(), 2)
                yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)


def _synchronize(torch_module: Any) -> None:
    if hasattr(torch_module, "npu"):
        torch_module.npu.synchronize()


def _read_profiler_metrics(
    profile_root: Path,
    active_count: int,
    kernel_names: list[str],
) -> PerfMetrics:
    operator_details_path = find_optional_profile_csv(profile_root, "operator_details.csv")
    operator_details_rows = None
    if operator_details_path is not None:
        operator_details_rows = parse_operator_details_csv(
            operator_details_path,
            active_count=active_count,
            kernel_names=kernel_names,
        )
        if operator_details_rows.total_time_us > 0:
            return resolve_perf_metrics(operator_details_rows.ops, kernel_names)

    kernel_details_path = find_optional_profile_csv(profile_root, "kernel_details.csv")
    kernel_details_rows = None
    if kernel_details_path is not None:
        kernel_details_rows = parse_kernel_details_csv(
            kernel_details_path,
            active_count=active_count,
        )
        if kernel_details_rows.total_time_us > 0:
            return resolve_perf_metrics(kernel_details_rows.ops, kernel_names)

    op_statistic_path = find_latest_op_statistic_csv(profile_root)
    if op_statistic_path is not None:
        return resolve_perf_metrics(parse_op_statistic_csv(op_statistic_path).ops, kernel_names)

    if operator_details_rows is not None:
        return resolve_perf_metrics(operator_details_rows.ops, kernel_names)

    if kernel_details_rows is not None:
        return resolve_perf_metrics(kernel_details_rows.ops, kernel_names)

    raise FileNotFoundError(
        f"No operator_details.csv, kernel_details.csv, or op_statistic.csv found under {profile_root}"
    )


def _profile_output_root(parent: Path, case_id: str) -> Path:
    return parent / f"PROF_{_sanitize_case_id(case_id)}_{int(time.time() * 1000)}"


def _resolve_local_bench_profile_output_root() -> tuple[str | None, str]:
    configured_root = os.environ.get(_LOCAL_BENCH_OUTPUT_DIR_ENV)
    if configured_root:
        return str(Path(configured_root).expanduser().resolve()), _LOCAL_BENCH_OUTPUT_DIR_ENV
    return None, _LOCAL_BENCH_OUTPUT_DIR_ENV


def _create_local_preserved_profile_run_dir(prefix: str) -> Path | None:
    configured_root, configured_env = _resolve_local_bench_profile_output_root()
    if not configured_root:
        return None
    root = Path(configured_root).expanduser()
    if root.exists() and not root.is_dir():
        raise ValueError(f"{configured_env} must point to a directory: {root}")
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        _set_directory_owner_only(root)
    run_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=str(root)))
    _set_directory_owner_only(run_dir)
    return run_dir


def create_local_preserved_profile_run_dir(prefix: str) -> Path | None:
    return _create_local_preserved_profile_run_dir(prefix)


def _create_local_standalone_profile_dir(
    case_id: str,
    preserved_run_dir: Path | None,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if preserved_run_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix=f"triton-agent-standalone-bench-{_sanitize_case_id(case_id)}-")
        return Path(temp_dir.name), temp_dir
    profile_root = preserved_run_dir.resolve() / f"case-{_sanitize_case_id(case_id)}"
    profile_root.mkdir(parents=True, exist_ok=False)
    _set_directory_owner_only(profile_root)
    return profile_root, None


def _set_directory_owner_only(path: Path) -> None:
    path.chmod(0o700)


def _sanitize_case_id(case_id: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in case_id)
    return sanitized or "case"


def _cleanup_local_bench_extra_info(workdir: Path) -> None:
    extra_info_dir = workdir / "extra-info"
    if not extra_info_dir.is_dir():
        return
    shutil.rmtree(extra_info_dir)
