from __future__ import annotations

import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol, cast


_LOCAL_OPTIMUM_WINDOW_ENV = "TRITON_AGENT_OPTIMIZE_LOCAL_OPTIMUM_WINDOW"
_LOCAL_OPTIMUM_MAX_GAIN_ENV = "TRITON_AGENT_OPTIMIZE_LOCAL_OPTIMUM_MAX_GEOMEAN_GAIN"
_DEFAULT_LOCAL_OPTIMUM_WINDOW = 3
_DEFAULT_LOCAL_OPTIMUM_MAX_GAIN = 0.02
_BATCH_OPTIMIZE_EXCLUDED_PREFIXES = ("test_", "differential_test_", "bench_", "opt_")
_BATCH_OPTIMIZE_EXCLUDED_NAMES = {"__init__.py"}


class PerfArtifactsModule(Protocol):
    def parse_perf_file(self, path: Path) -> dict[str, float]: ...

    def parse_required_perf_file(
        self,
        path: Path,
        required_latency_ids: dict[str, float],
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
        required_latency_ids: dict[str, float],
        *,
        metric_source: str = "auto",
    ) -> dict[str, float]: ...


@dataclass(frozen=True)
class LocalOptimumConfig:
    window: int
    max_geomean_gain: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RoundSpeedupSample:
    round_name: str
    round_number: int
    metric_basis: str
    geomean_speedup: float


def collect_local_optimum_warnings(
    current_round_dir: Path,
    *,
    baseline_perf_path: Path,
) -> tuple[str, ...]:
    config = load_local_optimum_config_from_env()
    warnings = list(config.warnings)
    candidate_round_dirs = _candidate_round_window(current_round_dir, window=config.window)
    if candidate_round_dirs is None:
        return tuple(warnings)

    baseline_cache: dict[str, dict[str, float]] = {}
    samples: list[RoundSpeedupSample] = []
    for round_dir in candidate_round_dirs:
        sample = _build_round_speedup_sample(
            round_dir,
            baseline_perf_path=baseline_perf_path,
            baseline_cache=baseline_cache,
        )
        if sample is None:
            return tuple(warnings)
        samples.append(sample)

    metric_basis = samples[0].metric_basis
    if any(sample.metric_basis != metric_basis for sample in samples[1:]):
        return tuple(warnings)

    adjacent_gains = [
        current.geomean_speedup - previous.geomean_speedup
        for previous, current in zip(samples, samples[1:])
    ]
    if all(gain <= config.max_geomean_gain for gain in adjacent_gains):
        warnings.append(_build_local_optimum_warning(samples, adjacent_gains))

    return tuple(warnings)


def load_local_optimum_config_from_env() -> LocalOptimumConfig:
    window = _DEFAULT_LOCAL_OPTIMUM_WINDOW
    raw_window = os.environ.get(_LOCAL_OPTIMUM_WINDOW_ENV)
    if raw_window is not None:
        try:
            parsed_window = int(raw_window)
            if parsed_window < 2:
                raise ValueError
            window = parsed_window
        except ValueError:
            pass

    max_geomean_gain = _DEFAULT_LOCAL_OPTIMUM_MAX_GAIN
    raw_gain = os.environ.get(_LOCAL_OPTIMUM_MAX_GAIN_ENV)
    if raw_gain is not None:
        try:
            parsed_gain = float(raw_gain)
            if parsed_gain < 0 or not math.isfinite(parsed_gain):
                raise ValueError
            max_geomean_gain = parsed_gain
        except ValueError:
            pass

    return LocalOptimumConfig(
        window=window,
        max_geomean_gain=max_geomean_gain,
        warnings=(),
    )


def _candidate_round_window(current_round_dir: Path, *, window: int) -> list[Path] | None:
    current_round_number = _round_number(current_round_dir.name)
    if current_round_number is None:
        return None

    workspace = current_round_dir.parent
    candidate_rounds = sorted(
        (
            round_dir
            for round_dir in workspace.glob("opt-round-*")
            if round_dir.is_dir()
            and (round_number := _round_number(round_dir.name)) is not None
            and round_number <= current_round_number
        ),
        key=lambda round_dir: cast(int, _round_number(round_dir.name)),
    )
    if len(candidate_rounds) < window:
        return None

    recent_rounds = candidate_rounds[-window:]
    round_numbers = [cast(int, _round_number(round_dir.name)) for round_dir in recent_rounds]
    if round_numbers[-1] != current_round_number:
        return None
    if any(current != previous + 1 for previous, current in zip(round_numbers, round_numbers[1:])):
        return None
    return recent_rounds


def _build_round_speedup_sample(
    round_dir: Path,
    *,
    baseline_perf_path: Path,
    baseline_cache: dict[str, dict[str, float]],
) -> RoundSpeedupSample | None:
    state = _load_round_local_optimum_state(round_dir)
    if state is None:
        return None
    perf_path = _resolve_round_perf_path(round_dir, declared_perf_artifact=state.perf_artifact)
    if perf_path is None:
        return None

    baseline_values = baseline_cache.get(state.metric_basis)
    if baseline_values is None:
        baseline_values = _parse_baseline_values(
            baseline_perf_path,
            metric_basis=state.metric_basis,
        )
        baseline_cache[state.metric_basis] = baseline_values

    round_values = _parse_round_values(
        perf_path,
        baseline_values=baseline_values,
        metric_basis=state.metric_basis,
    )
    latency_ids = sorted(baseline_values)
    if set(latency_ids) != set(round_values):
        return None

    speedup_values: list[float] = []
    for latency_id in latency_ids:
        baseline_value = baseline_values[latency_id]
        round_value = round_values[latency_id]
        if baseline_value <= 0 or round_value <= 0:
            return None
        speedup_values.append(baseline_value / round_value)
    if not speedup_values:
        return None

    return RoundSpeedupSample(
        round_name=f"round-{state.round_number}",
        round_number=state.round_number,
        metric_basis=state.metric_basis,
        geomean_speedup=math.exp(sum(math.log(value) for value in speedup_values) / len(speedup_values)),
    )


@dataclass(frozen=True)
class _RoundLocalOptimumState:
    round_number: int
    metric_basis: str
    perf_artifact: str | None


def _load_round_local_optimum_state(round_dir: Path) -> _RoundLocalOptimumState | None:
    round_number = _round_number(round_dir.name)
    if round_number is None:
        return None
    state_path = round_dir / "round-state.json"
    try:
        import json

        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    state = cast(dict[str, object], payload)
    metric_basis = _normalize_metric_basis(state.get("effective_metric_source"))
    if metric_basis is None:
        return None
    perf_artifact = state.get("perf_artifact")
    perf_artifact_value = str(perf_artifact) if isinstance(perf_artifact, str) else None
    return _RoundLocalOptimumState(
        round_number=round_number,
        metric_basis=metric_basis,
        perf_artifact=perf_artifact_value,
    )


def _normalize_metric_basis(value: object) -> str | None:
    if value == "kernel":
        return "kernel"
    if value == "total-op":
        return "total-op"
    if value == "mixed":
        return "auto"
    return None


def _resolve_round_perf_path(round_dir: Path, *, declared_perf_artifact: str | None) -> Path | None:
    if declared_perf_artifact is not None:
        declared_path = Path(declared_perf_artifact)
        declared = round_dir / declared_path
        if declared.is_file():
            return declared
        workspace_relative = round_dir.parent / declared_path
        if workspace_relative.is_file():
            return workspace_relative
    workspace = round_dir.parent
    expected_perf_name = _expected_round_perf_name(workspace)
    if expected_perf_name is not None:
        expected_perf_path = round_dir / expected_perf_name
        if expected_perf_path.is_file():
            return expected_perf_path
    legacy_perf_path = round_dir / "perf.txt"
    if legacy_perf_path.is_file():
        return legacy_perf_path
    perf_files = sorted(path for path in round_dir.glob("*_perf.txt") if path.is_file())
    if len(perf_files) == 1:
        return perf_files[0]
    return None


def _parse_baseline_values(baseline_perf_path: Path, *, metric_basis: str) -> dict[str, float]:
    module = _load_perf_artifacts_module()
    if metric_basis == "auto":
        return module.parse_perf_file(baseline_perf_path)
    return module.parse_perf_file_for_metric_source(
        baseline_perf_path,
        metric_source=metric_basis,
    )


def _parse_round_values(
    perf_path: Path,
    *,
    baseline_values: dict[str, float],
    metric_basis: str,
) -> dict[str, float]:
    module = _load_perf_artifacts_module()
    if metric_basis == "auto":
        return module.parse_required_perf_file(perf_path, baseline_values)
    return module.parse_required_perf_file_for_metric_source(
        perf_path,
        baseline_values,
        metric_source=metric_basis,
    )


def _build_local_optimum_warning(samples: list[RoundSpeedupSample], adjacent_gains: list[float]) -> str:
    start_round = samples[0].round_name
    end_round = samples[-1].round_name
    gains_text = ", ".join(_format_speedup_gain(gain) for gain in adjacent_gains)
    metric_basis = _describe_metric_basis(samples[0].metric_basis)
    return (
        "recent rounds show only marginal baseline-relative geomean speedup gains "
        f"on the same metric basis ({metric_basis}; {start_round} -> {end_round}: {gains_text}); "
        "optimization may be stagnating in the current direction and may be stuck in a local optimum. "
        "Review earlier rounds and consider resuming from a round before this flat sequence "
        "to explore a different optimization path."
    )


def _format_speedup_gain(value: float) -> str:
    return f"{value:+.2f}x"


def _describe_metric_basis(metric_basis: str) -> str:
    if metric_basis == "auto":
        return "mixed fallback"
    return metric_basis


def _round_number(name: str) -> int | None:
    prefix = "opt-round-"
    if not name.startswith(prefix):
        return None
    suffix = name.removeprefix(prefix)
    if not suffix.isdigit():
        return None
    return int(suffix)


def _expected_round_perf_name(workspace: Path) -> str | None:
    operator_file = _resolve_workspace_operator_file(workspace)
    if operator_file is None:
        return None
    return f"opt_{operator_file.stem}_perf.txt"


def _resolve_workspace_operator_file(workspace: Path) -> Path | None:
    candidates = [
        path
        for path in sorted(workspace.iterdir())
        if _is_batch_optimize_operator_candidate(path)
    ]
    if len(candidates) != 1:
        return None
    return candidates[0]


def _is_batch_optimize_operator_candidate(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix != ".py":
        return False
    if path.name in _BATCH_OPTIMIZE_EXCLUDED_NAMES:
        return False
    return not path.name.startswith(_BATCH_OPTIMIZE_EXCLUDED_PREFIXES)


@lru_cache(maxsize=1)
def _load_perf_artifacts_module() -> PerfArtifactsModule:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "npu-run-eval"
        / "scripts"
        / "perf_artifacts.py"
    )
    module_name = "skill_triton_npu_run_eval_perf_artifacts"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load perf artifacts script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    script_dir = str(script_path.parent)
    added = False
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        added = True
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        if added:
            sys.path.remove(script_dir)
    return cast(PerfArtifactsModule, module)
