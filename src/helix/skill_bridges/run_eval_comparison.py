"""Typed bridge for run-eval comparison and performance-artifact facades."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Protocol, TextIO, cast

from helix.skills.loader import load_operator_eval_script_module


class CompareResultApi(Protocol):
    def compare_result_payload_objects(self, ref_payload: object, new_payload: object, *, accuracy_mode: str | None = None) -> int: ...
    def compare_result_files(self, ref_result: Path, new_result: Path, *, accuracy_mode: str | None = None) -> int: ...
    def compare_remote_result_files(self, ref_result: Path, new_result: Path, remote: str, remote_workdir: str | None, *, accuracy_mode: str | None = None, verbose: bool = False, stderr: TextIO | None = None) -> int: ...
    def load_case_result_payload(self, path: Path, case_id: str) -> object: ...
    def find_case_result_payload(self, path: Path, case_id: str) -> object | None: ...


class PerfArtifactsApi(Protocol):
    def compare_perf_files(self, baseline_perf: Path, compare_perf: Path, *, skip_latency_errors: bool = False, metric_source: str = "auto") -> int: ...
    def parse_perf_file(self, path: Path) -> dict[str, float]: ...
    def parse_required_perf_file(self, path: Path, required_latency_ids: Iterable[str]) -> dict[str, float]: ...
    def parse_perf_file_for_metric_source(self, path: Path, *, metric_source: str = "auto") -> dict[str, float]: ...
    def parse_required_perf_file_for_metric_source(self, path: Path, required_latency_ids: Iterable[str], *, metric_source: str = "auto") -> dict[str, float]: ...
    def parse_perf_pair_for_comparison(self, baseline_perf: Path, compare_perf: Path, *, metric_source: str = "auto") -> tuple[dict[str, float], dict[str, float], dict[str, str]]: ...


@lru_cache(maxsize=1)
def _result_api() -> CompareResultApi:
    return cast(CompareResultApi, load_operator_eval_script_module("compare_result_api"))


@lru_cache(maxsize=1)
def _perf_api() -> PerfArtifactsApi:
    return cast(PerfArtifactsApi, load_operator_eval_script_module("perf_artifacts_api"))


def compare_result_payload_objects(ref_payload: object, new_payload: object, *, accuracy_mode: str | None = None) -> int:
    return int(_result_api().compare_result_payload_objects(ref_payload, new_payload, accuracy_mode=accuracy_mode))


def compare_result_files(ref_result: Path, new_result: Path, *, accuracy_mode: str | None = None) -> int:
    return int(_result_api().compare_result_files(ref_result, new_result, accuracy_mode=accuracy_mode))


def compare_remote_result_files(
    ref_result: Path,
    new_result: Path,
    remote: str,
    remote_workdir: str | None,
    *,
    accuracy_mode: str | None = None,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> int:
    return int(_result_api().compare_remote_result_files(ref_result, new_result, remote, remote_workdir, accuracy_mode=accuracy_mode, verbose=verbose, stderr=stderr))


def load_case_result_payload(path: Path, case_id: str) -> object:
    return _result_api().load_case_result_payload(path, case_id)


def find_case_result_payload(path: Path, case_id: str) -> object | None:
    return _result_api().find_case_result_payload(path, case_id)


def compare_perf_files(baseline_perf: Path, compare_perf: Path, *, skip_latency_errors: bool = False, metric_source: str = "auto") -> int:
    return int(_perf_api().compare_perf_files(baseline_perf, compare_perf, skip_latency_errors=skip_latency_errors, metric_source=metric_source))


def parse_perf_file(path: Path) -> dict[str, float]:
    return _perf_api().parse_perf_file(path)


def parse_required_perf_file(path: Path, required_latency_ids: Iterable[str]) -> dict[str, float]:
    return _perf_api().parse_required_perf_file(path, required_latency_ids)


def parse_perf_file_for_metric_source(path: Path, *, metric_source: str = "auto") -> dict[str, float]:
    return _perf_api().parse_perf_file_for_metric_source(path, metric_source=metric_source)


def parse_required_perf_file_for_metric_source(path: Path, required_latency_ids: Iterable[str], *, metric_source: str = "auto") -> dict[str, float]:
    return _perf_api().parse_required_perf_file_for_metric_source(path, required_latency_ids, metric_source=metric_source)


def parse_perf_pair_for_comparison(baseline_perf: Path, compare_perf: Path, *, metric_source: str = "auto") -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    return _perf_api().parse_perf_pair_for_comparison(baseline_perf, compare_perf, metric_source=metric_source)
