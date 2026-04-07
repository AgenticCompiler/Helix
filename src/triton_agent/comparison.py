from __future__ import annotations

from pathlib import Path
from typing import Protocol, TextIO, cast

from triton_agent.run_skill import load_run_skill_module


class CompareResultModule(Protocol):
    def compare_result_files(
        self,
        oracle_result: Path,
        new_result: Path,
        compare_level: str,
    ) -> int: ...

    def compare_remote_result_files(
        self,
        oracle_result: Path,
        new_result: Path,
        compare_level: str,
        remote: str,
        remote_workdir: str | None,
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> int: ...


class ComparePerfModule(Protocol):
    def compare_perf_files(self, baseline_perf: Path, compare_perf: Path) -> int: ...


def _load_compare_result() -> CompareResultModule:
    return cast(CompareResultModule, load_run_skill_module("compare_result"))


def _load_compare_perf() -> ComparePerfModule:
    return cast(ComparePerfModule, load_run_skill_module("compare_perf"))


def compare_result_files(oracle_result: Path, new_result: Path, compare_level: str) -> int:
    return _load_compare_result().compare_result_files(oracle_result, new_result, compare_level)


def compare_remote_result_files(
    oracle_result: Path,
    new_result: Path,
    compare_level: str,
    remote: str,
    remote_workdir: str | None,
    *,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> int:
    return _load_compare_result().compare_remote_result_files(
        oracle_result,
        new_result,
        compare_level,
        remote,
        remote_workdir,
        verbose=verbose,
        stderr=stderr,
    )


def compare_perf_files(baseline_perf: Path, compare_perf: Path) -> int:
    return _load_compare_perf().compare_perf_files(baseline_perf, compare_perf)
