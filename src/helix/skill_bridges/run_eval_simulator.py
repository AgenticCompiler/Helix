"""Typed bridge for the simulator skill facade."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Protocol, cast

from helix.skills.loader import load_operator_eval_script_module


class RunSimulatorApi(Protocol):
    def run_local_simulator(
        self,
        bench_file: Path,
        operator_file: Path,
        *,
        case_id: str | None = None,
        kernel_name: str | None = None,
    ) -> dict[str, object]: ...


@lru_cache(maxsize=1)
def _api() -> RunSimulatorApi:
    return cast(RunSimulatorApi, load_operator_eval_script_module("run_simulator_api"))


def run_local_simulator(
    bench_file: Path,
    operator_file: Path,
    *,
    case_id: str | None = None,
    kernel_name: str | None = None,
) -> dict[str, object]:
    return _api().run_local_simulator(
        bench_file,
        operator_file,
        case_id=case_id,
        kernel_name=kernel_name,
    )
