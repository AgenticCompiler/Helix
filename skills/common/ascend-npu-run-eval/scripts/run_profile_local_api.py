"""Parent-process API for isolated local benchmark profile execution."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional, cast

from env_registry import HELIX_PROFILE_TIMEOUT_SECONDS
from result_payload import ResultPayload, make_result
from run_profile_execution import profile_timeout
from run_runtime import local_python_executable, result_succeeded, run_streaming_process


SCRIPT_DIR = Path(__file__).resolve().parent


def run_local_profile_bench(
    bench_file: Path,
    operator_file: Path,
    case_id: str | None = None,
    kernel_name: str | None = None,
) -> tuple[ResultPayload, Path | None]:
    del kernel_name
    if case_id is None:
        raise ValueError("torch-npu-profiler benchmark profiling requires --case-id <id>.")
    with tempfile.TemporaryDirectory() as tmp:
        result_file = Path(tmp) / "local-profile-result.json"
        result = run_streaming_process(
            [
                local_python_executable(),
                str(SCRIPT_DIR / "run_profile_local_worker.py"),
                "--bench-file",
                str(bench_file.resolve()),
                "--operator-file",
                str(operator_file.resolve()),
                "--case-id",
                case_id,
                "--result-file",
                str(result_file),
            ],
            str(bench_file.resolve().parent),
            stall_timeout_seconds=0,
            timeout_seconds=profile_timeout(),
            timeout_env_name=HELIX_PROFILE_TIMEOUT_SECONDS,
        )
        if not result_succeeded(result):
            return result, None
        if not result_file.exists():
            return (
                make_result(
                    return_code=1,
                    stdout=str(result["stdout"]),
                    stderr=str(result["stderr"]) + f"Local profile worker did not write result payload: {result_file}",
                    stalled=bool(result["stalled"]),
                    session_id=result["session_id"],
                ),
                None,
            )
        return _read_payload(result_file)


def _read_payload(result_file: Path) -> tuple[ResultPayload, Path | None]:
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    raw_result = payload["result"]
    result = make_result(
        return_code=int(raw_result["return_code"]),
        stdout=str(raw_result["stdout"]),
        stderr=str(raw_result["stderr"]),
        stalled=bool(raw_result["stalled"]),
        session_id=cast(Optional[str], raw_result["session_id"]),
    )
    raw_profile = payload.get("profile_dir")
    return result, None if raw_profile is None else Path(str(raw_profile)).expanduser().resolve()
