"""Parent-process API for isolated local run-test execution."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Optional, cast

from debug_device import maybe_print_visible_devices
from run_runtime import (
    ResultPayload,
    eval_timeout_seconds,
    local_python_executable,
    make_result,
    result_succeeded,
    run_buffered_process,
    run_streaming_process,
)
from run_test_result import filter_result_payload
from test_contract import deserialize_payload_object, run_test_accuracy_env


SCRIPT_DIR = Path(__file__).resolve().parent
_LOCAL_TEST_WORKER_COMMAND = "local-test-worker"
_LOCAL_TEST_PAYLOAD_WORKER_COMMAND = "local-test-payload-worker"


def _local_test_worker_command(
    command_name: str,
    test_file: Path,
    operator_file: Path,
    test_mode: str,
    result_file: Path,
    *,
    case_id: str | None,
    verbose: bool,
) -> list[str]:
    command = [
        local_python_executable(),
        str(SCRIPT_DIR / "run_test_local_worker.py"),
        command_name,
        "--test-file",
        str(test_file.resolve()),
        "--operator-file",
        str(operator_file.resolve()),
        "--test-mode",
        test_mode,
        "--result-file",
        str(result_file),
    ]
    if case_id is not None:
        command.extend(["--case-id", case_id])
    if verbose:
        command.append("--verbose")
    return command


def run_local_test(
    test_file: Path,
    operator_file: Path,
    test_mode: str,
    *,
    case_id: str | None = None,
    accuracy_mode: str | None = None,
    extra_env: Mapping[str, str] | None = None,
    verbose: bool = False,
) -> tuple[ResultPayload, Path | None]:
    maybe_print_visible_devices()
    with tempfile.TemporaryDirectory() as tmp:
        result_file = Path(tmp) / "local-test-result.json"
        runner_result = _run_local_worker_process(
            _local_test_worker_command(
                _LOCAL_TEST_WORKER_COMMAND,
                test_file,
                operator_file,
                test_mode,
                result_file,
                case_id=case_id,
                verbose=verbose,
            ),
            test_file,
            accuracy_mode=accuracy_mode,
            extra_env=extra_env,
            verbose=verbose,
        )
        if not result_succeeded(runner_result):
            return runner_result, None
        if not result_file.exists():
            return _missing_worker_result(runner_result, result_file, "Local test worker"), None
        result, archived_result, _case_payload = _read_local_test_worker_payload(result_file)
        return filter_result_payload(result, verbose=verbose), archived_result


def run_local_test_case_payload(
    test_file: Path,
    operator_file: Path,
    *,
    case_id: str,
    accuracy_mode: str | None = None,
    extra_env: Mapping[str, str] | None = None,
    verbose: bool = False,
) -> tuple[ResultPayload, object | None]:
    maybe_print_visible_devices()
    with tempfile.TemporaryDirectory() as tmp:
        result_file = Path(tmp) / "local-test-result.json"
        runner_result = _run_local_worker_process(
            _local_test_worker_command(
                _LOCAL_TEST_PAYLOAD_WORKER_COMMAND,
                test_file,
                operator_file,
                "differential",
                result_file,
                case_id=case_id,
                verbose=verbose,
            ),
            test_file,
            accuracy_mode=accuracy_mode,
            extra_env=extra_env,
            verbose=verbose,
        )
        if not result_succeeded(runner_result):
            return runner_result, None
        if not result_file.exists():
            return _missing_worker_result(runner_result, result_file, "Local test payload worker"), None
        result, _archived_result, case_payload = _read_local_test_worker_payload(result_file)
        if case_payload is None:
            return (
                make_result(
                    return_code=1,
                    stdout=str(result["stdout"]),
                    stderr=str(result["stderr"]) + "Local test payload worker did not return case payload.",
                    stalled=bool(result["stalled"]),
                    session_id=result["session_id"],
                ),
                None,
            )
        return filter_result_payload(result, verbose=verbose), case_payload


def _run_local_worker_process(
    command: list[str],
    test_file: Path,
    *,
    accuracy_mode: str | None,
    extra_env: Mapping[str, str] | None,
    verbose: bool,
) -> ResultPayload:
    run_process = run_streaming_process if verbose else run_buffered_process
    return run_process(
        command,
        str(test_file.resolve().parent),
        stall_timeout_seconds=0,
        extra_env={**run_test_accuracy_env(accuracy_mode), **(extra_env or {})},
        timeout_seconds=eval_timeout_seconds(),
    )


def _missing_worker_result(
    runner_result: ResultPayload,
    result_file: Path,
    worker_name: str,
) -> ResultPayload:
    return make_result(
        return_code=1,
        stdout=str(runner_result["stdout"]),
        stderr=str(runner_result["stderr"]) + f"{worker_name} did not write result payload: {result_file}",
        stalled=bool(runner_result["stalled"]),
        session_id=runner_result["session_id"],
    )


def _read_local_test_worker_payload(result_file: Path) -> tuple[ResultPayload, Path | None, object | None]:
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    result_payload = payload["result"]
    result = make_result(
        return_code=int(result_payload["return_code"]),
        stdout=str(result_payload["stdout"]),
        stderr=str(result_payload["stderr"]),
        stalled=bool(result_payload["stalled"]),
        session_id=cast(Optional[str], result_payload["session_id"]),
    )
    archived_raw = payload.get("archived_result")
    archived_result = None if archived_raw is None else Path(str(archived_raw)).expanduser().resolve()
    serialized_payload = payload.get("serialized_payload")
    case_payload = None
    if serialized_payload is not None:
        case_payload = deserialize_payload_object(str(serialized_payload))
    return result, archived_result, case_payload
