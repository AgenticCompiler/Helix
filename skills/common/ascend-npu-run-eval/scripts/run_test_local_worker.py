"""Fixed local worker protocol for run-test execution."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Optional, cast

from result_payload import ResultPayload
from run_test_execution import run_differential_test, run_differential_test_payload, run_standalone_test
from run_test_result import differential_archive_path
from test_contract import serialize_payload_object


_LOCAL_TEST_WORKER_COMMAND = "local-test-worker"
_LOCAL_TEST_PAYLOAD_WORKER_COMMAND = "local-test-payload-worker"


def _build_local_test_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=Path(__file__).name)
    parser.add_argument("command", choices=[_LOCAL_TEST_WORKER_COMMAND, _LOCAL_TEST_PAYLOAD_WORKER_COMMAND])
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--operator-file", required=True)
    parser.add_argument("--test-mode", choices=["standalone", "differential"], required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _run_local_test_worker(
    test_file: Path,
    operator_file: Path,
    test_mode: str,
    result_file: Path,
    *,
    case_id: str | None,
    verbose: bool,
) -> int:
    if test_mode == "standalone":
        if case_id is not None:
            raise ValueError("--case-id is supported only with differential tests.")
        result = run_standalone_test(test_file, operator_file, verbose=verbose)
        archived_result = None
    elif test_mode == "differential":
        archive_path = differential_archive_path(operator_file)
        result = run_differential_test(
            test_file,
            operator_file,
            archive_path,
            case_id=case_id,
            verbose=verbose,
        )
        archived_result = archive_path if int(result["return_code"]) == 0 and archive_path.exists() else None
    else:
        raise ValueError(f"Unsupported test mode: {test_mode}")
    _write_local_test_worker_payload(result_file, result, archived_result)
    return 0


def _run_local_test_payload_worker(
    test_file: Path,
    operator_file: Path,
    test_mode: str,
    result_file: Path,
    *,
    case_id: str | None,
    verbose: bool,
) -> int:
    if test_mode != "differential":
        raise ValueError("Single-case payload execution is supported only with differential tests.")
    if case_id is None:
        raise ValueError("Single-case payload execution requires --case-id.")
    result, payload = run_differential_test_payload(
        test_file,
        operator_file,
        case_id=case_id,
        verbose=verbose,
    )
    _write_local_test_worker_payload(
        result_file,
        result,
        archived_result=None,
        serialized_payload=None if payload is None else serialize_payload_object(payload),
    )
    return 0


def run_local_test_worker_main(argv: list[str] | None = None) -> int:
    args = _build_local_test_worker_parser().parse_args(argv)
    test_file = Path(args.test_file).expanduser().resolve()
    operator_file = Path(args.operator_file).expanduser().resolve()
    test_mode = cast(str, args.test_mode)
    result_file = Path(args.result_file).expanduser().resolve()
    case_id = cast(Optional[str], args.case_id)
    verbose = bool(args.verbose)
    if args.command == _LOCAL_TEST_PAYLOAD_WORKER_COMMAND:
        return _run_local_test_payload_worker(
            test_file, operator_file, test_mode, result_file, case_id=case_id, verbose=verbose
        )
    return _run_local_test_worker(
        test_file, operator_file, test_mode, result_file, case_id=case_id, verbose=verbose
    )


def _write_local_test_worker_payload(
    result_file: Path,
    result: ResultPayload,
    archived_result: Path | None,
    serialized_payload: str | None = None,
) -> None:
    result_file.write_text(
        json.dumps(
            {
                "result": {
                    "return_code": int(result["return_code"]),
                    "stdout": str(result["stdout"]),
                    "stderr": str(result["stderr"]),
                    "stalled": bool(result["stalled"]),
                    "session_id": result["session_id"],
                },
                "archived_result": None if archived_result is None else str(archived_result.resolve()),
                "serialized_payload": serialized_payload,
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        raise SystemExit(run_local_test_worker_main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
