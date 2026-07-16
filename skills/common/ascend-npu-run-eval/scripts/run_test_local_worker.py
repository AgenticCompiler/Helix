from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib
import json
import os
import sys
import traceback
from io import StringIO
from pathlib import Path
from typing import Any, Optional, cast

from env_registry import TRITON_ALWAYS_COMPILE
from run_runtime import ResultPayload, make_result, result_succeeded
from run_test_result import differential_archive_path
from test_contract import (
    bootstrap_torch_npu as _bootstrap_torch_npu,
    compute_flag_from_metadata as _compute_flag_from_metadata,
    load_differential_test_cases,
    load_module as _load_module,
    parse_test_metadata,
    require_callable as _require_callable,
    resolve_operator_api as _resolve_operator_api,
    serialize_payload_object as _serialize_payload_object,
    temporary_sys_path_entries as _temporary_sys_path_entries,
)


SCRIPT_DIR = Path(__file__).resolve().parent
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
        result = _run_import_only_standalone_test(test_file, operator_file, verbose=verbose)
        archived_result = None
    elif test_mode == "differential":
        archive_path = differential_archive_path(operator_file)
        result = _run_declarative_differential_test(
            test_file,
            operator_file,
            archive_path,
            case_id=case_id,
            verbose=verbose,
        )
        archived_result = archive_path if result_succeeded(result) and archive_path.exists() else None
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
    result, case_payload = _run_declarative_differential_test_payload(
        test_file,
        operator_file,
        case_id=case_id,
        verbose=verbose,
    )
    _write_local_test_worker_payload(
        result_file,
        result,
        archived_result=None,
        serialized_payload=None if case_payload is None else _serialize_payload_object(case_payload),
    )
    return 0


def run_local_test_worker_main(argv: list[str] | None = None) -> int:
    parser = _build_local_test_worker_parser()
    args = parser.parse_args(argv)
    test_file = Path(args.test_file).expanduser().resolve()
    operator_file = Path(args.operator_file).expanduser().resolve()
    test_mode = cast(str, args.test_mode)
    result_file = Path(args.result_file).expanduser().resolve()
    case_id = cast(Optional[str], args.case_id)
    verbose = bool(args.verbose)
    if args.command == _LOCAL_TEST_PAYLOAD_WORKER_COMMAND:
        return _run_local_test_payload_worker(
            test_file,
            operator_file,
            test_mode,
            result_file,
            case_id=case_id,
            verbose=verbose,
        )
    return _run_local_test_worker(
        test_file,
        operator_file,
        test_mode,
        result_file,
        case_id=case_id,
        verbose=verbose,
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

def _run_import_only_standalone_test(
    test_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
) -> ResultPayload:
    real_stderr = sys.stderr
    prev = os.environ.get(TRITON_ALWAYS_COMPILE)
    os.environ[TRITON_ALWAYS_COMPILE] = "1"
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    try:
        test_path = test_file.resolve()
        operator_path = operator_file.resolve()
        metadata = parse_test_metadata(test_path)
        if verbose:
            print("[run-test] mode: standalone", file=real_stderr)
            print(f"[run-test] test file: {test_path}", file=real_stderr)
            print(f"[run-test] operator file: {operator_path}", file=real_stderr)
            print(f"[run-test] operator api: {metadata.get('api-name', '?')} ({metadata.get('api-kind', '?')})", file=real_stderr)
            print("[run-test] running...", file=real_stderr)
        _bootstrap_torch_npu(test_path.parent, operator_path.parent)
        with _temporary_sys_path_entries(test_path.parent, operator_path.parent, SCRIPT_DIR):
            test_module = _load_module(test_path, f"standalone_test_{test_path.stem}")
            main_fn = _require_callable(test_module, "main", test_path, kind="Standalone test module")
            operator_module = _load_module(operator_path, f"standalone_operator_{operator_path.stem}")
            operator_api = _resolve_operator_api(operator_module, metadata, operator_path)
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                main_fn(operator_api)
                _maybe_synchronize_torch()
        if verbose:
            print("[run-test] PASSED", file=real_stderr)
        return make_result(
            return_code=0,
            stdout=stdout_buffer.getvalue(),
            stderr=stderr_buffer.getvalue(),
        )
    except Exception:
        if verbose:
            print("[run-test] FAILED", file=real_stderr)
        return make_result(
            return_code=1,
            stdout=stdout_buffer.getvalue(),
            stderr=stderr_buffer.getvalue() + traceback.format_exc(),
        )
    finally:
        if prev is None:
            del os.environ[TRITON_ALWAYS_COMPILE]
        else:
            os.environ[TRITON_ALWAYS_COMPILE] = prev


def _run_declarative_differential_test(
    test_file: Path,
    operator_file: Path,
    archive_path: Path,
    *,
    case_id: str | None = None,
    verbose: bool = False,
) -> ResultPayload:
    result, case_payload = _run_declarative_differential_test_payload(
        test_file,
        operator_file,
        case_id=case_id,
        verbose=verbose,
    )
    if not result_succeeded(result) or case_payload is None:
        return result
    torch = importlib.import_module("torch")
    torch.save(case_payload, archive_path)
    if verbose:
        print(f"[run-test] archive saved: {archive_path}", file=sys.stderr)
    return result


def _run_declarative_differential_test_payload(
    test_file: Path,
    operator_file: Path,
    *,
    case_id: str | None = None,
    verbose: bool = False,
) -> tuple[ResultPayload, object | None]:
    real_stderr = sys.stderr
    try:
        _bootstrap_torch_npu(test_file.resolve().parent, operator_file.resolve().parent)
        torch = importlib.import_module("torch")
    except ImportError as exc:
        return (
            make_result(
                return_code=1,
                stdout="",
                stderr=f"Missing differential test dependency: {exc}",
            ),
            None,
        )
    prev = os.environ.get(TRITON_ALWAYS_COMPILE)
    os.environ[TRITON_ALWAYS_COMPILE] = "1"
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    try:
        metadata = parse_test_metadata(test_file)
        if verbose:
            print("[run-test] mode: differential", file=real_stderr)
            print(f"[run-test] test file: {test_file}", file=real_stderr)
            print(f"[run-test] operator file: {operator_file}", file=real_stderr)
            if case_id is not None:
                print(f"[run-test] case-id: {case_id}", file=real_stderr)
        compute = _compute_flag_from_metadata(metadata)
        cases = load_differential_test_cases(test_file, operator_file, case_id=case_id)
        if verbose:
            print(f"[run-test] total cases: {len(cases)}", file=real_stderr)
        records: list[dict[str, object]] = []
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            for i, case in enumerate(cases):
                if verbose:
                    print(
                        f"[run-test] [{i + 1}/{len(cases)}] {case.case_id} ...",
                        file=real_stderr, end="", flush=True,
                    )
                records.append(
                    {
                        "id": case.case_id,
                        "inputs": case.inputs,
                        "result": case.fn(),
                    }
                )
                _synchronize(torch)
                if verbose:
                    print(" ok", file=real_stderr)
        return (
            make_result(
                return_code=0,
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
            ),
            {"compute": compute, "cases": records},
        )
    except Exception:
        if verbose:
            print("[run-test] FAILED", file=real_stderr)
        return (
            make_result(
                return_code=1,
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue() + traceback.format_exc(),
            ),
            None,
        )
    finally:
        if prev is None:
            del os.environ[TRITON_ALWAYS_COMPILE]
        else:
            os.environ[TRITON_ALWAYS_COMPILE] = prev

def _maybe_synchronize_torch() -> None:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    _synchronize(torch)


def _synchronize(torch_module: Any) -> None:
    if hasattr(torch_module, "npu"):
        torch_module.npu.synchronize()


if __name__ == "__main__":
    try:
        raise SystemExit(run_local_test_worker_main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
