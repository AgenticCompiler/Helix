"""Concrete standalone and differential run-test execution."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib
import os
import sys
import traceback
from io import StringIO
from pathlib import Path
from typing import Any

from env_registry import TRITON_ALWAYS_COMPILE
from result_payload import ResultPayload, make_result
from test_contract import (
    SCRIPT_DIR,
    bootstrap_torch_npu,
    compute_flag_from_metadata,
    load_differential_test_cases,
    load_module,
    parse_test_metadata,
    require_callable,
    resolve_operator_api,
    temporary_sys_path_entries,
)


def run_standalone_test(
    test_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
) -> ResultPayload:
    real_stderr = sys.stderr
    previous = os.environ.get(TRITON_ALWAYS_COMPILE)
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
            print(
                f"[run-test] operator api: {metadata.get('api-name', '?')} "
                f"({metadata.get('api-kind', '?')})",
                file=real_stderr,
            )
            print("[run-test] running...", file=real_stderr)
        bootstrap_torch_npu(test_path.parent, operator_path.parent)
        with temporary_sys_path_entries(test_path.parent, operator_path.parent, SCRIPT_DIR):
            test_module = load_module(test_path, f"run_test_standalone_{test_path.stem}")
            main_fn = require_callable(test_module, "main", test_path, kind="Standalone test module")
            operator_module = load_module(operator_path, f"run_test_operator_{operator_path.stem}")
            operator_api = resolve_operator_api(operator_module, metadata, operator_path)
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                main_fn(operator_api)
                _maybe_synchronize_torch()
        if verbose:
            print("[run-test] PASSED", file=real_stderr)
        return make_result(return_code=0, stdout=stdout_buffer.getvalue(), stderr=stderr_buffer.getvalue())
    except Exception:
        if verbose:
            print("[run-test] FAILED", file=real_stderr)
        return make_result(
            return_code=1,
            stdout=stdout_buffer.getvalue(),
            stderr=stderr_buffer.getvalue() + traceback.format_exc(),
        )
    finally:
        _restore_always_compile(previous)


def run_differential_test(
    test_file: Path,
    operator_file: Path,
    archive_path: Path,
    *,
    case_id: str | None = None,
    verbose: bool = False,
) -> ResultPayload:
    result, payload = run_differential_test_payload(
        test_file,
        operator_file,
        case_id=case_id,
        verbose=verbose,
    )
    if int(result["return_code"]) != 0 or bool(result["stalled"]) or payload is None:
        return result
    torch = importlib.import_module("torch")
    torch.save(payload, archive_path)
    if verbose:
        print(f"[run-test] archive saved: {archive_path}", file=sys.stderr)
    return result


def run_differential_test_payload(
    test_file: Path,
    operator_file: Path,
    *,
    case_id: str | None = None,
    verbose: bool = False,
) -> tuple[ResultPayload, object | None]:
    real_stderr = sys.stderr
    try:
        bootstrap_torch_npu(test_file.resolve().parent, operator_file.resolve().parent)
        torch = importlib.import_module("torch")
    except ImportError as exc:
        return make_result(return_code=1, stdout="", stderr=f"Missing differential test dependency: {exc}"), None
    previous = os.environ.get(TRITON_ALWAYS_COMPILE)
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
        compute = compute_flag_from_metadata(metadata)
        cases = load_differential_test_cases(test_file, operator_file, case_id=case_id)
        if verbose:
            print(f"[run-test] total cases: {len(cases)}", file=real_stderr)
        records: list[dict[str, object]] = []
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            for index, case in enumerate(cases):
                if verbose:
                    print(
                        f"[run-test] [{index + 1}/{len(cases)}] {case.case_id} ...",
                        file=real_stderr,
                        end="",
                        flush=True,
                    )
                records.append({"id": case.case_id, "inputs": case.inputs, "result": case.fn()})
                _synchronize(torch)
                if verbose:
                    print(" ok", file=real_stderr)
        return make_result(return_code=0, stdout=stdout_buffer.getvalue(), stderr=stderr_buffer.getvalue()), {
            "compute": compute,
            "cases": records,
        }
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
        _restore_always_compile(previous)


def _restore_always_compile(previous: str | None) -> None:
    if previous is None:
        os.environ.pop(TRITON_ALWAYS_COMPILE, None)
    else:
        os.environ[TRITON_ALWAYS_COMPILE] = previous


def _maybe_synchronize_torch() -> None:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    _synchronize(torch)


def _synchronize(torch_module: Any) -> None:
    npu = getattr(torch_module, "npu", None)
    synchronize = getattr(npu, "synchronize", None)
    if callable(synchronize):
        synchronize()
