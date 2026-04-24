from __future__ import annotations

import shutil
import sys
from pathlib import Path

import compare_result_payloads as result_payload_compare
from run_runtime import (
    ResultPayload,
    cleanup_remote_workspace,
    copy_file_from_remote,
    copy_file_to_remote,
    create_remote_workspace,
    result_succeeded,
    run_streaming_process,
    run_remote_command_buffered,
    run_remote_command_streaming,
)


ORACLE_COMPARE_LEVELS = result_payload_compare.ORACLE_COMPARE_LEVELS
_compare_values = result_payload_compare._compare_values
_extract_ordered_results = result_payload_compare._extract_ordered_results
_load_result_payload = result_payload_compare._load_result_payload
_resolve_compare_tolerances = result_payload_compare._resolve_compare_tolerances


def run_local_test(
    test_file: Path,
    operator_file: Path,
    test_mode: str,
) -> tuple[ResultPayload, Path | None]:
    command = [sys.executable, str(test_file), "--operator-file", str(operator_file)]
    result = run_streaming_process(command, str(test_file.parent), stall_timeout_seconds=900)
    archived_result = None
    if test_mode == "differential" and result_succeeded(result):
        archived_result = archive_differential_result(test_file, operator_file)
    return result, archived_result


def parse_test_metadata(test_file: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in test_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            if metadata:
                break
            continue
        if not stripped.startswith("#"):
            break
        body = stripped[1:].strip()
        if ":" not in body:
            continue
        key, value = body.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def run_remote_test(
    test_file: Path,
    operator_file: Path,
    test_mode: str,
    remote: str,
    remote_workdir: str | None,
    keep_remote_workdir: bool = False,
    verbose: bool = False,
    stderr=None,
) -> tuple[ResultPayload, Path | None, str]:
    spec, remote_workspace = create_remote_workspace(
        remote, remote_workdir, verbose=verbose, stderr=stderr
    )
    remote_test = f"{remote_workspace}/{test_file.name}"
    remote_operator = f"{remote_workspace}/{operator_file.name}"
    try:
        copy_file_to_remote(spec, test_file, remote_test, verbose=verbose, stderr=stderr)
        copy_file_to_remote(spec, operator_file, remote_operator, verbose=verbose, stderr=stderr)
        result = run_remote_command_streaming(
            spec,
            remote_workspace,
            ["python3", test_file.name, "--operator-file", operator_file.name],
            verbose=verbose,
            stderr=stderr,
        )
        archived_result = None
        if test_mode == "differential" and result_succeeded(result):
            archived_result = _copy_remote_differential_result(
                spec,
                remote_workspace,
                test_file,
                operator_file,
                verbose=verbose,
                stderr=stderr,
            )
        return result, archived_result, remote_workspace
    finally:
        if not keep_remote_workdir:
            cleanup_remote_workspace(spec, remote_workspace, verbose=verbose, stderr=stderr)


def find_case_insensitive_result_file(directory: Path) -> Path | None:
    for candidate in sorted(directory.iterdir()):
        if candidate.is_file() and candidate.name.lower() == "test_result.pt":
            return candidate
    return None


def archive_differential_result(test_file: Path, operator_file: Path) -> Path:
    result_file = find_case_insensitive_result_file(operator_file.parent)
    if result_file is None and test_file.parent != operator_file.parent:
        result_file = find_case_insensitive_result_file(test_file.parent)
    if result_file is None:
        raise FileNotFoundError(
            f"Differential result payload not found beside operator or test file: {operator_file.parent}"
        )

    archive_name = f"{operator_file.stem}_result.pt"
    archive_path = operator_file.parent / archive_name
    shutil.copy2(result_file, archive_path)
    return archive_path


def compare_result_files(oracle_result: Path, new_result: Path, compare_level: str) -> int:
    return _compare_result_files_impl(oracle_result, new_result, compare_level)


def compare_remote_result_files(
    oracle_result: Path,
    new_result: Path,
    compare_level: str,
    remote: str,
    remote_workdir: str | None,
    verbose: bool = False,
    stderr=None,
) -> int:
    spec, remote_workspace = create_remote_workspace(
        remote, remote_workdir, verbose=verbose, stderr=stderr
    )
    compare_script = Path(__file__).resolve().with_name("compare_result_payloads.py")
    remote_script = f"{remote_workspace}/{compare_script.name}"
    remote_oracle = f"{remote_workspace}/{oracle_result.name}"
    remote_new = f"{remote_workspace}/{new_result.name}"
    try:
        copy_file_to_remote(spec, compare_script, remote_script, verbose=verbose, stderr=stderr)
        copy_file_to_remote(spec, oracle_result, remote_oracle, verbose=verbose, stderr=stderr)
        copy_file_to_remote(spec, new_result, remote_new, verbose=verbose, stderr=stderr)
        result = run_remote_command_streaming(
            spec,
            remote_workspace,
            [
                "python3",
                compare_script.name,
                "--oracle-result",
                oracle_result.name,
                "--new-result",
                new_result.name,
                "--compare-level",
                compare_level,
            ],
            verbose=verbose,
            stderr=stderr,
        )
        return int(result["return_code"])
    finally:
        cleanup_remote_workspace(spec, remote_workspace, verbose=verbose, stderr=stderr)


def _compare_result_files_impl(oracle_result: Path, new_result: Path, compare_level: str) -> int:
    try:
        rtol, atol = _resolve_compare_tolerances(compare_level)
    except ValueError:
        print(
            f"FAIL: invalid compare level '{compare_level}', "
            f"expected one of {sorted(ORACLE_COMPARE_LEVELS)}"
        )
        return 2

    expected_payload = _load_result_payload(oracle_result)
    actual_payload = _load_result_payload(new_result)

    expected, expected_error = _extract_ordered_results(expected_payload, "oracle")
    if expected_error:
        print(f"FAIL: {expected_error}")
        return 1

    actual, actual_error = _extract_ordered_results(actual_payload, "compare")
    if actual_error:
        print(f"FAIL: {actual_error}")
        return 1

    mismatch = _compare_values(expected, actual, "output", rtol, atol)
    if mismatch:
        print(f"FAIL: {mismatch}")
        return 1

    print(
        "PASS: ordered outputs match "
        f"(level={compare_level.strip().lower()}, rtol={rtol}, atol={atol})"
    )
    return 0


def _copy_remote_differential_result(
    spec,
    remote_workspace: str,
    test_file: Path,
    operator_file: Path,
    verbose: bool = False,
    stderr=None,
) -> Path:
    result = run_remote_command_buffered(
        spec,
        remote_workspace,
        (
            "python3 -c "
            + repr(
                "import pathlib; "
                "matches = sorted(p.name for p in pathlib.Path('.').iterdir() "
                "if p.is_file() and p.name.lower() == 'test_result.pt'); "
                "print(matches[0] if matches else '')"
            )
        ),
        verbose=verbose,
        stderr=stderr,
    )
    stdout = str(result["stdout"])
    remote_name = stdout.strip().splitlines()[-1].strip() if stdout.strip() else ""
    if not remote_name:
        raise FileNotFoundError(
            f"Differential result payload not found in remote workspace for {test_file.name}"
        )
    archive_path = operator_file.parent / f"{operator_file.stem}_result.pt"
    copy_file_from_remote(
        spec,
        f"{remote_workspace}/{remote_name}",
        archive_path,
        verbose=verbose,
        stderr=stderr,
    )
    return archive_path
