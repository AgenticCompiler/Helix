from __future__ import annotations

import re
from pathlib import Path
from typing import TextIO

from debug_device import maybe_print_visible_devices
from env_registry import TRITON_ALWAYS_COMPILE
from result_payload import ResultPayload, make_result
from run_runtime import (
    RemoteSpec,
    cleanup_remote_workspace,
    copy_file_from_remote,
    copy_file_to_remote,
    copy_files_to_remote,
    create_remote_workspace,
    eval_timeout_seconds,
    result_succeeded,
    run_remote_command_streaming,
)
from test_contract import deserialize_payload_object, run_test_accuracy_env
from run_test_result import differential_archive_path, filter_result_payload


SCRIPT_DIR = Path(__file__).resolve().parent


_SERIALIZED_PAYLOAD_BEGIN = "__HELIX_SERIALIZED_PAYLOAD_BEGIN__"
_SERIALIZED_PAYLOAD_END = "__HELIX_SERIALIZED_PAYLOAD_END__"
_BASE64_SERIALIZED_PAYLOAD_LINE = re.compile(r"^[A-Za-z0-9+/=]+$")


def run_remote_test(
    test_file: Path,
    operator_file: Path,
    test_mode: str,
    remote: str,
    remote_workdir: str | None,
    *,
    case_id: str | None = None,
    accuracy_mode: str | None = None,
    keep_remote_workdir: bool = False,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[ResultPayload, Path | None, str]:
    maybe_print_visible_devices()
    spec, remote_workspace = create_remote_workspace(
        remote, remote_workdir, verbose=verbose, stderr=stderr
    )
    try:
        _copy_run_test_runtime(spec, test_file, operator_file, remote_workspace, verbose, stderr)
        extra_env = {
            TRITON_ALWAYS_COMPILE: "1",
            **run_test_accuracy_env(accuracy_mode),
        }
        if test_mode == "standalone":
            if case_id is not None:
                raise ValueError("--case-id is supported only with differential tests.")
            result = run_remote_command_streaming(
                spec,
                remote_workspace,
                _build_remote_worker_command(
                    test_file.name, operator_file.name, "standalone"
                ),
                stall_timeout_seconds=eval_timeout_seconds(),
                verbose=verbose,
                stderr=stderr,
                extra_env=extra_env,
            )
            return filter_result_payload(result, verbose=verbose), None, remote_workspace
        if test_mode == "differential":
            archive_path = differential_archive_path(operator_file)
            result = run_remote_command_streaming(
                spec,
                remote_workspace,
                _build_remote_worker_command(
                    test_file.name, operator_file.name, "differential", case_id=case_id
                ),
                stall_timeout_seconds=eval_timeout_seconds(),
                verbose=verbose,
                stderr=stderr,
                extra_env=extra_env,
            )
            archived_result = None
            if result_succeeded(result):
                archived_result = _copy_remote_differential_archive(
                    spec,
                    remote_workspace,
                    archive_path,
                    verbose=verbose,
                    stderr=stderr,
                )
            return filter_result_payload(result, verbose=verbose), archived_result, remote_workspace
        raise ValueError(f"Unsupported test mode: {test_mode}")
    finally:
        if not keep_remote_workdir:
            cleanup_remote_workspace(spec, remote_workspace, verbose=verbose, stderr=stderr)


def run_remote_test_case_payload(
    test_file: Path,
    operator_file: Path,
    remote: str,
    remote_workdir: str | None,
    *,
    case_id: str,
    accuracy_mode: str | None = None,
    keep_remote_workdir: bool = False,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[ResultPayload, object | None, str]:
    maybe_print_visible_devices()
    spec, remote_workspace = create_remote_workspace(
        remote, remote_workdir, verbose=verbose, stderr=stderr
    )
    try:
        _copy_run_test_runtime(spec, test_file, operator_file, remote_workspace, verbose, stderr)
        result = run_remote_command_streaming(
            spec,
            remote_workspace,
            _build_remote_worker_command(
                test_file.name,
                operator_file.name,
                "differential",
                case_id=case_id,
                archive_result=False,
                emit_serialized_payload=True,
            ),
            stall_timeout_seconds=eval_timeout_seconds(),
            verbose=verbose,
            stderr=stderr,
            extra_env={
                TRITON_ALWAYS_COMPILE: "1",
                **run_test_accuracy_env(accuracy_mode),
            },
        )
        result, case_payload = _extract_serialized_payload_result(result)
        if result_succeeded(result) and case_payload is None:
            return (
                make_result(
                    return_code=1,
                    stdout=str(result["stdout"]),
                    stderr=str(result["stderr"]) + "Remote test payload helper did not return case payload.",
                    stalled=bool(result["stalled"]),
                    session_id=result["session_id"],
                ),
                None,
                remote_workspace,
            )
        return filter_result_payload(result, verbose=verbose), case_payload, remote_workspace
    finally:
        if not keep_remote_workdir:
            cleanup_remote_workspace(spec, remote_workspace, verbose=verbose, stderr=stderr)


def run_remote_differential_comparison(
    test_file: Path,
    ref_operator_file: Path,
    operator_file: Path,
    remote: str,
    remote_workdir: str | None,
    *,
    case_id: str | None = None,
    accuracy_mode: str | None = None,
    keep_remote_workdir: bool = False,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[ResultPayload, str]:
    """Run and compare differential archives without moving PT payloads locally."""
    maybe_print_visible_devices()
    spec, remote_workspace = create_remote_workspace(
        remote, remote_workdir, verbose=verbose, stderr=stderr
    )
    remote_test = f"{remote_workspace}/{test_file.name}"
    remote_ref_operator = f"{remote_workspace}/reference_{ref_operator_file.name}"
    remote_operator = f"{remote_workspace}/candidate_{operator_file.name}"
    extra_env = {
        TRITON_ALWAYS_COMPILE: "1",
        **run_test_accuracy_env(accuracy_mode),
    }
    try:
        compare_script = SCRIPT_DIR / "compare_result.py"
        copy_file_to_remote(spec, test_file, remote_test, verbose=verbose, stderr=stderr)
        copy_file_to_remote(
            spec, ref_operator_file, remote_ref_operator, verbose=verbose, stderr=stderr
        )
        copy_file_to_remote(spec, operator_file, remote_operator, verbose=verbose, stderr=stderr)
        copy_files_to_remote(
            spec,
            [*(SCRIPT_DIR / filename for filename in _RUN_TEST_RUNTIME_FILENAMES), compare_script],
            remote_workspace,
            verbose=verbose,
            stderr=stderr,
        )
        reference_result = run_remote_command_streaming(
            spec,
            remote_workspace,
            _build_remote_worker_command(
                test_file.name,
                Path(remote_ref_operator).name,
                "differential",
                case_id=case_id,
            ),
            stall_timeout_seconds=eval_timeout_seconds(),
            verbose=verbose,
            stderr=stderr,
            extra_env=extra_env,
        )
        if not result_succeeded(reference_result):
            return filter_result_payload(reference_result, verbose=verbose), remote_workspace
        candidate_result = run_remote_command_streaming(
            spec,
            remote_workspace,
            _build_remote_worker_command(
                test_file.name,
                Path(remote_operator).name,
                "differential",
                case_id=case_id,
            ),
            stall_timeout_seconds=eval_timeout_seconds(),
            verbose=verbose,
            stderr=stderr,
            extra_env=extra_env,
        )
        if not result_succeeded(candidate_result):
            return filter_result_payload(candidate_result, verbose=verbose), remote_workspace
        compare_command = [
            "python3",
            compare_script.name,
            "--ref-result",
            f"reference_{ref_operator_file.stem}_result.pt",
            "--new-result",
            f"candidate_{operator_file.stem}_result.pt",
        ]
        if accuracy_mode is not None:
            compare_command.extend(["--accuracy-mode", accuracy_mode])
        compare_result = run_remote_command_streaming(
            spec,
            remote_workspace,
            compare_command,
            stall_timeout_seconds=eval_timeout_seconds(),
            verbose=verbose,
            stderr=stderr,
            extra_env=extra_env,
        )
        return filter_result_payload(compare_result, verbose=verbose), remote_workspace
    finally:
        if not keep_remote_workdir:
            cleanup_remote_workspace(spec, remote_workspace, verbose=verbose, stderr=stderr)


def _copy_remote_differential_archive(
    spec: RemoteSpec,
    remote_workspace: str,
    archive_path: Path,
    *,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> Path:
    copy_file_from_remote(
        spec,
        f"{remote_workspace}/{archive_path.name}",
        archive_path,
        verbose=verbose,
        stderr=stderr,
    )
    return archive_path


_RUN_TEST_RUNTIME_FILENAMES = (
    "npu_compare.py",
    "dtype_close_compare.py",
    "npu_compare_common.py",
    "npu_contract_compare.py",
    "env_registry.py",
    "run_test_remote_worker.py",
    "test_contract.py",
    "torch_npu_warnings.py",
)


def _copy_run_test_runtime(
    spec: RemoteSpec,
    test_file: Path,
    operator_file: Path,
    remote_workspace: str,
    verbose: bool,
    stderr: TextIO | None,
) -> None:
    copy_files_to_remote(
        spec,
        [
            test_file,
            operator_file,
            *(SCRIPT_DIR / filename for filename in _RUN_TEST_RUNTIME_FILENAMES),
        ],
        remote_workspace,
        verbose=verbose,
        stderr=stderr,
    )


def _build_remote_worker_command(
    test_name: str,
    operator_name: str,
    test_mode: str,
    *,
    case_id: str | None = None,
    archive_result: bool = True,
    emit_serialized_payload: bool = False,
) -> list[str]:
    command = [
        "python3",
        "run_test_remote_worker.py",
        "--test-file",
        test_name,
        "--operator-file",
        operator_name,
        "--test-mode",
        test_mode,
    ]
    if case_id is not None:
        command.extend(["--case-id", case_id])
    if not archive_result:
        command.append("--no-archive")
    if emit_serialized_payload:
        command.append("--emit-serialized-payload")
    return command


def _extract_serialized_payload_result(result: ResultPayload) -> tuple[ResultPayload, object | None]:
    stdout = str(result["stdout"])
    match = re.search(
        rf"(?s)(?P<prefix>.*?){re.escape(_SERIALIZED_PAYLOAD_BEGIN)}\r?\n"
        rf"(?P<payload>.*?)\r?\n{re.escape(_SERIALIZED_PAYLOAD_END)}(?P<suffix>.*)",
        stdout,
    )
    if match is None:
        return result, None
    prefix = match.group("prefix")
    serialized_payload_block = match.group("payload")
    suffix = match.group("suffix")
    clean_stdout = prefix + suffix.lstrip("\r\n")
    clean_result = make_result(
        return_code=int(result["return_code"]),
        stdout=clean_stdout,
        stderr=str(result["stderr"]),
        stalled=bool(result["stalled"]),
        session_id=result["session_id"],
    )
    return clean_result, _extract_serialized_payload_object(serialized_payload_block)


def _extract_serialized_payload_object(serialized_payload_block: str) -> object | None:
    for line in reversed(serialized_payload_block.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        if _BASE64_SERIALIZED_PAYLOAD_LINE.fullmatch(candidate) is None:
            continue
        try:
            return deserialize_payload_object(candidate)
        except Exception:
            continue
    return None
