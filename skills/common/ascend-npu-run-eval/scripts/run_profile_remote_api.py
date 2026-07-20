"""Remote workspace API for benchmark profile execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, TextIO, cast

from run_bench_modes import stream_target_for_verbosity
from env_registry import HELIX_PROFILE_TIMEOUT_SECONDS, TRITON_ALWAYS_COMPILE
from remote_python_bundle import stage_remote_python_bundle
from run_profile_execution import profile_timeout, validate_profile_dir
from result_payload import ResultPayload, make_result
from run_runtime import (
    cleanup_remote_workspace,
    copy_directory_from_remote,
    copy_file_to_remote,
    create_remote_workspace,
    result_succeeded,
    run_remote_command_buffered,
    run_remote_command_streaming,
    RemoteSpec,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def run_remote_profile_bench(
    bench_file: Path,
    operator_file: Path,
    remote: str,
    remote_workdir: str | None,
    case_id: str | None = None,
    kernel_name: str | None = None,
    keep_remote_workdir: bool = False,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[ResultPayload, Path | None, str]:
    del kernel_name
    if case_id is None:
        raise ValueError("torch-npu-profiler benchmark profiling requires --case-id <id>.")
    spec, workspace = create_remote_workspace(remote, remote_workdir, verbose=verbose, stderr=stderr)
    try:
        _stage_remote_python_bundle(
            spec,
            workspace,
            verbose=verbose,
            stderr=stderr,
        )
        for source in [bench_file, operator_file]:
            copy_file_to_remote(spec, source, f"{workspace}/{source.name}", verbose=verbose, stderr=stderr)
        result_name = "profile-result.json"
        with stream_target_for_verbosity(verbose) as stream_target:
            worker_result = run_remote_command_streaming(
                spec,
                workspace,
                [
                    "python3",
                    "run_profile_remote_worker.py",
                    "--bench-file",
                    bench_file.name,
                    "--operator-file",
                    operator_file.name,
                    "--case-id",
                    case_id,
                    "--result-file",
                    result_name,
                ],
                stdout=stream_target,
                stall_timeout_seconds=profile_timeout(),
                timeout_env_name=HELIX_PROFILE_TIMEOUT_SECONDS,
                verbose=verbose,
                stderr=stderr,
                extra_env={TRITON_ALWAYS_COMPILE: "1"},
            )
        if not result_succeeded(worker_result):
            return worker_result, None, workspace
        payload_result = run_remote_command_buffered(
            spec, workspace, ["cat", result_name], verbose=verbose, stderr=stderr
        )
        if not result_succeeded(payload_result):
            return payload_result, None, workspace
        result, profile_name = _read_payload(payload_result)
        if not result_succeeded(result) or profile_name is None:
            return result, None, workspace
        local_profile_dir = operator_file.parent / profile_name
        if local_profile_dir.exists():
            raise FileExistsError(f"Local profile directory already exists: {local_profile_dir}")
        copy_directory_from_remote(
            spec, f"{workspace}/{profile_name}", local_profile_dir, verbose=verbose, stderr=stderr
        )
        validate_profile_dir(local_profile_dir)
        return result, local_profile_dir, workspace
    finally:
        if not keep_remote_workdir:
            cleanup_remote_workspace(spec, workspace, verbose=verbose, stderr=stderr)


def _read_payload(payload_result: ResultPayload) -> tuple[ResultPayload, str | None]:
    payload_text = str(payload_result["stdout"])
    payload_line = next(
        (line for line in reversed(payload_text.splitlines()) if line.lstrip().startswith("{")),
        payload_text,
    )
    payload = json.loads(payload_line)
    raw = payload["result"]
    result = make_result(
        return_code=int(raw["return_code"]),
        stdout=str(raw["stdout"]),
        stderr=str(raw["stderr"]),
        stalled=bool(raw["stalled"]),
        session_id=cast(Optional[str], raw["session_id"]),
    )
    profile_name = payload.get("profile_name")
    return result, None if profile_name is None else str(profile_name)


def _stage_remote_python_bundle(
    spec: RemoteSpec,
    workspace: str,
    *,
    verbose: bool,
    stderr: TextIO | None,
) -> None:
    stage_remote_python_bundle(
        [SCRIPT_DIR / "run_profile_remote_worker.py"],
        workspace,
        lambda source, target: copy_file_to_remote(
            spec, source, target, verbose=verbose, stderr=stderr
        ),
    )
