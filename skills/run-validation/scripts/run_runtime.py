from __future__ import annotations

import errno
import os
import pty
import select
import shlex
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Optional, TextIO, TypedDict


class ResultPayload(TypedDict):
    return_code: int
    stdout: str
    stderr: str
    stalled: bool
    session_id: str | None


class RemoteSpec(TypedDict):
    user_host: str
    port: int | None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def emit_verbose(stderr: TextIO, category: str, message: str) -> None:
    print(f"[{category}] {message}", file=stderr)


def run_buffered_process(
    command: list[str],
    workdir: str,
    stall_timeout_seconds: int,
) -> ResultPayload:
    process = subprocess.Popen(
        command,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout_lines: list[str] = []
    start = time.monotonic()

    while True:
        line = process.stdout.readline() if process.stdout is not None else ""
        if line:
            stdout_lines.append(line)
            start = time.monotonic()
        elif process.poll() is not None:
            break
        elif time.monotonic() - start > stall_timeout_seconds:
            process.terminate()
            stderr_text = process.stderr.read() if process.stderr is not None else ""
            return make_result(
                return_code=1,
                stdout="".join(stdout_lines),
                stderr=stderr_text,
                stalled=True,
            )

    stderr_text = process.stderr.read() if process.stderr is not None else ""
    return make_result(
        return_code=process.returncode or 0,
        stdout="".join(stdout_lines),
        stderr=stderr_text,
    )


def run_streaming_process(
    command: list[str],
    workdir: str,
    stall_timeout_seconds: int,
    stdout: Optional[TextIO] = None,
) -> ResultPayload:
    master_fd, slave_fd = pty.openpty()
    output_chunks: list[str] = []
    process = subprocess.Popen(
        command,
        cwd=workdir,
        stdin=subprocess.DEVNULL,
        stdout=slave_fd,
        stderr=slave_fd,
        text=False,
        close_fds=True,
    )
    os.close(slave_fd)
    start = time.monotonic()

    try:
        while True:
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if ready:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError as error:
                    if error.errno == errno.EIO and process.poll() is not None:
                        break
                    raise
                if chunk:
                    text = chunk.decode(errors="replace")
                    output_chunks.append(text)
                    print(text, file=stdout or sys.stdout, end="")
                    start = time.monotonic()
                elif process.poll() is not None:
                    break
            elif process.poll() is not None:
                break
            elif time.monotonic() - start > stall_timeout_seconds:
                process.terminate()
                return make_result(
                    return_code=1,
                    stdout="".join(output_chunks),
                    stderr="",
                    stalled=True,
                )
        return make_result(return_code=process.wait(), stdout="".join(output_chunks), stderr="")
    finally:
        os.close(master_fd)


def parse_remote_spec(raw: str) -> RemoteSpec:
    if "@" not in raw:
        raise ValueError(f"Remote target must be in user@host[:port] form: {raw}")
    if ":" not in raw:
        return {"user_host": raw, "port": None}

    user_host, possible_port = raw.rsplit(":", 1)
    if not possible_port.isdigit():
        raise ValueError(f"Remote target port must be numeric: {raw}")
    return {"user_host": user_host, "port": int(possible_port)}


def create_remote_workspace(
    remote: str,
    remote_workdir: str | None,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[RemoteSpec, str]:
    spec = parse_remote_spec(remote)
    if remote_workdir:
        root = shlex.quote(remote_workdir)
        pattern = shlex.quote(str(PurePosixPath(remote_workdir) / "triton-agent-XXXXXX"))
        remote_command = f"mkdir -p {root} && mktemp -d {pattern}"
    else:
        remote_command = "mktemp -d"
    command = _ssh_command(spec, remote_command)
    _maybe_emit_remote_command(command, verbose, stderr)
    result = run_buffered_process(command, ".", stall_timeout_seconds=120)
    if not result_succeeded(result):
        raise RuntimeError(result["stderr"] or result["stdout"] or "Failed to create remote workspace.")
    workspace = result["stdout"].strip().splitlines()[-1].strip()
    if not workspace:
        raise RuntimeError("Remote workspace command did not return a path.")
    return spec, workspace


def cleanup_remote_workspace(
    spec: RemoteSpec,
    remote_workspace: str,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> None:
    command = _ssh_command(spec, f"rm -rf {shlex.quote(remote_workspace)}")
    _maybe_emit_remote_command(command, verbose, stderr)
    run_buffered_process(command, ".", stall_timeout_seconds=120)


def copy_file_to_remote(
    spec: RemoteSpec,
    local_path: Path,
    remote_path: str,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> None:
    command = _scp_to_remote_command(spec, local_path, remote_path)
    _maybe_emit_remote_command(command, verbose, stderr)
    result = run_buffered_process(command, ".", stall_timeout_seconds=300)
    if not result_succeeded(result):
        raise RuntimeError(result["stderr"] or result["stdout"] or f"Failed to copy {local_path} to remote.")


def copy_file_from_remote(
    spec: RemoteSpec,
    remote_path: str,
    local_path: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> None:
    command = _scp_from_remote_command(spec, remote_path, local_path)
    _maybe_emit_remote_command(command, verbose, stderr)
    result = run_buffered_process(command, ".", stall_timeout_seconds=300)
    if not result_succeeded(result):
        raise RuntimeError(result["stderr"] or result["stdout"] or f"Failed to copy {remote_path} from remote.")


def run_remote_command_streaming(
    spec: RemoteSpec,
    remote_workspace: str,
    remote_command: str,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> ResultPayload:
    command = _ssh_command(spec, f"cd {shlex.quote(remote_workspace)} && {remote_command}")
    _maybe_emit_remote_command(command, verbose, stderr)
    return run_streaming_process(command, ".", stall_timeout_seconds=900)


def run_remote_command_buffered(
    spec: RemoteSpec,
    remote_workspace: str,
    remote_command: str,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> ResultPayload:
    command = _ssh_command(spec, f"cd {shlex.quote(remote_workspace)} && {remote_command}")
    _maybe_emit_remote_command(command, verbose, stderr)
    return run_buffered_process(command, ".", stall_timeout_seconds=900)


def _ssh_command(spec: RemoteSpec, remote_command: str) -> list[str]:
    command = ["ssh"]
    if spec["port"] is not None:
        command.extend(["-p", str(spec["port"])])
    command.extend([spec["user_host"], f"bash -lc {shlex.quote(remote_command)}"])
    return command


def _scp_to_remote_command(spec: RemoteSpec, local_path: Path, remote_path: str) -> list[str]:
    command = ["scp"]
    if spec["port"] is not None:
        command.extend(["-P", str(spec["port"])])
    command.extend([str(local_path), f"{spec['user_host']}:{remote_path}"])
    return command


def _scp_from_remote_command(spec: RemoteSpec, remote_path: str, local_path: Path) -> list[str]:
    command = ["scp"]
    if spec["port"] is not None:
        command.extend(["-P", str(spec["port"])])
    command.extend([f"{spec['user_host']}:{remote_path}", str(local_path)])
    return command


def _maybe_emit_remote_command(command: list[str], verbose: bool, stderr: TextIO | None) -> None:
    if not verbose or stderr is None:
        return
    emit_verbose(stderr, "remote", f"command: {shlex.join(command)}")


def make_result(
    *,
    return_code: int,
    stdout: str,
    stderr: str,
    stalled: bool = False,
    session_id: str | None = None,
) -> ResultPayload:
    return {
        "return_code": return_code,
        "stdout": stdout,
        "stderr": stderr,
        "stalled": stalled,
        "session_id": session_id,
    }


def result_succeeded(result: ResultPayload) -> bool:
    return result["return_code"] == 0 and not result["stalled"]
