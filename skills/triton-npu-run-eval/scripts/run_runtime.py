from __future__ import annotations

import errno
import os
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Optional, TextIO, TypedDict, cast

from result_payload import ResultPayload, make_result

_IS_WINDOWS = sys.platform == "win32"
_BLOCKS_PARALLEL_ENV = "TRITON_ALL_BLOCKS_PARALLEL"
_BLOCKS_PARALLEL_UNSAFE_VALUE = "1"
_BLOCKS_PARALLEL_SAFE_VALUE = "0"

if not _IS_WINDOWS:
    import pty
    import select
else:
    pty = None
    select = None

class RemoteSpec(TypedDict):
    user_host: str
    port: int | None


def local_python_executable() -> str:
    configured = os.environ.get("TRITON_AGENT_PYTHON", "").strip()
    if configured:
        return configured
    if getattr(sys, "frozen", False):
        return "python"
    return sys.executable

def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {raw!r}")
    return value


def _ssh_timeout() -> int:
    return env_int("TRITON_AGENT_SSH_TIMEOUT_SECONDS", 120)


def _scp_timeout() -> int:
    return env_int("TRITON_AGENT_SCP_TIMEOUT_SECONDS", 300)


def eval_stall_timeout_seconds() -> int:
    return env_int("TRITON_AGENT_EVAL_TIMEOUT_SECONDS", 300)


def emit_verbose(stderr: TextIO, category: str, message: str) -> None:
    print(f"[{category}] {message}", file=stderr)


def run_buffered_process(
    command: list[str],
    workdir: str,
    stall_timeout_seconds: int,
    extra_env: dict[str, str] | None = None,
) -> ResultPayload:
    process = subprocess.Popen(
        command,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_merged_env(extra_env),
    )
    stdout_pipe = process.stdout
    stderr_pipe = process.stderr
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def _drain_stderr() -> None:
        if stderr_pipe is None:
            return
        try:
            for line in stderr_pipe:
                stderr_lines.append(line)
        except ValueError:
            pass  # pipe closed by parent

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    start = time.monotonic()

    try:
        while True:
            line = stdout_pipe.readline() if stdout_pipe is not None else ""
            if line:
                stdout_lines.append(line)
                start = time.monotonic()
            elif process.poll() is not None:
                break
            elif stall_timeout_seconds > 0 and time.monotonic() - start > stall_timeout_seconds:
                process.terminate()
                stderr_thread.join(timeout=5)
                return make_result(
                    return_code=1,
                    stdout="".join(stdout_lines),
                    stderr="".join(stderr_lines),
                    stalled=True,
                )

        stderr_thread.join(timeout=5)
        return make_result(
            return_code=_resolved_returncode(process.returncode),
            stdout="".join(stdout_lines),
            stderr="".join(stderr_lines),
        )
    finally:
        if stdout_pipe is not None:
            close_stdout = getattr(stdout_pipe, "close", None)
            if callable(close_stdout):
                close_stdout()
        if stderr_pipe is not None:
            close_stderr = getattr(stderr_pipe, "close", None)
            if callable(close_stderr):
                close_stderr()


def run_streaming_process(
    command: list[str],
    workdir: str,
    stall_timeout_seconds: int,
    stdout: Optional[TextIO] = None,
    extra_env: dict[str, str] | None = None,
) -> ResultPayload:
    if _IS_WINDOWS:
        return _run_streaming_windows(command, workdir, stall_timeout_seconds, stdout, extra_env)
    return _run_streaming_pty(command, workdir, stall_timeout_seconds, stdout, extra_env)


def _run_streaming_windows(
    command: list[str],
    workdir: str,
    stall_timeout_seconds: int,
    stdout: Optional[TextIO] = None,
    extra_env: dict[str, str] | None = None,
) -> ResultPayload:
    process = subprocess.Popen(
        command,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        env=_merged_env(extra_env),
    )
    output_chunks: list[str] = []
    start_ref: list[float] = [time.monotonic()]
    stalled_ref: list[bool] = [False]
    lock = threading.Lock()

    def reader() -> None:
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode(errors="replace")
            with lock:
                output_chunks.append(text)
                print(text, file=stdout or sys.stdout, end="", flush=True)
                start_ref[0] = time.monotonic()

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    while True:
        reader_thread.join(timeout=0.1)
        if not reader_thread.is_alive() and process.poll() is not None:
            break
        with lock:
            elapsed = time.monotonic() - start_ref[0]
        if stall_timeout_seconds > 0 and elapsed > stall_timeout_seconds:
            process.terminate()
            stalled_ref[0] = True
            break

    reader_thread.join()
    rc = process.wait() if not stalled_ref[0] else 1
    return make_result(
        return_code=rc,
        stdout="".join(output_chunks),
        stderr="",
        stalled=stalled_ref[0],
    )


def _run_streaming_pty(
    command: list[str],
    workdir: str,
    stall_timeout_seconds: int,
    stdout: Optional[TextIO] = None,
    extra_env: dict[str, str] | None = None,
) -> ResultPayload:
    pty_module = pty
    select_module = select
    if pty_module is None or select_module is None:
        raise RuntimeError("PTY streaming is unavailable on this platform")
    master_fd, slave_fd = cast(Any, pty_module).openpty()
    output_chunks: list[str] = []
    process = subprocess.Popen(
        command,
        cwd=workdir,
        stdin=subprocess.DEVNULL,
        stdout=slave_fd,
        stderr=slave_fd,
        text=False,
        close_fds=True,
        env=_merged_env(extra_env),
    )
    os.close(slave_fd)
    start = time.monotonic()

    try:
        while True:
            ready, _, _ = cast(Any, select_module).select([master_fd], [], [], 0.1)
            if ready:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError as error:
                    if error.errno == errno.EIO:
                        # PTY slave closed — normal child exit signal.
                        # poll() may not reflect termination yet due to a race,
                        # so wait briefly before treating this as clean shutdown.
                        if process.poll() is None:
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                pass
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
            elif stall_timeout_seconds > 0 and time.monotonic() - start > stall_timeout_seconds:
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
    result = run_buffered_process(command, ".", stall_timeout_seconds=_ssh_timeout())
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
    run_buffered_process(command, ".", stall_timeout_seconds=_ssh_timeout())


def copy_file_to_remote(
    spec: RemoteSpec,
    local_path: Path,
    remote_path: str,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> None:
    command = _scp_to_remote_command(spec, local_path, remote_path)
    _maybe_emit_remote_command(command, verbose, stderr)
    result = run_buffered_process(command, ".", stall_timeout_seconds=_scp_timeout())
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
    result = run_buffered_process(command, ".", stall_timeout_seconds=_scp_timeout())
    if not result_succeeded(result):
        raise RuntimeError(result["stderr"] or result["stdout"] or f"Failed to copy {remote_path} from remote.")


def copy_directory_from_remote(
    spec: RemoteSpec,
    remote_path: str,
    local_path: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    command = _scp_from_remote_command(spec, remote_path, local_path, recursive=True)
    _maybe_emit_remote_command(command, verbose, stderr)
    result = run_buffered_process(command, ".", stall_timeout_seconds=_scp_timeout())
    if not result_succeeded(result):
        raise RuntimeError(
            result["stderr"] or result["stdout"] or f"Failed to copy directory {remote_path} from remote."
        )


def run_remote_command_streaming(
    spec: RemoteSpec,
    remote_workspace: str,
    remote_command: str | Sequence[str],
    stdout: TextIO | None = None,
    verbose: bool = False,
    stderr: TextIO | None = None,
    extra_env: dict[str, str] | None = None,
    stall_timeout_seconds: int | None = None,
) -> ResultPayload:
    env_prefix = _shell_env_prefix(extra_env)
    command_text = _normalize_remote_command(remote_command)
    command = _ssh_command(
        spec,
        f"cd {shlex.quote(remote_workspace)} && {env_prefix + ' ' if env_prefix else ''}{command_text}",
    )
    _maybe_emit_remote_command(command, verbose, stderr)
    timeout = stall_timeout_seconds if stall_timeout_seconds is not None else eval_stall_timeout_seconds()
    return run_streaming_process(command, ".", stall_timeout_seconds=timeout, stdout=stdout)


def run_remote_command_buffered(
    spec: RemoteSpec,
    remote_workspace: str,
    remote_command: str | Sequence[str],
    verbose: bool = False,
    stderr: TextIO | None = None,
    extra_env: dict[str, str] | None = None,
    stall_timeout_seconds: int | None = None,
) -> ResultPayload:
    env_prefix = _shell_env_prefix(extra_env)
    command_text = _normalize_remote_command(remote_command)
    command = _ssh_command(
        spec,
        f"cd {shlex.quote(remote_workspace)} && {env_prefix + ' ' if env_prefix else ''}{command_text}",
    )
    _maybe_emit_remote_command(command, verbose, stderr)
    timeout = stall_timeout_seconds if stall_timeout_seconds is not None else eval_stall_timeout_seconds()
    return run_buffered_process(command, ".", stall_timeout_seconds=timeout)


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
    command.extend([local_path.as_posix(), f"{spec['user_host']}:{remote_path}"])
    return command


def _scp_from_remote_command(
    spec: RemoteSpec,
    remote_path: str,
    local_path: Path,
    recursive: bool = False,
) -> list[str]:
    command = ["scp"]
    if recursive:
        command.append("-r")
    if spec["port"] is not None:
        command.extend(["-P", str(spec["port"])])
    command.extend([f"{spec['user_host']}:{remote_path}", local_path.as_posix()])
    return command


def _maybe_emit_remote_command(command: list[str], verbose: bool, stderr: TextIO | None) -> None:
    if not verbose or stderr is None:
        return
    emit_verbose(stderr, "remote", f"command: {shlex.join(command)}")


def result_succeeded(result: ResultPayload) -> bool:
    return result["return_code"] == 0 and not result["stalled"]


def _normalize_remote_command(remote_command: str | Sequence[str]) -> str:
    if isinstance(remote_command, str):
        return remote_command
    return shlex.join(str(part) for part in remote_command)


def _resolved_returncode(returncode: int | None) -> int:
    return returncode if returncode is not None else 1


def _normalized_execution_extra_env(extra_env: dict[str, str] | None) -> dict[str, str]:
    normalized = {} if extra_env is None else dict(extra_env)
    blocks_parallel = normalized.get(_BLOCKS_PARALLEL_ENV)
    if blocks_parallel == _BLOCKS_PARALLEL_UNSAFE_VALUE:
        normalized[_BLOCKS_PARALLEL_ENV] = _BLOCKS_PARALLEL_SAFE_VALUE
    elif blocks_parallel is None and os.environ.get(_BLOCKS_PARALLEL_ENV) == _BLOCKS_PARALLEL_SAFE_VALUE:
        normalized[_BLOCKS_PARALLEL_ENV] = _BLOCKS_PARALLEL_SAFE_VALUE
    return normalized


def _merged_env(extra_env: dict[str, str] | None) -> dict[str, str] | None:
    normalized = _normalized_execution_extra_env(extra_env)
    if extra_env is None and not normalized:
        return None
    merged = dict(os.environ)
    merged.update(normalized)
    return merged


def _shell_env_prefix(extra_env: dict[str, str] | None) -> str:
    normalized = _normalized_execution_extra_env(extra_env)
    if not normalized:
        return ""
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(normalized.items()))
