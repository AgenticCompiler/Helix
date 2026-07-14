from __future__ import annotations

import errno
import locale
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Optional, TextIO, TypedDict, cast

from env_registry import (
    HELIX_EVAL_TIMEOUT_SECONDS,
    HELIX_PYTHON,
    HELIX_SCP_TIMEOUT_SECONDS,
    HELIX_SSH_TIMEOUT_SECONDS,
    TRITON_ALL_BLOCKS_PARALLEL,
)
from result_payload import ResultPayload, make_result

_IS_WINDOWS = sys.platform == "win32"
_BLOCKS_PARALLEL_UNSAFE_VALUE = "1"
_BLOCKS_PARALLEL_SAFE_VALUE = "0"
_NPU_COMPARE_RUNTIME_FILES = (
    "npu_compare.py",
    "dtype_close_compare.py",
    "npu_compare_common.py",
    "npu_contract_compare.py",
    "env_registry.py",
)

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
    configured = os.environ.get(HELIX_PYTHON, "").strip()
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
    return env_int(HELIX_SSH_TIMEOUT_SECONDS, 120)


def _scp_timeout() -> int:
    return env_int(HELIX_SCP_TIMEOUT_SECONDS, 300)


def eval_timeout_seconds() -> int:
    return env_int(HELIX_EVAL_TIMEOUT_SECONDS, 300)


def emit_verbose(stderr: TextIO, category: str, message: str) -> None:
    print(f"[{category}] {message}", file=stderr)


def _timeout_message(timeout_seconds: float) -> str:
    return (
        f"Evaluation timed out after {timeout_seconds:g} seconds "
        "(HELIX_EVAL_TIMEOUT_SECONDS); the current operator execution exceeded the limit.\n"
    )


def _iter_pipe_chunks(stream: Any) -> Iterator[bytes | str]:
    if stream is None:
        return
    read = getattr(stream, "read", None)
    if callable(read):
        while True:
            try:
                try:
                    chunk = read(4096)
                except TypeError:
                    chunk = read()
            except ValueError:
                return
            if not chunk:
                return
            if isinstance(chunk, (bytes, str)):
                yield chunk
        return
    readline = getattr(stream, "readline", None)
    if callable(readline):
        while True:
            try:
                chunk = readline()
            except ValueError:
                return
            if not chunk:
                return
            if isinstance(chunk, (bytes, str)):
                yield chunk
        return
    try:
        for chunk in stream:
            if isinstance(chunk, (bytes, str)):
                yield chunk
    except ValueError:
        return


def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if _IS_WINDOWS:
        try:
            completed = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if completed.returncode != 0:
                process.terminate()
        except OSError:
            process.terminate()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=1)
        return
    except (AttributeError, TypeError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        kill = getattr(process, "kill", None)
        if callable(kill):
            kill()


def _process_group_popen_kwargs() -> dict[str, Any]:
    if _IS_WINDOWS:
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def run_buffered_process(
    command: list[str],
    workdir: str,
    stall_timeout_seconds: int,
    extra_env: dict[str, str] | None = None,
    *,
    timeout_seconds: float | None = None,
) -> ResultPayload:
    process = subprocess.Popen(
        command,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        env=_merged_env(extra_env),
        **_process_group_popen_kwargs(),
    )
    stdout_pipe = process.stdout
    stderr_pipe = process.stderr
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    lock = threading.Lock()
    start = time.monotonic()
    last_output = [start]

    def _drain(stream: Any, chunks: list[str]) -> None:
        for chunk in _iter_pipe_chunks(stream):
            text = _coerce_output_text(chunk)
            with lock:
                chunks.append(text)
                last_output[0] = time.monotonic()

    stdout_thread = threading.Thread(target=_drain, args=(stdout_pipe, stdout_lines), daemon=True)
    stderr_thread = threading.Thread(target=_drain, args=(stderr_pipe, stderr_lines), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    readers = (stdout_thread, stderr_thread)

    try:
        while True:
            if process.poll() is not None:
                break
            elapsed = time.monotonic() - start
            if timeout_seconds is not None and timeout_seconds > 0 and elapsed > timeout_seconds:
                _terminate_process_tree(process)
                for reader in readers:
                    reader.join(timeout=1)
                return make_result(
                    return_code=1,
                    stdout="".join(stdout_lines),
                    stderr="".join(stderr_lines) + _timeout_message(timeout_seconds),
                    stalled=True,
                )
            if stall_timeout_seconds > 0:
                with lock:
                    elapsed_since_output = time.monotonic() - last_output[0]
                if elapsed_since_output > stall_timeout_seconds:
                    _terminate_process_tree(process)
                    for reader in readers:
                        reader.join(timeout=1)
                    return make_result(
                        return_code=1,
                        stdout="".join(stdout_lines),
                        stderr="".join(stderr_lines),
                        stalled=True,
                    )
            time.sleep(0.05)

        for reader in readers:
            reader.join(timeout=1)
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
    *,
    timeout_seconds: float | None = None,
) -> ResultPayload:
    if _IS_WINDOWS:
        return _run_streaming_windows(
            command,
            workdir,
            stall_timeout_seconds,
            stdout,
            extra_env,
            timeout_seconds=timeout_seconds,
        )
    return _run_streaming_pty(
        command,
        workdir,
        stall_timeout_seconds,
        stdout,
        extra_env,
        timeout_seconds=timeout_seconds,
    )


def _run_streaming_windows(
    command: list[str],
    workdir: str,
    stall_timeout_seconds: int,
    stdout: Optional[TextIO] = None,
    extra_env: dict[str, str] | None = None,
    *,
    timeout_seconds: float | None = None,
) -> ResultPayload:
    process = subprocess.Popen(
        command,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        env=_merged_env(extra_env),
        **_process_group_popen_kwargs(),
    )
    output_chunks: list[str] = []
    started_at = time.monotonic()
    start = started_at
    start_ref: list[float] = [start]
    stalled_ref: list[bool] = [False]
    timed_out_ref: list[bool] = [False]
    lock = threading.Lock()

    def reader() -> None:
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            text = _coerce_output_text(chunk)
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
        if timeout_seconds is not None and timeout_seconds > 0 and time.monotonic() - start > timeout_seconds:
            _terminate_process_tree(process)
            stalled_ref[0] = True
            timed_out_ref[0] = True
            break
        if stall_timeout_seconds > 0 and elapsed > stall_timeout_seconds:
            _terminate_process_tree(process)
            stalled_ref[0] = True
            break

    reader_thread.join()
    rc = process.wait() if not stalled_ref[0] else 1
    return make_result(
        return_code=rc,
        stdout="".join(output_chunks),
        stderr=_timeout_message(timeout_seconds) if timed_out_ref[0] and timeout_seconds is not None else "",
        stalled=stalled_ref[0],
    )


def _run_streaming_pty(
    command: list[str],
    workdir: str,
    stall_timeout_seconds: int,
    stdout: Optional[TextIO] = None,
    extra_env: dict[str, str] | None = None,
    *,
    timeout_seconds: float | None = None,
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
        **_process_group_popen_kwargs(),
    )
    os.close(slave_fd)
    started_at = time.monotonic()
    start = started_at

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
                    text = _coerce_output_text(chunk)
                    output_chunks.append(text)
                    print(text, file=stdout or sys.stdout, end="")
                    start = time.monotonic()
                elif process.poll() is not None:
                    break
            elif process.poll() is not None:
                break
            elif timeout_seconds is not None and timeout_seconds > 0 and time.monotonic() - started_at > timeout_seconds:
                _terminate_process_tree(process)
                return make_result(
                    return_code=1,
                    stdout="".join(output_chunks),
                    stderr=_timeout_message(timeout_seconds),
                    stalled=True,
                )
            elif stall_timeout_seconds > 0 and time.monotonic() - start > stall_timeout_seconds:
                _terminate_process_tree(process)
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
    candidate = raw.strip()
    if not candidate:
        raise ValueError("Remote target must not be empty.")
    if ":" not in candidate:
        return {"user_host": candidate, "port": None}

    user_host, possible_port = candidate.rsplit(":", 1)
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
        pattern = shlex.quote(str(PurePosixPath(remote_workdir) / "helix-XXXXXX"))
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
    local_workdir, local_arg = _scp_local_operand(local_path)
    command = _scp_to_remote_command(spec, local_arg, remote_path)
    _maybe_emit_remote_command(command, verbose, stderr)
    result = run_buffered_process(command, local_workdir, stall_timeout_seconds=_scp_timeout())
    if not result_succeeded(result):
        raise RuntimeError(result["stderr"] or result["stdout"] or f"Failed to copy {local_path} to remote.")


def copy_file_from_remote(
    spec: RemoteSpec,
    remote_path: str,
    local_path: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> None:
    local_workdir, local_arg = _scp_local_operand(local_path)
    command = _scp_from_remote_command(spec, remote_path, local_arg)
    _maybe_emit_remote_command(command, verbose, stderr)
    result = run_buffered_process(command, local_workdir, stall_timeout_seconds=_scp_timeout())
    if not result_succeeded(result):
        raise RuntimeError(result["stderr"] or result["stdout"] or f"Failed to copy {remote_path} from remote.")


def copy_npu_compare_runtime_to_remote(
    spec: RemoteSpec,
    script_dir: Path,
    remote_workspace: str,
    verbose: bool = False,
    stderr: TextIO | None = None,
    copy_file_fn: Callable[[RemoteSpec, Path, str, bool, TextIO | None], None] | None = None,
) -> None:
    copier = copy_file_to_remote if copy_file_fn is None else copy_file_fn
    for filename in _NPU_COMPARE_RUNTIME_FILES:
        copier(
            spec,
            script_dir / filename,
            f"{remote_workspace}/{filename}",
            verbose,
            stderr,
        )


def copy_directory_from_remote(
    spec: RemoteSpec,
    remote_path: str,
    local_path: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_workdir, local_arg = _scp_local_operand(local_path)
    command = _scp_from_remote_command(spec, remote_path, local_arg, recursive=True)
    _maybe_emit_remote_command(command, verbose, stderr)
    result = run_buffered_process(command, local_workdir, stall_timeout_seconds=_scp_timeout())
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
    timeout = stall_timeout_seconds if stall_timeout_seconds is not None else eval_timeout_seconds()
    return run_streaming_process(
        command,
        ".",
        stall_timeout_seconds=0,
        stdout=stdout,
        timeout_seconds=timeout,
    )


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
    timeout = stall_timeout_seconds if stall_timeout_seconds is not None else eval_timeout_seconds()
    return run_buffered_process(
        command,
        ".",
        stall_timeout_seconds=0,
        timeout_seconds=timeout,
    )


def _ssh_command(spec: RemoteSpec, remote_command: str) -> list[str]:
    command = ["ssh"]
    if spec["port"] is not None:
        command.extend(["-p", str(spec["port"])])
    command.extend([spec["user_host"], f"bash -lc {shlex.quote(remote_command)}"])
    return command


def _scp_to_remote_command(spec: RemoteSpec, local_path_arg: str, remote_path: str) -> list[str]:
    command = ["scp"]
    if spec["port"] is not None:
        command.extend(["-P", str(spec["port"])])
    command.extend([local_path_arg, f"{spec['user_host']}:{remote_path}"])
    return command


def _scp_from_remote_command(
    spec: RemoteSpec,
    remote_path: str,
    local_path_arg: str,
    recursive: bool = False,
) -> list[str]:
    command = ["scp"]
    if recursive:
        command.append("-r")
    if spec["port"] is not None:
        command.extend(["-P", str(spec["port"])])
    command.extend([f"{spec['user_host']}:{remote_path}", local_path_arg])
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
    blocks_parallel = normalized.get(TRITON_ALL_BLOCKS_PARALLEL)
    if blocks_parallel == _BLOCKS_PARALLEL_UNSAFE_VALUE:
        normalized[TRITON_ALL_BLOCKS_PARALLEL] = _BLOCKS_PARALLEL_SAFE_VALUE
    elif (
        blocks_parallel is None
        and os.environ.get(TRITON_ALL_BLOCKS_PARALLEL) == _BLOCKS_PARALLEL_SAFE_VALUE
    ):
        normalized[TRITON_ALL_BLOCKS_PARALLEL] = _BLOCKS_PARALLEL_SAFE_VALUE
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


def _coerce_output_text(data: bytes | str) -> str:
    # Keep this decoder local to the skill helper instead of importing a shared
    # helix utility: skill-side scripts must stay self-contained.
    if isinstance(data, str):
        return data
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        preferred = locale.getpreferredencoding(False) or "utf-8"
        return data.decode(preferred, errors="replace")


def _scp_local_operand(local_path: Path) -> tuple[str, str]:
    return str(local_path.parent), local_path.name
