from __future__ import annotations

from dataclasses import dataclass
import errno
import os
import signal as _signal
import shutil
import subprocess
import sys
import time
import threading
from typing import Any, Callable, Optional, Protocol, TextIO, cast

from triton_agent.models import AgentResult
from triton_agent.transient_failures import contains_transient_agent_failure_text

_IS_WINDOWS = sys.platform == "win32"

if not _IS_WINDOWS:
    import pty
    import select
else:
    pty = None
    select = None


_PTY_EIO_EXIT_GRACE_SECONDS = 0.1


def _resolve_command(command: list[str]) -> list[str]:
    """On Windows, wrap .cmd/.bat executables with 'cmd /c' so they can be launched
    by subprocess.Popen without shell=True."""
    if not _IS_WINDOWS or not command:
        return command
    resolved = shutil.which(command[0])
    if resolved and resolved.lower().endswith((".cmd", ".bat")):
        return ["cmd", "/c", resolved] + command[1:]
    return command


class OutputFilter(Protocol):
    def feed(self, text: str, *, flush: bool = False) -> str: ...


@dataclass(frozen=True)
class InterruptPolicy:
    first_sigint_grace_seconds: float = 2.0
    second_sigint_grace_seconds: float = 1.0
    interrupted_return_code: int = 130


def run_process(
    command: list[str],
    workdir: str,
    mode: str,
    stall_timeout_seconds: int = 0,
    session_id_extractor: Optional[Callable[[str], Optional[str]]] = None,
    stdout: Optional[TextIO] = None,
    output_filter: Optional[OutputFilter] = None,
    interrupt_policy: Optional[InterruptPolicy] = None,
    extra_env: Optional[dict[str, str]] = None,
    *,
    rendered_chunk_sink: Optional[Callable[[str], None]] = None,
    collect_stdout: bool = True,
) -> AgentResult:
    if mode == "interactive":
        return run_interactive_process(command, workdir, extra_env=extra_env)
    if mode == "streaming":
        return run_streaming_process(
            command,
            workdir,
            stall_timeout_seconds=stall_timeout_seconds,
            stdout=stdout,
            output_filter=output_filter,
            session_id_extractor=session_id_extractor or (lambda _text: None),
            interrupt_policy=interrupt_policy,
            extra_env=extra_env,
            rendered_chunk_sink=rendered_chunk_sink,
            collect_stdout=collect_stdout,
        )
    if mode == "buffered":
        return run_buffered_process(
            command,
            workdir,
            stall_timeout_seconds=stall_timeout_seconds,
            session_id_extractor=session_id_extractor or (lambda _line: None),
            output_filter=output_filter,
            interrupt_policy=interrupt_policy,
            extra_env=extra_env,
        )
    raise ValueError(f"Unsupported process runner mode: {mode}")


def run_interactive_process(
    command: list[str],
    workdir: str,
    *,
    extra_env: Optional[dict[str, str]] = None,
) -> AgentResult:
    completed = subprocess.run(_resolve_command(command), cwd=workdir, env=_merged_env(extra_env))
    return AgentResult(
        return_code=completed.returncode,
        stdout="",
        stderr="",
        stalled=False,
        session_id=None,
    )


def run_buffered_process(
    command: list[str],
    workdir: str,
    stall_timeout_seconds: int,
    session_id_extractor: Callable[[str], Optional[str]],
    output_filter: Optional[OutputFilter] = None,
    interrupt_policy: Optional[InterruptPolicy] = None,
    extra_env: Optional[dict[str, str]] = None,
) -> AgentResult:
    process = subprocess.Popen(
        _resolve_command(command),
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_merged_env(extra_env),
        start_new_session=interrupt_policy is not None and not _IS_WINDOWS,
    )
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    session_id: Optional[str] = None
    session_buffer = ""
    start_ref: list[float] = [time.monotonic()]
    lock = threading.Lock()

    def record_stdout_chunk(chunk: str) -> None:
        nonlocal session_id, session_buffer
        filtered = output_filter.feed(chunk) if output_filter is not None else chunk
        with lock:
            if filtered:
                stdout_chunks.append(filtered)
            start_ref[0] = time.monotonic()
        if session_id is None:
            session_id, session_buffer = _extract_session_id_from_text(
                session_id_extractor,
                session_id,
                session_buffer,
                chunk,
            )

    def record_stderr_chunk(chunk: str) -> None:
        with lock:
            if chunk:
                stderr_chunks.append(chunk)
            start_ref[0] = time.monotonic()

    stdout_reader = threading.Thread(
        target=_read_text_stream,
        args=(process.stdout, record_stdout_chunk),
        daemon=True,
    )
    stderr_reader = threading.Thread(
        target=_read_text_stream,
        args=(process.stderr, record_stderr_chunk),
        daemon=True,
    )
    readers = (stdout_reader, stderr_reader)
    for reader in readers:
        reader.start()

    try:
        while True:
            if process.poll() is not None:
                for reader in readers:
                    reader.join(timeout=1.0)
                break
            if stall_timeout_seconds > 0 and time.monotonic() - start_ref[0] > stall_timeout_seconds:
                process.terminate()
                if not _wait_for_process_exit(process, 1.0):
                    process.kill()
                for reader in readers:
                    reader.join(timeout=1.0)
                session_id = _flush_session_id_buffer(session_id_extractor, session_id, session_buffer)
                if output_filter is not None:
                    trailing = output_filter.feed("", flush=True)
                    if trailing:
                        stdout_chunks.append(trailing)
                return AgentResult(
                    return_code=1,
                    stdout="".join(stdout_chunks),
                    stderr="".join(stderr_chunks),
                    stalled=True,
                    session_id=session_id,
                )
            time.sleep(0.05)
    except KeyboardInterrupt:
        if interrupt_policy is None:
            raise
        _interrupt_process(process, interrupt_policy)
        for reader in readers:
            reader.join(timeout=1.0)
        session_id = _flush_session_id_buffer(session_id_extractor, session_id, session_buffer)
        if output_filter is not None:
            trailing = output_filter.feed("", flush=True)
            if trailing:
                stdout_chunks.append(trailing)
        return AgentResult(
            return_code=interrupt_policy.interrupted_return_code,
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
            stalled=False,
            session_id=session_id,
        )

    session_id = _flush_session_id_buffer(session_id_extractor, session_id, session_buffer)
    trailing = output_filter.feed("", flush=True) if output_filter is not None else ""
    with lock:
        if trailing:
            stdout_chunks.append(trailing)
        captured_stdout = "".join(stdout_chunks)
        captured_stderr = "".join(stderr_chunks)
    return AgentResult(
        return_code=_resolved_returncode(process.returncode),
        stdout=captured_stdout,
        stderr=captured_stderr,
        stalled=False,
        session_id=session_id,
    )


def run_streaming_process(
    command: list[str],
    workdir: str,
    stall_timeout_seconds: int,
    stdout: Optional[TextIO] = None,
    output_filter: Optional[OutputFilter] = None,
    session_id_extractor: Optional[Callable[[str], Optional[str]]] = None,
    interrupt_policy: Optional[InterruptPolicy] = None,
    extra_env: Optional[dict[str, str]] = None,
    *,
    rendered_chunk_sink: Optional[Callable[[str], None]] = None,
    collect_stdout: bool = True,
) -> AgentResult:
    if _IS_WINDOWS:
        return _run_streaming_process_windows(
            command,
            workdir,
            stall_timeout_seconds=stall_timeout_seconds,
            stdout=stdout,
            output_filter=output_filter,
            session_id_extractor=session_id_extractor or (lambda _text: None),
            interrupt_policy=interrupt_policy,
            extra_env=extra_env,
            rendered_chunk_sink=rendered_chunk_sink,
            collect_stdout=collect_stdout,
        )
    return _run_streaming_process_pty(
        command,
        workdir,
        stall_timeout_seconds=stall_timeout_seconds,
        stdout=stdout,
        output_filter=output_filter,
        session_id_extractor=session_id_extractor or (lambda _text: None),
        interrupt_policy=interrupt_policy,
        extra_env=extra_env,
        rendered_chunk_sink=rendered_chunk_sink,
        collect_stdout=collect_stdout,
    )


def _run_streaming_process_windows(
    command: list[str],
    workdir: str,
    stall_timeout_seconds: int,
    stdout: Optional[TextIO] = None,
    output_filter: Optional[OutputFilter] = None,
    session_id_extractor: Callable[[str], Optional[str]] = lambda _text: None,
    interrupt_policy: Optional[InterruptPolicy] = None,
    extra_env: Optional[dict[str, str]] = None,
    *,
    rendered_chunk_sink: Optional[Callable[[str], None]] = None,
    collect_stdout: bool = True,
) -> AgentResult:
    """Windows-compatible streaming using threads to drain stdout and stderr."""
    process = subprocess.Popen(
        _resolve_command(command),
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        env=_merged_env(extra_env),
    )
    output_chunks: list[str] = []
    start_ref: list[float] = [time.monotonic()]
    stalled_ref: list[bool] = [False]
    session_id_ref: list[str | None] = [None]
    retryable_failure_ref: list[bool] = [False]
    lock = threading.Lock()
    rolling_window = ""

    def reader() -> None:
        nonlocal rolling_window
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode(errors="replace")
            rolling_window = (rolling_window + text.lower())[-4096:]
            retryable_failure_ref[0] = (
                retryable_failure_ref[0] or contains_transient_agent_failure_text(rolling_window)
            )
            filtered = output_filter.feed(text) if output_filter is not None else text
            with lock:
                if filtered:
                    if collect_stdout:
                        output_chunks.append(filtered)
                    print(filtered, file=stdout or sys.stdout, end="", flush=True)
                    if rendered_chunk_sink is not None:
                        rendered_chunk_sink(filtered)
                start_ref[0] = time.monotonic()
                session_id_ref[0] = session_id_ref[0] or session_id_extractor(text)

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    try:
        while True:
            reader_thread.join(timeout=0.1)
            if process.poll() is not None:
                break
            with lock:
                elapsed = time.monotonic() - start_ref[0]
            if stall_timeout_seconds > 0 and elapsed > stall_timeout_seconds:
                process.terminate()
                stalled_ref[0] = True
                break
    except KeyboardInterrupt:
        if interrupt_policy is None:
            raise
        _interrupt_process(process, interrupt_policy)
        reader_thread.join(timeout=2.0)
        if output_filter is not None:
            trailing = output_filter.feed("", flush=True)
            if trailing:
                if collect_stdout:
                    output_chunks.append(trailing)
                print(trailing, file=stdout or sys.stdout, end="")
                if rendered_chunk_sink is not None:
                    rendered_chunk_sink(trailing)
        return AgentResult(
            return_code=interrupt_policy.interrupted_return_code,
            stdout="".join(output_chunks) if collect_stdout else "",
            stderr="",
            stalled=False,
            session_id=session_id_ref[0],
            retryable_failure=retryable_failure_ref[0],
        )

    # Give the reader thread a short grace period to finish reading
    # any remaining buffered output after the child process exits.
    # Use a timeout to avoid hanging if the child's stdout pipe is
    # held open by grandchildren (e.g. sub-agents).
    reader_thread.join(timeout=1.0)
    trailing = output_filter.feed("", flush=True) if output_filter is not None else ""
    with lock:
        if trailing:
            if collect_stdout:
                output_chunks.append(trailing)
            print(trailing, file=stdout or sys.stdout, end="")
            if rendered_chunk_sink is not None:
                rendered_chunk_sink(trailing)
        captured_stdout = "".join(output_chunks) if collect_stdout else ""
    rc = process.wait() if not stalled_ref[0] else 1
    return AgentResult(
        return_code=rc,
        stdout=captured_stdout,
        stderr="",
        stalled=stalled_ref[0],
        session_id=session_id_ref[0],
        retryable_failure=retryable_failure_ref[0],
    )


def _run_streaming_process_pty(
    command: list[str],
    workdir: str,
    stall_timeout_seconds: int,
    stdout: Optional[TextIO] = None,
    output_filter: Optional[OutputFilter] = None,
    session_id_extractor: Callable[[str], Optional[str]] = lambda _text: None,
    interrupt_policy: Optional[InterruptPolicy] = None,
    extra_env: Optional[dict[str, str]] = None,
    *,
    rendered_chunk_sink: Optional[Callable[[str], None]] = None,
    collect_stdout: bool = True,
) -> AgentResult:
    """Unix PTY-backed streaming so the child sees a terminal and flushes incrementally."""
    if pty is None or select is None:
        raise RuntimeError("PTY streaming is unavailable on this platform")
    openpty = cast(Callable[[], tuple[int, int]], getattr(pty, "openpty"))
    select_fn = cast(
        Callable[[list[int], list[int], list[int], float], tuple[list[int], list[int], list[int]]],
        getattr(select, "select"),
    )
    master_fd, slave_fd = openpty()
    output_chunks: list[str] = []
    process = subprocess.Popen(
        _resolve_command(command),
        cwd=workdir,
        stdin=subprocess.DEVNULL,
        stdout=slave_fd,
        stderr=slave_fd,
        text=False,
        close_fds=True,
        env=_merged_env(extra_env),
        start_new_session=interrupt_policy is not None,
    )
    os.close(slave_fd)
    start = time.monotonic()
    session_id: Optional[str] = None
    retryable_failure = False
    rolling_window = ""

    try:
        while True:
            ready, _, _ = select_fn([master_fd], [], [], 0.1)
            if ready:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError as error:
                    if _is_clean_pty_eof(error, process):
                        break
                    raise
                if chunk:
                    text = chunk.decode(errors="replace")
                    rolling_window = (rolling_window + text.lower())[-4096:]
                    retryable_failure = (
                        retryable_failure or contains_transient_agent_failure_text(rolling_window)
                    )
                    filtered = output_filter.feed(text) if output_filter is not None else text
                    if filtered:
                        if collect_stdout:
                            output_chunks.append(filtered)
                        print(filtered, file=stdout or sys.stdout, end="")
                        if rendered_chunk_sink is not None:
                            rendered_chunk_sink(filtered)
                    start = time.monotonic()
                    session_id = session_id or session_id_extractor(text)
                elif process.poll() is not None:
                    break
            elif process.poll() is not None:
                break
            elif stall_timeout_seconds > 0 and time.monotonic() - start > stall_timeout_seconds:
                process.terminate()
                return AgentResult(
                    return_code=1,
                    stdout="".join(output_chunks) if collect_stdout else "",
                    stderr="",
                    stalled=True,
                    session_id=session_id,
                    retryable_failure=retryable_failure,
                )
        if output_filter is not None:
            trailing = output_filter.feed("", flush=True)
            if trailing:
                if collect_stdout:
                    output_chunks.append(trailing)
                print(trailing, file=stdout or sys.stdout, end="")
                if rendered_chunk_sink is not None:
                    rendered_chunk_sink(trailing)
        return AgentResult(
            return_code=process.wait(),
            stdout="".join(output_chunks) if collect_stdout else "",
            stderr="",
            stalled=False,
            session_id=session_id,
            retryable_failure=retryable_failure,
        )
    except KeyboardInterrupt:
        if interrupt_policy is None:
            raise
        _interrupt_process(process, interrupt_policy)
        if output_filter is not None:
            trailing = output_filter.feed("", flush=True)
            if trailing:
                if collect_stdout:
                    output_chunks.append(trailing)
                print(trailing, file=stdout or sys.stdout, end="")
                if rendered_chunk_sink is not None:
                    rendered_chunk_sink(trailing)
        return AgentResult(
            return_code=interrupt_policy.interrupted_return_code,
            stdout="".join(output_chunks) if collect_stdout else "",
            stderr="",
            stalled=False,
            session_id=session_id,
            retryable_failure=retryable_failure,
        )
    finally:
        os.close(master_fd)


def _is_clean_pty_eof(error: OSError, process: Any) -> bool:
    if error.errno != errno.EIO:
        return False
    if process.poll() is not None:
        return True
    try:
        process.wait(timeout=_PTY_EIO_EXIT_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        return False
    return True


def _resolved_returncode(returncode: int | None) -> int:
    return returncode if returncode is not None else 1


def _read_text_stream(
    stream: TextIO | None,
    on_chunk: Callable[[str], None],
) -> None:
    if stream is None:
        return
    while True:
        chunk = stream.read(4096)
        if not chunk:
            break
        on_chunk(chunk)


def _extract_session_id_from_text(
    extractor: Callable[[str], Optional[str]],
    current_session_id: Optional[str],
    pending_text: str,
    chunk: str,
) -> tuple[Optional[str], str]:
    if current_session_id is not None:
        return current_session_id, pending_text
    pending_text += chunk
    while True:
        newline_index = pending_text.find("\n")
        if newline_index == -1:
            break
        line = pending_text[: newline_index + 1]
        pending_text = pending_text[newline_index + 1 :]
        extracted = extractor(line)
        if extracted is not None:
            return extracted, pending_text
    return None, pending_text


def _flush_session_id_buffer(
    extractor: Callable[[str], Optional[str]],
    current_session_id: Optional[str],
    pending_text: str,
) -> Optional[str]:
    if current_session_id is not None or not pending_text:
        return current_session_id
    return extractor(pending_text)


def _merged_env(extra_env: Optional[dict[str, str]]) -> Optional[dict[str, str]]:
    if extra_env is None:
        return None
    merged = dict(os.environ)
    merged.update(extra_env)
    return merged


def _interrupt_process(process: subprocess.Popen[Any], policy: InterruptPolicy) -> None:
    if _IS_WINDOWS:
        _interrupt_process_windows(process, policy)
    else:
        _interrupt_process_unix(process, policy)


def _interrupt_process_windows(process: subprocess.Popen[Any], policy: InterruptPolicy) -> None:
    if process.poll() is not None:
        return
    ctrl_c_event = getattr(_signal, "CTRL_C_EVENT", None)
    if ctrl_c_event is None:
        process.kill()
        return
    try:
        process.send_signal(ctrl_c_event)
    except (OSError, KeyboardInterrupt):
        pass
    if _wait_for_process_exit(process, policy.first_sigint_grace_seconds):
        return
    try:
        process.send_signal(ctrl_c_event)
    except (OSError, KeyboardInterrupt):
        pass
    if _wait_for_process_exit(process, policy.second_sigint_grace_seconds):
        return
    process.kill()


def _interrupt_process_unix(process: subprocess.Popen[Any], policy: InterruptPolicy) -> None:
    _signal_process_group(process, _signal.SIGINT)
    if _wait_for_process_exit(process, policy.first_sigint_grace_seconds):
        return
    _signal_process_group(process, _signal.SIGINT)
    if _wait_for_process_exit(process, policy.second_sigint_grace_seconds):
        return
    sigkill = cast(Any, getattr(_signal, "SIGKILL", _signal.SIGTERM))
    _signal_process_group(process, sigkill)


def _signal_process_group(process: subprocess.Popen[Any], sig: Any) -> None:
    if process.poll() is not None:
        return
    killpg = cast(Callable[[int, Any], None] | None, getattr(os, "killpg", None))
    if killpg is None:
        process.kill()
        return
    try:
        killpg(process.pid, sig)
    except ProcessLookupError:
        return


def _wait_for_process_exit(process: subprocess.Popen[Any], timeout_seconds: float) -> bool:
    if timeout_seconds <= 0:
        return process.poll() is not None
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return True
        remaining = deadline - time.monotonic()
        time.sleep(min(0.05, max(remaining, 0.0)))
    return process.poll() is not None
