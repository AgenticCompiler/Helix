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
from typing import Any, Callable, Optional, Protocol, TextIO

from triton_agent.models import AgentResult

_IS_WINDOWS = sys.platform == "win32"

if not _IS_WINDOWS:
    import pty
    import select
else:
    pty = None
    select = None


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
) -> AgentResult:
    if mode == "interactive":
        return run_interactive_process(command, workdir)
    if mode == "streaming":
        return run_streaming_process(
            command,
            workdir,
            stall_timeout_seconds=stall_timeout_seconds,
            stdout=stdout,
            output_filter=output_filter,
            session_id_extractor=session_id_extractor or (lambda _text: None),
            interrupt_policy=interrupt_policy,
        )
    if mode == "buffered":
        return run_buffered_process(
            command,
            workdir,
            stall_timeout_seconds=stall_timeout_seconds,
            session_id_extractor=session_id_extractor or (lambda _line: None),
            output_filter=output_filter,
            interrupt_policy=interrupt_policy,
        )
    raise ValueError(f"Unsupported process runner mode: {mode}")


def run_interactive_process(command: list[str], workdir: str) -> AgentResult:
    completed = subprocess.run(_resolve_command(command), cwd=workdir)
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
) -> AgentResult:
    process = subprocess.Popen(
        _resolve_command(command),
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=interrupt_policy is not None and not _IS_WINDOWS,
    )
    stdout_lines: list[str] = []
    session_id: Optional[str] = None
    start = time.monotonic()

    try:
        while True:
            line = process.stdout.readline() if process.stdout is not None else ""
            if line:
                filtered = output_filter.feed(line) if output_filter is not None else line
                if filtered:
                    stdout_lines.append(filtered)
                start = time.monotonic()
                session_id = session_id or session_id_extractor(line)
            elif process.poll() is not None:
                break
            elif stall_timeout_seconds > 0 and time.monotonic() - start > stall_timeout_seconds:
                process.terminate()
                stderr_text = process.stderr.read() if process.stderr is not None else ""
                return AgentResult(
                    return_code=1,
                    stdout="".join(stdout_lines),
                    stderr=stderr_text,
                    stalled=True,
                    session_id=session_id,
                )
    except KeyboardInterrupt:
        if interrupt_policy is None:
            raise
        _interrupt_process(process, interrupt_policy)
        stderr_text = process.stderr.read() if process.stderr is not None else ""
        if output_filter is not None:
            trailing = output_filter.feed("", flush=True)
            if trailing:
                stdout_lines.append(trailing)
        return AgentResult(
            return_code=interrupt_policy.interrupted_return_code,
            stdout="".join(stdout_lines),
            stderr=stderr_text,
            stalled=False,
            session_id=session_id,
        )

    stderr_text = process.stderr.read() if process.stderr is not None else ""
    if output_filter is not None:
        trailing = output_filter.feed("", flush=True)
        if trailing:
            stdout_lines.append(trailing)
    return AgentResult(
        return_code=_resolved_returncode(process.returncode),
        stdout="".join(stdout_lines),
        stderr=stderr_text,
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
        )
    return _run_streaming_process_pty(
        command,
        workdir,
        stall_timeout_seconds=stall_timeout_seconds,
        stdout=stdout,
        output_filter=output_filter,
        session_id_extractor=session_id_extractor or (lambda _text: None),
        interrupt_policy=interrupt_policy,
    )


def _run_streaming_process_windows(
    command: list[str],
    workdir: str,
    stall_timeout_seconds: int,
    stdout: Optional[TextIO] = None,
    output_filter: Optional[OutputFilter] = None,
    session_id_extractor: Callable[[str], Optional[str]] = lambda _text: None,
    interrupt_policy: Optional[InterruptPolicy] = None,
) -> AgentResult:
    """Windows-compatible streaming using threads to drain stdout and stderr."""
    process = subprocess.Popen(
        _resolve_command(command),
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
    )
    output_chunks: list[str] = []
    start_ref: list[float] = [time.monotonic()]
    stalled_ref: list[bool] = [False]
    session_id_ref: list[str | None] = [None]
    lock = threading.Lock()

    def reader() -> None:
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode(errors="replace")
            filtered = output_filter.feed(text) if output_filter is not None else text
            with lock:
                if filtered:
                    output_chunks.append(filtered)
                    print(filtered, file=stdout or sys.stdout, end="", flush=True)
                start_ref[0] = time.monotonic()
                session_id_ref[0] = session_id_ref[0] or session_id_extractor(text)

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    try:
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
    except KeyboardInterrupt:
        if interrupt_policy is None:
            raise
        _interrupt_process(process, interrupt_policy)
        reader_thread.join(timeout=2.0)
        if output_filter is not None:
            trailing = output_filter.feed("", flush=True)
            if trailing:
                output_chunks.append(trailing)
                print(trailing, file=stdout or sys.stdout, end="")
        return AgentResult(
            return_code=interrupt_policy.interrupted_return_code,
            stdout="".join(output_chunks),
            stderr="",
            stalled=False,
            session_id=session_id_ref[0],
        )

    reader_thread.join()
    if output_filter is not None:
        trailing = output_filter.feed("", flush=True)
        if trailing:
            output_chunks.append(trailing)
            print(trailing, file=stdout or sys.stdout, end="")
    rc = process.wait() if not stalled_ref[0] else 1
    return AgentResult(
        return_code=rc,
        stdout="".join(output_chunks),
        stderr="",
        stalled=stalled_ref[0],
        session_id=session_id_ref[0],
    )


def _run_streaming_process_pty(
    command: list[str],
    workdir: str,
    stall_timeout_seconds: int,
    stdout: Optional[TextIO] = None,
    output_filter: Optional[OutputFilter] = None,
    session_id_extractor: Callable[[str], Optional[str]] = lambda _text: None,
    interrupt_policy: Optional[InterruptPolicy] = None,
) -> AgentResult:
    """Unix PTY-backed streaming so the child sees a terminal and flushes incrementally."""
    if pty is None or select is None:
        raise RuntimeError("PTY streaming is unavailable on this platform")
    master_fd, slave_fd = pty.openpty()
    output_chunks: list[str] = []
    process = subprocess.Popen(
        _resolve_command(command),
        cwd=workdir,
        stdin=subprocess.DEVNULL,
        stdout=slave_fd,
        stderr=slave_fd,
        text=False,
        close_fds=True,
        start_new_session=interrupt_policy is not None,
    )
    os.close(slave_fd)
    start = time.monotonic()
    session_id: Optional[str] = None

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
                    filtered = output_filter.feed(text) if output_filter is not None else text
                    if filtered:
                        output_chunks.append(filtered)
                        print(filtered, file=stdout or sys.stdout, end="")
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
                    stdout="".join(output_chunks),
                    stderr="",
                    stalled=True,
                    session_id=session_id,
                )
        if output_filter is not None:
            trailing = output_filter.feed("", flush=True)
            if trailing:
                output_chunks.append(trailing)
                print(trailing, file=stdout or sys.stdout, end="")
        return AgentResult(
            return_code=process.wait(),
            stdout="".join(output_chunks),
            stderr="",
            stalled=False,
            session_id=session_id,
        )
    except KeyboardInterrupt:
        if interrupt_policy is None:
            raise
        _interrupt_process(process, interrupt_policy)
        if output_filter is not None:
            trailing = output_filter.feed("", flush=True)
            if trailing:
                output_chunks.append(trailing)
                print(trailing, file=stdout or sys.stdout, end="")
        return AgentResult(
            return_code=interrupt_policy.interrupted_return_code,
            stdout="".join(output_chunks),
            stderr="",
            stalled=False,
            session_id=session_id,
        )
    finally:
        os.close(master_fd)


def _resolved_returncode(returncode: int | None) -> int:
    return returncode if returncode is not None else 1


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
    _signal_process_group(process, _signal.SIGKILL)


def _signal_process_group(process: subprocess.Popen[Any], sig: Any) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
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
