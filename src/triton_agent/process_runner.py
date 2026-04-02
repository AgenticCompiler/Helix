from __future__ import annotations

import errno
import os
import pty
import select
import subprocess
import sys
import time
from typing import Callable, Optional, Protocol, TextIO

from triton_agent.models import AgentResult


class OutputFilter(Protocol):
    def feed(self, text: str, *, flush: bool = False) -> str: ...


def run_process(
    command: list[str],
    workdir: str,
    mode: str,
    stall_timeout_seconds: int = 0,
    session_id_extractor: Optional[Callable[[str], Optional[str]]] = None,
    stdout: Optional[TextIO] = None,
    output_filter: Optional[OutputFilter] = None,
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
        )
    if mode == "buffered":
        return run_buffered_process(
            command,
            workdir,
            stall_timeout_seconds=stall_timeout_seconds,
            session_id_extractor=session_id_extractor or (lambda _line: None),
            output_filter=output_filter,
        )
    raise ValueError(f"Unsupported process runner mode: {mode}")


def run_interactive_process(command: list[str], workdir: str) -> AgentResult:
    completed = subprocess.run(command, cwd=workdir)
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
) -> AgentResult:
    process = subprocess.Popen(
        command,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout_lines: list[str] = []
    session_id: Optional[str] = None
    start = time.monotonic()

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
) -> AgentResult:
    # Route stdout/stderr through one PTY so the child behaves as if it were
    # attached to a terminal and flushes output incrementally.
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
            # Poll the PTY frequently so streamed output feels live without
            # dropping the existing stall timeout behavior.
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if ready:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError as error:
                    # Linux PTYs may report EOF as EIO once the child side has
                    # closed. Treat that as a normal shutdown after the process
                    # has already exited, but preserve any other read failure.
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
                    session_id=None,
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
            session_id=None,
        )
    finally:
        os.close(master_fd)


def _resolved_returncode(returncode: int | None) -> int:
    return returncode if returncode is not None else 1
