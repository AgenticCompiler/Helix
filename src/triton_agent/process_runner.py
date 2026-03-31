from __future__ import annotations

import os
import pty
import select
import subprocess
import sys
import time
from typing import Callable, Optional, TextIO

from triton_agent.models import AgentResult


def run_process(
    command: list[str],
    workdir: str,
    mode: str,
    stall_timeout_seconds: int = 0,
    session_id_extractor: Optional[Callable[[str], Optional[str]]] = None,
    stdout: Optional[TextIO] = None,
) -> AgentResult:
    if mode == "interactive":
        return run_interactive_process(command, workdir)
    if mode == "streaming":
        return run_streaming_process(
            command,
            workdir,
            stall_timeout_seconds=stall_timeout_seconds,
            stdout=stdout,
        )
    if mode == "buffered":
        return run_buffered_process(
            command,
            workdir,
            stall_timeout_seconds=stall_timeout_seconds,
            session_id_extractor=session_id_extractor or (lambda _line: None),
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
            stdout_lines.append(line)
            start = time.monotonic()
            session_id = session_id or session_id_extractor(line)
        elif process.poll() is not None:
            break
        elif time.monotonic() - start > stall_timeout_seconds:
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
    return AgentResult(
        return_code=process.returncode or 0,
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
                chunk = os.read(master_fd, 4096)
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
                return AgentResult(
                    return_code=1,
                    stdout="".join(output_chunks),
                    stderr="",
                    stalled=True,
                    session_id=None,
                )

        return AgentResult(
            return_code=process.wait(),
            stdout="".join(output_chunks),
            stderr="",
            stalled=False,
            session_id=None,
        )
    finally:
        os.close(master_fd)
