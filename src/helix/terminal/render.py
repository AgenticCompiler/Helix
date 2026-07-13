from __future__ import annotations

import sys
from typing import Optional, TextIO

from helix.models import AgentResult


def render_result(
    result: AgentResult,
    skip_stdout: bool,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> None:
    stdout_stream = stdout or sys.stdout
    stderr_stream = stderr or sys.stderr
    if result.stdout and not skip_stdout:
        print(result.stdout, file=stdout_stream, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, file=stderr_stream, end="" if result.stderr.endswith("\n") else "\n")
