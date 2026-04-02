from __future__ import annotations

import json
import sys
import uuid
from typing import List, Optional, TextIO

from triton_agent.agent import AgentRunner
from triton_agent.models import AgentRequest, AgentResult
from triton_agent.process_runner import run_process
from triton_agent.verbose import emit_verbose_lines, format_command_messages


class CodexRunner(AgentRunner):
    def __init__(self, executable: str = "codex", stall_timeout_seconds: int = 900) -> None:
        self.executable = executable
        self.stall_timeout_seconds = stall_timeout_seconds

    def build_command(self, request: AgentRequest) -> List[str]:
        if request.interact:
            return [self.executable, "--cd", str(request.workdir), request.prompt]
        return [
            self.executable,
            "exec",
            "--cd",
            str(request.workdir),
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "danger-full-access",
            request.prompt,
        ]

    def run(
        self,
        request: AgentRequest,
        stdout: Optional[TextIO] = None,
        stderr: Optional[TextIO] = None,
    ) -> AgentResult:
        command = self.build_command(request)
        if request.verbose:
            self._log_launch_command(command, stderr or sys.stderr)
        return run_process(
            command,
            str(request.workdir),
            mode=self._select_mode(request),
            stall_timeout_seconds=self.stall_timeout_seconds,
            session_id_extractor=_extract_session_id,
            stdout=stdout,
            output_filter=self._build_output_filter(request),
        )

    def resume(self, request: AgentRequest, summary: str) -> AgentResult:
        resumed_prompt = (
            "Continue the existing optimize task instead of restarting from scratch.\n"
            "Read `opt-note.md`, existing `opt-round-*` directories, and any round summaries "
            "or attempt logs before making the next change.\n\n"
            f"Progress summary:\n{summary}"
        )
        return self.run(
            AgentRequest(
                command_kind=request.command_kind,
                input_path=request.input_path,
                operator_path=request.operator_path,
                output_path=request.output_path,
                test_mode=request.test_mode,
                bench_mode=request.bench_mode,
                interact=request.interact,
                verbose=request.verbose,
                show_output=request.show_output,
                force_overwrite=request.force_overwrite,
                agent_name=request.agent_name,
                skill_name=request.skill_name,
                prompt=resumed_prompt,
                workdir=request.workdir,
                min_rounds=request.min_rounds,
            )
        )

    def _log_launch_command(self, command: List[str], stream: TextIO) -> None:
        emit_verbose_lines(stream, "agent", format_command_messages(command))

    def _select_mode(self, request: AgentRequest) -> str:
        if request.interact:
            return "interactive"
        if request.show_output:
            # A PTY encourages line-buffered behavior from the child process, which makes
            # `--show-output` feel genuinely live instead of flushing in large chunks.
            return "streaming"
        return "buffered"

    def _build_output_filter(self, request: AgentRequest) -> "_UnifiedDiffFilter | None":
        if request.interact:
            return None
        return _UnifiedDiffFilter()


class _UnifiedDiffFilter:
    _DIFF_PREFIXES = (
        "diff --git ",
        "index ",
        "--- ",
        "+++ ",
        "@@ ",
        "new file mode ",
        "deleted file mode ",
        "similarity index ",
        "rename from ",
        "rename to ",
        "old mode ",
        "new mode ",
        "Binary files ",
        "\\ No newline at end of file",
    )

    def __init__(self) -> None:
        self._buffer = ""
        self._in_diff = False

    def feed(self, text: str, *, flush: bool = False) -> str:
        self._buffer += text
        emitted: list[str] = []

        while True:
            newline_index = self._buffer.find("\n")
            if newline_index == -1:
                break
            line = self._buffer[: newline_index + 1]
            self._buffer = self._buffer[newline_index + 1 :]
            kept = self._process_line(line)
            if kept:
                emitted.append(kept)

        if flush and self._buffer:
            kept = self._process_line(self._buffer)
            self._buffer = ""
            if kept:
                emitted.append(kept)

        return "".join(emitted)

    def _process_line(self, line: str) -> str:
        bare = line.rstrip("\n")
        if not self._in_diff:
            if bare.startswith("diff --git "):
                self._in_diff = True
                return ""
            return line

        if self._is_diff_line(bare):
            return ""

        self._in_diff = False
        if bare.startswith("diff --git "):
            self._in_diff = True
            return ""
        return line

    def _is_diff_line(self, line: str) -> bool:
        if line.startswith(self._DIFF_PREFIXES):
            return True
        if line.startswith(("+", "-", " ")):
            return True
        return False



def _extract_session_id(line: str) -> Optional[str]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        payload = None

    candidates = []
    if isinstance(payload, dict):
        candidates.extend(
            [
                payload.get("session_id"),
                payload.get("sessionId"),
                payload.get("thread_id"),
                payload.get("threadId"),
            ]
        )
    candidates.append(line.strip())

    for candidate in candidates:
        if not candidate:
            continue
        text = str(candidate)
        for token in text.replace('"', " ").split():
            try:
                return str(uuid.UUID(token))
            except ValueError:
                continue
    return None
