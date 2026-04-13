from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from typing import Callable, Optional, cast

from triton_agent.backends.base import AgentRunner
from triton_agent.models import AgentRequest


class CodexRunner(AgentRunner):
    def __init__(self, executable: str = "codex", stall_timeout_seconds: int = 900) -> None:
        super().__init__(executable, stall_timeout_seconds)

    def build_command(self, request: AgentRequest) -> list[str]:
        if request.interact:
            command = [self.executable, "--cd", str(request.workdir)]
            if request.command_kind == request.command_kind.OPTIMIZE and request.no_agent_session:
                command.append("--ephemeral")
            command.append(request.prompt)
            return command
        command = [
            self.executable,
            "exec",
            "--cd",
            str(request.workdir),
            "--skip-git-repo-check",
            "--sandbox",
            "danger-full-access",
        ]
        if request.command_kind != request.command_kind.OPTIMIZE or request.no_agent_session:
            command.append("--ephemeral")
        command.append(request.prompt)
        return command

    def session_id_extractor(self) -> Callable[[str], str | None]:
        return _extract_session_id

    def output_filter(self, request: AgentRequest) -> "_UnifiedDiffFilter | None":
        if request.interact:
            return None
        return _UnifiedDiffFilter()


class _UnifiedDiffFilter:
    _DIFF_METADATA_PREFIXES = (
        "diff --git ",
        "index ",
        "--- ",
        "+++ ",
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
        self._in_hunk = False

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
                self._in_hunk = False
                return ""
            return line

        if bare.startswith("@@ "):
            self._in_hunk = True
            return ""
        if bare.startswith(self._DIFF_METADATA_PREFIXES):
            self._in_hunk = False
            return ""
        if self._in_hunk and self._is_hunk_line(bare):
            return ""

        self._in_diff = False
        self._in_hunk = False
        if bare.startswith("diff --git "):
            self._in_diff = True
            self._in_hunk = False
            return ""
        return line

    def _is_hunk_line(self, line: str) -> bool:
        if line.startswith(("+", "-")):
            return True
        if line == "\\ No newline at end of file":
            return True
        if line.startswith(" "):
            return len(line) == 1 or not line.startswith("  ")
        return False


def _extract_session_id(line: str) -> Optional[str]:
    try:
        payload: object = json.loads(line)
    except json.JSONDecodeError:
        payload = None

    candidates: list[object] = []
    if isinstance(payload, Mapping):
        payload_map = cast(Mapping[str, object], payload)
        for key in ("session_id", "sessionId", "thread_id", "threadId"):
            candidates.append(payload_map.get(key))
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
