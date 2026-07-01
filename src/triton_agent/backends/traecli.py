from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from triton_agent.backends.base import AgentRunner
from triton_agent.models import AgentRequest

if TYPE_CHECKING:
    from triton_agent.backends.traecli_trace import TraeCliJsonOutputFilter


class TraeCLIRunner(AgentRunner):
    def __init__(self, executable: str = "traecli", stall_timeout_seconds: int = 900) -> None:
        super().__init__(executable, stall_timeout_seconds)

    def build_command(self, request: AgentRequest) -> list[str]:
        if request.interact:
            return [self.executable, request.prompt]
        command = [self.executable, "--print", "--yolo"]
        if request.log_tools:
            command.extend(
                [
                    "--output-format",
                    "stream-json",
                    "--include-partial-messages",
                ]
            )
        command.append(request.prompt)
        return command

    def output_filter(self, request: AgentRequest) -> "TraeCliJsonOutputFilter | None":
        if request.interact or not request.log_tools:
            return None
        from triton_agent.backends.traecli_trace import TraeCliJsonOutputFilter

        return TraeCliJsonOutputFilter(request.extra_env)

    def session_id_extractor(self) -> Callable[[str], str | None]:
        return _extract_traecli_session_id


def _extract_traecli_session_id(text: str) -> str | None:
    match = re.search(r'"session_id"\s*:\s*"([^"]+)"', text)
    if match:
        return match.group(1)
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            event = cast(dict[str, Any], json.loads(stripped))
        except json.JSONDecodeError:
            continue
        session_id = event.get("session_id")
        if isinstance(session_id, str) and session_id:
            return session_id
    return None
