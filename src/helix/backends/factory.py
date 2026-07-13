from __future__ import annotations

from helix.backends.base import AgentRunner
from helix.backends.claude import ClaudeRunner
from helix.backends.codex import CodexRunner
from helix.backends.openhands import OpenHandsRunner
from helix.backends.opencode import OpenCodeRunner
from helix.backends.pi import PiRunner
from helix.backends.traecli import TraeCLIRunner


def create_runner(agent_name: str) -> AgentRunner:
    if agent_name == "codex":
        return CodexRunner()
    if agent_name == "opencode":
        return OpenCodeRunner()
    if agent_name == "pi":
        return PiRunner()
    if agent_name == "claude":
        return ClaudeRunner()
    if agent_name == "openhands":
        return OpenHandsRunner()
    if agent_name == "traecli":
        return TraeCLIRunner()
    raise ValueError(f"Unsupported agent backend: {agent_name}")
