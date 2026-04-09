from __future__ import annotations

from triton_agent.backends.base import AgentRunner
from triton_agent.backends.claude import ClaudeRunner
from triton_agent.backends.codex import CodexRunner
from triton_agent.backends.opencode import OpenCodeRunner
from triton_agent.backends.pi import PiRunner


def create_runner(agent_name: str) -> AgentRunner:
    if agent_name == "codex":
        return CodexRunner()
    if agent_name == "opencode":
        return OpenCodeRunner()
    if agent_name == "pi":
        return PiRunner()
    if agent_name == "claude":
        return ClaudeRunner()
    raise ValueError(f"Unsupported agent backend: {agent_name}")
