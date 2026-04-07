from __future__ import annotations

from triton_agent.agent import AgentRunner
from triton_agent.claude_runner import ClaudeRunner
from triton_agent.codex_runner import CodexRunner
from triton_agent.opencode_runner import OpenCodeRunner
from triton_agent.pi_runner import PiRunner


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
