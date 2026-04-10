from __future__ import annotations

from abc import ABC, abstractmethod

from triton_agent.models import AgentRequest, AgentResult
from triton_agent.process_runner import InterruptPolicy


class AgentRunner(ABC):
    _OPTIMIZE_INTERRUPT_POLICY = InterruptPolicy()

    @abstractmethod
    def run(self, request: AgentRequest) -> AgentResult:
        raise NotImplementedError

    def interrupt_policy(self, request: AgentRequest) -> InterruptPolicy | None:
        if request.interact or request.command_kind != request.command_kind.OPTIMIZE:
            return None
        return self._OPTIMIZE_INTERRUPT_POLICY

    def resume(self, request: AgentRequest, summary: str) -> AgentResult:
        resumed_request = request.with_prompt(
            f"{request.prompt}\n\nContinue from this progress summary:\n{summary}"
        )
        return self.run(resumed_request)
