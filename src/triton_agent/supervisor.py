from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from triton_agent.models import AgentRequest, AgentResult


class SupportsOptimizeRecovery(Protocol):
    def run(self, request: AgentRequest) -> AgentResult:
        ...

    def resume(self, request: AgentRequest, summary: str) -> AgentResult:
        ...


class OptimizeSupervisor:
    def __init__(self, max_recovery_attempts: int = 2) -> None:
        self.max_recovery_attempts = max_recovery_attempts

    def run(self, runner: SupportsOptimizeRecovery, request: AgentRequest) -> AgentResult:
        attempt = 0
        current_request = request

        while True:
            result = runner.run(current_request)
            if result.succeeded:
                return result
            if current_request.interact:
                return result
            if not result.stalled or attempt >= self.max_recovery_attempts:
                return result

            attempt += 1
            summary = self._build_summary(result)
            current_request = replace(
                current_request,
                prompt=f"{request.prompt}\n\nContinue from this progress summary:\n{summary}",
            )
            result = runner.resume(current_request, summary)
            if result.succeeded or not result.stalled or attempt >= self.max_recovery_attempts:
                return result
            current_request = replace(
                current_request,
                prompt=f"{request.prompt}\n\nRetry optimize task after stall.\n\nKnown progress:\n{summary}",
            )

    def _build_summary(self, result: AgentResult) -> str:
        output = result.stdout.strip() or result.stderr.strip() or "No progress output was captured."
        return output[-2000:]
