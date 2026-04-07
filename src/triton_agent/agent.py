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
        resumed_request = AgentRequest(
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
            prompt=f"{request.prompt}\n\nContinue from this progress summary:\n{summary}",
            workdir=request.workdir,
            min_rounds=request.min_rounds,
            continue_optimize=request.continue_optimize,
            no_agent_session=request.no_agent_session,
        )
        return self.run(resumed_request)
