from __future__ import annotations

from abc import ABC, abstractmethod

from triton_agent.models import AgentRequest, AgentResult


class AgentRunner(ABC):
    @abstractmethod
    def run(self, request: AgentRequest) -> AgentResult:
        raise NotImplementedError

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
        )
        return self.run(resumed_request)
