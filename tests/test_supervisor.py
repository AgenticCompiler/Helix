import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.models import AgentRequest, AgentResult, CommandKind
from triton_agent.supervisor import OptimizeSupervisor


class FakeRunner:
    def __init__(self, results):
        self.results = list(results)
        self.prompts = []

    def run(self, request: AgentRequest) -> AgentResult:
        self.prompts.append(request.prompt)
        return self.results.pop(0)

    def resume(self, request: AgentRequest, summary: str) -> AgentResult:
        self.prompts.append(summary)
        return self.results.pop(0)


class OptimizeSupervisorTests(unittest.TestCase):
    def test_retries_with_progress_summary_after_stall(self) -> None:
        request = AgentRequest(
            command_kind=CommandKind.OPTIMIZE,
            input_path=Path("/tmp/op.py"),
            operator_path=Path("/tmp/op.py"),
            output_path=Path("/tmp/opt_op.py"),
            test_mode=None,
            bench_mode=None,
            interact=False,
            verbose=False,
            show_output=False,
            force_overwrite=False,
            agent_name="codex",
            skill_name="optimize",
            prompt="Optimize this operator",
            workdir=Path("/tmp"),
        )
        runner = FakeRunner(
            [
                AgentResult(
                    return_code=1,
                    stdout="working...\n",
                    stderr="",
                    stalled=True,
                    session_id=None,
                ),
                AgentResult(
                    return_code=0,
                    stdout="done",
                    stderr="",
                    stalled=False,
                    session_id=None,
                ),
            ]
        )
        supervisor = OptimizeSupervisor(max_recovery_attempts=1)
        result = supervisor.run(runner, request)
        self.assertEqual(result.return_code, 0)
        self.assertEqual(len(runner.prompts), 2)
        self.assertIn("working...", runner.prompts[1])


if __name__ == "__main__":
    unittest.main()
