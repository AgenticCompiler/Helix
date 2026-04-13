import sys
import unittest
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.models import AgentRequest, AgentResult, CommandKind
from triton_agent.optimize.models import GateDecision, GateResult
from triton_agent.supervisor import OptimizeSupervisor


class FakeRunner:
    def __init__(self, results):
        self.results = list(results)
        self.prompts = []
        self.resume_requests = []

    def run(self, request: AgentRequest) -> AgentResult:
        self.prompts.append(request.prompt)
        return self.results.pop(0)

    def resume(self, request: AgentRequest, summary: str) -> AgentResult:
        self.resume_requests.append(request)
        self.prompts.append(summary)
        return self.results.pop(0)


class RoundCreatingRunner(FakeRunner):
    def __init__(self, results, workdir: Path) -> None:
        super().__init__(results)
        self.workdir = workdir

    def resume(self, request: AgentRequest, summary: str) -> AgentResult:
        next_round = self.workdir / "opt-round-2"
        next_round.mkdir(exist_ok=True)
        return super().resume(request, summary)


class OptimizeSupervisorTests(unittest.TestCase):
    def test_round_gate_runner_stops_after_pass_stop(self) -> None:
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

        class LoopRunner:
            def __init__(self) -> None:
                self.events: list[str] = []

            def run_worker(self, request: AgentRequest) -> AgentResult:
                self.events.append("worker-run")
                return AgentResult(return_code=0, stdout="round complete", stderr="")

            def run_supervisor(
                self, request: AgentRequest, result: AgentResult
            ) -> GateResult:
                self.events.append("supervisor-run")
                return GateResult(decision=GateDecision.PASS_STOP, blocking_issues=())

        runner = LoopRunner()

        result = OptimizeSupervisor().run(runner, request)

        self.assertEqual(result.return_code, 0)
        self.assertEqual(runner.events, ["worker-run", "supervisor-run"])

    def test_round_gate_runner_relaunches_worker_after_revise_required(self) -> None:
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

        class LoopRunner:
            def __init__(self) -> None:
                self.worker_requests: list[AgentRequest] = []
                self._gate_calls = 0

            def run_worker(self, request: AgentRequest) -> AgentResult:
                self.worker_requests.append(request)
                return AgentResult(return_code=0, stdout="round complete", stderr="")

            def run_supervisor(
                self, request: AgentRequest, result: AgentResult
            ) -> GateResult:
                self._gate_calls += 1
                if self._gate_calls == 1:
                    return GateResult(
                        decision=GateDecision.REVISE_REQUIRED,
                        blocking_issues=("required evidence is missing",),
                    )
                return GateResult(decision=GateDecision.PASS_STOP, blocking_issues=())

        runner = LoopRunner()

        result = OptimizeSupervisor().run(runner, request)

        self.assertEqual(result.return_code, 0)
        self.assertEqual(len(runner.worker_requests), 2)
        self.assertIn("required evidence is missing", runner.worker_requests[1].prompt)
        self.assertEqual(runner.worker_requests[1].optimize_role, "worker")

    def test_round_gate_runner_retries_transient_worker_failure(self) -> None:
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

        class LoopRunner:
            def __init__(self) -> None:
                self.worker_calls = 0
                self.supervisor_calls = 0

            def run_worker(self, request: AgentRequest) -> AgentResult:
                del request
                self.worker_calls += 1
                if self.worker_calls == 1:
                    return AgentResult(
                        return_code=1,
                        stdout="",
                        stderr="ERROR: exceeded retry limit, last status: 429 Too Many Requests",
                    )
                return AgentResult(return_code=0, stdout="round complete", stderr="")

            def run_supervisor(
                self, request: AgentRequest, result: AgentResult
            ) -> GateResult:
                del request, result
                self.supervisor_calls += 1
                return GateResult(decision=GateDecision.PASS_STOP, blocking_issues=())

        runner = LoopRunner()

        result = OptimizeSupervisor(max_recovery_attempts=1).run(runner, request)

        self.assertEqual(result.return_code, 0)
        self.assertEqual(runner.worker_calls, 2)
        self.assertEqual(runner.supervisor_calls, 1)

    def test_round_gate_runner_retries_transient_supervisor_failure(self) -> None:
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

        class LoopRunner:
            def __init__(self) -> None:
                self.worker_calls = 0
                self.supervisor_calls = 0

            def run_worker(self, request: AgentRequest) -> AgentResult:
                del request
                self.worker_calls += 1
                return AgentResult(return_code=0, stdout="round complete", stderr="")

            def run_supervisor(
                self, request: AgentRequest, result: AgentResult
            ) -> GateResult:
                del request, result
                self.supervisor_calls += 1
                if self.supervisor_calls == 1:
                    return GateResult(
                        decision=GateDecision.HARD_FAIL,
                        blocking_issues=(
                            "ERROR: exceeded retry limit, last status: 429 Too Many Requests",
                        ),
                    )
                return GateResult(decision=GateDecision.PASS_STOP, blocking_issues=())

        runner = LoopRunner()

        result = OptimizeSupervisor(max_recovery_attempts=1).run(runner, request)

        self.assertEqual(result.return_code, 0)
        self.assertEqual(runner.worker_calls, 1)
        self.assertEqual(runner.supervisor_calls, 2)

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
            min_rounds=None,
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

    def test_repeated_stalls_keep_using_resume_path(self) -> None:
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
            min_rounds=None,
        )

        class RepeatedStallRunner:
            def __init__(self) -> None:
                self.calls: list[str] = []
                self.resume_summaries: list[str] = []
                self._resume_count = 0

            def run(self, request: AgentRequest) -> AgentResult:
                self.calls.append("run")
                return AgentResult(return_code=1, stdout="first stall", stderr="", stalled=True)

            def resume(self, request: AgentRequest, summary: str) -> AgentResult:
                self.calls.append("resume")
                self.resume_summaries.append(summary)
                self._resume_count += 1
                if self._resume_count == 1:
                    return AgentResult(return_code=1, stdout="second stall", stderr="", stalled=True)
                return AgentResult(return_code=0, stdout="done", stderr="", stalled=False)

        runner = RepeatedStallRunner()
        supervisor = OptimizeSupervisor(max_recovery_attempts=2)

        result = supervisor.run(runner, request)

        self.assertEqual(result.return_code, 0)
        self.assertEqual(runner.calls, ["run", "resume", "resume"])
        self.assertEqual(runner.resume_summaries, ["first stall", "second stall"])

    def test_restarts_when_successful_run_has_too_few_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "opt-round-1").mkdir()
            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=workspace / "op.py",
                operator_path=workspace / "op.py",
                output_path=workspace / "opt_op.py",
                test_mode=None,
                bench_mode=None,
                interact=False,
                verbose=False,
                show_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="optimize",
                prompt="Optimize this operator",
                workdir=workspace,
                min_rounds=2,
            )
            runner = RoundCreatingRunner(
                [
                    AgentResult(return_code=0, stdout="finished one round", stderr=""),
                    AgentResult(return_code=0, stdout="finished two rounds", stderr=""),
                ],
                workspace,
            )

            supervisor = OptimizeSupervisor(max_recovery_attempts=1)
            result = supervisor.run(runner, request)

            self.assertEqual(result.return_code, 0)
            self.assertEqual(len(runner.resume_requests), 1)
            self.assertEqual(runner.resume_requests[0].optimize_role, "worker")
            self.assertIn("Continue the existing optimize task", runner.resume_requests[0].prompt)
            self.assertIn("Read `opt-note.md`", runner.resume_requests[0].prompt)

    def test_restarts_with_strict_analysis_wording_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "opt-round-1").mkdir()
            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=workspace / "op.py",
                operator_path=workspace / "op.py",
                output_path=workspace / "opt_op.py",
                test_mode=None,
                bench_mode=None,
                interact=False,
                verbose=False,
                show_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="optimize",
                prompt="Optimize this operator",
                workdir=workspace,
                min_rounds=2,
                require_analysis=True,
            )
            runner = RoundCreatingRunner(
                [
                    AgentResult(return_code=0, stdout="finished one round", stderr=""),
                    AgentResult(return_code=0, stdout="finished two rounds", stderr=""),
                ],
                workspace,
            )

            result = OptimizeSupervisor(max_recovery_attempts=1).run(runner, request)

            self.assertEqual(result.return_code, 0)
            self.assertEqual(len(runner.resume_requests), 1)
            self.assertIn("profiling or IR-backed evidence", runner.resume_requests[0].prompt)

    def test_user_interrupt_does_not_trigger_recovery(self) -> None:
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
            min_rounds=None,
        )
        runner = FakeRunner(
            [
                AgentResult(
                    return_code=130,
                    stdout="",
                    stderr="Interrupted",
                    stalled=False,
                    session_id=None,
                )
            ]
        )

        result = OptimizeSupervisor(max_recovery_attempts=2).run(runner, request)

        self.assertEqual(result.return_code, 130)
        self.assertEqual(runner.prompts, ["Optimize this operator"])
        self.assertEqual(runner.resume_requests, [])


if __name__ == "__main__":
    unittest.main()
