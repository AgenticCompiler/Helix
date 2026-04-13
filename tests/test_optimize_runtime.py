import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from typing import Any, List, Optional, cast
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.models import AgentRequest, AgentResult, CommandKind
from triton_agent.optimize.batch import run_optimize_batch
from triton_agent.optimize.models import GateDecision, OptimizeRunOptions
from triton_agent.optimize.runtime import (
    OptimizeLoopRunner,
    _count_round_directories,
    _latest_round_dir,
    run_optimize_request,
)
from triton_agent.optimize_guidance import OptimizeGuidanceState


class OptimizeRuntimeTests(unittest.TestCase):
    def _build_guidance_state(self, workdir: Path) -> OptimizeGuidanceState:
        triton_dir = workdir / ".triton-agent"
        triton_dir.mkdir(parents=True, exist_ok=True)
        role_dir = triton_dir / "roles"
        role_dir.mkdir(parents=True, exist_ok=True)
        history_dir = triton_dir / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        round_brief_path = triton_dir / "round-brief.md"
        supervisor_report_path = triton_dir / "supervisor-report.md"
        round_brief_path.write_text("brief\n", encoding="utf-8")
        supervisor_report_path.write_text("report\n", encoding="utf-8")
        archive_root = workdir / "optimize-logs" / "triton-agent"
        run_archive_dir = archive_root / "run-001"
        shared_guidance_snapshot_path = run_archive_dir / "shared-guidance.md"
        return OptimizeGuidanceState(
            guidance_path=workdir / "AGENTS.md",
            backup_path=None,
            created_guidance=False,
            role_dir=role_dir,
            worker_brief_path=role_dir / "optimize-worker.md",
            supervisor_brief_path=role_dir / "optimize-supervisor.md",
            round_brief_path=round_brief_path,
            supervisor_report_path=supervisor_report_path,
            history_dir=history_dir,
            archive_root=archive_root,
            run_archive_dir=run_archive_dir,
            shared_guidance_snapshot_path=shared_guidance_snapshot_path,
            created_paths=(round_brief_path, supervisor_report_path),
        )

    def test_run_optimize_request_invokes_worker_then_supervisor_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="standalone",
                interact=False,
                verbose=False,
                show_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                supervise="on",
                optimize_role="worker",
            )

            class FakeRunner:
                def __init__(self) -> None:
                    self.requests: List[AgentRequest] = []

                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del stdout, stderr
                    self.requests.append(request)
                    if request.optimize_role == "worker":
                        round_dir = workdir / "opt-round-1"
                        round_dir.mkdir(exist_ok=True)
                        (workdir / "opt-note.md").write_text("## Round 1\n", encoding="utf-8")
                        (round_dir / "kernel.py").write_text("print('optimized')\n", encoding="utf-8")
                        (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
                        (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
                        (round_dir / "perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
                        (round_dir / "round-state.json").write_text(
                            json.dumps(
                                {
                                    "round": "opt-round-1",
                                    "parent_round": "round-0",
                                    "hypothesis": "vectorize loads",
                                    "evidence_sources": ["benchmark"],
                                    "correctness_status": "passed",
                                    "benchmark_status": "passed",
                                    "perf_artifact": "perf.txt",
                                    "perf_summary_source": "compare-perf",
                                    "summary_path": "summary.md",
                                    "opt_note_updated": True,
                                    "next_recommendation": "stop",
                                }
                            ),
                            encoding="utf-8",
                        )
                    return AgentResult(return_code=0, stdout="ok", stderr="")

            runner = FakeRunner()

            with patch("triton_agent.optimize.runtime.create_runner", return_value=runner):
                result = run_optimize_request(request)

            self.assertEqual(result.return_code, 0)
            self.assertEqual(len(runner.requests), 2)
            worker_request, supervisor_request = runner.requests
            self.assertEqual(worker_request.optimize_role, "worker")
            self.assertTrue(worker_request.no_agent_session)
            self.assertIsNotNone(worker_request.round_brief_path)
            self.assertIsNotNone(worker_request.supervisor_report_path)
            self.assertEqual(supervisor_request.optimize_role, "supervisor")
            self.assertEqual(supervisor_request.skill_name, "optimize-supervisor")
            self.assertFalse(supervisor_request.interact)
            self.assertTrue(supervisor_request.no_agent_session)
            self.assertEqual(supervisor_request.round_brief_path, worker_request.round_brief_path)
            self.assertEqual(supervisor_request.supervisor_report_path, worker_request.supervisor_report_path)
            self.assertFalse((workdir / ".triton-agent").exists())
            archive_root = workdir / "optimize-logs" / "triton-agent"
            self.assertTrue(archive_root.exists())
            run_archives = [path for path in archive_root.iterdir() if path.is_dir()]
            self.assertEqual(len(run_archives), 1)
            run_archive = run_archives[0]
            self.assertTrue((run_archive / "shared-guidance.md").exists())
            self.assertTrue((run_archive / "roles" / "optimize-worker.md").exists())
            self.assertTrue((run_archive / "roles" / "optimize-supervisor.md").exists())
            self.assertTrue((run_archive / "final" / "round-brief.md").exists())
            self.assertTrue((run_archive / "final" / "supervisor-report.md").exists())
            self.assertTrue((run_archive / "history").exists())

    def test_run_supervisor_appends_history_snapshots_across_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            round_one = workdir / "opt-round-1"
            round_one.mkdir()
            (round_one / "attempts.md").write_text("attempts\n", encoding="utf-8")
            (round_one / "summary.md").write_text("summary\n", encoding="utf-8")
            (round_one / "perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
            (round_one / "kernel.py").write_text("print('optimized-1')\n", encoding="utf-8")
            (round_one / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-1",
                        "parent_round": "round-0",
                        "hypothesis": "vectorize loads",
                        "evidence_sources": ["benchmark"],
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "perf_artifact": "perf.txt",
                        "perf_summary_source": "compare-perf",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                        "next_recommendation": "continue",
                    }
                ),
                encoding="utf-8",
            )

            guidance_state = self._build_guidance_state(workdir)
            existing_brief = guidance_state.history_dir / "round-004-brief.md"
            existing_report = guidance_state.history_dir / "round-004-supervisor-report.md"
            existing_brief.write_text("existing brief\n", encoding="utf-8")
            existing_report.write_text("existing report\n", encoding="utf-8")

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="standalone",
                interact=False,
                verbose=False,
                show_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                supervise="on",
                optimize_role="worker",
            )

            class FakeRunner:
                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del request, stdout, stderr
                    return AgentResult(return_code=0, stdout="ok", stderr="")

            loop_runner = OptimizeLoopRunner(cast(Any, FakeRunner()), guidance_state)

            first_result = loop_runner.run_supervisor(
                request,
                AgentResult(return_code=0, stdout="worker ok", stderr=""),
            )

            round_two = workdir / "opt-round-2"
            round_two.mkdir()
            (round_two / "attempts.md").write_text("attempts\n", encoding="utf-8")
            (round_two / "summary.md").write_text("summary\n", encoding="utf-8")
            (round_two / "perf.txt").write_text("case0: 0.8\n", encoding="utf-8")
            (round_two / "kernel.py").write_text("print('optimized-2')\n", encoding="utf-8")
            (round_two / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-2",
                        "parent_round": "opt-round-1",
                        "hypothesis": "fuse epilogue",
                        "evidence_sources": ["benchmark"],
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "perf_artifact": "perf.txt",
                        "perf_summary_source": "compare-perf",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                        "next_recommendation": "stop",
                    }
                ),
                encoding="utf-8",
            )

            second_result = loop_runner.run_supervisor(
                request,
                AgentResult(return_code=0, stdout="worker ok", stderr=""),
            )

            self.assertEqual(first_result.decision, GateDecision.PASS_CONTINUE)
            self.assertEqual(second_result.decision, GateDecision.PASS_STOP)
            history_dir = guidance_state.history_dir
            self.assertEqual(existing_brief.read_text(encoding="utf-8"), "existing brief\n")
            self.assertEqual(existing_report.read_text(encoding="utf-8"), "existing report\n")
            self.assertTrue((history_dir / "round-005-brief.md").exists())
            self.assertTrue((history_dir / "round-006-brief.md").exists())
            self.assertTrue((history_dir / "round-005-supervisor-report.md").exists())
            self.assertTrue((history_dir / "round-006-supervisor-report.md").exists())
            self.assertNotEqual(
                (history_dir / "round-005-brief.md").read_text(encoding="utf-8"),
                (history_dir / "round-006-brief.md").read_text(encoding="utf-8"),
            )

    def test_run_optimize_request_unsupervised_uses_single_agent_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="standalone",
                interact=False,
                verbose=False,
                show_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                supervise="off",
            )

            class FakeRunner:
                def __init__(self) -> None:
                    self.calls: List[AgentRequest] = []

                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del stdout, stderr
                    self.calls.append(request)
                    return AgentResult(return_code=0, stdout="ok", stderr="")

                def resume(
                    self,
                    request: AgentRequest,
                    summary: str,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del summary, stdout, stderr
                    self.calls.append(request)
                    return AgentResult(return_code=0, stdout="ok", stderr="")

            runner = FakeRunner()
            with patch("triton_agent.optimize.runtime.create_runner", return_value=runner):
                with patch("triton_agent.optimize.runtime.OptimizeGuidanceManager.prepare") as mocked_prepare:
                    result = run_optimize_request(request)

            self.assertEqual(result.return_code, 0)
            self.assertEqual(len(runner.calls), 1)
            self.assertEqual(runner.calls[0].supervise, "off")
            self.assertIsNone(runner.calls[0].round_brief_path)
            self.assertIsNone(runner.calls[0].supervisor_report_path)
            self.assertFalse((workdir / ".triton-agent" / "roles").exists())
            self.assertFalse((workdir / "optimize-logs").exists())
            mocked_prepare.assert_not_called()

    def test_run_optimize_request_unsupervised_retries_with_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="standalone",
                interact=False,
                verbose=False,
                show_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                supervise="off",
            )

            class RecordingRecoveryRunner:
                def __init__(self) -> None:
                    self.calls: List[str] = []
                    self.resume_summaries: List[str] = []

                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del stdout, stderr
                    self.calls.append("run")
                    return AgentResult(
                        return_code=1,
                        stdout="first stall",
                        stderr="",
                        stalled=True,
                        session_id=None,
                    )

                def resume(
                    self,
                    request: AgentRequest,
                    summary: str,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del stdout, stderr
                    self.calls.append("resume")
                    self.resume_summaries.append(summary)
                    return AgentResult(return_code=0, stdout="done", stderr="", stalled=False)

            runner = RecordingRecoveryRunner()
            with patch("triton_agent.optimize.runtime.create_runner", return_value=runner):
                result = run_optimize_request(request)

            self.assertEqual(result.return_code, 0)
            self.assertEqual(runner.calls, ["run", "resume"])
            self.assertEqual(runner.resume_summaries, ["first stall"])

    def test_run_optimize_batch_preserves_supervise_mode(self) -> None:
        for supervise_mode in ("on", "off"):
            with self.subTest(supervise=supervise_mode):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    workspace = root / "kernel_workspace"
                    workspace.mkdir()
                    operator = workspace / "kernel.py"
                    operator.write_text("print('x')\n", encoding="utf-8")

                    options = OptimizeRunOptions(
                        agent_name="codex",
                        interact=False,
                        verbose=False,
                        show_output=False,
                        remote=None,
                        remote_workdir=None,
                        min_rounds=None,
                        resume_mode="auto",
                        require_analysis=False,
                        no_agent_session=False,
                        supervise=supervise_mode,
                        output=None,
                        test_mode=None,
                        bench_mode=None,
                    )

                    captured_requests: List[AgentRequest] = []

                    def fake_run_request(
                        request: AgentRequest,
                        stdout: Optional[object] = None,
                        stderr: Optional[object] = None,
                    ) -> AgentResult:
                        del stdout, stderr
                        captured_requests.append(request)
                        return AgentResult(return_code=0, stdout="ok", stderr="")

                    with patch(
                        "triton_agent.optimize.batch.render_batch_optimize_results", return_value=0
                    ):
                        exit_code = run_optimize_batch(
                            root,
                            options,
                            max_concurrency=1,
                            stdout=StringIO(),
                            run_request=fake_run_request,
                        )

                    self.assertEqual(exit_code, 0)
                    self.assertEqual(len(captured_requests), 1)
                    batch_request = captured_requests[0]
                    self.assertEqual(batch_request.supervise, supervise_mode)
                    expected_role = "worker" if supervise_mode == "on" else None
                    self.assertEqual(batch_request.optimize_role, expected_role)

    def test_run_optimize_request_keeps_interactive_only_for_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="standalone",
                interact=True,
                verbose=False,
                show_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                supervise="on",
                optimize_role="worker",
            )

            class FakeRunner:
                def __init__(self) -> None:
                    self.requests: List[AgentRequest] = []

                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del stdout, stderr
                    self.requests.append(request)
                    if request.optimize_role == "worker":
                        round_dir = workdir / "opt-round-1"
                        round_dir.mkdir(exist_ok=True)
                        (workdir / "opt-note.md").write_text("## Round 1\n", encoding="utf-8")
                        (round_dir / "kernel.py").write_text("print('optimized')\n", encoding="utf-8")
                        (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
                        (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
                        (round_dir / "perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
                        (round_dir / "round-state.json").write_text(
                            json.dumps(
                                {
                                    "round": "opt-round-1",
                                    "parent_round": "round-0",
                                    "hypothesis": "vectorize loads",
                                    "evidence_sources": ["benchmark"],
                                    "correctness_status": "passed",
                                    "benchmark_status": "passed",
                                    "perf_artifact": "perf.txt",
                                    "perf_summary_source": "compare-perf",
                                    "summary_path": "summary.md",
                                    "opt_note_updated": True,
                                    "next_recommendation": "stop",
                                }
                            ),
                            encoding="utf-8",
                        )
                    return AgentResult(return_code=0, stdout="ok", stderr="")

            runner = FakeRunner()

            with patch("triton_agent.optimize.runtime.create_runner", return_value=runner):
                result = run_optimize_request(request)

            self.assertEqual(result.return_code, 0)
            self.assertEqual(len(runner.requests), 2)
            worker_request, supervisor_request = runner.requests
            self.assertEqual(worker_request.optimize_role, "worker")
            self.assertTrue(worker_request.interact)
            self.assertTrue(worker_request.no_agent_session)
            self.assertEqual(supervisor_request.optimize_role, "supervisor")
            self.assertFalse(supervisor_request.interact)
            self.assertTrue(supervisor_request.no_agent_session)

    def test_run_optimize_request_end_to_end_converts_gate_eval_value_error_to_gate_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="standalone",
                interact=False,
                verbose=False,
                show_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                supervise="on",
                optimize_role="worker",
            )

            class FakeRunner:
                def __init__(self) -> None:
                    self.requests: List[AgentRequest] = []
                    self.worker_calls = 0

                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del stdout, stderr
                    self.requests.append(request)
                    if request.optimize_role == "worker":
                        self.worker_calls += 1
                        if self.worker_calls == 1:
                            round_dir = workdir / "opt-round-1"
                            round_dir.mkdir(exist_ok=True)
                            (workdir / "opt-note.md").write_text("## Round 1\n", encoding="utf-8")
                            return AgentResult(return_code=0, stdout="worker ok", stderr="")
                        return AgentResult(return_code=1, stdout="", stderr="worker stopped for test")
                    return AgentResult(return_code=0, stdout="supervisor ok", stderr="")

            runner = FakeRunner()

            with patch("triton_agent.optimize.runtime.create_runner", return_value=runner):
                with patch(
                    "triton_agent.optimize.runtime.evaluate_round_gate",
                    side_effect=ValueError("invalid round-state.json in opt-round-1: missing fields"),
                ):
                    result = run_optimize_request(request)

            self.assertEqual(result.return_code, 1)
            self.assertEqual(len(runner.requests), 3)
            self.assertEqual(runner.requests[0].optimize_role, "worker")
            self.assertEqual(runner.requests[1].optimize_role, "supervisor")
            self.assertEqual(runner.requests[2].optimize_role, "worker")
            self.assertIn("Gate decision: revise-metadata", runner.requests[2].prompt)
            self.assertIn("invalid round-state.json in opt-round-1", runner.requests[2].prompt)

    def test_run_supervisor_converts_gate_eval_failures_to_gate_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            round_dir = workdir / "opt-round-1"
            round_dir.mkdir()

            guidance_state = self._build_guidance_state(workdir)

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="standalone",
                interact=False,
                verbose=False,
                show_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                supervise="on",
                optimize_role="worker",
            )

            class FakeRunner:
                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del request, stdout, stderr
                    return AgentResult(return_code=0, stdout="ok", stderr="")

            loop_runner = OptimizeLoopRunner(cast(Any, FakeRunner()), guidance_state)

            with patch(
                "triton_agent.optimize.runtime.evaluate_round_gate",
                side_effect=ValueError("invalid round-state.json in opt-round-1: missing fields"),
            ):
                gate_result = loop_runner.run_supervisor(
                    request,
                    AgentResult(return_code=0, stdout="worker ok", stderr=""),
                )

            self.assertEqual(gate_result.decision, GateDecision.REVISE_METADATA)
            self.assertIn("invalid round-state.json", gate_result.blocking_issues[0])
            self.assertIn(
                "Decision: revise-metadata",
                guidance_state.supervisor_report_path.read_text(encoding="utf-8"),
            )

    def test_run_supervisor_does_not_swallow_unexpected_gate_eval_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            round_dir = workdir / "opt-round-1"
            round_dir.mkdir()

            guidance_state = self._build_guidance_state(workdir)

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="standalone",
                interact=False,
                verbose=False,
                show_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                supervise="on",
                optimize_role="worker",
            )

            class FakeRunner:
                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del request, stdout, stderr
                    return AgentResult(return_code=0, stdout="ok", stderr="")

            loop_runner = OptimizeLoopRunner(cast(Any, FakeRunner()), guidance_state)

            with patch(
                "triton_agent.optimize.runtime.evaluate_round_gate",
                side_effect=RuntimeError("unexpected failure"),
            ):
                with self.assertRaises(RuntimeError):
                    loop_runner.run_supervisor(
                        request,
                        AgentResult(return_code=0, stdout="worker ok", stderr=""),
                    )

    def test_run_supervisor_updates_live_and_history_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            round_dir = workdir / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
            (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
            (round_dir / "perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
            (round_dir / "kernel.py").write_text("print('optimized')\n", encoding="utf-8")
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-1",
                        "parent_round": "round-0",
                        "hypothesis": "vectorize loads",
                        "evidence_sources": ["benchmark"],
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "perf_artifact": "perf.txt",
                        "perf_summary_source": "compare-perf",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                        "next_recommendation": "stop",
                    }
                ),
                encoding="utf-8",
            )

            guidance_state = self._build_guidance_state(workdir)

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="standalone",
                interact=False,
                verbose=False,
                show_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                supervise="on",
                optimize_role="worker",
            )

            class FakeRunner:
                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del stdout, stderr
                    return AgentResult(return_code=0, stdout="ok", stderr="")

            loop_runner = OptimizeLoopRunner(cast(Any, FakeRunner()), guidance_state)

            gate_result = loop_runner.run_supervisor(
                request,
                AgentResult(return_code=0, stdout="worker ok", stderr=""),
            )

            self.assertEqual(gate_result.decision, GateDecision.PASS_STOP)
            history_dir = guidance_state.history_dir
            brief_path = history_dir / "round-001-brief.md"
            report_path = history_dir / "round-001-supervisor-report.md"
            self.assertTrue(brief_path.exists())
            self.assertTrue(report_path.exists())
            self.assertEqual(
                guidance_state.round_brief_path.read_text(encoding="utf-8"),
                brief_path.read_text(encoding="utf-8"),
            )
            self.assertEqual(
                guidance_state.supervisor_report_path.read_text(encoding="utf-8"),
                report_path.read_text(encoding="utf-8"),
            )

    def test_latest_round_dir_prefers_highest_numeric_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "opt-round-2").mkdir()
            (workdir / "opt-round-10").mkdir()

            latest = _latest_round_dir(workdir)

            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.name, "opt-round-10")

    def test_round_helpers_ignore_non_numeric_opt_round_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "opt-round-2").mkdir()
            (workdir / "opt-round-final").mkdir()
            (workdir / "opt-round-notes").mkdir()

            latest = _latest_round_dir(workdir)

            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.name, "opt-round-2")
            self.assertEqual(_count_round_directories(workdir), 1)


if __name__ == "__main__":
    unittest.main()
