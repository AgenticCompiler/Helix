import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.models import AgentRequest, AgentResult, CommandKind
from triton_agent.optimize.runtime import run_optimize_request


class OptimizeRuntimeTests(unittest.TestCase):
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
                optimize_role="worker",
            )

            class FakeRunner:
                def __init__(self) -> None:
                    self.roles: List[Optional[str]] = []
                    self.prompts: List[str] = []

                def run(self, request: AgentRequest) -> AgentResult:
                    self.roles.append(request.optimize_role)
                    self.prompts.append(request.prompt)
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
            self.assertEqual(runner.roles, ["worker", "supervisor"])
            self.assertIn("This invocation is the optimize supervisor role.", runner.prompts[1])

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
                optimize_role="worker",
            )

            class FakeRunner:
                def __init__(self) -> None:
                    self.roles: List[Optional[str]] = []
                    self.interacts: List[bool] = []

                def run(self, request: AgentRequest) -> AgentResult:
                    self.roles.append(request.optimize_role)
                    self.interacts.append(request.interact)
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
            self.assertEqual(runner.roles, ["worker", "supervisor"])
            self.assertEqual(runner.interacts, [True, False])


if __name__ == "__main__":
    unittest.main()
