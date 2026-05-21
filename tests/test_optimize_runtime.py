import importlib.util
import json
import os
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
import triton_agent.optimize.execution as execution_module
from triton_agent.optimize.compiler_source import CompilerSourceInfo
from triton_agent.optimize.models import GateDecision, OptimizeRunOptions
from triton_agent.optimize.execution import (
    SupervisedOptimizeAdapter,
    _count_round_directories,
    _latest_round_dir,
)
from triton_agent.optimize.orchestration import build_optimize_request, run_optimize_request
from triton_agent.optimize.pt_cleanup import cleanup_workspace_pt_files
from triton_agent.optimize.archive import ArchiveState
from triton_agent.optimize.memory_file import MemoryFileState
from triton_agent.optimize.resume import reset_optimize_workspace
from triton_agent.optimize.runtime_handoff import RuntimeHandoffState
from triton_agent.optimize.session_artifacts import OptimizeSessionArtifactsState


class OptimizeRuntimeTests(unittest.TestCase):
    def test_optimize_orchestration_module_replaces_runtime_module(self) -> None:
        self.assertIsNotNone(importlib.util.find_spec("triton_agent.optimize.orchestration"))
        self.assertIsNone(importlib.util.find_spec("triton_agent.optimize.runtime"))

    def test_optimize_run_loop_module_replaces_supervisor_module(self) -> None:
        self.assertIsNotNone(importlib.util.find_spec("triton_agent.optimize.run_loop"))
        self.assertIsNone(importlib.util.find_spec("triton_agent.optimize.supervisor"))

    def test_optimize_gate_module_has_been_removed(self) -> None:
        self.assertIsNone(importlib.util.find_spec("triton_agent.optimize.gate"))

    def _build_guidance_state(self, workdir: Path) -> OptimizeSessionArtifactsState:
        runtime_root = workdir / ".triton-agent"
        runtime_root.mkdir(parents=True, exist_ok=True)
        guidance_path = workdir / "AGENTS.md"
        guidance_path.write_text("shared guidance\n", encoding="utf-8")
        history_dir = runtime_root / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        round_brief_path = runtime_root / "round-brief.md"
        supervisor_report_path = runtime_root / "supervisor-report.md"
        round_brief_path.write_text("brief\n", encoding="utf-8")
        supervisor_report_path.write_text("report\n", encoding="utf-8")
        archive_root = workdir / "triton-agent-logs" / "triton-agent"
        run_archive_dir = archive_root / "run-001"
        shared_guidance_snapshot_path = run_archive_dir / "shared-guidance.md"
        return OptimizeSessionArtifactsState(
            memory_file=MemoryFileState(
                guidance_path=guidance_path,
                backup_path=None,
                created_guidance=False,
            ),
            archive=ArchiveState(
                archive_root=archive_root,
                run_archive_dir=run_archive_dir,
                agent_sessions_path=run_archive_dir / "agent-sessions.jsonl",
                shared_guidance_snapshot_path=shared_guidance_snapshot_path,
            ),
            runtime_handoff=RuntimeHandoffState(
                runtime_root=runtime_root,
                round_brief_path=round_brief_path,
                supervisor_report_path=supervisor_report_path,
                history_dir=history_dir,
                created_paths=(round_brief_path, supervisor_report_path),
            ),
        )

    def _write_baseline(self, workdir: Path) -> None:
        baseline_dir = workdir / "baseline"
        baseline_dir.mkdir(exist_ok=True)
        (workdir / "kernel.py").write_text("print('source')\n", encoding="utf-8")
        (baseline_dir / "state.json").write_text(
            json.dumps(
                {
                    "baseline_kind": "prepared",
                    "source_operator": "kernel.py",
                    "baseline_operator": "baseline/kernel.py",
                    "test_file": "differential_test_kernel.py",
                    "test_mode": "differential",
                    "bench_file": "bench_kernel.py",
                    "bench_mode": "standalone",
                    "perf_artifact": "baseline/perf.txt",
                    "correctness_status": "passed",
                    "benchmark_status": "passed",
                    "baseline_established": True,
                }
            ),
            encoding="utf-8",
        )
        (baseline_dir / "perf.txt").write_text("latency-a: 1.0\n", encoding="utf-8")
        (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")

    def _write_round(
        self,
        workdir: Path,
        round_name: str,
        *,
        parent_round: str,
        round_disposition: str,
        perf_text: str = "case0: 1.0\n",
    ) -> Path:
        round_dir = workdir / round_name
        round_dir.mkdir(exist_ok=True)
        (workdir / "opt-note.md").write_text("## Round\n", encoding="utf-8")
        (round_dir / "opt_kernel.py").write_text(f"print('{round_name}')\n", encoding="utf-8")
        (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
        (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
        (round_dir / "opt_kernel_perf.txt").write_text(perf_text, encoding="utf-8")
        (round_dir / "round-state.json").write_text(
            json.dumps(
                {
                    "round": round_name,
                    "parent_round": parent_round,
                    "hypothesis": "vectorize loads",
                    "evidence_sources": ["benchmark"],
                    "correctness_status": "passed",
                    "benchmark_status": "passed",
                    "perf_artifact": "opt_kernel_perf.txt",
                    "canonical_baseline": "baseline",
                    "comparison_target": "baseline/perf.txt",
                    "perf_summary_source": "compare-perf",
                    "effective_metric_source": "kernel",
                    "summary_path": "summary.md",
                    "opt_note_updated": True,
                    "round_disposition": round_disposition,
                }
            ),
            encoding="utf-8",
        )
        return round_dir

    def _write_supervisor_handoff(
        self,
        guidance_state: OptimizeSessionArtifactsState,
        *,
        decision: str,
        issues: tuple[str, ...] = (),
        brief_lines: tuple[str, ...] = (),
        latest_round: Optional[str] = None,
    ) -> None:
        report_lines = [
            "# Optimize Supervisor Report",
            "",
            f"Decision: {decision}",
            f"Blocking issues: {', '.join(issues) if issues else 'none'}",
        ]
        if latest_round is not None:
            report_lines.append(f"Latest round: {latest_round}")
        guidance_state.supervisor_report_path.write_text(
            "\n".join(report_lines) + "\n",
            encoding="utf-8",
        )
        brief_content = "# Optimize Round Brief\n\n"
        if brief_lines:
            brief_content += "\n".join(brief_lines) + "\n"
        else:
            brief_content += "No additional guidance.\n"
        guidance_state.round_brief_path.write_text(brief_content, encoding="utf-8")

    def test_run_optimize_request_delegates_supervised_flow_to_helper(self) -> None:
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
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                supervise="on",
                optimize_role="worker",
            )

            expected = AgentResult(return_code=0, stdout="ok", stderr="")
            fake_runner = object()

            with patch("triton_agent.optimize.orchestration.create_runner", return_value=fake_runner):
                with patch("triton_agent.optimize.orchestration.SkillLinkManager.prepare_skills", return_value=()):
                    with patch("triton_agent.optimize.orchestration.SkillLinkManager.cleanup", return_value=[]):
                        with patch.object(
                            execution_module,
                            "execute_supervised_optimize",
                            return_value=expected,
                        ) as mocked:
                            result = run_optimize_request(request)

            self.assertIs(result, expected)
            mocked.assert_called_once()

    def test_run_optimize_request_delegates_unsupervised_flow_to_helper(self) -> None:
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
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                supervise="off",
            )

            expected = AgentResult(return_code=0, stdout="ok", stderr="")
            fake_runner = object()

            with patch("triton_agent.optimize.orchestration.create_runner", return_value=fake_runner):
                with patch("triton_agent.optimize.orchestration.SkillLinkManager.prepare_skills", return_value=()):
                    with patch("triton_agent.optimize.orchestration.SkillLinkManager.cleanup", return_value=[]):
                        with patch.object(
                            execution_module,
                            "execute_unsupervised_optimize",
                            return_value=expected,
                        ) as mocked:
                            result = run_optimize_request(request)

            self.assertIs(result, expected)
            mocked.assert_called_once()

    def test_build_optimize_request_skips_compiler_source_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
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
                reset_optimize=False,
                no_agent_session=False,
                supervise="off",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
            )

            with patch("triton_agent.optimize.orchestration.prepare_compiler_source") as mocked:
                request = build_optimize_request(operator, workdir, options)

            mocked.assert_not_called()
            self.assertEqual(request.compiler_source_analysis, "off")
            self.assertIsNone(request.compiler_source_path)
            self.assertIsNone(request.compiler_source_commit)

    def test_build_optimize_request_disables_agent_hooks_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
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
                reset_optimize=False,
                no_agent_session=False,
                supervise="off",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
            )

            request = build_optimize_request(operator, workdir, options)

            self.assertFalse(request.enable_agent_hooks)

    def test_build_optimize_request_enables_agent_hooks_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
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
                reset_optimize=False,
                no_agent_session=False,
                supervise="off",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
                enable_agent_hooks=True,
            )

            request = build_optimize_request(operator, workdir, options)

            self.assertTrue(request.enable_agent_hooks)

    def test_build_optimize_request_uses_explicit_optimize_skill_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
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
                reset_optimize=False,
                no_agent_session=False,
                supervise="off",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
            )

            request = build_optimize_request(operator, workdir, options)

            self.assertEqual(
                request.staged_skill_names,
                (
                    "triton-npu-optimize",
                    "triton-npu-optimize-knowledge",
                    "triton-npu-prepare-optimize-baseline",
                    "triton-npu-gen-test",
                    "triton-npu-gen-bench",
                    "triton-npu-run-eval",
                    "triton-npu-optimize-check",
                    "triton-npu-profile-operator",
                    "triton-npu-analyze-round-performance",
                    "triton-npu-analyze-ir",
                    "triton-npu-analyze-compiler-source",
                    "triton-npu-repair-guide",
                ),
            )
            self.assertIsNone(request.staged_skill_sources)
            self.assertNotIn(
                "triton-npu-convert-pytorch-operator",
                request.staged_skill_names or (),
            )
            self.assertNotIn(
                "triton-npu-cann-ext-api-patterns",
                request.staged_skill_names or (),
            )

    def test_build_optimize_request_defaults_optimize_knowledge_to_v1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
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
                reset_optimize=False,
                no_agent_session=False,
                supervise="off",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
                optimize_knowledge="v1",
            )

            request = build_optimize_request(operator, workdir, options)

            self.assertIsNone(request.staged_skill_sources)

    def test_build_optimize_request_defaults_optimize_target_to_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
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
                reset_optimize=False,
                no_agent_session=False,
                supervise="off",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
                optimize_target="kernel",
            )

            request = build_optimize_request(operator, workdir, options)

            self.assertEqual(request.optimize_target, "kernel")

    def test_build_optimize_request_preserves_operator_optimize_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
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
                reset_optimize=False,
                no_agent_session=False,
                supervise="off",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
                optimize_target="operator",
            )

            request = build_optimize_request(operator, workdir, options)

            self.assertEqual(request.optimize_target, "operator")
            self.assertIn(
                "torch-npu-optimize-knowledge",
                request.staged_skill_names or (),
            )

    def test_build_optimize_request_maps_v2_knowledge_to_stable_staged_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
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
                reset_optimize=False,
                no_agent_session=False,
                supervise="off",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
                optimize_knowledge="v2",
            )

            request = build_optimize_request(operator, workdir, options)

            self.assertIn(
                "triton-npu-optimize-knowledge",
                request.staged_skill_names or (),
            )
            self.assertEqual(
                request.staged_skill_sources,
                {"triton-npu-optimize-knowledge": "triton-npu-optimize-knowledge-v2"},
            )

    def test_build_optimize_request_maps_v3_knowledge_to_stable_staged_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
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
                reset_optimize=False,
                no_agent_session=False,
                supervise="off",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
                optimize_knowledge="v3",
            )

            request = build_optimize_request(operator, workdir, options)

            self.assertIn(
                "triton-npu-optimize-knowledge",
                request.staged_skill_names or (),
            )
            self.assertEqual(
                request.staged_skill_sources,
                {"triton-npu-optimize-knowledge": "triton-npu-optimize-knowledge-v3"},
            )

    def test_build_optimize_request_stages_cann_ext_api_skill_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
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
                reset_optimize=False,
                no_agent_session=False,
                supervise="off",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
                enable_cann_ext_api=True,
            )

            request = build_optimize_request(operator, workdir, options)

            self.assertIn(
                "triton-npu-cann-ext-api-patterns",
                request.staged_skill_names or (),
            )
            self.assertIn(
                "CANN Triton extension API pattern access is enabled for this optimize run.",
                request.prompt,
            )

    def test_build_optimize_request_maps_v2_knowledge_and_cann_ext_api_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
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
                reset_optimize=False,
                no_agent_session=False,
                supervise="off",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
                optimize_knowledge="v2",
                enable_cann_ext_api=True,
            )

            request = build_optimize_request(operator, workdir, options)

            self.assertIn("triton-npu-cann-ext-api-patterns", request.staged_skill_names or ())
            self.assertEqual(
                request.staged_skill_sources,
                {"triton-npu-optimize-knowledge": "triton-npu-optimize-knowledge-v2"},
            )

    def test_build_optimize_request_maps_v3_knowledge_and_cann_ext_api_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
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
                reset_optimize=False,
                no_agent_session=False,
                supervise="off",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
                optimize_knowledge="v3",
                enable_cann_ext_api=True,
            )

            request = build_optimize_request(operator, workdir, options)

            self.assertIn("triton-npu-cann-ext-api-patterns", request.staged_skill_names or ())
            self.assertEqual(
                request.staged_skill_sources,
                {"triton-npu-optimize-knowledge": "triton-npu-optimize-knowledge-v3"},
            )

    def test_build_optimize_request_provisions_compiler_source_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            source_path = (workdir / "AscendNPU-IR").resolve()
            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                show_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=None,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                supervise="off",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
                compiler_source_analysis="auto",
            )

            with patch(
                "triton_agent.optimize.orchestration.prepare_compiler_source",
                return_value=CompilerSourceInfo(
                    path=source_path,
                    commit="abc123",
                ),
            ) as mocked:
                request = build_optimize_request(operator, workdir, options)

            mocked.assert_called_once_with(
                mode="auto",
            )
            self.assertEqual(request.compiler_source_analysis, "auto")
            self.assertEqual(request.compiler_source_path, source_path)
            self.assertEqual(request.compiler_source_commit, "abc123")
            self.assertIn("Compiler source path: ", request.prompt)
            self.assertNotIn("https://gitcode.com/Ascend/AscendNPU-IR.git", request.prompt)

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
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                supervise="on",
                optimize_role="worker",
            )

            class FakeRunner:
                def __init__(self) -> None:
                    self.requests: List[AgentRequest] = []
                    self.supervisor_calls = 0

                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del stdout, stderr
                    self.requests.append(request)
                    if request.optimize_role == "worker":
                        self_outer._write_baseline(workdir)
                        self_outer._write_round(
                            workdir,
                            "opt-round-1",
                            parent_round="round-0",
                            round_disposition="stop",
                        )
                    else:
                        self.supervisor_calls += 1
                        self_outer._write_supervisor_handoff(
                            guidance_state,
                            decision="pass-stop",
                            latest_round="opt-round-1",
                            brief_lines=("Stop after auditing opt-round-1.",),
                        )
                    session_id = (
                        "019da9c2-dfcb-7c71-a2f9-7a90bab2e0f5"
                        if request.optimize_role == "worker"
                        else "119da9c2-dfcb-7c71-a2f9-7a90bab2e0f5"
                    )
                    return AgentResult(return_code=0, stdout="ok", stderr="", session_id=session_id)

            self_outer = self
            runner = FakeRunner()
            guidance_state = self._build_guidance_state(workdir)

            with patch("triton_agent.optimize.orchestration.create_runner", return_value=runner):
                with patch(
                    "triton_agent.optimize.orchestration.OptimizeSessionArtifactsManager.prepare_supervised_session",
                    return_value=guidance_state,
                ):
                    result = run_optimize_request(request)

            self.assertEqual(result.return_code, 0)
            self.assertEqual(len(runner.requests), 2)
            worker_request, supervisor_request = runner.requests
            self.assertEqual(worker_request.optimize_role, "worker")
            self.assertTrue(worker_request.no_agent_session)
            self.assertIsNotNone(worker_request.round_brief_path)
            self.assertIsNotNone(worker_request.supervisor_report_path)
            self.assertEqual(supervisor_request.optimize_role, "supervisor")
            self.assertEqual(supervisor_request.skill_name, "triton-npu-optimize")
            self.assertFalse(supervisor_request.interact)
            self.assertTrue(supervisor_request.no_agent_session)
            self.assertEqual(supervisor_request.round_brief_path, worker_request.round_brief_path)
            self.assertEqual(
                supervisor_request.supervisor_report_path,
                worker_request.supervisor_report_path,
            )
            self.assertFalse((workdir / ".triton-agent").exists())
            archive_root = workdir / "triton-agent-logs" / "triton-agent"
            self.assertTrue(archive_root.exists())
            run_archives = [path for path in archive_root.iterdir() if path.is_dir()]
            self.assertEqual(len(run_archives), 1)
            run_archive = run_archives[0]
            self.assertTrue((run_archive / "shared-guidance.md").exists())
            self.assertTrue((run_archive / "final" / "round-brief.md").exists())
            self.assertTrue((run_archive / "final" / "supervisor-report.md").exists())
            self.assertTrue((run_archive / "history").exists())
            session_lines = (run_archive / "agent-sessions.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(session_lines), 2)
            session_entries = [json.loads(line) for line in session_lines]
            self.assertEqual(
                [(entry["role"], entry["session_id"], entry["agent"]) for entry in session_entries],
                [
                    ("worker", "019da9c2-dfcb-7c71-a2f9-7a90bab2e0f5", "codex"),
                    ("supervisor", "119da9c2-dfcb-7c71-a2f9-7a90bab2e0f5", "codex"),
                ],
            )

    def test_run_supervisor_appends_history_snapshots_across_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)
            self._write_round(
                workdir,
                "opt-round-1",
                parent_round="round-0",
                round_disposition="continue",
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
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                supervise="on",
                optimize_role="worker",
            )

            class FakeRunner:
                def __init__(self) -> None:
                    self.supervisor_calls = 0

                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del stdout, stderr
                    self.supervisor_calls += 1
                    if self.supervisor_calls == 1:
                        self_outer._write_supervisor_handoff(
                            guidance_state,
                            decision="pass-continue",
                            latest_round="opt-round-1",
                            brief_lines=("Continue from opt-round-1 with narrower changes.",),
                        )
                    else:
                        self_outer._write_supervisor_handoff(
                            guidance_state,
                            decision="pass-stop",
                            latest_round="opt-round-2",
                            brief_lines=("Stop after verifying opt-round-2.",),
                        )
                    return AgentResult(return_code=0, stdout="ok", stderr="")

            loop_runner = SupervisedOptimizeAdapter(cast(Any, FakeRunner()), guidance_state)
            self_outer = self

            first_result = loop_runner.run_supervisor(
                request,
                AgentResult(return_code=0, stdout="worker ok", stderr=""),
            )

            self._write_round(
                workdir,
                "opt-round-2",
                parent_round="opt-round-1",
                round_disposition="stop",
                perf_text="case0: 0.8\n",
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
            self.assertIn(
                "Continue from opt-round-1 with narrower changes.",
                (history_dir / "round-005-brief.md").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "Stop after verifying opt-round-2.",
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
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                supervise="off",
            )

            class FakeRunner:
                def __init__(self) -> None:
                    self.calls: List[AgentRequest] = []
                    self.saw_guidance_file = False
                    self.guidance_content = ""

                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del stdout, stderr
                    self.calls.append(request)
                    self.saw_guidance_file = (workdir / "AGENTS.md").exists()
                    if self.saw_guidance_file:
                        self.guidance_content = (workdir / "AGENTS.md").read_text(encoding="utf-8")
                    return AgentResult(return_code=0, stdout="ok", stderr="", session_id=None)

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
            with patch("triton_agent.optimize.orchestration.create_runner", return_value=runner):
                with patch(
                    "triton_agent.optimize.orchestration.OptimizeSessionArtifactsManager.prepare_supervised_session"
                ) as mocked_prepare:
                    result = run_optimize_request(request)

            self.assertEqual(result.return_code, 0)
            self.assertEqual(len(runner.calls), 1)
            self.assertEqual(runner.calls[0].supervise, "off")
            self.assertTrue(runner.saw_guidance_file)
            self.assertIn("Own the end-to-end optimize session.", runner.guidance_content)
            self.assertNotIn("Read the role brief", runner.guidance_content)
            self.assertIsNone(runner.calls[0].round_brief_path)
            self.assertIsNone(runner.calls[0].supervisor_report_path)
            self.assertFalse((workdir / "AGENTS.md").exists())
            self.assertFalse((workdir / ".triton-agent").exists())
            archive_root = workdir / "triton-agent-logs" / "triton-agent"
            run_archives = [path for path in archive_root.iterdir() if path.is_dir()]
            self.assertEqual(len(run_archives), 1)
            session_lines = (run_archives[0] / "agent-sessions.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(session_lines), 1)
            session_entry = json.loads(session_lines[0])
            self.assertEqual(session_entry["role"], "worker")
            self.assertEqual(session_entry["session_id"], "unknown")
            self.assertEqual(session_entry["agent"], "codex")
            mocked_prepare.assert_not_called()

    def test_run_optimize_request_unsupervised_operator_target_guidance(self) -> None:
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
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                supervise="off",
                optimize_target="operator",
            )

            class FakeRunner:
                def __init__(self) -> None:
                    self.guidance_content = ""

                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del request, stdout, stderr
                    self.guidance_content = (workdir / "AGENTS.md").read_text(encoding="utf-8")
                    return AgentResult(return_code=0, stdout="ok", stderr="", session_id=None)

                def resume(
                    self,
                    request: AgentRequest,
                    summary: str,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del request, summary, stdout, stderr
                    return AgentResult(return_code=0, stdout="ok", stderr="")

            runner = FakeRunner()
            with patch("triton_agent.optimize.orchestration.create_runner", return_value=runner):
                result = run_optimize_request(request)

            self.assertEqual(result.return_code, 0)
            self.assertIn("Target optimization scope: operator.", runner.guidance_content)
            self.assertIn("Optimize end-to-end operator latency.", runner.guidance_content)

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
                skill_name="triton-npu-optimize",
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
                    del request, stdout, stderr
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
                    del request, stdout, stderr
                    self.calls.append("resume")
                    self.resume_summaries.append(summary)
                    return AgentResult(return_code=0, stdout="done", stderr="", stalled=False)

            runner = RecordingRecoveryRunner()
            with patch("triton_agent.optimize.orchestration.create_runner", return_value=runner):
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
                        reset_optimize=False,
                        no_agent_session=False,
                        supervise=supervise_mode,
                        output=None,
                        test_mode=None,
                        bench_mode=None,
                        prompt=None,
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
                        "triton_agent.optimize.batch.render_batch_optimize_results",
                        return_value=0,
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

    def test_run_optimize_batch_applies_user_prompt_to_each_workspace_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("kernel_a", "kernel_b"):
                workspace = root / name
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
                reset_optimize=False,
                no_agent_session=False,
                supervise="off",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt="Avoid changing numerics.",
            )

            captured_prompts: List[str] = []

            def fake_run_request(
                request: AgentRequest,
                stdout: Optional[object] = None,
                stderr: Optional[object] = None,
            ) -> AgentResult:
                del stdout, stderr
                captured_prompts.append(request.prompt)
                return AgentResult(return_code=0, stdout="ok", stderr="")

            with patch(
                "triton_agent.optimize.batch.render_batch_optimize_results",
                return_value=0,
            ):
                exit_code = run_optimize_batch(
                    root,
                    options,
                    max_concurrency=1,
                    stdout=StringIO(),
                    run_request=fake_run_request,
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(captured_prompts), 2)
            for prompt in captured_prompts:
                self.assertIn("Additional user instructions:", prompt)
                self.assertIn("Avoid changing numerics.", prompt)

    def test_run_optimize_batch_assigns_distinct_affinity_env_per_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("alpha", "beta"):
                workspace = root / name
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
                reset_optimize=False,
                no_agent_session=False,
                supervise="off",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
            )
            seen_envs: list[dict[str, str]] = []

            def fake_run_request(
                request: AgentRequest,
                stdout: Optional[object] = None,
                stderr: Optional[object] = None,
            ) -> AgentResult:
                del stdout, stderr
                seen_envs.append(request.extra_env or {})
                return AgentResult(return_code=0, stdout="ok", stderr="")

            with patch.dict(os.environ, {"TRITON_AGENT_BATCH_NPU_DEVICES": "0,1"}, clear=False):
                with patch(
                    "triton_agent.optimize.batch.render_batch_optimize_results",
                    return_value=0,
                ):
                    exit_code = run_optimize_batch(
                        root,
                        options,
                        max_concurrency=2,
                        stdout=StringIO(),
                        run_request=fake_run_request,
                    )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                {env["ASCEND_RT_VISIBLE_DEVICES"] for env in seen_envs},
                {"0", "1"},
            )

    def test_run_optimize_batch_rejects_concurrency_larger_than_affinity_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "alpha"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                show_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=None,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                supervise="off",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
            )

            with patch.dict(os.environ, {"TRITON_AGENT_BATCH_NPU_DEVICES": "0"}, clear=False):
                with self.assertRaisesRegex(ValueError, "--max-concurrency"):
                    run_optimize_batch(
                        root,
                        options,
                        max_concurrency=2,
                        stdout=StringIO(),
                        run_request=lambda request, stdout=None, stderr=None: AgentResult(
                            return_code=0,
                            stdout="ok",
                            stderr="",
                        ),
                    )

    def test_run_optimize_batch_passes_compiler_source_settings_to_each_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root / "AscendNPU-IR"
            for name in ("kernel_a", "kernel_b"):
                workspace = root / name
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
                reset_optimize=False,
                no_agent_session=False,
                supervise="off",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
                compiler_source_analysis="auto",
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
                "triton_agent.optimize.orchestration.prepare_compiler_source",
                return_value=CompilerSourceInfo(
                    path=source_path,
                    commit="abc123",
                ),
            ):
                with patch(
                    "triton_agent.optimize.batch.render_batch_optimize_results",
                    return_value=0,
                ):
                    exit_code = run_optimize_batch(
                        root,
                        options,
                        max_concurrency=1,
                        stdout=StringIO(),
                        run_request=fake_run_request,
                    )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(captured_requests), 2)
            for request in captured_requests:
                self.assertEqual(request.compiler_source_analysis, "auto")
                self.assertEqual(request.compiler_source_path, source_path)
                self.assertEqual(request.compiler_source_commit, "abc123")

    def test_run_optimize_batch_skips_completed_workspace_from_root_status_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            alpha = root / "alpha"
            beta = root / "beta"
            alpha.mkdir()
            beta.mkdir()
            (alpha / "kernel.py").write_text("print('alpha')\n", encoding="utf-8")
            (beta / "kernel.py").write_text("print('beta')\n", encoding="utf-8")
            (root / "optimize-batch-status.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "workspaces": {
                            "alpha": {
                                "status": "completed",
                                "operator_file": "kernel.py",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                show_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=None,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                supervise="off",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
            )

            seen_inputs: list[Path] = []
            captured_results = []

            def fake_run_request(
                request: AgentRequest,
                stdout: Optional[object] = None,
                stderr: Optional[object] = None,
            ) -> AgentResult:
                del stdout, stderr
                seen_inputs.append(request.input_path)
                return AgentResult(return_code=0, stdout="ok", stderr="")

            def fake_render(results, stdout=None):
                del stdout
                captured_results.extend(results)
                return 0

            with patch("triton_agent.optimize.batch.render_batch_optimize_results", side_effect=fake_render):
                exit_code = run_optimize_batch(
                    root,
                    options,
                    max_concurrency=1,
                    stdout=StringIO(),
                    run_request=fake_run_request,
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual([path.parent.name for path in seen_inputs], ["beta"])
            self.assertEqual([item.workspace.name for item in captured_results], ["alpha", "beta"])
            self.assertEqual(captured_results[0].message, "already completed")

    def test_run_optimize_batch_writes_completed_status_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "alpha"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('alpha')\n", encoding="utf-8")

            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                show_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=None,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                supervise="off",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
            )

            with patch(
                "triton_agent.optimize.batch.render_batch_optimize_results",
                return_value=0,
            ):
                exit_code = run_optimize_batch(
                    root,
                    options,
                    max_concurrency=1,
                    stdout=StringIO(),
                    run_request=lambda request, stdout=None, stderr=None: AgentResult(
                        return_code=0,
                        stdout="ok",
                        stderr="",
                    ),
                )

            self.assertEqual(exit_code, 0)
            status = json.loads((root / "optimize-batch-status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["version"], 1)
            self.assertEqual(status["workspaces"]["alpha"]["status"], "completed")
            self.assertEqual(status["workspaces"]["alpha"]["operator_file"], "kernel.py")

    def test_run_optimize_batch_writes_incomplete_status_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "alpha"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('alpha')\n", encoding="utf-8")

            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                show_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=None,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                supervise="off",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
            )

            with patch(
                "triton_agent.optimize.batch.render_batch_optimize_results",
                return_value=1,
            ):
                exit_code = run_optimize_batch(
                    root,
                    options,
                    max_concurrency=1,
                    stdout=StringIO(),
                    run_request=lambda request, stdout=None, stderr=None: AgentResult(
                        return_code=7,
                        stdout="",
                        stderr="failed optimize",
                    ),
                )

            self.assertEqual(exit_code, 1)
            status = json.loads((root / "optimize-batch-status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["workspaces"]["alpha"]["status"], "incomplete")
            self.assertEqual(status["workspaces"]["alpha"]["operator_file"], "kernel.py")

    def test_run_optimize_batch_ignores_malformed_status_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "alpha"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('alpha')\n", encoding="utf-8")
            (root / "optimize-batch-status.json").write_text("{not json\n", encoding="utf-8")

            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                show_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=None,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                supervise="off",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
            )

            seen_inputs: list[Path] = []

            def fake_run_request(
                request: AgentRequest,
                stdout: Optional[object] = None,
                stderr: Optional[object] = None,
            ) -> AgentResult:
                del stdout, stderr
                seen_inputs.append(request.input_path)
                return AgentResult(return_code=0, stdout="ok", stderr="")

            with patch(
                "triton_agent.optimize.batch.render_batch_optimize_results",
                return_value=0,
            ):
                exit_code = run_optimize_batch(
                    root,
                    options,
                    max_concurrency=1,
                    stdout=StringIO(),
                    run_request=fake_run_request,
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual([path.parent.name for path in seen_inputs], ["alpha"])

    def test_run_optimize_batch_reset_optimize_clears_root_status_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "alpha"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('alpha')\n", encoding="utf-8")
            (root / "optimize-batch-status.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "workspaces": {
                            "alpha": {
                                "status": "completed",
                                "operator_file": "kernel.py",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                show_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=None,
                resume_mode="fresh",
                reset_optimize=True,
                no_agent_session=False,
                supervise="off",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
            )

            with patch(
                "triton_agent.optimize.batch.render_batch_optimize_results",
                return_value=0,
            ):
                exit_code = run_optimize_batch(
                    root,
                    options,
                    max_concurrency=1,
                    stdout=StringIO(),
                    run_request=lambda request, stdout=None, stderr=None: AgentResult(
                        return_code=0,
                        stdout="ok",
                        stderr="",
                    ),
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue((root / "optimize-batch-status.json").exists())
            status = json.loads((root / "optimize-batch-status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["workspaces"]["alpha"]["status"], "completed")

    def test_reset_optimize_workspace_unlinks_symlinked_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            outside = workdir / "outside"
            outside.mkdir()
            for name in ("baseline-real", "runtime-real", "logs-real", "round-real"):
                target = outside / name
                target.mkdir()
                (target / "marker.txt").write_text(name, encoding="utf-8")

            (workdir / "opt-note.md").write_text("note\n", encoding="utf-8")
            (workdir / "learned_lessons.md").write_text("lesson\n", encoding="utf-8")
            try:
                (workdir / "baseline").symlink_to(outside / "baseline-real", target_is_directory=True)
                (workdir / ".triton-agent").symlink_to(outside / "runtime-real", target_is_directory=True)
                (workdir / "triton-agent-logs").symlink_to(outside / "logs-real", target_is_directory=True)
                (workdir / "opt-round-1").symlink_to(outside / "round-real", target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlinks are unavailable: {exc}")
            optimized = workdir / "opt_kernel.py"
            optimized.write_text("print('opt')\n", encoding="utf-8")

            reset_optimize_workspace(operator, workdir)

            self.assertFalse((workdir / "opt-note.md").exists())
            self.assertFalse((workdir / "learned_lessons.md").exists())
            self.assertFalse((workdir / "baseline").exists())
            self.assertFalse((workdir / ".triton-agent").exists())
            self.assertFalse((workdir / "triton-agent-logs").exists())
            self.assertFalse((workdir / "opt-round-1").exists())
            self.assertFalse(optimized.exists())

            self.assertTrue((outside / "baseline-real").exists())
            self.assertTrue((outside / "runtime-real").exists())
            self.assertTrue((outside / "logs-real").exists())
            self.assertTrue((outside / "round-real").exists())

    def test_cleanup_workspace_pt_files_preserves_pt_files_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            round_dir = workdir / "opt-round-1"
            round_dir.mkdir()
            root_pt = workdir / "kernel_result.pt"
            round_pt = round_dir / "test_result.pt"
            root_pt.write_text("root\n", encoding="utf-8")
            round_pt.write_text("round\n", encoding="utf-8")

            cleaned = cleanup_workspace_pt_files(workdir)

            self.assertEqual(cleaned, [])
            self.assertTrue(root_pt.exists())
            self.assertTrue(round_pt.exists())

    def test_cleanup_workspace_pt_files_deletes_pt_files_when_env_var_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            round_dir = workdir / "opt-round-1"
            round_dir.mkdir()
            root_pt = workdir / "kernel_result.pt"
            round_pt = round_dir / "test_result.pt"
            root_pt.write_text("root\n", encoding="utf-8")
            round_pt.write_text("round\n", encoding="utf-8")

            with patch.dict(os.environ, {"TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES": "1"}, clear=False):
                cleaned = cleanup_workspace_pt_files(workdir)

            self.assertEqual(cleaned, ["kernel_result.pt", "opt-round-1/test_result.pt"])
            self.assertFalse(root_pt.exists())
            self.assertFalse(round_pt.exists())

    def test_reset_optimize_workspace_deletes_result_pt_files_regardless_of_env_var(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            result_pt = workdir / "kernel_result.pt"
            result_pt.write_text("stub\n", encoding="utf-8")

            with patch.dict(os.environ, {"TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES": "0"}, clear=False):
                reset_optimize_workspace(operator, workdir)

            self.assertFalse(result_pt.exists())

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
                skill_name="triton-npu-optimize",
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
                        self_outer._write_baseline(workdir)
                        self_outer._write_round(
                            workdir,
                            "opt-round-1",
                            parent_round="round-0",
                            round_disposition="stop",
                        )
                    else:
                        assert request.round_brief_path is not None
                        assert request.supervisor_report_path is not None
                        request.round_brief_path.write_text(
                            "# Optimize Round Brief\n\nStop after verifying opt-round-1.\n",
                            encoding="utf-8",
                        )
                        request.supervisor_report_path.write_text(
                            "# Optimize Supervisor Report\n\n"
                            "Decision: pass-stop\n"
                            "Blocking issues: none\n"
                            "Latest round: opt-round-1\n",
                            encoding="utf-8",
                        )
                    return AgentResult(return_code=0, stdout="ok", stderr="")

            self_outer = self
            runner = FakeRunner()

            with patch("triton_agent.optimize.orchestration.create_runner", return_value=runner):
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

    def test_run_optimize_request_supervisor_prompt_excludes_user_instructions(self) -> None:
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
                skill_name="triton-npu-optimize",
                prompt=(
                    "Optimize this operator\n\n"
                    "Additional user instructions:\n"
                    "Focus on occupancy."
                ),
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
                        self_outer._write_baseline(workdir)
                        self_outer._write_round(
                            workdir,
                            "opt-round-1",
                            parent_round="round-0",
                            round_disposition="stop",
                        )
                    else:
                        assert request.round_brief_path is not None
                        assert request.supervisor_report_path is not None
                        request.round_brief_path.write_text(
                            "# Optimize Round Brief\n\nStop after verifying opt-round-1.\n",
                            encoding="utf-8",
                        )
                        request.supervisor_report_path.write_text(
                            "# Optimize Supervisor Report\n\n"
                            "Decision: pass-stop\n"
                            "Blocking issues: none\n"
                            "Latest round: opt-round-1\n",
                            encoding="utf-8",
                        )
                    return AgentResult(return_code=0, stdout="ok", stderr="")

            self_outer = self
            runner = FakeRunner()

            with patch("triton_agent.optimize.orchestration.create_runner", return_value=runner):
                result = run_optimize_request(request)

            self.assertEqual(result.return_code, 0)
            self.assertEqual(len(runner.requests), 2)
            supervisor_request = runner.requests[1]
            self.assertEqual(supervisor_request.optimize_role, "supervisor")
            self.assertNotIn("Additional user instructions:", supervisor_request.prompt)
            self.assertNotIn("Focus on occupancy.", supervisor_request.prompt)

    def test_run_optimize_request_end_to_end_uses_supervisor_report_for_continue_prompt(
        self,
    ) -> None:
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
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                supervise="on",
                optimize_role="worker",
            )

            class FakeRunner:
                def __init__(self) -> None:
                    self.requests: List[AgentRequest] = []
                    self.worker_calls = 0
                    self.supervisor_calls = 0

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
                            self_outer._write_baseline(workdir)
                            self_outer._write_round(
                                workdir,
                                "opt-round-1",
                                parent_round="round-0",
                                round_disposition="continue",
                            )
                            return AgentResult(return_code=0, stdout="worker ok", stderr="")
                        return AgentResult(return_code=1, stdout="", stderr="worker stopped for test")
                    self.supervisor_calls += 1
                    self_outer._write_supervisor_handoff(
                        guidance_state,
                        decision="revise-metadata",
                        latest_round="opt-round-1",
                        issues=("round summary is missing the compare-perf conclusion",),
                        brief_lines=("Repair opt-round-1 metadata before starting a new round.",),
                    )
                    return AgentResult(return_code=0, stdout="supervisor ok", stderr="")

            self_outer = self
            runner = FakeRunner()
            guidance_state = self._build_guidance_state(workdir)

            with patch("triton_agent.optimize.orchestration.create_runner", return_value=runner):
                with patch(
                    "triton_agent.optimize.orchestration.OptimizeSessionArtifactsManager.prepare_supervised_session",
                    return_value=guidance_state,
                ):
                    result = run_optimize_request(request)

            self.assertEqual(result.return_code, 1)
            self.assertEqual(len(runner.requests), 3)
            self.assertEqual(runner.requests[0].optimize_role, "worker")
            self.assertEqual(runner.requests[1].optimize_role, "supervisor")
            self.assertEqual(runner.requests[2].optimize_role, "worker")
            self.assertIn("Gate decision: revise-metadata", runner.requests[2].prompt)
            self.assertIn("round summary is missing the compare-perf conclusion", runner.requests[2].prompt)

    def test_run_supervisor_converts_invalid_supervisor_report_to_gate_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)
            self._write_round(
                workdir,
                "opt-round-1",
                parent_round="round-0",
                round_disposition="continue",
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
                skill_name="triton-npu-optimize",
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
                    self_outer._write_supervisor_handoff(
                        guidance_state,
                        decision="invalid-decision",
                        latest_round="opt-round-1",
                    )
                    return AgentResult(return_code=0, stdout="ok", stderr="")

            loop_runner = SupervisedOptimizeAdapter(cast(Any, FakeRunner()), guidance_state)
            self_outer = self

            gate_result = loop_runner.run_supervisor(
                request,
                AgentResult(return_code=0, stdout="worker ok", stderr=""),
            )

            self.assertEqual(gate_result.decision, GateDecision.REVISE_METADATA)
            self.assertIn("invalid supervisor decision", gate_result.blocking_issues[0])
            self.assertIn(
                "Decision: revise-metadata",
                guidance_state.supervisor_report_path.read_text(encoding="utf-8"),
            )

    def test_run_supervisor_converts_missing_decision_line_to_gate_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)
            self._write_round(
                workdir,
                "opt-round-1",
                parent_round="round-0",
                round_disposition="continue",
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
                skill_name="triton-npu-optimize",
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
                    guidance_state.supervisor_report_path.write_text(
                        "# Optimize Supervisor Report\n\nBlocking issues: missing decision line\n",
                        encoding="utf-8",
                    )
                    guidance_state.round_brief_path.write_text(
                        "# Optimize Round Brief\n\nRepair metadata before the next round.\n",
                        encoding="utf-8",
                    )
                    return AgentResult(return_code=0, stdout="ok", stderr="")

            loop_runner = SupervisedOptimizeAdapter(cast(Any, FakeRunner()), guidance_state)

            gate_result = loop_runner.run_supervisor(
                request,
                AgentResult(return_code=0, stdout="worker ok", stderr=""),
            )

            self.assertEqual(gate_result.decision, GateDecision.REVISE_METADATA)
            self.assertIn("missing supervisor decision line", gate_result.blocking_issues[0])

    def test_run_supervisor_updates_live_and_history_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)
            self._write_round(
                workdir,
                "opt-round-1",
                parent_round="round-0",
                round_disposition="stop",
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
                skill_name="triton-npu-optimize",
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
                    self_outer._write_supervisor_handoff(
                        guidance_state,
                        decision="pass-stop",
                        latest_round="opt-round-1",
                        brief_lines=("Stop after verifying opt-round-1.",),
                    )
                    return AgentResult(return_code=0, stdout="ok", stderr="")

            loop_runner = SupervisedOptimizeAdapter(cast(Any, FakeRunner()), guidance_state)
            self_outer = self

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
            self.assertIn(
                "Stop after verifying opt-round-1.",
                guidance_state.round_brief_path.read_text(encoding="utf-8"),
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
