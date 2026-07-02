import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, cast
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.models import AgentRequest, AgentResult, CommandKind
from triton_agent.optimize.batch import run_optimize_batch
import triton_agent.optimize.execution as execution_module
from triton_agent.optimize.compiler_source import CompilerSourceInfo
from triton_agent.optimize.models import OptimizeRunOptions
from triton_agent.optimize.execution import (
    _count_round_directories,
    _latest_round_dir,
)
from triton_agent.optimize.orchestration import build_optimize_request, run_optimize_request
from triton_agent.optimize.prompts import build_optimize_round_prompt
from triton_agent.optimize.pt_cleanup import cleanup_workspace_pt_files
from triton_agent.optimize.archive import ArchiveState
from triton_agent.optimize.memory_file import MemoryFileState
from triton_agent.optimize.resume import reset_optimize_workspace
from triton_agent.optimize.session_artifacts import OptimizeSessionArtifactsState
from triton_agent.remote.env import remote_target_env_name, remote_workdir_env_name


def _optimize_invocation_kind(request: AgentRequest) -> str:
    if "Do not open a new optimization round yet." in request.prompt:
        return "baseline"
    if "This invocation is an audit and handoff pass" in request.prompt:
        return "supervisor"
    return "worker"


class OptimizeRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._report_patcher = patch(
            "triton_agent.report.workspace.generate_workspace_report",
            return_value=(True, "ok"),
        )
        self._report_patcher.start()

    def tearDown(self) -> None:
        self._report_patcher.stop()
        super().tearDown()

    def test_optimize_recovery_classifies_stall_before_transient_text(self) -> None:
        recovery = importlib.import_module("triton_agent.optimize.recovery")

        self.assertEqual(
            recovery.classify_worker_failure(
                AgentResult(
                    return_code=1,
                    stdout="ERROR: exceeded retry limit, last status: 429 Too Many Requests",
                    stderr="",
                    stalled=True,
                )
            ),
            "stall",
        )

    def test_optimize_recovery_classifies_transient_and_fatal_failures(self) -> None:
        recovery = importlib.import_module("triton_agent.optimize.recovery")

        self.assertEqual(
            recovery.classify_worker_failure(
                AgentResult(
                    return_code=1,
                    stdout="",
                    stderr="ERROR: exceeded retry limit, last status: 429 Too Many Requests",
                )
            ),
            "transient",
        )
        self.assertEqual(
            recovery.classify_worker_failure(
                AgentResult(
                    return_code=1,
                    stdout="",
                    stderr="plain error",
                )
            ),
            "fatal",
        )

    def test_optimize_recovery_progress_path_allowlist_includes_business_artifacts(self) -> None:
        recovery = importlib.import_module("triton_agent.optimize.recovery")

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            round_dir = workspace / "opt-round-3"
            round_dir.mkdir()
            round_state = round_dir / "round-state.json"
            summary = round_dir / "summary.md"
            attempts = round_dir / "attempts.md"
            round_state.write_text("{}", encoding="utf-8")
            summary.write_text("summary\n", encoding="utf-8")
            attempts.write_text("attempts\n", encoding="utf-8")

            self.assertTrue(recovery.is_optimize_progress_path(round_state, workspace))
            self.assertTrue(recovery.is_optimize_progress_path(summary, workspace))
            self.assertTrue(recovery.is_optimize_progress_path(attempts, workspace))

    def test_optimize_recovery_progress_path_allowlist_excludes_logs_and_hidden_runtime_state(self) -> None:
        recovery = importlib.import_module("triton_agent.optimize.recovery")

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            log_file = workspace / "triton-agent-logs" / "optimize.show-output.log"
            hidden_file = workspace / "supervisor-report.md"
            log_file.parent.mkdir(parents=True)
            log_file.write_text("log\n", encoding="utf-8")
            hidden_file.write_text("report\n", encoding="utf-8")

            self.assertFalse(recovery.is_optimize_progress_path(log_file, workspace))
            self.assertFalse(recovery.is_optimize_progress_path(hidden_file, workspace))

    def test_optimize_recovery_scan_ignores_directory_mtime_only_changes(self) -> None:
        recovery = importlib.import_module("triton_agent.optimize.recovery")

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            round_dir = workspace / "opt-round-3"
            round_dir.mkdir()

            before = recovery.scan_optimize_progress(workspace)
            os.utime(round_dir, None)
            after = recovery.scan_optimize_progress(workspace)

            self.assertEqual(after, before)

    def test_optimize_recovery_scan_tolerates_transient_non_progress_directory_disappearing(self) -> None:
        recovery = importlib.import_module("triton_agent.optimize.recovery")

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "opt-note.md").write_text("note\n", encoding="utf-8")
            round_dir = workspace / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
            disappearing_path = workspace / "kernel_meta"

            real_rglob = Path.rglob

            def fake_rglob(self: Path, pattern: str):  # type: ignore[no-untyped-def]
                if self == workspace and pattern == "*":
                    def generator():
                        yield workspace / "opt-note.md"
                        yield round_dir / "summary.md"
                        raise FileNotFoundError(str(disappearing_path))

                    return generator()
                return real_rglob(self, pattern)

            with patch("pathlib.Path.rglob", new=fake_rglob):
                snapshot = recovery.scan_optimize_progress(workspace)

            self.assertEqual(
                tuple(path for path, _size, _mtime in snapshot.file_fingerprints),
                ("opt-note.md", "opt-round-1/summary.md"),
            )

    def test_optimize_recovery_compute_range_progress_is_range_local(self) -> None:
        recovery = importlib.import_module("triton_agent.optimize.recovery")

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            for round_name in ("opt-round-1", "opt-round-2", "opt-round-11", "opt-round-12", "opt-round-13", "opt-round-14"):
                (workdir / round_name).mkdir()

            def fake_check_round(
                round_dir: Path,
                *,
                current_round: Optional[int] = None,
                final_round: Optional[int] = None,
                optimize_target: Optional[str] = None,
            ) -> SimpleNamespace:
                del final_round, optimize_target
                status = "pass" if current_round in (11, 12, 13) else "fail"
                return SimpleNamespace(kind="round", status=status, issues=(), summary=status)

            with patch("triton_agent.optimize.recovery.check_round", side_effect=fake_check_round):
                progress = recovery.compute_range_progress(
                    workdir,
                    batch_start=11,
                    batch_end=15,
                    optimize_target="kernel",
                )

            self.assertEqual(progress.last_accepted_round, 13)
            self.assertEqual(progress.first_unresolved_round, 14)
            self.assertEqual(progress.next_batch_start, 14)
            self.assertEqual(progress.next_batch_end, 15)

    def test_optimize_recovery_budget_tracks_attempts_per_round(self) -> None:
        recovery = importlib.import_module("triton_agent.optimize.recovery")

        budget = recovery.RecoveryBudget(max_attempts=3)
        budget.consume(14)
        budget.consume(14)
        self.assertEqual(budget.remaining(14), 1)
        self.assertFalse(budget.exhausted(14))

        budget.consume(14)
        self.assertTrue(budget.exhausted(14))

        budget.consume(15)
        self.assertEqual(budget.remaining(15), 2)

    def test_optimize_orchestration_module_replaces_runtime_module(self) -> None:
        self.assertIsNotNone(importlib.util.find_spec("triton_agent.optimize.orchestration"))
        self.assertIsNone(importlib.util.find_spec("triton_agent.optimize.runtime"))

    def test_optimize_run_loop_and_supervisor_modules_have_been_removed(self) -> None:
        self.assertIsNone(importlib.util.find_spec("triton_agent.optimize.run_loop"))
        self.assertIsNone(importlib.util.find_spec("triton_agent.optimize.supervisor"))

    def test_optimize_gate_module_has_been_removed(self) -> None:
        self.assertIsNone(importlib.util.find_spec("triton_agent.optimize.gate"))

    def test_optimize_execution_module_no_longer_exports_legacy_supervised_entrypoints(
        self,
    ) -> None:
        self.assertFalse(hasattr(execution_module, "SupervisedOptimizeAdapter"))
        self.assertFalse(hasattr(execution_module, "execute_supervised_optimize"))

    def test_build_optimize_round_prompt_includes_phase_summary_when_present(self) -> None:
        prompt = build_optimize_round_prompt(
            Path("kernel.py"),
            None,
            test_mode="differential",
            bench_mode="torch-npu-profiler",
            round_mode="checked",
            current_round=1,
            final_round=1,
            workflow_phase_summary="Current phase: round_active\nCurrent round: 1",
        )

        self.assertIn("Workflow phase summary:", prompt)
        self.assertIn("Current phase: round_active", prompt)

    def test_build_optimize_request_preserves_enable_agent_hooks_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
                enable_agent_hooks=True,
            )

            request = build_optimize_request(operator, workdir, options)

        self.assertTrue(request.enable_agent_hooks)

    def _build_guidance_state(self, workdir: Path) -> OptimizeSessionArtifactsState:
        hidden_triton_agent_dir = workdir / ".triton-agent"
        hidden_triton_agent_dir.mkdir(parents=True, exist_ok=True)
        guidance_path = workdir / "AGENTS.md"
        guidance_path.write_text("shared guidance\n", encoding="utf-8")
        handoff_dir = hidden_triton_agent_dir / "supervisor-handoffs"
        handoff_dir.mkdir(parents=True, exist_ok=True)
        supervisor_report_path = workdir / "supervisor-report.md"
        supervisor_report_path.write_text("report\n", encoding="utf-8")
        run_archive_dir = workdir / "triton-agent-logs" / "run-001"
        shared_guidance_snapshot_path = run_archive_dir / "shared-guidance.md"
        return OptimizeSessionArtifactsState(
            memory_file=MemoryFileState(
                guidance_path=guidance_path,
                backup_path=None,
                created_guidance=False,
            ),
            archive=ArchiveState(
                run_archive_dir=run_archive_dir,
                shared_guidance_snapshot_path=shared_guidance_snapshot_path,
            ),
            hidden_triton_agent_dir=hidden_triton_agent_dir,
            supervisor_report_path=supervisor_report_path,
            supervisor_handoff_dir=handoff_dir,
        )

    def _build_checked_guidance_state(self, workdir: Path) -> OptimizeSessionArtifactsState:
        guidance_path = workdir / "AGENTS.md"
        guidance_path.write_text("shared guidance\n", encoding="utf-8")
        run_archive_dir = workdir / "triton-agent-logs" / "run-checked"
        shared_guidance_snapshot_path = run_archive_dir / "shared-guidance.md"
        return OptimizeSessionArtifactsState(
            memory_file=MemoryFileState(
                guidance_path=guidance_path,
                backup_path=None,
                created_guidance=False,
            ),
            archive=ArchiveState(
                run_archive_dir=run_archive_dir,
                shared_guidance_snapshot_path=shared_guidance_snapshot_path,
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
                    "bench_mode": "torch-npu-profiler",
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
        perf_text: str = "case0: 1.0\n",
        operator_source: Optional[str] = None,
    ) -> Path:
        round_dir = workdir / round_name
        round_dir.mkdir(exist_ok=True)
        (workdir / "opt-note.md").write_text("## Round\n", encoding="utf-8")
        round_operator_source = operator_source or (
            "import triton\n"
            "import triton.language as tl\n\n"
            "@triton.jit\n"
            "def _kernel(X, Y):\n"
            "    return\n\n"
            "def launch(x, y):\n"
            "    _kernel[1](x, y)\n"
        )
        (round_dir / "opt_kernel.py").write_text(round_operator_source, encoding="utf-8")
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
                    "comparison_target": "baseline/perf.txt",
                    "effective_metric_source": "kernel",
                    "summary_path": "summary.md",
                    "opt_note_updated": True,
                }
            ),
            encoding="utf-8",
        )
        return round_dir

    def _write_supervisor_handoff(
        self,
        guidance_state: OptimizeSessionArtifactsState,
        *,
        status: str,
        issues: tuple[str, ...] = (),
        latest_round: Optional[str] = None,
    ) -> None:
        report_lines = [
            "# Optimize Supervisor Report",
            "",
            f"Status: {status}",
            f"Blocking issues: {', '.join(issues) if issues else 'none'}",
        ]
        if latest_round is not None:
            report_lines.append(f"Latest round: {latest_round}")
        assert guidance_state.supervisor_report_path is not None
        guidance_state.supervisor_report_path.write_text(
            "\n".join(report_lines) + "\n",
            encoding="utf-8",
        )

    def test_run_optimize_request_delegates_multi_invocation_flow_to_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            (workdir / "opt-round-1").mkdir()

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                min_rounds=1,
                round_mode="checked",
            )

            expected = AgentResult(return_code=0, stdout="ok", stderr="")
            fake_runner = object()

            with patch("triton_agent.optimize.orchestration.create_runner", return_value=fake_runner):
                with patch("triton_agent.optimize.orchestration.SkillLinkManager.prepare_skills", return_value=()):
                    with patch("triton_agent.optimize.orchestration.SkillLinkManager.cleanup", return_value=[]):
                        with patch.object(
                            execution_module,
                            "execute_multi_invocation_optimize",
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
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
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
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
            )

            request = build_optimize_request(operator, workdir, options)

            self.assertFalse(request.enable_agent_hooks)
            self.assertFalse(request.log_tools)

    def test_build_optimize_request_omits_run_eval_mcp_server_name_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
            )

            request = build_optimize_request(operator, workdir, options)

        self.assertIsNone(request.mcp_servers)

    def test_build_optimize_request_attaches_run_eval_mcp_server_name_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
                enable_mcp=True,
            )

            request = build_optimize_request(operator, workdir, options)

        self.assertEqual(
            request.staged_skill_sources,
            {"ascend-npu-run-eval": "ascend-npu-run-eval-mcp"},
        )
        self.assertEqual(request.mcp_servers, ("triton-agent-run-eval",))

    def test_build_optimize_request_omits_mcp_servers_without_run_eval_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
            )

            with patch(
                "triton_agent.optimize.orchestration.resolve_staged_skills",
                return_value=(("triton-npu-optimize",), None),
            ):
                request = build_optimize_request(operator, workdir, options)

        self.assertIsNone(request.mcp_servers)

    def test_build_optimize_request_enables_agent_hooks_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
                enable_agent_hooks=True,
            )

            request = build_optimize_request(operator, workdir, options)

            self.assertTrue(request.enable_agent_hooks)

    def test_build_optimize_request_carries_enable_subagent_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
                enable_subagent=True,
            )

            request = build_optimize_request(operator, workdir, options)

            self.assertTrue(request.enable_subagent)

    def test_build_optimize_request_enables_log_tools_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
                log_tools=True,
            )

            request = build_optimize_request(operator, workdir, options)

            self.assertTrue(request.log_tools)

    def test_build_optimize_request_injects_remote_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                stream_output=False,
                remote="alice@example.com:2200",
                remote_workdir="/tmp/triton-agent",
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
            )

            request = build_optimize_request(operator, workdir, options)

            self.assertEqual(request.remote, "alice@example.com:2200")
            self.assertEqual(request.remote_workdir, "/tmp/triton-agent")
            self.assertIsNotNone(request.extra_env)
            assert request.extra_env is not None
            self.assertEqual(request.extra_env[remote_target_env_name()], "alice@example.com:2200")
            self.assertEqual(request.extra_env[remote_workdir_env_name()], "/tmp/triton-agent")

    def test_build_optimize_request_uses_explicit_optimize_skill_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
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
                    "ascend-npu-prepare-optimize-baseline",
                    "ascend-npu-gen-test",
                    "ascend-npu-gen-bench",
                    "ascend-npu-run-eval",
                    "ascend-npu-optimize-state",
                    "ascend-npu-profile-operator",
                    "ascend-npu-analyze-round-performance",
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
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
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
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
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
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
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
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
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
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
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
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
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
            self.assertEqual(request.prompt, "")

    def test_build_optimize_request_maps_v2_knowledge_and_cann_ext_api_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
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
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
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
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
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
            self.assertEqual(request.prompt, "")

    def test_build_optimize_request_provisions_compiler_source_and_cann_ext_api_together(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            source_path = (workdir / "AscendNPU-IR").resolve()
            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
                output=None,
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                prompt="Prefer occupancy-safe changes.",
                compiler_source_analysis="auto",
                enable_cann_ext_api=True,
            )

            with patch(
                "triton_agent.optimize.orchestration.prepare_compiler_source",
                return_value=CompilerSourceInfo(
                    path=source_path,
                    commit="abc123",
                ),
            ):
                request = build_optimize_request(operator, workdir, options)

            self.assertEqual(request.round_mode, "checked")
            self.assertEqual(request.prompt, "")
            self.assertEqual(request.user_prompt, "Prefer occupancy-safe changes.")

    def test_build_optimize_request_sets_initial_batch_bounds_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=5,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
                round_batch_size=2,
                output=None,
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                prompt="Prefer occupancy-safe changes.",
            )

            request = build_optimize_request(operator, workdir, options)

            self.assertEqual(request.round_mode, "checked")
            self.assertEqual(request.round_batch_size, 2)
            self.assertEqual(request.current_round, 1)
            self.assertEqual(request.final_round, 2)
            self.assertEqual(request.user_prompt, "Prefer occupancy-safe changes.")
            self.assertEqual(request.prompt, "")
            self.assertFalse(request.disable_backend_retry)
            self.assertIsNone(request.progress_probe)

    def test_build_optimize_request_interactive_uses_single_long_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            options = OptimizeRunOptions(
                agent_name="codex",
                interact=True,
                verbose=False,
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=30,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
                round_batch_size=99,
                output=None,
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                prompt="Stay attached.",
            )

            request = build_optimize_request(operator, workdir, options)

            self.assertTrue(request.interact)
            self.assertEqual(request.round_batch_size, 99)
            self.assertEqual(request.current_round, 1)
            self.assertEqual(request.final_round, 30)

    def test_run_optimize_request_invokes_worker_then_supervisor_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            # Pre-create a valid baseline so the preflight returns READY
            self._write_baseline(workdir)

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                min_rounds=1,
                round_mode="supervised",
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
                    if _optimize_invocation_kind(request) == "worker":
                        self_outer._write_round(
                            workdir,
                            "opt-round-1",
                            parent_round="round-0",
                        )
                    else:
                        self.supervisor_calls += 1
                        self_outer._write_supervisor_handoff(
                            guidance_state,
                            status="pass",
                            latest_round="opt-round-1",
                        )
                    session_id = (
                        "019da9c2-dfcb-7c71-a2f9-7a90bab2e0f5"
                        if _optimize_invocation_kind(request) == "worker"
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
            self.assertEqual(_optimize_invocation_kind(worker_request), "worker")
            self.assertFalse(worker_request.no_agent_session)
            self.assertIsNotNone(worker_request.supervisor_report_path)
            self.assertTrue(worker_request.disable_backend_retry)
            self.assertIsNotNone(worker_request.progress_probe)
            self.assertEqual(_optimize_invocation_kind(supervisor_request), "supervisor")
            self.assertEqual(supervisor_request.skill_name, "triton-npu-optimize")
            self.assertFalse(supervisor_request.interact)
            self.assertFalse(supervisor_request.no_agent_session)
            self.assertFalse(supervisor_request.disable_backend_retry)
            self.assertIsNone(supervisor_request.progress_probe)
            self.assertEqual(
                supervisor_request.supervisor_report_path,
                worker_request.supervisor_report_path,
            )
            self.assertEqual(worker_request.supervisor_report_path, workdir / "supervisor-report.md")
            self.assertFalse((workdir / "supervisor-report.md").exists())
            self.assertFalse((workdir / ".triton-agent").exists())
            archive_root = workdir / "triton-agent-logs"
            self.assertTrue(archive_root.exists())
            run_archives = [path for path in archive_root.iterdir() if path.is_dir()]
            self.assertEqual(len(run_archives), 1)
            run_archive = run_archives[0]
            self.assertTrue((run_archive / "shared-guidance.md").exists())
            self.assertTrue((run_archive / "supervisor-report.md").exists())
            self.assertTrue((run_archive / "supervisor-handoffs").exists())
            self.assertEqual(
                [
                    (
                        json.loads((run_archive / "agent-session-batch-1-1-r1.json").read_text(encoding="utf-8"))["session_id"],
                        json.loads((run_archive / "agent-session-batch-1-1-r1.json").read_text(encoding="utf-8"))["agent"],
                    ),
                    (
                        json.loads((run_archive / "agent-session-supervisor.json").read_text(encoding="utf-8"))["session_id"],
                        json.loads((run_archive / "agent-session-supervisor.json").read_text(encoding="utf-8"))["agent"],
                    ),
                ],
                [
                    ("019da9c2-dfcb-7c71-a2f9-7a90bab2e0f5", "codex"),
                    ("119da9c2-dfcb-7c71-a2f9-7a90bab2e0f5", "codex"),
                ],
            )
            self.assertFalse((run_archive / "agent-sessions.jsonl").exists())

    def test_run_optimize_request_preserves_explicit_no_agent_session_for_worker_and_supervisor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="claude",
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                min_rounds=1,
                round_mode="supervised",
                no_agent_session=True,
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
                    if _optimize_invocation_kind(request) == "worker":
                        self_outer._write_round(
                            workdir,
                            "opt-round-1",
                            parent_round="round-0",
                        )
                    else:
                        self_outer._write_supervisor_handoff(
                            guidance_state,
                            status="pass",
                            latest_round="opt-round-1",
                        )
                    return AgentResult(return_code=0, stdout="ok", stderr="")

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
            self.assertTrue(worker_request.no_agent_session)
            self.assertTrue(supervisor_request.no_agent_session)

    def test_run_optimize_request_supervised_runs_one_supervisor_per_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                min_rounds=4,
                round_mode="supervised",
                round_batch_size=2,
                current_round=1,
                final_round=2,
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
                    if _optimize_invocation_kind(request) == "worker":
                        self.worker_calls += 1
                        if self.worker_calls == 1:
                            self_outer._write_round(
                                workdir,
                                "opt-round-1",
                                parent_round="round-0",
                                )
                            self_outer._write_round(
                                workdir,
                                "opt-round-2",
                                parent_round="round-1",
                                )
                        else:
                            self_outer._write_round(
                                workdir,
                                "opt-round-3",
                                parent_round="round-2",
                                )
                            self_outer._write_round(
                                workdir,
                                "opt-round-4",
                                parent_round="round-3",
                                )
                        return AgentResult(return_code=0, stdout="worker ok", stderr="")
                    latest_round_dir = _latest_round_dir(workdir)
                    self_outer._write_supervisor_handoff(
                        guidance_state,
                        status="pass",
                        latest_round=latest_round_dir.name if latest_round_dir is not None else None,
                    )
                    return AgentResult(return_code=0, stdout="supervisor ok", stderr="")

            def fake_check_round(
                round_dir: Path,
                *,
                current_round: Optional[int] = None,
                final_round: Optional[int] = None,
                optimize_target: Optional[str] = None,
            ) -> SimpleNamespace:
                del round_dir, current_round, final_round, optimize_target
                return SimpleNamespace(
                    kind="round",
                    status="pass",
                    issues=(),
                    summary="round check passed",
                )

            self_outer = self
            runner = FakeRunner()
            guidance_state = self._build_guidance_state(workdir)

            with patch("triton_agent.optimize.orchestration.create_runner", return_value=runner):
                with patch(
                    "triton_agent.optimize.orchestration.OptimizeSessionArtifactsManager.prepare_supervised_session",
                    return_value=guidance_state,
                ):
                    with patch("triton_agent.optimize.execution.check_round", side_effect=fake_check_round):
                        result = run_optimize_request(request)

            self.assertEqual(result.return_code, 0)
            self.assertEqual(
                [_optimize_invocation_kind(record) for record in runner.requests],
                ["worker", "supervisor", "worker", "supervisor"],
            )

    def test_run_optimize_request_supervised_loop_keeps_running_until_min_rounds_are_satisfied(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                min_rounds=2,
                round_mode="supervised",
            )

            class FakeRunner:
                def __init__(self) -> None:
                    self.requests: list[AgentRequest] = []
                    self.worker_calls = 0

                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del stdout, stderr
                    self.requests.append(request)
                    if _optimize_invocation_kind(request) == "worker":
                        self.worker_calls += 1
                        if self.worker_calls == 1:
                            self_outer._write_round(
                                workdir,
                                "opt-round-1",
                                parent_round="round-0",
                                )
                            return AgentResult(return_code=0, stdout="worker ok", stderr="")
                        return AgentResult(return_code=1, stdout="", stderr="stop after second prompt for test")
                    self_outer._write_supervisor_handoff(
                        guidance_state,
                        status="pass",
                        latest_round="opt-round-1",
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
            self.assertEqual(_optimize_invocation_kind(runner.requests[0]), "worker")
            self.assertEqual(_optimize_invocation_kind(runner.requests[1]), "supervisor")
            self.assertEqual(_optimize_invocation_kind(runner.requests[2]), "worker")
            self.assertNotIn("CLI batch follow-up from the previous worker batch:", runner.requests[2].prompt)
            self.assertNotIn("\"status\": \"pass\"", runner.requests[2].prompt)
            self.assertNotIn("Supervisor guidance:", runner.requests[2].prompt)

    def test_determine_batch_followup_runs_supervisor_and_merges_report_in_supervised_mode(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)
            guidance_state = self._build_guidance_state(workdir)
            self._write_round(
                workdir,
                "opt-round-1",
                parent_round="round-0",
                            )

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                min_rounds=2,
                round_mode="supervised",
                round_batch_size=1,
                current_round=1,
                final_round=1,
            )

            class FakeRunner:
                def __init__(self) -> None:
                    self.requests: list[AgentRequest] = []

                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del stdout, stderr
                    self.requests.append(request)
                    if _optimize_invocation_kind(request) == "supervisor":
                        self_outer._write_supervisor_handoff(
                            guidance_state,
                            status="fail",
                            latest_round="opt-round-1",
                            issues=("round summary is missing the compare-perf conclusion",),
                        )
                        return AgentResult(return_code=0, stdout="supervisor ok", stderr="")
                    return AgentResult(return_code=0, stdout="worker ok", stderr="")

            self_outer = self
            runner = FakeRunner()
            controller = execution_module.MultiInvocationOptimizeController(
                cast(Any, runner),
                execution_module.OptimizeSessionArtifactsManager(),
                guidance_state,
                verbose_stream=StringIO(),
            )

            with patch(
                "triton_agent.optimize.execution.check_round",
                return_value=SimpleNamespace(
                    kind="round",
                    status="pass",
                    issues=(),
                    summary="round check passed",
                ),
            ):
                followup = controller.check_batch_round(
                    request,
                    batch_start=1,
                    batch_end=1,
                )

            self.assertEqual([_optimize_invocation_kind(record) for record in runner.requests], ["supervisor"])
            self.assertTrue(followup.has_failures)
            self.assertIn("Supervisor guidance:", followup.summary)
            self.assertIn("round summary is missing the compare-perf conclusion", followup.summary)

    def test_multi_invocation_controller_baseline_phase_preserves_request_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            output_path = workdir / "opt_kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            guidance_state = self._build_checked_guidance_state(workdir)
            recorded_requests: list[AgentRequest] = []

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=output_path,
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt="",
                workdir=workdir,
                remote="alice@example.com:2200",
                remote_workdir="/tmp/remote",
                min_rounds=1,
                round_mode="checked",
                user_prompt="Focus on occupancy.",
            )

            class FakeRunner:
                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del stdout, stderr
                    recorded_requests.append(request)
                    return AgentResult(return_code=0, stdout="baseline ready", stderr="")

            controller = execution_module.MultiInvocationOptimizeController(
                cast(Any, FakeRunner()),
                execution_module.OptimizeSessionArtifactsManager(),
                guidance_state,
                verbose_stream=StringIO(),
            )

            result = controller.run_baseline_phase(
                request,
                execution_module.BaselinePreflightResult(
                    state=execution_module.BaselinePreflightState.NEEDS_PREPARE,
                    issues=("baseline/ directory does not exist",),
                ),
            )

            self.assertEqual(result.return_code, 0)
            self.assertEqual(len(recorded_requests), 1)
            baseline_request = recorded_requests[0]
            self.assertEqual(_optimize_invocation_kind(baseline_request), "baseline")
            self.assertIn(f"Operator input: {operator.as_posix()}", baseline_request.prompt)
            self.assertNotIn("Requested output:", baseline_request.prompt)
            self.assertIn("Requested test mode: differential", baseline_request.prompt)
            self.assertIn("Requested bench mode: torch-npu-profiler", baseline_request.prompt)
            self.assertIn("Remote execution target: alice@example.com:2200", baseline_request.prompt)
            self.assertIn("Remote execution root: /tmp/remote", baseline_request.prompt)
            self.assertIn("Additional user instructions:", baseline_request.prompt)
            self.assertIn("Focus on occupancy.", baseline_request.prompt)
            self.assertIn("Do not open a new optimization round yet.", baseline_request.prompt)

    def test_multi_invocation_controller_checked_batch_validates_all_new_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)
            guidance_state = self._build_checked_guidance_state(workdir)

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt="",
                workdir=workdir,
                min_rounds=3,
                round_mode="checked",
                round_batch_size=2,
                current_round=1,
                final_round=2,
            )

            class FakeRunner:
                def __init__(self) -> None:
                    self.requests: list[AgentRequest] = []
                    self.worker_calls = 0

                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del stdout, stderr
                    self.requests.append(request)
                    self.worker_calls += 1
                    if self.worker_calls == 1:
                        self_outer._write_round(
                            workdir,
                            "opt-round-1",
                            parent_round="round-0",
                        )
                        self_outer._write_round(
                            workdir,
                            "opt-round-2",
                            parent_round="round-1",
                        )
                        return AgentResult(return_code=0, stdout="worker ok", stderr="")
                    return AgentResult(return_code=1, stdout="", stderr="stop after second batch prompt for test")

            checked_rounds: list[tuple[str, Optional[int], Optional[int], Optional[str]]] = []

            def fake_check_round(
                round_dir: Path,
                *,
                current_round: Optional[int] = None,
                final_round: Optional[int] = None,
                optimize_target: Optional[str] = None,
            ) -> SimpleNamespace:
                checked_rounds.append((round_dir.name, current_round, final_round, optimize_target))
                return SimpleNamespace(
                    kind="round",
                    status="pass",
                    issues=(),
                    summary="round check passed",
                )

            self_outer = self
            runner = FakeRunner()
            controller = execution_module.MultiInvocationOptimizeController(
                cast(Any, runner),
                execution_module.OptimizeSessionArtifactsManager(),
                guidance_state,
                verbose_stream=StringIO(),
            )

            with patch("triton_agent.optimize.execution.check_round", side_effect=fake_check_round):
                with patch("triton_agent.optimize.recovery.check_round", side_effect=fake_check_round):
                    result = controller.run_round_loop(request)

            self.assertEqual(result.return_code, 1)
            self.assertEqual(
                checked_rounds,
                [
                    ("opt-round-1", 1, 2, "kernel"),
                    ("opt-round-2", 2, 2, "kernel"),
                ],
            )
            self.assertEqual(len(runner.requests), 2)
            self.assertIn("This invocation owns rounds 1 through 2.", runner.requests[0].prompt)
            self.assertIn("Execute those rounds strictly one at a time.", runner.requests[0].prompt)
            self.assertIn("This invocation owns rounds 3 through 3.", runner.requests[1].prompt)
            self.assertNotIn("CLI batch follow-up from the previous worker batch:", runner.requests[1].prompt)

    def test_multi_invocation_controller_failed_worker_run_preserves_pt_files_with_round_cleanup_policy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)
            guidance_state = self._build_checked_guidance_state(workdir)
            stray_result = workdir / "TEST_RESULT.pt"

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                min_rounds=1,
                round_mode="checked",
            )

            class FakeRunner:
                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del request, stdout, stderr
                    stray_result.write_text("payload\n", encoding="utf-8")
                    return AgentResult(return_code=1, stdout="", stderr="worker failed")

            controller = execution_module.MultiInvocationOptimizeController(
                cast(Any, FakeRunner()),
                execution_module.OptimizeSessionArtifactsManager(),
                guidance_state,
                verbose_stream=StringIO(),
            )

            with patch.dict(os.environ, {"TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES": "round"}, clear=False):
                result = controller.run_round_loop(request)

            self.assertEqual(result.return_code, 1)
            self.assertTrue(stray_result.exists())

    def test_multi_invocation_controller_checked_batch_carries_failures_to_next_batch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)
            guidance_state = self._build_checked_guidance_state(workdir)

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                min_rounds=4,
                round_mode="checked",
                round_batch_size=3,
                current_round=1,
                final_round=3,
            )

            class FakeRunner:
                def __init__(self) -> None:
                    self.requests: list[AgentRequest] = []
                    self.worker_calls = 0

                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del stdout, stderr
                    self.requests.append(request)
                    self.worker_calls += 1
                    if self.worker_calls == 1:
                        self_outer._write_round(
                            workdir,
                            "opt-round-1",
                            parent_round="round-0",
                        )
                        self_outer._write_round(
                            workdir,
                            "opt-round-2",
                            parent_round="round-1",
                        )
                        self_outer._write_round(
                            workdir,
                            "opt-round-3",
                            parent_round="round-2",
                        )
                        return AgentResult(return_code=0, stdout="worker ok", stderr="")
                    return AgentResult(return_code=1, stdout="", stderr="stop after repair prompt for test")

            checked_rounds: list[tuple[str, Optional[int], Optional[int]]] = []

            def fake_check_round(
                round_dir: Path,
                *,
                current_round: Optional[int] = None,
                final_round: Optional[int] = None,
                optimize_target: Optional[str] = None,
            ) -> SimpleNamespace:
                del optimize_target
                checked_rounds.append((round_dir.name, current_round, final_round))
                if round_dir.name == "opt-round-2":
                    return SimpleNamespace(
                        kind="round",
                        status="fail",
                        issues=("round 2 metadata is incomplete",),
                        summary="round check failed",
                    )
                return SimpleNamespace(
                    kind="round",
                    status="pass",
                    issues=(),
                    summary="round check passed",
                )

            self_outer = self
            runner = FakeRunner()
            controller = execution_module.MultiInvocationOptimizeController(
                cast(Any, runner),
                execution_module.OptimizeSessionArtifactsManager(),
                guidance_state,
                verbose_stream=StringIO(),
            )

            with patch("triton_agent.optimize.execution.check_round", side_effect=fake_check_round):
                result = controller.run_round_loop(request)

            self.assertEqual(result.return_code, 1)
            self.assertEqual(
                checked_rounds,
                [
                    ("opt-round-1", 1, 3),
                    ("opt-round-2", 2, 3),
                    ("opt-round-3", 3, 3),
                ],
            )
            self.assertEqual(len(runner.requests), 2)
            self.assertIn("This invocation owns rounds 4 through 4.", runner.requests[1].prompt)
            self.assertIn("CLI batch follow-up from the previous worker batch:", runner.requests[1].prompt)
            self.assertIn("opt-round-2", runner.requests[1].prompt)
            self.assertIn("opt-round-3", runner.requests[1].prompt)
            self.assertNotIn("not yet accepted as session progress", runner.requests[1].prompt)

    def test_multi_invocation_controller_recovers_transient_worker_failure_by_retrying_same_range(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)
            guidance_state = self._build_checked_guidance_state(workdir)

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                min_rounds=2,
                round_mode="checked",
                round_batch_size=2,
                current_round=1,
                final_round=2,
            )

            class FakeRunner:
                def __init__(self) -> None:
                    self.requests: list[AgentRequest] = []
                    self.worker_calls = 0

                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del stdout, stderr
                    self.requests.append(request)
                    self.worker_calls += 1
                    if self.worker_calls == 1:
                        return AgentResult(
                            return_code=1,
                            stdout="",
                            stderr="ERROR: exceeded retry limit, last status: 429 Too Many Requests",
                        )
                    self_outer._write_round(
                        workdir,
                        "opt-round-1",
                        parent_round="round-0",
                    )
                    self_outer._write_round(
                        workdir,
                        "opt-round-2",
                        parent_round="round-1",
                    )
                    return AgentResult(return_code=0, stdout="worker ok", stderr="")

            self_outer = self
            runner = FakeRunner()
            controller = execution_module.MultiInvocationOptimizeController(
                cast(Any, runner),
                execution_module.OptimizeSessionArtifactsManager(),
                guidance_state,
                verbose_stream=StringIO(),
            )

            with patch(
                "triton_agent.optimize.execution.check_round",
                return_value=SimpleNamespace(
                    kind="round",
                    status="pass",
                    issues=(),
                    summary="round check passed",
                ),
            ):
                result = controller.run_round_loop(request)

            self.assertEqual(result.return_code, 0)
            self.assertEqual(len(runner.requests), 2)
            self.assertEqual((runner.requests[0].current_round, runner.requests[0].final_round), (1, 2))
            self.assertEqual((runner.requests[1].current_round, runner.requests[1].final_round), (1, 2))
            self.assertEqual(runner.requests[0].show_output_label, "batch-1-2-r1")
            self.assertEqual(runner.requests[1].show_output_label, "batch-1-2-r2")
            self.assertTrue(runner.requests[0].disable_backend_retry)
            self.assertIsNotNone(runner.requests[0].progress_probe)
            self.assertTrue(runner.requests[1].disable_backend_retry)
            self.assertIsNotNone(runner.requests[1].progress_probe)
            self.assertIn("previous invocation ended in a transient backend failure", runner.requests[1].prompt)
            self.assertIn("This invocation owns rounds 1 through 2.", runner.requests[1].prompt)

    def test_multi_invocation_controller_transient_recovery_records_distinct_launch_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)
            guidance_state = self._build_checked_guidance_state(workdir)

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                min_rounds=2,
                round_mode="checked",
                round_batch_size=2,
                current_round=1,
                final_round=2,
            )

            class FakeRunner:
                def __init__(self) -> None:
                    self.worker_calls = 0

                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del stdout, stderr
                    self.worker_calls += 1
                    if self.worker_calls == 1:
                        return AgentResult(
                            return_code=1,
                            stdout="",
                            stderr="ERROR: exceeded retry limit, last status: 429 Too Many Requests",
                            session_id="session-r1",
                        )
                    self_outer._write_round(
                        workdir,
                        "opt-round-1",
                        parent_round="round-0",
                    )
                    self_outer._write_round(
                        workdir,
                        "opt-round-2",
                        parent_round="round-1",
                    )
                    return AgentResult(
                        return_code=0,
                        stdout="worker ok",
                        stderr="",
                        session_id="session-r2",
                    )

            self_outer = self
            controller = execution_module.MultiInvocationOptimizeController(
                cast(Any, FakeRunner()),
                execution_module.OptimizeSessionArtifactsManager(),
                guidance_state,
                verbose_stream=StringIO(),
            )

            with patch(
                "triton_agent.optimize.execution.check_round",
                return_value=SimpleNamespace(
                    kind="round",
                    status="pass",
                    issues=(),
                    summary="round check passed",
                ),
            ):
                result = controller.run_round_loop(request)

            self.assertEqual(result.return_code, 0)
            self.assertTrue(guidance_state.agent_session_path("batch-1-2-r1").exists())
            self.assertTrue(guidance_state.agent_session_path("batch-1-2-r2").exists())
            self.assertEqual(
                json.loads(guidance_state.agent_session_path("batch-1-2-r1").read_text(encoding="utf-8"))["session_id"],
                "session-r1",
            )
            self.assertEqual(
                json.loads(guidance_state.agent_session_path("batch-1-2-r2").read_text(encoding="utf-8"))["session_id"],
                "session-r2",
            )

    def test_multi_invocation_controller_recovers_stalled_worker_from_first_unresolved_round(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)
            guidance_state = self._build_checked_guidance_state(workdir)

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                min_rounds=15,
                round_mode="checked",
                round_batch_size=5,
                current_round=11,
                final_round=15,
            )

            class FakeRunner:
                def __init__(self) -> None:
                    self.requests: list[AgentRequest] = []
                    self.worker_calls = 0

                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del stdout, stderr
                    self.requests.append(request)
                    self.worker_calls += 1
                    if self.worker_calls == 1:
                        self_outer._write_round(
                            workdir,
                            "opt-round-11",
                            parent_round="opt-round-10",
                        )
                        self_outer._write_round(
                            workdir,
                            "opt-round-12",
                            parent_round="opt-round-11",
                        )
                        self_outer._write_round(
                            workdir,
                            "opt-round-13",
                            parent_round="opt-round-12",
                        )
                        (workdir / "opt-round-14").mkdir(exist_ok=True)
                        return AgentResult(
                            return_code=1,
                            stdout="",
                            stderr="worker stalled",
                            stalled=True,
                        )
                    self_outer._write_round(
                        workdir,
                        "opt-round-14",
                        parent_round="opt-round-13",
                    )
                    self_outer._write_round(
                        workdir,
                        "opt-round-15",
                        parent_round="opt-round-14",
                    )
                    return AgentResult(return_code=0, stdout="worker ok", stderr="")

            def fake_check_round(
                round_dir: Path,
                *,
                current_round: Optional[int] = None,
                final_round: Optional[int] = None,
                optimize_target: Optional[str] = None,
            ) -> SimpleNamespace:
                del final_round, optimize_target
                status = "pass" if current_round in (11, 12, 13) else "fail"
                if current_round in (14, 15) and (round_dir / "round-state.json").exists():
                    status = "pass"
                return SimpleNamespace(
                    kind="round",
                    status=status,
                    issues=() if status == "pass" else ("round check failed",),
                    summary="round check passed" if status == "pass" else "round check failed",
                )

            self_outer = self
            runner = FakeRunner()
            controller = execution_module.MultiInvocationOptimizeController(
                cast(Any, runner),
                execution_module.OptimizeSessionArtifactsManager(),
                guidance_state,
                verbose_stream=StringIO(),
            )

            with patch("triton_agent.optimize.execution.check_round", side_effect=fake_check_round):
                with patch("triton_agent.optimize.recovery.check_round", side_effect=fake_check_round):
                    result = controller.run_round_loop(request)

            self.assertEqual(result.return_code, 0)
            self.assertEqual(len(runner.requests), 2)
            self.assertEqual((runner.requests[0].current_round, runner.requests[0].final_round), (11, 15))
            self.assertEqual((runner.requests[1].current_round, runner.requests[1].final_round), (14, 15))
            self.assertIn("previous invocation stalled", runner.requests[1].prompt)
            self.assertIn("Resume from round 14", runner.requests[1].prompt)
            self.assertIn("Inspect existing artifacts for round 14", runner.requests[1].prompt)
            self.assertNotIn("This invocation owns rounds 11 through 15.", runner.requests[1].prompt)

    def test_multi_invocation_controller_non_recoverable_worker_failure_exits_immediately(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)
            guidance_state = self._build_checked_guidance_state(workdir)

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                min_rounds=1,
                round_mode="checked",
            )

            class FakeRunner:
                def __init__(self) -> None:
                    self.requests: list[AgentRequest] = []

                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del stdout, stderr
                    self.requests.append(request)
                    return AgentResult(return_code=1, stdout="", stderr="plain worker launch failure")

            runner = FakeRunner()
            controller = execution_module.MultiInvocationOptimizeController(
                cast(Any, runner),
                execution_module.OptimizeSessionArtifactsManager(),
                guidance_state,
                verbose_stream=StringIO(),
            )

            result = controller.run_round_loop(request)

            self.assertEqual(result.return_code, 1)
            self.assertEqual(len(runner.requests), 1)
            self.assertEqual(result.stderr, "plain worker launch failure")

    def test_multi_invocation_controller_batch_validation_failure_does_not_enter_recovery(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)
            guidance_state = self._build_checked_guidance_state(workdir)

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                min_rounds=1,
                round_mode="checked",
            )

            class FakeRunner:
                def __init__(self) -> None:
                    self.requests: list[AgentRequest] = []

                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del stdout, stderr
                    self.requests.append(request)
                    self_outer._write_round(
                        workdir,
                        "opt-round-1",
                        parent_round="round-0",
                        operator_source="print('broken round')\n",
                    )
                    return AgentResult(return_code=0, stdout="worker ok", stderr="")

            self_outer = self
            runner = FakeRunner()
            controller = execution_module.MultiInvocationOptimizeController(
                cast(Any, runner),
                execution_module.OptimizeSessionArtifactsManager(),
                guidance_state,
                verbose_stream=StringIO(),
            )

            with patch(
                "triton_agent.optimize.execution.check_round",
                return_value=SimpleNamespace(
                    kind="round",
                    status="fail",
                    issues=("round metadata is incomplete",),
                    summary="round check failed",
                ),
            ):
                result = controller.run_round_loop(request)

            self.assertEqual(result.return_code, 1)
            self.assertEqual(len(runner.requests), 1)
            self.assertIn("round metadata is incomplete", result.stderr)

    def test_run_optimize_batch_preserves_round_mode_mode(self) -> None:
        for round_mode_mode in ("checked", "supervised"):
            with self.subTest(round_mode=round_mode_mode):
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
                        stream_output=False,
                        remote=None,
                        remote_workdir=None,
                        min_rounds=1,
                        resume_mode="auto",
                        reset_optimize=False,
                        no_agent_session=False,
                        round_mode=round_mode_mode,
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
                    self.assertEqual(batch_request.round_mode, round_mode_mode)
                    self.assertEqual(_optimize_invocation_kind(batch_request), "worker")

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
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt="Avoid changing numerics.",
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
            self.assertEqual(len(captured_requests), 2)
            for request in captured_requests:
                self.assertEqual(request.prompt, "")
                self.assertEqual(request.user_prompt, "Avoid changing numerics.")

    def test_run_optimize_batch_passes_agent_hooks_to_each_workspace_request(self) -> None:
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
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
                enable_agent_hooks=True,
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
            self.assertEqual(len(captured_requests), 2)
            for request in captured_requests:
                self.assertTrue(request.enable_agent_hooks)

    def test_run_optimize_batch_executes_post_optimize_command_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "kernel_a"
            workspace.mkdir()
            operator = workspace / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
                post_optimize_command="echo done",
                upload_enabled=False,
            )
            seen_post_commands: list[tuple[str, Path]] = []

            def fake_post_optimize_command(command: str, workdir: Path) -> subprocess.CompletedProcess[str]:
                seen_post_commands.append((command, workdir))
                return subprocess.CompletedProcess(
                    args=command,
                    returncode=0,
                    stdout="done\n",
                    stderr="",
                )

            with patch(
                "triton_agent.optimize.batch.run_post_optimize_command",
                side_effect=fake_post_optimize_command,
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
                        run_request=lambda request, stdout=None, stderr=None: AgentResult(
                            return_code=0,
                            stdout="ok",
                            stderr="",
                        ),
                    )

            self.assertEqual(exit_code, 0)
            self.assertEqual(seen_post_commands, [("echo done", workspace)])
            status = json.loads((root / "optimize-batch-status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["workspaces"]["kernel_a"]["status"], "completed")

    def test_run_optimize_batch_marks_workspace_failed_when_post_optimize_command_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "kernel_a"
            workspace.mkdir()
            operator = workspace / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
                post_optimize_command="echo done",
                upload_enabled=False,
            )
            captured_results = []

            def fake_render(results, stdout=None):
                del stdout
                captured_results.extend(results)
                return 1

            with patch(
                "triton_agent.optimize.batch.run_post_optimize_command",
                return_value=subprocess.CompletedProcess(
                    args="echo done",
                    returncode=3,
                    stdout="",
                    stderr="post command failed\n",
                ),
            ):
                with patch(
                    "triton_agent.optimize.batch.render_batch_optimize_results",
                    side_effect=fake_render,
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

            self.assertEqual(exit_code, 1)
            self.assertEqual(len(captured_results), 1)
            self.assertEqual(captured_results[0].status, "failed")
            self.assertIn("post command failed", captured_results[0].message)
            status = json.loads((root / "optimize-batch-status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["workspaces"]["kernel_a"]["status"], "incomplete")

    def test_run_optimize_batch_operator_filter_selects_matching_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "kernel_workspace"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")
            (workspace / "kernel_fp16.py").write_text("print('y')\n", encoding="utf-8")

            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
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
                    operator_filter="*_fp16.py",
                    stdout=StringIO(),
                    run_request=fake_run_request,
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(captured_requests), 1)
            self.assertEqual(captured_requests[0].input_path.name, "kernel_fp16.py")

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
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
            )
            seen_devices: list[Optional[str]] = []

            def fake_run_request(
                request: AgentRequest,
                stdout: Optional[object] = None,
                stderr: Optional[object] = None,
            ) -> AgentResult:
                del stdout, stderr
                seen_devices.append((request.extra_env or {}).get("ASCEND_RT_VISIBLE_DEVICES"))
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
            self.assertCountEqual(seen_devices, ["0", "1"])

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
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
            )

            with patch.dict(os.environ, {"TRITON_AGENT_BATCH_NPU_DEVICES": "0"}, clear=False):
                with self.assertRaisesRegex(ValueError, "--concurrency"):
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

    def test_run_optimize_batch_allows_same_device_when_workers_per_npu_gt_1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("alpha", "beta"):
                workspace = root / name
                workspace.mkdir()
                (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
            )
            seen_devices: list[Optional[str]] = []

            def fake_run_request(
                request: AgentRequest,
                stdout: Optional[object] = None,
                stderr: Optional[object] = None,
            ) -> AgentResult:
                del stdout, stderr
                seen_devices.append((request.extra_env or {}).get("ASCEND_RT_VISIBLE_DEVICES"))
                return AgentResult(return_code=0, stdout="ok", stderr="")

            env_vars = {
                "TRITON_AGENT_BATCH_NPU_DEVICES": "0",
                "TRITON_AGENT_BATCH_WORKERS_PER_NPU": "2",
            }
            with patch.dict(os.environ, env_vars, clear=False):
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
            self.assertEqual(len(seen_devices), 2)
            self.assertEqual(seen_devices, ["0", "0"])

    def test_run_optimize_batch_does_not_inject_affinity_env_when_mcp_enabled(self) -> None:
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
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
                enable_mcp=True,
            )
            seen_devices: list[Optional[str]] = []

            def fake_run_request(
                request: AgentRequest,
                stdout: Optional[object] = None,
                stderr: Optional[object] = None,
            ) -> AgentResult:
                del stdout, stderr
                seen_devices.append((request.extra_env or {}).get("ASCEND_RT_VISIBLE_DEVICES"))
                return AgentResult(return_code=0, stdout="ok", stderr="")

            with patch.dict(os.environ, {"TRITON_AGENT_BATCH_NPU_DEVICES": "0"}, clear=False):
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
            self.assertEqual(seen_devices, [None, None])

    def test_run_optimize_batch_rejects_concurrency_beyond_effective_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "alpha"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
            )

            env_vars = {
                "TRITON_AGENT_BATCH_NPU_DEVICES": "0",
                "TRITON_AGENT_BATCH_WORKERS_PER_NPU": "2",
            }
            with patch.dict(os.environ, env_vars, clear=False):
                with self.assertRaisesRegex(ValueError, "TRITON_AGENT_BATCH_WORKERS_PER_NPU"):
                    run_optimize_batch(
                        root,
                        options,
                        max_concurrency=3,
                        stdout=StringIO(),
                        run_request=lambda request, stdout=None, stderr=None: AgentResult(
                            return_code=0, stdout="ok", stderr=""
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
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
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
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
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
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
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
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
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
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
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
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="fresh",
                reset_optimize=True,
                no_agent_session=False,
                round_mode="checked",
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

    def test_cleanup_workspace_pt_files_deletes_pt_files_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            baseline_dir = workdir / "baseline"
            round_dir = workdir / "opt-round-1"
            baseline_dir.mkdir()
            round_dir.mkdir()
            root_pt = workdir / "kernel_result.pt"
            baseline_pt = baseline_dir / "TEST_RESULT.pt"
            round_pt = round_dir / "test_result.pt"
            root_pt.write_text("root\n", encoding="utf-8")
            baseline_pt.write_text("baseline\n", encoding="utf-8")
            round_pt.write_text("round\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES", None)
                cleaned = cleanup_workspace_pt_files(workdir)

            self.assertEqual(
                cleaned,
                [
                    "kernel_result.pt",
                    "baseline/TEST_RESULT.pt",
                    "opt-round-1/test_result.pt",
                ],
            )
            self.assertFalse(root_pt.exists())
            self.assertFalse(baseline_pt.exists())
            self.assertFalse(round_pt.exists())

    def test_cleanup_workspace_pt_files_deletes_pt_files_when_env_var_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            baseline_dir = workdir / "baseline"
            round_dir = workdir / "opt-round-1"
            baseline_dir.mkdir()
            round_dir.mkdir()
            root_pt = workdir / "kernel_result.pt"
            baseline_pt = baseline_dir / "TEST_RESULT.pt"
            round_pt = round_dir / "test_result.pt"
            root_pt.write_text("root\n", encoding="utf-8")
            baseline_pt.write_text("baseline\n", encoding="utf-8")
            round_pt.write_text("round\n", encoding="utf-8")

            with patch.dict(os.environ, {"TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES": "round"}, clear=False):
                cleaned = cleanup_workspace_pt_files(workdir)

            self.assertEqual(
                cleaned,
                [
                    "kernel_result.pt",
                    "baseline/TEST_RESULT.pt",
                    "opt-round-1/test_result.pt",
                ],
            )
            self.assertFalse(root_pt.exists())
            self.assertFalse(baseline_pt.exists())
            self.assertFalse(round_pt.exists())

    def test_cleanup_workspace_pt_files_preserves_pt_files_for_run_test_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            round_dir = workdir / "opt-round-1"
            round_dir.mkdir()
            root_pt = workdir / "kernel_result.pt"
            round_pt = round_dir / "test_result.pt"
            root_pt.write_text("root\n", encoding="utf-8")
            round_pt.write_text("round\n", encoding="utf-8")

            with patch.dict(os.environ, {"TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES": "run-test"}, clear=False):
                cleaned = cleanup_workspace_pt_files(workdir)

            self.assertEqual(cleaned, [])
            self.assertTrue(root_pt.exists())
            self.assertTrue(round_pt.exists())

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

    def test_run_optimize_request_interactive_runs_only_worker_without_post_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=True,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                min_rounds=1,
                round_mode="supervised",
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
                    return AgentResult(return_code=0, stdout="ok", stderr="")

            runner = FakeRunner()

            with patch("triton_agent.optimize.orchestration.create_runner", return_value=runner):
                result = run_optimize_request(request)

            self.assertEqual(result.return_code, 0)
            self.assertEqual(len(runner.requests), 1)
            worker_request = runner.requests[0]
            self.assertEqual(_optimize_invocation_kind(worker_request), "worker")
            self.assertTrue(worker_request.interact)
            self.assertFalse(worker_request.no_agent_session)
            self.assertIn(
                "You must run the staged `ascend-npu-optimize-state` skill's `submit-round` subcommand after each completed round.",
                worker_request.prompt,
            )
            self.assertIn(
                "When a round in this invocation is complete, run `submit-round --round-dir opt-round-N --current-round N --final-round M`",
                worker_request.prompt,
            )
            self.assertNotIn("Interactive mode will not run CLI round checks", worker_request.prompt)

    def test_run_optimize_request_interactive_skips_baseline_phase_and_updates_worker_prompt(self) -> None:
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
                bench_mode="torch-npu-profiler",
                interact=True,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                min_rounds=30,
                round_mode="checked",
                round_batch_size=99,
                current_round=1,
                final_round=30,
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
                    self_outer._write_baseline(workdir)
                    round_dir = workdir / "opt-round-1"
                    round_dir.mkdir()
                    (round_dir / "attempts.md").write_text("started\n", encoding="utf-8")
                    return AgentResult(return_code=0, stdout="ok", stderr="")

            self_outer = self
            runner = FakeRunner()

            with patch("triton_agent.optimize.orchestration.create_runner", return_value=runner):
                result = run_optimize_request(request)

            self.assertEqual(result.return_code, 0)
            self.assertEqual(len(runner.requests), 1)
            worker_request = runner.requests[0]
            self.assertEqual(_optimize_invocation_kind(worker_request), "worker")
            self.assertTrue(worker_request.interact)
            self.assertIn("This invocation owns rounds 1 through 30.", worker_request.prompt)
            self.assertIn("repair or establish `baseline/` before `opt-round-1`", worker_request.prompt)
            self.assertIn("Do not rely on a separate baseline-preflight invocation", worker_request.prompt)
            self.assertIn(
                "You must run the staged `ascend-npu-optimize-state` skill's `submit-round` subcommand after each completed round.",
                worker_request.prompt,
            )
            self.assertNotIn("Interactive mode will not run CLI round checks", worker_request.prompt)
            self.assertNotIn("The baseline has already been validated before this worker batch.", worker_request.prompt)

    def test_run_optimize_request_supervisor_prompt_excludes_user_instructions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt=(
                    "Optimize this operator\n\n"
                    "Additional user instructions:\n"
                    "Focus on occupancy."
                ),
                workdir=workdir,
                min_rounds=1,
                round_mode="supervised",
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
                    if _optimize_invocation_kind(request) == "worker":
                        self_outer._write_round(
                            workdir,
                            "opt-round-1",
                            parent_round="round-0",
                        )
                    else:
                        assert request.supervisor_report_path is not None
                        request.supervisor_report_path.write_text(
                            "# Optimize Supervisor Report\n\n"
                            "Status: pass\n"
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
            self.assertEqual(_optimize_invocation_kind(supervisor_request), "supervisor")
            self.assertNotIn("Additional user instructions:", supervisor_request.prompt)
            self.assertNotIn("Focus on occupancy.", supervisor_request.prompt)

    def test_run_optimize_request_end_to_end_uses_supervisor_report_for_continue_prompt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                min_rounds=2,
                round_mode="supervised",
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
                    if _optimize_invocation_kind(request) == "worker":
                        self.worker_calls += 1
                        if self.worker_calls == 1:
                            self_outer._write_round(
                                workdir,
                                "opt-round-1",
                                parent_round="round-0",
                                )
                            return AgentResult(return_code=0, stdout="worker ok", stderr="")
                        return AgentResult(return_code=1, stdout="", stderr="worker stopped for test")
                    self.supervisor_calls += 1
                    self_outer._write_supervisor_handoff(
                        guidance_state,
                        status="fail",
                        latest_round="opt-round-1",
                        issues=("round summary is missing the compare-perf conclusion",),
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
                    with patch(
                        "triton_agent.optimize.execution.check_round",
                        return_value=SimpleNamespace(
                            kind="round",
                            status="pass",
                            issues=(
                                "recent rounds show only marginal baseline-relative geomean speedup gains; optimization may be stagnating in the current direction and may be stuck in a local optimum.",
                            ),
                            summary="round check passed",
                        ),
                    ):
                        result = run_optimize_request(request)

            self.assertEqual(result.return_code, 1)
            self.assertEqual(len(runner.requests), 3)
            self.assertEqual(_optimize_invocation_kind(runner.requests[0]), "worker")
            self.assertEqual(_optimize_invocation_kind(runner.requests[1]), "supervisor")
            self.assertEqual(_optimize_invocation_kind(runner.requests[2]), "worker")
            self.assertIn("Read this CLI round follow-up summary before auditing the round:", runner.requests[1].prompt)
            self.assertIn("may be stuck in a local optimum", runner.requests[1].prompt)
            self.assertIn("CLI batch follow-up from the previous worker batch:", runner.requests[2].prompt)
            self.assertIn("may be stuck in a local optimum", runner.requests[2].prompt)
            self.assertIn("\"status\": \"fail\"", runner.requests[2].prompt)
            self.assertIn("round summary is missing the compare-perf conclusion", runner.requests[2].prompt)

    def test_multi_invocation_controller_checked_continue_writes_handoff_and_resume_prompt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)
            guidance_state = self._build_checked_guidance_state(workdir)

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt="",
                workdir=workdir,
                remote="alice@example.com:2200",
                min_rounds=2,
                round_mode="checked",
                user_prompt="Focus on occupancy.",
            )

            class FakeRunner:
                def __init__(self) -> None:
                    self.requests: list[AgentRequest] = []
                    self.worker_calls = 0

                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del stdout, stderr
                    self.requests.append(request)
                    self.worker_calls += 1
                    if self.worker_calls == 1:
                        self_outer._write_round(
                            workdir,
                            "opt-round-1",
                            parent_round="round-0",
                        )
                        return AgentResult(return_code=0, stdout="worker ok", stderr="")
                    return AgentResult(return_code=1, stdout="", stderr="stop after second prompt for test")

            self_outer = self
            runner = FakeRunner()
            controller = execution_module.MultiInvocationOptimizeController(
                cast(Any, runner),
                execution_module.OptimizeSessionArtifactsManager(),
                guidance_state,
                verbose_stream=StringIO(),
            )

            result = controller.run_round_loop(request)

            self.assertEqual(result.return_code, 1)
            self.assertEqual(len(runner.requests), 2)
            self.assertIn("Remote execution target: alice@example.com:2200", runner.requests[0].prompt)
            self.assertIn("Additional user instructions:", runner.requests[0].prompt)
            self.assertIn("Focus on occupancy.", runner.requests[0].prompt)
            self.assertNotIn("CLI batch follow-up from the previous worker batch:", runner.requests[1].prompt)
            self.assertIn("This invocation owns rounds 2 through 2.", runner.requests[1].prompt)
            self.assertIn(
                "Before each round, re-evaluate the next bottleneck and choose the right analysis depth from the current evidence.",
                runner.requests[1].prompt,
            )
            self.assertIn(
                "Do not pre-plan the full batch before acting.",
                runner.requests[1].prompt,
            )
            self.assertIn(
                "State the optimization hypothesis and why it may help before editing code for each round.",
                runner.requests[1].prompt,
            )

    def test_multi_invocation_controller_checked_continue_carries_local_optimum_warning(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)
            guidance_state = self._build_checked_guidance_state(workdir)

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                min_rounds=2,
                round_mode="checked",
            )

            class FakeRunner:
                def __init__(self) -> None:
                    self.requests: list[AgentRequest] = []
                    self.worker_calls = 0

                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del stdout, stderr
                    self.requests.append(request)
                    self.worker_calls += 1
                    if self.worker_calls == 1:
                        self_outer._write_round(
                            workdir,
                            "opt-round-1",
                            parent_round="round-0",
                        )
                        return AgentResult(return_code=0, stdout="worker ok", stderr="")
                    return AgentResult(return_code=1, stdout="", stderr="stop after second prompt for test")

            self_outer = self
            runner = FakeRunner()
            controller = execution_module.MultiInvocationOptimizeController(
                cast(Any, runner),
                execution_module.OptimizeSessionArtifactsManager(),
                guidance_state,
                verbose_stream=StringIO(),
            )

            with patch(
                "triton_agent.optimize.execution.check_round",
                return_value=SimpleNamespace(
                    kind="round",
                    status="pass",
                    issues=(
                        "recent rounds show only marginal baseline-relative geomean speedup gains; optimization may be stagnating in the current direction and may be stuck in a local optimum.",
                    ),
                    summary="round check passed",
                ),
            ):
                result = controller.run_round_loop(request)

            self.assertEqual(result.return_code, 1)
            self.assertEqual(len(runner.requests), 2)
            self.assertNotIn("CLI batch follow-up from the previous worker batch:", runner.requests[1].prompt)
            self.assertNotIn("may be stuck in a local optimum", runner.requests[1].prompt)

    def test_run_optimize_request_final_failed_batch_stops_without_repair_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                min_rounds=1,
                round_mode="checked",
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
                    self.worker_calls += 1
                    if self.worker_calls > 1:
                        raise AssertionError("final failed batch should not trigger a repair rerun")
                    self_outer._write_round(
                        workdir,
                        "opt-round-1",
                        parent_round="round-0",
                                                operator_source="print('broken round')\n",
                    )
                    return AgentResult(return_code=0, stdout="ok", stderr="")

            self_outer = self
            runner = FakeRunner()

            with patch("triton_agent.optimize.orchestration.create_runner", return_value=runner):
                with patch(
                    "triton_agent.optimize.execution.check_round",
                    return_value=SimpleNamespace(
                        kind="round",
                        status="fail",
                        issues=("round metadata is incomplete",),
                        summary="round check failed",
                    ),
                ):
                    result = run_optimize_request(request)

            self.assertEqual(result.return_code, 1)
            self.assertEqual(runner.worker_calls, 1)
            self.assertIn("round metadata is incomplete", result.stderr)

    def test_multi_invocation_controller_converts_invalid_supervisor_report_to_gate_result(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)
            self._write_round(
                workdir,
                "opt-round-1",
                parent_round="round-0",
                            )

            guidance_state = self._build_guidance_state(workdir)

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                min_rounds=1,
                round_mode="supervised",
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
                        status="invalid-status",
                        latest_round="opt-round-1",
                    )
                    return AgentResult(return_code=0, stdout="ok", stderr="")

            self_outer = self
            controller = execution_module.MultiInvocationOptimizeController(
                cast(Any, FakeRunner()),
                execution_module.OptimizeSessionArtifactsManager(),
                guidance_state,
                verbose_stream=StringIO(),
            )

            gate_result = controller._run_supervisor_batch(
                request,
                batch_start=1,
                batch_end=1,
                batch_round_summary="opt-round-1: {}",
            )

            self.assertEqual(gate_result.payload["status"], "fail")
            issues = cast(list[object], gate_result.payload["issues"])
            self.assertIn("invalid supervisor status", str(issues[0]))

    def test_multi_invocation_controller_converts_missing_decision_line_to_gate_result(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)
            self._write_round(
                workdir,
                "opt-round-1",
                parent_round="round-0",
                            )

            guidance_state = self._build_guidance_state(workdir)

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                min_rounds=1,
                round_mode="supervised",
            )

            class FakeRunner:
                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del request, stdout, stderr
                    assert guidance_state.supervisor_report_path is not None
                    guidance_state.supervisor_report_path.write_text(
                        "# Optimize Supervisor Report\n\nBlocking issues: missing decision line\n",
                        encoding="utf-8",
                    )
                    return AgentResult(return_code=0, stdout="ok", stderr="")

            controller = execution_module.MultiInvocationOptimizeController(
                cast(Any, FakeRunner()),
                execution_module.OptimizeSessionArtifactsManager(),
                guidance_state,
                verbose_stream=StringIO(),
            )

            gate_result = controller._run_supervisor_batch(
                request,
                batch_start=1,
                batch_end=1,
                batch_round_summary="opt-round-1: {}",
            )

            self.assertEqual(gate_result.payload["status"], "fail")
            issues = cast(list[object], gate_result.payload["issues"])
            self.assertIn("missing supervisor status line", str(issues[0]))

    def test_multi_invocation_controller_snapshots_supervisor_handoff_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            self._write_baseline(workdir)
            self._write_round(
                workdir,
                "opt-round-1",
                parent_round="round-0",
                            )

            guidance_state = self._build_guidance_state(workdir)
            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=operator,
                operator_path=operator,
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt="Optimize this operator",
                workdir=workdir,
                min_rounds=1,
                round_mode="supervised",
            )

            class FakeRunner:
                def run(
                    self,
                    request: AgentRequest,
                    stdout: Optional[object] = None,
                    stderr: Optional[object] = None,
                ) -> AgentResult:
                    del stdout, stderr
                    assert request.supervisor_report_path is not None
                    request.supervisor_report_path.write_text(
                        "# Optimize Supervisor Report\n\n"
                        "Status: pass\n"
                        "Blocking issues: none\n"
                        "Latest round: opt-round-1\n",
                        encoding="utf-8",
                    )
                    return AgentResult(return_code=0, stdout="ok", stderr="")

            controller = execution_module.MultiInvocationOptimizeController(
                cast(Any, FakeRunner()),
                execution_module.OptimizeSessionArtifactsManager(),
                guidance_state,
                verbose_stream=StringIO(),
            )

            gate_result = controller._run_supervisor_batch(
                request,
                batch_start=1,
                batch_end=1,
                batch_round_summary="opt-round-1: {}",
            )

            self.assertEqual(gate_result.payload["status"], "pass")
            assert guidance_state.supervisor_handoff_dir is not None
            report_snapshot = guidance_state.supervisor_handoff_dir / "round-001-supervisor-report.md"
            self.assertTrue(report_snapshot.exists())
            assert guidance_state.supervisor_report_path is not None
            self.assertEqual(
                guidance_state.supervisor_report_path.read_text(encoding="utf-8"),
                report_snapshot.read_text(encoding="utf-8"),
            )

    def test_latest_round_dir_prefers_highest_numeric_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            self._write_round(
                workdir,
                "opt-round-2",
                parent_round="round-1",
                            )
            self._write_round(
                workdir,
                "opt-round-10",
                parent_round="round-9",
                            )

            latest = _latest_round_dir(workdir)

            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.name, "opt-round-10")

    def test_round_helpers_ignore_non_numeric_opt_round_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            self._write_round(
                workdir,
                "opt-round-2",
                parent_round="round-1",
                            )
            (workdir / "opt-round-final").mkdir()
            (workdir / "opt-round-notes").mkdir()

            latest = _latest_round_dir(workdir)

            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.name, "opt-round-2")
            self.assertEqual(_count_round_directories(workdir), 1)

    def test_round_helpers_ignore_incomplete_precreated_round_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            self._write_round(
                workdir,
                "opt-round-1",
                parent_round="round-0",
                            )
            (workdir / "opt-round-2").mkdir()

            latest = _latest_round_dir(workdir)

            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.name, "opt-round-1")
            self.assertEqual(_count_round_directories(workdir), 1)

    def test_run_optimize_batch_auto_upload_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "alpha"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('alpha')\n", encoding="utf-8")

            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
                upload_enabled=True,
            )

            with patch(
                "triton_agent.optimize.batch.render_batch_optimize_results",
                return_value=0,
            ):
                with patch("triton_agent.optimize.batch.upload_optimize_workspace") as mock_upload:
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
            mock_upload.assert_called_once()

    def test_run_optimize_batch_no_auto_upload_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "alpha"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('alpha')\n", encoding="utf-8")

            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
                upload_enabled=False,
            )

            with patch(
                "triton_agent.optimize.batch.render_batch_optimize_results",
                return_value=0,
            ):
                with patch("triton_agent.optimize.batch.upload_optimize_workspace") as mock_upload:
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
            mock_upload.assert_not_called()

    def test_run_optimize_batch_no_auto_upload_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "alpha"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('alpha')\n", encoding="utf-8")

            options = OptimizeRunOptions(
                agent_name="codex",
                interact=False,
                verbose=False,
                stream_output=False,
                remote=None,
                remote_workdir=None,
                min_rounds=1,
                resume_mode="auto",
                reset_optimize=False,
                no_agent_session=False,
                round_mode="checked",
                output=None,
                test_mode=None,
                bench_mode=None,
                prompt=None,
                upload_enabled=True,
            )

            with patch(
                "triton_agent.optimize.batch.render_batch_optimize_results",
                return_value=0,
            ):
                with patch("triton_agent.optimize.batch.upload_optimize_workspace") as mock_upload:
                    exit_code = run_optimize_batch(
                        root,
                        options,
                        max_concurrency=1,
                        stdout=StringIO(),
                        run_request=lambda request, stdout=None, stderr=None: AgentResult(
                            return_code=1,
                            stdout="",
                            stderr="error",
                        ),
                    )

            self.assertEqual(exit_code, 0)
            mock_upload.assert_not_called()

    def test_run_optimize_request_enters_managed_mcp_scope_when_request_requires_mcp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=workdir / "kernel.py",
                operator_path=workdir / "kernel.py",
                output_path=workdir / "opt_kernel.py",
                test_mode="differential",
                bench_mode="torch-npu-profiler",
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-optimize",
                prompt="Prompt body",
                workdir=workdir,
                mcp_servers=("triton-agent-run-eval",),
            )

            entered: list[str] = []

            class _DummyScope:
                def __enter__(self):
                    entered.append("enter")
                    return None

                def __exit__(self, exc_type, exc, tb):
                    entered.append("exit")
                    return False

            class DummyRunner:
                pass

            with patch("triton_agent.optimize.orchestration.SkillLinkManager.prepare_skills", return_value=()):
                with patch("triton_agent.optimize.orchestration.SkillLinkManager.cleanup", return_value=[]):
                    with patch("triton_agent.optimize.orchestration.managed_mcp_scope", return_value=_DummyScope()):
                        with patch("triton_agent.optimize.orchestration.create_runner", return_value=DummyRunner()):
                            with patch(
                                "triton_agent.optimize.orchestration.optimize_execution.execute_multi_invocation_optimize",
                                return_value=AgentResult(return_code=0, stdout="", stderr=""),
                            ):
                                    result = run_optimize_request(request)

            self.assertEqual(result.return_code, 0)
            self.assertEqual(entered, ["enter", "exit"])


if __name__ == "__main__":
    unittest.main()
