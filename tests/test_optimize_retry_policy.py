from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.models import AgentRequest, AgentResult, CommandKind
from triton_agent.optimize.agent_exit_log import write_agent_exit_log
from triton_agent.optimize.execution import MultiInvocationOptimizeController
from triton_agent.optimize.session_artifacts import OptimizeSessionArtifactsManager
from triton_agent.transient_failures import (
    is_optimize_worker_retryable,
    is_transient_agent_failure,
)


class OptimizeRetryPolicyTests(unittest.TestCase):
    def test_is_transient_agent_failure_detects_rate_limit_text(self) -> None:
        result = AgentResult(
            return_code=1,
            stdout="429 too many requests",
            stderr="",
        )
        self.assertTrue(is_transient_agent_failure(result))

    def test_is_optimize_worker_retryable_rejects_stalled_and_interrupt(self) -> None:
        stalled = AgentResult(return_code=1, stdout="", stderr="", stalled=True)
        interrupted = AgentResult(return_code=130, stdout="", stderr="")
        self.assertFalse(is_optimize_worker_retryable(stalled))
        self.assertFalse(is_optimize_worker_retryable(interrupted))

    def test_is_optimize_worker_retryable_rejects_transient_failures(self) -> None:
        result = AgentResult(return_code=1, stdout="rate limit exceeded", stderr="")
        self.assertFalse(is_optimize_worker_retryable(result))

    def test_is_optimize_worker_retryable_accepts_generic_agent_failure(self) -> None:
        result = AgentResult(return_code=1, stdout="agent crashed", stderr="")
        self.assertTrue(is_optimize_worker_retryable(result))

    def test_write_agent_exit_log_records_time_and_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            write_agent_exit_log(
                workdir=workdir,
                run_id="run-001",
                label="batch-1-5",
                return_code=1,
                stderr="boom",
                stalled=False,
                session_id="session-123",
                duration_ms=3210,
            )
            payload = json.loads(
                (workdir / "triton-agent-logs" / "run-001" / "agent-exit-batch-1-5.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["return_code"], 1)
            self.assertEqual(payload["duration_ms"], 3210)
            self.assertIn("ended_at", payload)

    def test_worker_batch_retries_only_retryable_failures(self) -> None:
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
                interact=False,
                verbose=True,
                stream_output=False,
                force_overwrite=False,
                agent_name="traecli",
                skill_name="triton-npu-optimize",
                prompt="worker prompt",
                workdir=workdir,
                min_rounds=1,
                current_round=1,
                final_round=1,
            )
            artifacts_state = OptimizeSessionArtifactsManager().prepare_checked_session(
                workdir,
                agent_name="traecli",
                optimize_target="kernel",
                compiler_source_path=None,
                compiler_source_commit=None,
                enable_cann_ext_api=False,
                enable_subagent=False,
                optimize_knowledge_skill_name=None,
            )
            calls: list[str] = []

            class FakeRunner:
                def run(self, _request: AgentRequest, **_kwargs: Any) -> AgentResult:
                    calls.append("run")
                    return AgentResult(return_code=1, stdout="agent crashed", stderr="")

            controller = MultiInvocationOptimizeController(
                cast(Any, FakeRunner()),
                OptimizeSessionArtifactsManager(),
                artifacts_state=artifacts_state,
                verbose_stream=StringIO(),
            )
            with patch("triton_agent.optimize.execution.time.sleep") as sleep_mock:
                result = controller._run_request_with_retry(
                    request,
                    batch_start=1,
                    batch_end=1,
                )

            self.assertFalse(result.succeeded)
            self.assertEqual(len(calls), 1 + 3)
            self.assertEqual(sleep_mock.call_count, 3)

    def test_worker_batch_does_not_retry_stalled_failure(self) -> None:
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
                interact=False,
                verbose=True,
                stream_output=False,
                force_overwrite=False,
                agent_name="traecli",
                skill_name="triton-npu-optimize",
                prompt="worker prompt",
                workdir=workdir,
                min_rounds=1,
                current_round=1,
                final_round=1,
            )
            artifacts_state = OptimizeSessionArtifactsManager().prepare_checked_session(
                workdir,
                agent_name="traecli",
                optimize_target="kernel",
                compiler_source_path=None,
                compiler_source_commit=None,
                enable_cann_ext_api=False,
                enable_subagent=False,
                optimize_knowledge_skill_name=None,
            )
            calls: list[str] = []

            class FakeRunner:
                def run(self, _request: AgentRequest, **_kwargs: Any) -> AgentResult:
                    calls.append("run")
                    return AgentResult(return_code=1, stdout="", stderr="", stalled=True)

            controller = MultiInvocationOptimizeController(
                cast(Any, FakeRunner()),
                OptimizeSessionArtifactsManager(),
                artifacts_state=artifacts_state,
                verbose_stream=StringIO(),
            )
            with patch("triton_agent.optimize.execution.time.sleep") as sleep_mock:
                result = controller._run_request_with_retry(
                    request,
                    batch_start=1,
                    batch_end=1,
                )

            self.assertTrue(result.stalled)
            self.assertEqual(calls, ["run"])
            sleep_mock.assert_not_called()

    def test_worker_recovery_retries_fatal_generic_failures(self) -> None:
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
                interact=False,
                verbose=True,
                stream_output=False,
                force_overwrite=False,
                agent_name="traecli",
                skill_name="triton-npu-optimize",
                prompt="worker prompt",
                workdir=workdir,
                min_rounds=1,
                current_round=1,
                final_round=1,
            )
            worker_request = replace(request, prompt="worker prompt")
            artifacts_state = OptimizeSessionArtifactsManager().prepare_checked_session(
                workdir,
                agent_name="traecli",
                optimize_target="kernel",
                compiler_source_path=None,
                compiler_source_commit=None,
                enable_cann_ext_api=False,
                enable_subagent=False,
                optimize_knowledge_skill_name=None,
            )
            calls: list[str] = []

            class FakeRunner:
                def run(self, _request: AgentRequest, **_kwargs: Any) -> AgentResult:
                    calls.append("run")
                    return AgentResult(return_code=1, stdout="agent crashed", stderr="")

            controller = MultiInvocationOptimizeController(
                cast(Any, FakeRunner()),
                OptimizeSessionArtifactsManager(),
                artifacts_state=artifacts_state,
                verbose_stream=StringIO(),
            )
            with patch("triton_agent.optimize.execution.time.sleep") as sleep_mock:
                _worker_request, result = controller._run_worker_with_recovery(
                    request,
                    worker_request,
                    issues=None,
                    original_batch_start=1,
                )

            self.assertFalse(result.succeeded)
            self.assertEqual(len(calls), 1 + 3)
            self.assertEqual(sleep_mock.call_count, 3)


if __name__ == "__main__":
    unittest.main()
