import sys
import tempfile
import unittest
from os import environ
from contextlib import contextmanager
from pathlib import Path
from io import StringIO
from typing import Optional, TextIO
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.backends.base import AgentRunner
from triton_agent.backends.hook_common import HookStageOptions
from triton_agent.models import AgentRequest, AgentResult, CommandKind
from triton_agent.otel_trace import TRACE_PATH_ENV, TRACE_RUN_ID_ENV
from triton_agent.prompts import build_prompt


class SharedRunnerBaseTests(unittest.TestCase):
    def test_base_runner_shares_process_execution_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = _DummyRunner()
            request = AgentRequest(
                command_kind=CommandKind.GEN_TEST,
                input_path=workspace / "op.py",
                operator_path=workspace / "op.py",
                output_path=workspace / "test_op.py",
                test_mode=None,
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=True,
                force_overwrite=False,
                agent_name="dummy",
                skill_name="ascend-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )

            with patch("triton_agent.backends.base.run_process", return_value=_ok_result()) as mocked:
                result = runner.run(request)

            self.assertEqual(result.return_code, 0)
            self.assertEqual(mocked.call_args.args[0], ["dummy", "Prompt body"])
            self.assertEqual(mocked.call_args.args[1], str(workspace))
            self.assertEqual(mocked.call_args.kwargs["mode"], "streaming")
            self.assertEqual(mocked.call_args.kwargs["stall_timeout_seconds"], 123)

    def test_base_runner_passes_request_extra_env_to_process_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = _DummyRunner()
            request = AgentRequest(
                command_kind=CommandKind.GEN_TEST,
                input_path=workspace / "op.py",
                operator_path=workspace / "op.py",
                output_path=workspace / "test_op.py",
                test_mode=None,
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="dummy",
                skill_name="ascend-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
                extra_env={"ASCEND_RT_VISIBLE_DEVICES": "2"},
            )

            with patch("triton_agent.backends.base.run_process", return_value=_ok_result()) as mocked:
                runner.run(request)

        self.assertEqual(mocked.call_args.kwargs["extra_env"], {"ASCEND_RT_VISIBLE_DEVICES": "2"})

    def test_base_runner_uses_explicit_progress_probe_from_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = _DummyRunner()
            def probe() -> float:
                return 1.0

            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=workspace / "op.py",
                operator_path=workspace / "op.py",
                output_path=workspace / "opt_op.py",
                test_mode=None,
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="dummy",
                skill_name="triton-npu-optimize",
                prompt="Prompt body",
                workdir=workspace,
                progress_probe=probe,
            )

            with patch("triton_agent.backends.base.run_process", return_value=_ok_result()) as mocked_run:
                runner.run(request)

            self.assertIs(mocked_run.call_args.kwargs["progress_probe"], probe)

    def test_base_runner_omits_progress_probe_without_request_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = _DummyRunner()
            request = AgentRequest(
                command_kind=CommandKind.GEN_TEST,
                input_path=workspace / "op.py",
                operator_path=workspace / "op.py",
                output_path=workspace / "test_op.py",
                test_mode=None,
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="dummy",
                skill_name="ascend-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )

            with patch("triton_agent.backends.base.run_process", return_value=_ok_result()) as mocked_run:
                runner.run(request)

            self.assertIsNone(mocked_run.call_args.kwargs["progress_probe"])

    def test_base_runner_rejects_request_scoped_mcp_servers_when_backend_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = _DummyRunner()
            request = AgentRequest(
                command_kind=CommandKind.GEN_TEST,
                input_path=workspace / "op.py",
                operator_path=workspace / "op.py",
                output_path=workspace / "test_op.py",
                test_mode=None,
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="dummy",
                skill_name="ascend-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
                mcp_servers=("triton-agent-run-eval",),
            )

            with patch("triton_agent.backends.base.run_process") as mocked:
                result = runner.run(request)

        self.assertEqual(result.return_code, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("does not support request-scoped MCP servers", result.stderr)
        mocked.assert_not_called()

    def test_base_runner_wraps_command_build_and_execution_in_prepare_cleanup_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            events: list[str] = []
            runner = _PreparedDummyRunner(events)
            request = AgentRequest(
                command_kind=CommandKind.GEN_TEST,
                input_path=workspace / "op.py",
                operator_path=workspace / "op.py",
                output_path=workspace / "test_op.py",
                test_mode=None,
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="dummy",
                skill_name="ascend-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )

            def _inspect_run(*args, **kwargs):
                del args, kwargs
                events.append("run")
                return _ok_result()

            with patch("triton_agent.backends.base.run_process", side_effect=_inspect_run):
                result = runner.run(request)

            self.assertEqual(result.return_code, 0)
            self.assertEqual(
                events,
                ["prepare-enter", "build-command", "run", "prepare-exit"],
            )

    def test_base_runner_uses_prepare_run_context_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = _DummyRunner()
            request = AgentRequest(
                command_kind=CommandKind.GEN_TEST,
                input_path=workspace / "op.py",
                operator_path=workspace / "op.py",
                output_path=workspace / "test_op.py",
                test_mode=None,
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="dummy",
                skill_name="ascend-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )

            with (
                patch.object(runner, "_prepare_run_context") as mocked_prepare,
                patch("triton_agent.backends.base.run_process", return_value=_ok_result()),
            ):
                result = runner.run(request)

            self.assertEqual(result.return_code, 0)
            mocked_prepare.assert_called_once()

    def test_base_runner_backend_prepare_context_wraps_run_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            events: list[str] = []
            runner = _PreparedRunHookDummyRunner(events)
            request = AgentRequest(
                command_kind=CommandKind.GEN_TEST,
                input_path=workspace / "op.py",
                operator_path=workspace / "op.py",
                output_path=workspace / "test_op.py",
                test_mode=None,
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="dummy",
                skill_name="ascend-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
                enable_agent_hooks=True,
            )

            def _inspect_run(*args, **kwargs):
                del args, kwargs
                events.append("run")
                return _ok_result()

            with patch("triton_agent.backends.base.run_process", side_effect=_inspect_run):
                result = runner.run(request)

            self.assertEqual(result.return_code, 0)
            self.assertEqual(events, ["hooks-enter", "run", "hooks-exit"])

    def test_base_runner_builds_hook_options_from_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = _DummyRunner()
            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=workspace / "op.py",
                operator_path=workspace / "op.py",
                output_path=workspace / "opt_op.py",
                test_mode=None,
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="dummy",
                skill_name="triton-npu-optimize",
                prompt="Prompt body",
                workdir=workspace,
                extra_env={
                    TRACE_PATH_ENV: str(workspace / "triton-agent-logs" / "otel" / "run-001" / "trace.jsonl"),
                    TRACE_RUN_ID_ENV: "run-001",
                },
                run_id="run-override",
                enable_agent_hooks=True,
                log_tools=True,
            )

            options = runner._hook_options(request)

            self.assertIsInstance(options, HookStageOptions)
            self.assertTrue(options.trace_enabled)
            self.assertTrue(options.guard_enabled)
            self.assertEqual(options.trace_path, workspace / "triton-agent-logs" / "otel" / "run-001" / "trace.jsonl")
            self.assertEqual(options.run_id, "run-override")
            self.assertFalse(hasattr(options, "role"))

    def test_base_runner_returns_extra_allowed_read_roots_from_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            compiler_source = workspace / "compiler-sources" / "AscendNPU-IR"
            runner = _DummyRunner()
            request = AgentRequest(
                command_kind=CommandKind.GEN_TEST,
                input_path=workspace / "op.py",
                operator_path=workspace / "op.py",
                output_path=workspace / "test_op.py",
                test_mode=None,
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="dummy",
                skill_name="ascend-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
                compiler_source_path=compiler_source,
            )

            roots = runner._extra_allowed_read_roots(request)

            self.assertEqual(roots, (compiler_source,))

    def test_base_runner_cleans_backend_prepare_context_when_run_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            events: list[str] = []
            runner = _PreparedRunHookDummyRunner(events)
            request = AgentRequest(
                command_kind=CommandKind.GEN_TEST,
                input_path=workspace / "op.py",
                operator_path=workspace / "op.py",
                output_path=workspace / "test_op.py",
                test_mode=None,
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="dummy",
                skill_name="ascend-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
                enable_agent_hooks=True,
            )

            with (
                patch("triton_agent.backends.base.run_process", side_effect=RuntimeError("boom")),
            ):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    runner.run(request)

            self.assertEqual(events, ["hooks-enter", "hooks-exit"])

    def test_base_runner_retries_transient_failures_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = _DummyRunner()
            request = AgentRequest(
                command_kind=CommandKind.GEN_TEST,
                input_path=workspace / "op.py",
                operator_path=workspace / "op.py",
                output_path=workspace / "test_op.py",
                test_mode=None,
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="dummy",
                skill_name="ascend-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )

            with (
                patch.dict(environ, {}, clear=False),
                patch(
                    "triton_agent.backends.base.run_process",
                    side_effect=[
                        AgentResult(
                            return_code=1,
                            stdout="",
                            stderr="ERROR: exceeded retry limit, last status: 429 Too Many Requests",
                        ),
                        AgentResult(
                            return_code=1,
                            stdout="",
                            stderr="rate limit hit again",
                        ),
                        _ok_result(),
                    ],
                ) as mocked_run,
                patch("time.sleep") as mocked_sleep,
            ):
                result = runner.run(request)

            self.assertEqual(result.return_code, 0)
            self.assertEqual(mocked_run.call_count, 3)
            self.assertEqual([call.args[0] for call in mocked_sleep.call_args_list], [1.0, 2.0])

    def test_show_output_streams_rendered_chunks_directly_to_workspace_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = _DummyRunner()
            request = AgentRequest(
                command_kind=CommandKind.GEN_TEST,
                input_path=workspace / "op.py",
                operator_path=workspace / "op.py",
                output_path=workspace / "test_op.py",
                test_mode=None,
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=True,
                force_overwrite=False,
                agent_name="dummy",
                skill_name="ascend-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )

            def _run_process(*args, **kwargs):
                sink = kwargs["rendered_chunk_sink"]
                sink("first streamed output\n")
                return AgentResult(
                    return_code=0,
                    stdout="",
                    stderr="",
                    session_id="session-1",
                )

            with patch("triton_agent.backends.base.run_process", side_effect=_run_process):
                result = runner.run(request, stdout=StringIO())

            self.assertEqual(result.return_code, 0)
            log_path = workspace / "triton-agent-logs" / "gen-test.show-output.log"
            self.assertTrue(log_path.exists())
            content = log_path.read_text(encoding="utf-8")
            self.assertEqual(content, "first streamed output\n")
            self.assertEqual(result.stdout, "")

    def test_show_output_passes_incremental_sink_and_disables_stdout_collection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = _DummyRunner()
            request = AgentRequest(
                command_kind=CommandKind.GEN_TEST,
                input_path=workspace / "op.py",
                operator_path=workspace / "op.py",
                output_path=workspace / "test_op.py",
                test_mode=None,
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=True,
                force_overwrite=False,
                agent_name="claude",
                skill_name="ascend-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )

            with patch("triton_agent.backends.base.run_process", return_value=_ok_result()) as mocked:
                runner.run(request, stdout=StringIO())

            self.assertFalse(mocked.call_args.kwargs["collect_stdout"])
            self.assertIsNotNone(mocked.call_args.kwargs["rendered_chunk_sink"])

    def test_show_output_retries_from_explicit_retryable_failure_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = _DummyRunner()
            request = AgentRequest(
                command_kind=CommandKind.GEN_TEST,
                input_path=workspace / "op.py",
                operator_path=workspace / "op.py",
                output_path=workspace / "test_op.py",
                test_mode=None,
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=True,
                force_overwrite=False,
                agent_name="dummy",
                skill_name="ascend-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )

            with (
                patch.dict(environ, {"TRITON_AGENT_CODE_AGENT_MAX_RETRIES": "1"}, clear=False),
                patch(
                    "triton_agent.backends.base.run_process",
                    side_effect=[
                        AgentResult(
                            return_code=1,
                            stdout="",
                            stderr="",
                            retryable_failure=True,
                        ),
                        AgentResult(
                            return_code=0,
                            stdout="",
                            stderr="",
                            retryable_failure=False,
                        ),
                    ],
                ) as mocked_run,
                patch("time.sleep"),
            ):
                result = runner.run(request)

            self.assertEqual(result.return_code, 0)
            self.assertEqual(mocked_run.call_count, 2)

    def test_base_runner_honors_zero_retry_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = _DummyRunner()
            request = AgentRequest(
                command_kind=CommandKind.GEN_TEST,
                input_path=workspace / "op.py",
                operator_path=workspace / "op.py",
                output_path=workspace / "test_op.py",
                test_mode=None,
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="dummy",
                skill_name="ascend-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )

            transient = AgentResult(
                return_code=1,
                stdout="",
                stderr="ERROR: exceeded retry limit, last status: 429 Too Many Requests",
            )
            with (
                patch.dict(environ, {"TRITON_AGENT_CODE_AGENT_MAX_RETRIES": "0"}, clear=False),
                patch("triton_agent.backends.base.run_process", return_value=transient) as mocked_run,
                patch("time.sleep") as mocked_sleep,
            ):
                result = runner.run(request)

            self.assertEqual(result.return_code, 1)
            self.assertEqual(mocked_run.call_count, 1)
            mocked_sleep.assert_not_called()

    def test_base_runner_can_disable_shared_retry_per_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = _DummyRunner()
            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=workspace / "op.py",
                operator_path=workspace / "op.py",
                output_path=workspace / "opt_op.py",
                test_mode=None,
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="dummy",
                skill_name="triton-npu-optimize",
                prompt="Prompt body",
                workdir=workspace,
                disable_backend_retry=True,
            )

            transient = AgentResult(
                return_code=1,
                stdout="",
                stderr="ERROR: exceeded retry limit, last status: 429 Too Many Requests",
            )
            with (
                patch.dict(environ, {"TRITON_AGENT_CODE_AGENT_MAX_RETRIES": "5"}, clear=False),
                patch("triton_agent.backends.base.run_process", return_value=transient) as mocked_run,
                patch("time.sleep") as mocked_sleep,
            ):
                result = runner.run(request)

            self.assertEqual(result.return_code, 1)
            self.assertEqual(mocked_run.call_count, 1)
            mocked_sleep.assert_not_called()

    def test_base_runner_does_not_retry_interactive_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = _DummyRunner()
            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=workspace / "op.py",
                operator_path=workspace / "op.py",
                output_path=workspace / "opt_op.py",
                test_mode=None,
                bench_mode=None,
                interact=True,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="dummy",
                skill_name="triton-npu-optimize",
                prompt="Prompt body",
                workdir=workspace,
            )

            transient = AgentResult(
                return_code=1,
                stdout="",
                stderr="ERROR: exceeded retry limit, last status: 429 Too Many Requests",
            )
            with (
                patch.dict(environ, {"TRITON_AGENT_CODE_AGENT_MAX_RETRIES": "5"}, clear=False),
                patch("triton_agent.backends.base.run_process", return_value=transient) as mocked_run,
                patch("time.sleep") as mocked_sleep,
            ):
                result = runner.run(request)

            self.assertEqual(result.return_code, 1)
            self.assertEqual(mocked_run.call_count, 1)
            mocked_sleep.assert_not_called()

    def test_base_runner_rejects_invalid_retry_env_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = _DummyRunner()
            request = AgentRequest(
                command_kind=CommandKind.GEN_TEST,
                input_path=workspace / "op.py",
                operator_path=workspace / "op.py",
                output_path=workspace / "test_op.py",
                test_mode=None,
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="dummy",
                skill_name="ascend-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )

            with (
                patch.dict(environ, {"TRITON_AGENT_CODE_AGENT_MAX_RETRIES": "abc"}, clear=False),
                patch("triton_agent.backends.base.run_process", return_value=_ok_result()),
            ):
                with self.assertRaisesRegex(ValueError, "TRITON_AGENT_CODE_AGENT_MAX_RETRIES"):
                    runner.run(request)

    def test_base_runner_resume_uses_shared_optimize_resume_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = _DummyRunner()
            request = AgentRequest(
                command_kind=CommandKind.OPTIMIZE,
                input_path=workspace / "op.py",
                operator_path=workspace / "op.py",
                output_path=workspace / "opt_op.py",
                test_mode=None,
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="dummy",
                skill_name="triton-npu-optimize",
                prompt=build_prompt(
                    CommandKind.OPTIMIZE,
                    workspace / "op.py",
                    workspace / "op.py",
                    workspace / "opt_op.py",
                    "differential",
                    "standalone",
                    False,
                    round_mode="checked",
                ),
                workdir=workspace,
                round_mode="checked",
            )

            with patch("triton_agent.backends.base.run_process", return_value=_ok_result()) as mocked:
                runner.resume(request, "one round done")

            resumed_prompt = mocked.call_args.args[0][-1]
            self.assertIn("Continue the existing optimize task", resumed_prompt)
            self.assertIn("This invocation owns rounds 1 through 5.", resumed_prompt)
            self.assertIn(
                "Escalate analysis in this order: pattern triage, profiling diagnosis, IR attribution, compiler-source escalation.",
                resumed_prompt,
            )
            self.assertIn(
                "`learned_lessons.md` is only for reusable, evidence-backed optimization or profiling rules",
                resumed_prompt,
            )
            self.assertIn("Do not put round narrative, command failures, or operator-specific details", resumed_prompt)


class _DummyRunner(AgentRunner):
    def __init__(self) -> None:
        self.stall_timeout_seconds = 123

    def build_command(self, request: AgentRequest) -> list[str]:
        return ["dummy", request.prompt]


class _PreparedDummyRunner(_DummyRunner):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self._events = events

    @contextmanager
    def _prepare_run_context(
        self,
        request: AgentRequest,
        stderr: Optional[TextIO] = None,
    ):
        del request, stderr
        self._events.append("prepare-enter")
        try:
            yield
        finally:
            self._events.append("prepare-exit")

    def build_command(self, request: AgentRequest) -> list[str]:
        self._events.append("build-command")
        return super().build_command(request)


class _PreparedRunHookDummyRunner(_DummyRunner):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self._events = events

    @contextmanager
    def _prepare_run_context(
        self,
        request: AgentRequest,
        stderr: Optional[TextIO] = None,
    ):
        del request, stderr
        self._events.append("hooks-enter")
        try:
            yield
        finally:
            self._events.append("hooks-exit")


def _ok_result() -> AgentResult:
    return AgentResult(return_code=0, stdout="", stderr="")


if __name__ == "__main__":
    unittest.main()
