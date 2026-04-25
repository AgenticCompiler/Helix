import sys
import tempfile
import unittest
from os import environ
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.backends.base import AgentRunner
from triton_agent.models import AgentRequest, AgentResult, CommandKind
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
                show_output=True,
                force_overwrite=False,
                agent_name="dummy",
                skill_name="triton-npu-gen-test",
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
                show_output=False,
                force_overwrite=False,
                agent_name="dummy",
                skill_name="triton-npu-gen-test",
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
                show_output=False,
                force_overwrite=False,
                agent_name="dummy",
                skill_name="triton-npu-gen-test",
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
                show_output=False,
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
                show_output=False,
                force_overwrite=False,
                agent_name="dummy",
                skill_name="triton-npu-gen-test",
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
                show_output=False,
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
                    supervise="on",
                ),
                workdir=workspace,
                supervise="on",
            )

            with patch("triton_agent.backends.base.run_process", return_value=_ok_result()) as mocked:
                runner.resume(request, "one round done")

            resumed_prompt = mocked.call_args.args[0][-1]
            self.assertIn("Continue the existing optimize task", resumed_prompt)
            self.assertIn("This invocation is the optimize worker role.", resumed_prompt)
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


def _ok_result() -> AgentResult:
    return AgentResult(return_code=0, stdout="", stderr="")


if __name__ == "__main__":
    unittest.main()
