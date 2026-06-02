import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.backends.traecli import TraeCLIRunner
from triton_agent.models import AgentRequest, AgentResult, CommandKind
from triton_agent.prompts import build_prompt


class TraeCLIRunnerTests(unittest.TestCase):
    def test_non_interactive_command_uses_print_and_yolo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = TraeCLIRunner()
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
                agent_name="traecli",
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )
            command = runner.build_command(request)
            self.assertEqual(command, ["traecli", "--print", "--yolo", "Prompt body"])

    def test_interactive_command_uses_prompt_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = TraeCLIRunner()
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
                agent_name="traecli",
                skill_name="triton-npu-optimize",
                prompt="Continue work",
                workdir=workspace,
            )
            command = runner.build_command(request)
            self.assertEqual(command, ["traecli", "Continue work"])

    def test_optimize_no_agent_session_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = TraeCLIRunner()
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
                agent_name="traecli",
                skill_name="triton-npu-optimize",
                prompt="Continue work",
                workdir=workspace,
                no_agent_session=True,
            )
            command = runner.build_command(request)
            self.assertEqual(command, ["traecli", "--print", "--yolo", "Continue work"])

    def test_run_uses_unified_process_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = TraeCLIRunner()
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
                agent_name="traecli",
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )
            with patch("triton_agent.backends.base.run_process", return_value=_ok_result()) as mocked:
                runner.run(request)
            mocked.assert_called_once()

    def test_verbose_logging_prints_launch_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = TraeCLIRunner()
            request = AgentRequest(
                command_kind=CommandKind.GEN_TEST,
                input_path=workspace / "op.py",
                operator_path=workspace / "op.py",
                output_path=workspace / "test_op.py",
                test_mode=None,
                bench_mode=None,
                interact=False,
                verbose=True,
                show_output=False,
                force_overwrite=False,
                agent_name="traecli",
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )
            stderr = StringIO()
            with patch("triton_agent.backends.base.run_process", return_value=_ok_result()):
                result = runner.run(request, stderr=stderr)
            self.assertEqual(result.return_code, 0)
            self.assertIn("[command]", stderr.getvalue())
            self.assertIn("traecli --print --yolo", stderr.getvalue())

    def test_resume_prompt_preserves_base_context_and_supervised_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = TraeCLIRunner()
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
                agent_name="traecli",
                skill_name="triton-npu-optimize",
                prompt=build_prompt(
                    CommandKind.OPTIMIZE,
                    workspace / "op.py",
                    workspace / "op.py",
                    workspace / "opt_op.py",
                    "differential",
                    "standalone",
                    False,
                    remote="alice@example.com:2200",
                    remote_workdir="/tmp/remote",
                    round_mode="checked",
                ),
                workdir=workspace,
                min_rounds=3,
                round_mode="checked",
            )
            with patch("triton_agent.backends.base.run_process", return_value=_ok_result()) as mocked:
                runner.resume(request, "one round done")

            resumed_request = mocked.call_args.args[0][-1]
            self.assertIn("This invocation owns exactly one round.", resumed_request)
            self.assertIn("Continue the existing optimize task", resumed_request)
            self.assertIn("Read `opt-note.md`", resumed_request)
            self.assertIn("existing `opt-round-*` directories", resumed_request)
            self.assertIn(
                "Escalate analysis in this order: pattern triage, profiling diagnosis, IR attribution, compiler-source escalation.",
                resumed_request,
            )


def _ok_result() -> AgentResult:
    return AgentResult(return_code=0, stdout="", stderr="")


if __name__ == "__main__":
    unittest.main()
