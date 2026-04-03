import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.models import AgentRequest, AgentResult, CommandKind
from triton_agent.opencode_runner import OpenCodeRunner


class OpenCodeRunnerTests(unittest.TestCase):
    def test_non_interactive_command_uses_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = OpenCodeRunner()
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
                agent_name="opencode",
                skill_name="test-gen",
                prompt="Prompt body",
                workdir=workspace,
            )
            command = runner.build_command(request)
            self.assertEqual(command[:3], ["opencode", "run", "--dir"])
            self.assertIn("--pure", command)
            self.assertIn("--thinking", command)
            self.assertEqual(command[-1], "Prompt body")

    def test_interactive_command_uses_project_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = OpenCodeRunner()
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
                agent_name="opencode",
                skill_name="optimize",
                prompt="Continue work",
                workdir=workspace,
            )
            command = runner.build_command(request)
            self.assertEqual(command[0], "opencode")
            self.assertEqual(command[1], str(workspace))
            self.assertIn("--pure", command)
            self.assertIn("--thinking", command)
            self.assertIn("--prompt", command)

    def test_optimize_no_agent_session_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = OpenCodeRunner()
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
                agent_name="opencode",
                skill_name="optimize",
                prompt="Continue work",
                workdir=workspace,
                no_agent_session=True,
            )
            command = runner.build_command(request)
            self.assertEqual(command[:3], ["opencode", "run", "--dir"])
            self.assertNotIn("--no-session", command)
            self.assertNotIn("--ephemeral", command)

    def test_run_uses_unified_process_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = OpenCodeRunner()
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
                agent_name="opencode",
                skill_name="test-gen",
                prompt="Prompt body",
                workdir=workspace,
            )
            with patch("triton_agent.opencode_runner.run_process", return_value=_ok_result()) as mocked:
                runner.run(request)
            mocked.assert_called_once()

    def test_verbose_logging_prints_launch_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = OpenCodeRunner()
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
                agent_name="opencode",
                skill_name="test-gen",
                prompt="Prompt body",
                workdir=workspace,
            )
            stderr = StringIO()
            with patch("triton_agent.opencode_runner.run_process", return_value=_ok_result()):
                runner.run(request, stderr=stderr)
            self.assertIn("[agent]", stderr.getvalue())
            self.assertIn("opencode run", stderr.getvalue())
            self.assertIn("--pure", stderr.getvalue())
            self.assertIn("--thinking", stderr.getvalue())


def _ok_result() -> AgentResult:
    return AgentResult(return_code=0, stdout="", stderr="")


if __name__ == "__main__":
    unittest.main()
