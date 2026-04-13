import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.backends.codex import CodexRunner
from triton_agent.models import AgentRequest, AgentResult, CommandKind
from triton_agent.prompts import build_prompt


class CodexRunnerTests(unittest.TestCase):
    def test_non_interactive_command_uses_exec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = CodexRunner()
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
                agent_name="codex",
                skill_name="test-gen",
                prompt="Prompt body",
                workdir=workspace,
            )
            command = runner.build_command(request)
            self.assertEqual(command[:2], ["codex", "exec"])
            self.assertIn("--cd", command)
            self.assertIn("--ephemeral", command)
            self.assertIn("--skip-git-repo-check", command)
            sandbox_index = command.index("--sandbox")
            self.assertEqual(command[sandbox_index + 1], "danger-full-access")
            self.assertNotIn("--json", command)
            self.assertEqual(command[-1], "Prompt body")

    def test_run_test_non_interactive_uses_danger_full_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = CodexRunner()
            request = AgentRequest(
                command_kind=CommandKind.RUN_TEST,
                input_path=workspace / "test_op.py",
                operator_path=workspace / "op.py",
                output_path=None,
                test_mode="standalone",
                bench_mode=None,
                interact=False,
                verbose=False,
                show_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="",
                prompt="Run tests",
                workdir=workspace,
            )
            command = runner.build_command(request)
            sandbox_index = command.index("--sandbox")
            self.assertEqual(command[sandbox_index + 1], "danger-full-access")

    def test_run_bench_non_interactive_uses_danger_full_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = CodexRunner()
            request = AgentRequest(
                command_kind=CommandKind.RUN_BENCH,
                input_path=workspace / "bench_op.py",
                operator_path=workspace / "op.py",
                output_path=None,
                test_mode=None,
                bench_mode="standalone",
                interact=False,
                verbose=False,
                show_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="",
                prompt="Run benchmark",
                workdir=workspace,
            )
            command = runner.build_command(request)
            sandbox_index = command.index("--sandbox")
            self.assertEqual(command[sandbox_index + 1], "danger-full-access")

    def test_interactive_command_uses_tui(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = CodexRunner()
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
                agent_name="codex",
                skill_name="optimize",
                prompt="Continue work",
                workdir=workspace,
            )
            command = runner.build_command(request)
            self.assertEqual(command[0], "codex")
            self.assertNotIn("exec", command[:2])
            self.assertEqual(command[:3], ["codex", "--cd", str(workspace)])
            self.assertNotIn("--ephemeral", command)
            self.assertNotIn("--skip-git-repo-check", command)
            self.assertNotIn("--sandbox", command)
            self.assertNotIn("--ask-for-approval", command)
            self.assertEqual(command[-1], "Continue work")

    def test_optimize_non_interactive_omits_ephemeral_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = CodexRunner()
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
                prompt="Continue work",
                workdir=workspace,
                no_agent_session=False,
            )
            command = runner.build_command(request)
            self.assertNotIn("--ephemeral", command)
            self.assertIn("--skip-git-repo-check", command)

    def test_optimize_no_agent_session_adds_ephemeral(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = CodexRunner()
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
                prompt="Continue work",
                workdir=workspace,
                no_agent_session=True,
            )
            command = runner.build_command(request)
            self.assertIn("--ephemeral", command)

    def test_optimize_run_enables_graceful_interrupt_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = CodexRunner()
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
                prompt="Continue work",
                workdir=workspace,
            )
            with patch("triton_agent.backends.codex.run_process", return_value=_ok_result()) as mocked:
                runner.run(request)

            self.assertIsNotNone(mocked.call_args.kwargs["interrupt_policy"])

    def test_interactive_mode_uses_unified_process_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = CodexRunner()
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
                agent_name="codex",
                skill_name="optimize",
                prompt="Continue work",
                workdir=workspace,
            )
            with patch("triton_agent.backends.codex.run_process", return_value=_ok_result()) as mocked:
                runner.run(request)
            mocked.assert_called_once()

    def test_verbose_logging_prints_launch_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = CodexRunner()
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
                agent_name="codex",
                skill_name="test-gen",
                prompt="Prompt body",
                workdir=workspace,
            )
            stderr = StringIO()
            with patch("triton_agent.backends.codex.run_process", return_value=_ok_result()):
                result = runner.run(request, stderr=stderr)
            self.assertEqual(result.return_code, 0)
            self.assertIn("[agent]", stderr.getvalue())
            self.assertIn("command:", stderr.getvalue())
            self.assertIn("prompt:", stderr.getvalue())
            self.assertIn("codex exec", stderr.getvalue())
            self.assertIn("<prompt>", stderr.getvalue())

    def test_show_output_streams_non_interactive_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = CodexRunner()
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
                agent_name="codex",
                skill_name="test-gen",
                prompt="Prompt body",
                workdir=workspace,
            )
            with patch("triton_agent.backends.codex.run_process", return_value=_ok_result()) as mocked:
                result = runner.run(request)
            self.assertEqual(result.return_code, 0)
            mocked.assert_called_once()

    def test_show_output_uses_streaming_process_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = CodexRunner()
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
                agent_name="codex",
                skill_name="test-gen",
                prompt="Prompt body",
                workdir=workspace,
            )
            with patch("triton_agent.backends.codex.run_process", return_value=_ok_result()) as mocked:
                runner.run(request)
            mocked.assert_called_once()

    def test_buffered_mode_uses_buffered_process_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = CodexRunner()
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
                agent_name="codex",
                skill_name="test-gen",
                prompt="Prompt body",
                workdir=workspace,
            )
            with patch("triton_agent.backends.codex.run_process", return_value=_ok_result()) as mocked:
                runner.run(request)
            mocked.assert_called_once()

    def test_resume_prompt_preserves_base_context_and_supervised_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = CodexRunner()
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
                    supervise="on",
                ),
                workdir=workspace,
                min_rounds=3,
                require_analysis=True,
                supervise="on",
            )
            with patch("triton_agent.backends.codex.run_process", return_value=_ok_result()) as mocked:
                runner.resume(request, "one round done")

            resumed_request = mocked.call_args.args[0][-1]
            self.assertIn("Remote execution target: alice@example.com:2200", resumed_request)
            self.assertIn("Remote execution root: /tmp/remote", resumed_request)
            self.assertIn("This invocation is the optimize worker role.", resumed_request)
            self.assertIn("Continue the existing optimize task", resumed_request)
            self.assertIn("Read `opt-note.md`", resumed_request)
            self.assertIn("existing `opt-round-*` directories", resumed_request)
            self.assertIn("profiling or IR-backed evidence", resumed_request)


def _ok_result() -> AgentResult:
    return AgentResult(return_code=0, stdout="", stderr="")


if __name__ == "__main__":
    unittest.main()
