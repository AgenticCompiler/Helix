import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.backends.claude import ClaudeRunner
from triton_agent.models import AgentRequest, AgentResult, CommandKind
from triton_agent.prompts import build_prompt


class ClaudeRunnerTests(unittest.TestCase):
    def test_non_interactive_command_uses_print_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = ClaudeRunner()
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
                agent_name="claude",
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )
            command = runner.build_command(request)
            self.assertEqual(command[:2], ["claude", "--print"])
            self.assertIn("--dangerously-skip-permissions", command)
            self.assertIn("--output-format", command)
            self.assertIn("stream-json", command)
            self.assertIn("--verbose", command)
            self.assertEqual(command[-1], "Prompt body")

    def test_plain_non_interactive_gets_json_output_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = ClaudeRunner()
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
                agent_name="claude",
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
                log_tools=False,
            )
            output_filter = runner.output_filter(request)
            self.assertIsNotNone(output_filter)
            assert output_filter is not None
            rendered = output_filter.feed(json.dumps({
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "Created test_op.py."},
                    ],
                },
            }) + "\n", flush=True)
            self.assertIn("Created test_op.py.", rendered)
            self.assertNotIn('"type":"assistant"', rendered)

    def test_show_output_enables_stream_json_without_log_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = ClaudeRunner()
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
                agent_name="claude",
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
                log_tools=False,
            )
            command = runner.build_command(request)
            self.assertIn("--output-format", command)
            self.assertIn("stream-json", command)
            self.assertIn("--verbose", command)

    def test_show_output_gets_json_output_filter_without_log_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = ClaudeRunner()
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
                agent_name="claude",
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
                log_tools=False,
            )
            self.assertIsNotNone(runner.output_filter(request))

    def test_log_tools_enables_stream_json_without_show_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = ClaudeRunner()
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
                agent_name="claude",
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
                log_tools=True,
            )
            command = runner.build_command(request)
            self.assertIn("--output-format", command)
            self.assertIn("stream-json", command)
            self.assertIn("--verbose", command)

    def test_interactive_output_filter_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = ClaudeRunner()
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
                agent_name="claude",
                skill_name="triton-npu-optimize",
                prompt="Continue work",
                workdir=workspace,
            )
            self.assertIsNone(runner.output_filter(request))

    def test_interactive_command_uses_prompt_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = ClaudeRunner()
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
                agent_name="claude",
                skill_name="triton-npu-optimize",
                prompt="Continue work",
                workdir=workspace,
            )
            command = runner.build_command(request)
            self.assertEqual(command[0], "claude")
            self.assertNotIn("--print", command)
            self.assertNotIn("--dangerously-skip-permissions", command)
            self.assertEqual(command[-1], "Continue work")

    def test_optimize_no_agent_session_adds_no_session_persistence_in_print_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = ClaudeRunner()
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
                agent_name="claude",
                skill_name="triton-npu-optimize",
                prompt="Continue work",
                workdir=workspace,
                no_agent_session=True,
            )
            command = runner.build_command(request)
            self.assertIn("--no-session-persistence", command)

    def test_optimize_no_agent_session_is_ignored_in_interactive_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = ClaudeRunner()
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
                agent_name="claude",
                skill_name="triton-npu-optimize",
                prompt="Continue work",
                workdir=workspace,
                no_agent_session=True,
            )
            command = runner.build_command(request)
            self.assertNotIn("--no-session-persistence", command)

    def test_run_uses_unified_process_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = ClaudeRunner()
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
                agent_name="claude",
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )
            with patch("triton_agent.backends.base.run_process", return_value=_ok_result()) as mocked:
                runner.run(request)
            mocked.assert_called_once()
            self.assertIsNotNone(mocked.call_args.kwargs["output_filter"])

    def test_run_stages_mcp_server_config_and_passes_mcp_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = ClaudeRunner()
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
                agent_name="claude",
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
                mcp_servers=("triton-agent-run-eval",),
            )

            def _inspect_run(*args, **kwargs):
                command = args[0]
                config_path = workspace / ".claude" / "mcp.json"
                self.assertTrue(config_path.exists())
                self.assertIn("--mcp-config", command)
                self.assertIn(str(config_path), command)
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                server = payload["mcpServers"]["triton-agent-run-eval"]
                self.assertEqual(server["type"], "http")
                self.assertTrue(server["url"].startswith("http://127.0.0.1:"))
                self.assertIn("/mcp?workspace=", server["url"])
                self.assertIn(str(workspace), server["url"])
                return _ok_result()

            with patch.dict(
                "os.environ",
                {
                    "TRITON_AGENT_BATCH_NPU_DEVICES": "0,1",
                    "TRITON_AGENT_BATCH_WORKERS_PER_NPU": "2",
                },
                clear=False,
            ):
                with patch("triton_agent.backends.base.run_process", side_effect=_inspect_run):
                    result = runner.run(request)

            self.assertEqual(result.return_code, 0)
            self.assertFalse((workspace / ".claude" / "mcp.json").exists())

    def test_verbose_logging_prints_launch_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = ClaudeRunner()
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
                agent_name="claude",
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )
            stderr = StringIO()
            with patch("triton_agent.backends.base.run_process", return_value=_ok_result()):
                result = runner.run(request, stderr=stderr)
            self.assertEqual(result.return_code, 0)
            self.assertIn("[command]", stderr.getvalue())
            self.assertIn("claude --print", stderr.getvalue())
            self.assertIn("--dangerously-skip-permissions", stderr.getvalue())

    def test_resume_prompt_preserves_base_context_and_supervised_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = ClaudeRunner()
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
                agent_name="claude",
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
