import sys
import tempfile
import tomllib
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.backends.codex import CodexRunner, _extract_session_id
from triton_agent.backends.codex_trace import CodexJsonOutputFilter
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
                skill_name="triton-npu-gen-test",
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

    def test_show_output_command_uses_json_stdout(self) -> None:
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
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )

            command = runner.build_command(request)

            self.assertIn("--json", command)

    def test_show_output_uses_codex_json_output_filter_without_log_tools(self) -> None:
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
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )

            output_filter = runner.output_filter(request)

            self.assertIsInstance(output_filter, CodexJsonOutputFilter)

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
                skill_name="triton-npu-optimize",
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
                skill_name="triton-npu-optimize",
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
                skill_name="triton-npu-optimize",
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
                skill_name="triton-npu-optimize",
                prompt="Continue work",
                workdir=workspace,
            )
            with patch("triton_agent.backends.base.run_process", return_value=_ok_result()) as mocked:
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
                skill_name="triton-npu-optimize",
                prompt="Continue work",
                workdir=workspace,
            )
            with patch("triton_agent.backends.base.run_process", return_value=_ok_result()) as mocked:
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
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )
            stderr = StringIO()
            with patch("triton_agent.backends.base.run_process", return_value=_ok_result()):
                result = runner.run(request, stderr=stderr)
            self.assertEqual(result.return_code, 0)
            self.assertIn("[command]", stderr.getvalue())
            self.assertIn("codex exec", stderr.getvalue())
            self.assertIn("prompt:", stderr.getvalue())
            self.assertIn("<prompt>", stderr.getvalue())
            self.assertNotIn("[command] command:", stderr.getvalue())
            self.assertIn("\n  Prompt body\n", stderr.getvalue())
            self.assertNotIn("[command]   Prompt body", stderr.getvalue())

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
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )
            with patch("triton_agent.backends.base.run_process", return_value=_ok_result()) as mocked:
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
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )
            with patch("triton_agent.backends.base.run_process", return_value=_ok_result()) as mocked:
                runner.run(request)
            mocked.assert_called_once()

    def test_run_stages_mcp_server_config_when_requested(self) -> None:
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
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
                mcp_servers=("triton-agent-run-eval",),
            )

            def _inspect_config(*args, **kwargs):
                del args, kwargs
                config_path = workspace / ".codex" / "config.toml"
                self.assertTrue(config_path.exists())
                content = config_path.read_text(encoding="utf-8")
                self.assertIn("[mcp_servers.triton-agent-run-eval]", content)
                parsed = tomllib.loads(content)
                server = parsed["mcp_servers"]["triton-agent-run-eval"]
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
                with patch("triton_agent.backends.base.run_process", side_effect=_inspect_config):
                    result = runner.run(request)

            self.assertEqual(result.return_code, 0)
            self.assertFalse((workspace / ".codex" / "config.toml").exists())

    def test_buffered_output_filters_bare_hunk_fragments_from_result(self) -> None:
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
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )
            helper = workspace / "emit_sample.py"
            helper.write_text(
                "import sys\n"
                "chunks = [\n"
                "    'baseline/perf.txt 368.0119\\n',\n"
                "    'opt-round-1/opt_triton_10_SwigluQuant_perf.txt 299.39092500000004\\n\\n',\n"
                "    '     tl.store(scale_ptr + row, inv_scale)\\n\\n',\n"
                "    '@@ -10,0 +11,3 @@\\n',\n"
                "    '+@triton.jit\\n',\n"
                "    '+def _round_half_to_even_tl(values):\\n',\n"
                "    '+    abs_values = tl.abs(values)\\n',\n"
                "    'done\\n',\n"
                "]\n"
                "for chunk in chunks:\n"
                "    sys.stdout.write(chunk)\n"
                "    sys.stdout.flush()\n",
                encoding="utf-8",
            )

            with patch.object(runner, "build_command", return_value=[sys.executable, str(helper)]):
                result = runner.run(request)

            self.assertEqual(result.return_code, 0)
            result_text = _normalize_newlines(result.stdout)

            self.assertIn("baseline/perf.txt 368.0119", result_text)
            self.assertIn("opt-round-1/opt_triton_10_SwigluQuant_perf.txt 299.39092500000004", result_text)
            self.assertIn("     tl.store(scale_ptr + row, inv_scale)", result_text)
            self.assertIn("done", result_text)
            self.assertNotIn("@@ -10,0 +11,3 @@", result_text)
            self.assertNotIn("+@triton.jit", result_text)
            self.assertNotIn("+def _round_half_to_even_tl(values):", result_text)

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
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )
            with patch("triton_agent.backends.base.run_process", return_value=_ok_result()) as mocked:
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

    def test_session_id_extractor_reads_codex_startup_text(self) -> None:
        line = "session id: 019da9c2-dfcb-7c71-a2f9-7a90bab2e0f5\n"

        self.assertEqual(
            _extract_session_id(line),
            "019da9c2-dfcb-7c71-a2f9-7a90bab2e0f5",
        )


def _ok_result() -> AgentResult:
    return AgentResult(return_code=0, stdout="", stderr="")


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


if __name__ == "__main__":
    unittest.main()
