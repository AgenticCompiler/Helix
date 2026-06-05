import sys
import tempfile
import unittest
import json
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.backends.opencode import OpenCodeRunner
from triton_agent.models import AgentRequest, AgentResult, CommandKind
from triton_agent.prompts import build_prompt


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
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )
            command = runner.build_command(request)
            self.assertEqual(command[:3], ["opencode", "run", "--dir"])
            self.assertIn("--pure", command)
            self.assertIn("--thinking", command)
            self.assertEqual(command[-1], "Prompt body")

    def test_non_interactive_command_omits_pure_when_hooks_are_enabled(self) -> None:
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
                skill_name="triton-npu-optimize",
                prompt="Prompt body",
                workdir=workspace,
                enable_agent_hooks=True,
            )
            command = runner.build_command(request)
            self.assertEqual(command[:3], ["opencode", "run", "--dir"])
            self.assertNotIn("--pure", command)
            self.assertIn("--thinking", command)

    def test_non_interactive_command_omits_pure_when_log_tools_enabled(self) -> None:
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
                skill_name="triton-npu-optimize",
                prompt="Prompt body",
                workdir=workspace,
                log_tools=True,
            )
            command = runner.build_command(request)
            self.assertEqual(command[:3], ["opencode", "run", "--dir"])
            self.assertNotIn("--pure", command)
            self.assertIn("--thinking", command)

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
                skill_name="triton-npu-optimize",
                prompt="Continue work",
                workdir=workspace,
            )
            command = runner.build_command(request)
            self.assertEqual(command[0], "opencode")
            self.assertEqual(command[1], str(workspace))
            self.assertEqual(command[2], "--pure")
            self.assertEqual(command[3], "--prompt")
            self.assertEqual(command[4], "Continue work")
            self.assertIn("--pure", command)
            self.assertIn("--prompt", command)
            self.assertNotIn("--thinking", command)

    def test_interactive_command_omits_pure_when_hooks_are_enabled(self) -> None:
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
                skill_name="triton-npu-optimize",
                prompt="Continue work",
                workdir=workspace,
                enable_agent_hooks=True,
            )
            command = runner.build_command(request)
            self.assertEqual(command[0], "opencode")
            self.assertEqual(command[1], str(workspace))
            self.assertEqual(command[2], "--prompt")
            self.assertEqual(command[3], "Continue work")
            self.assertNotIn("--pure", command)
            self.assertIn("--prompt", command)
            self.assertNotIn("--thinking", command)

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
                skill_name="triton-npu-optimize",
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
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )
            with patch("triton_agent.backends.base.run_process", return_value=_ok_result()) as mocked:
                runner.run(request)
            mocked.assert_called_once()

    def test_run_stages_general_and_explore_subagent_deny_config_and_cleans_it_up(self) -> None:
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
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )

            def _inspect_config(*args, **kwargs):
                del args, kwargs
                config_path = workspace / ".opencode" / "opencode.json"
                self.assertTrue(config_path.exists())
                config = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertEqual(config["$schema"], "https://opencode.ai/config.json")
                self.assertEqual(config["agent"]["build"]["mode"], "primary")
                self.assertEqual(config["agent"]["plan"]["mode"], "primary")
                self.assertEqual(config["agent"]["build"]["permission"]["task"]["general"], "deny")
                self.assertEqual(config["agent"]["build"]["permission"]["task"]["explore"], "deny")
                self.assertEqual(config["agent"]["plan"]["permission"]["task"]["general"], "deny")
                self.assertEqual(config["agent"]["plan"]["permission"]["task"]["explore"], "deny")
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
            self.assertFalse((workspace / ".opencode" / "opencode.json").exists())

    def test_run_warns_and_skips_existing_opencode_workspace_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            config_path = workspace / ".opencode" / "opencode.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("{}\n", encoding="utf-8")
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
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )

            stderr = StringIO()
            with patch("triton_agent.backends.base.run_process", return_value=_ok_result()) as mocked:
                result = runner.run(request, stderr=stderr)

            self.assertEqual(result.return_code, 0)
            mocked.assert_called_once()
            self.assertIn("Warning:", stderr.getvalue())
            self.assertIn("Existing OpenCode workspace config", stderr.getvalue())
            self.assertEqual(config_path.read_text(encoding="utf-8"), "{}\n")

    def test_run_stages_mcp_server_config_into_workspace_config(self) -> None:
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
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
                mcp_servers=("triton-agent-run-eval",),
            )

            def _inspect_config(*args, **kwargs):
                del args, kwargs
                config_path = workspace / ".opencode" / "opencode.json"
                self.assertTrue(config_path.exists())
                config = json.loads(config_path.read_text(encoding="utf-8"))
                server = config["mcp"]["triton-agent-run-eval"]
                self.assertEqual(server["type"], "remote")
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
            self.assertFalse((workspace / ".opencode" / "opencode.json").exists())

    def test_run_stages_backend_hooks_when_enabled(self) -> None:
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
                skill_name="triton-npu-optimize",
                prompt="Continue work",
                workdir=workspace,
                enable_agent_hooks=True,
            )

            def _inspect_hooks(*args, **kwargs):
                del args, kwargs
                plugin_file = workspace / ".opencode" / "plugins" / "triton-agent-hook-guard.js"
                hook_dir = workspace / ".opencode" / "triton-agent-hooks"
                self.assertTrue(plugin_file.exists())
                self.assertTrue((hook_dir / "policy.json").exists())
                return _ok_result()

            with patch("triton_agent.backends.base.run_process", side_effect=_inspect_hooks):
                result = runner.run(request)

            self.assertEqual(result.return_code, 0)
            self.assertFalse((workspace / ".opencode" / "plugins" / "triton-agent-hook-guard.js").exists())
            self.assertFalse((workspace / ".opencode" / "triton-agent-hooks").exists())

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
                skill_name="triton-npu-gen-test",
                prompt="Prompt body",
                workdir=workspace,
            )
            stderr = StringIO()
            with patch("triton_agent.backends.base.run_process", return_value=_ok_result()):
                runner.run(request, stderr=stderr)
            self.assertIn("[command]", stderr.getvalue())
            self.assertIn("opencode run", stderr.getvalue())
            self.assertIn("--pure", stderr.getvalue())
            self.assertIn("--thinking", stderr.getvalue())

    def test_resume_prompt_preserves_base_context_and_supervised_contract(self) -> None:
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
