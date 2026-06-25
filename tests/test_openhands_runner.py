import os
import io
import sys
import tempfile
import types
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Optional
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.backends.openhands import OpenHandsRunner, OpenHandsSetupError
from triton_agent.models import AgentRequest, AgentResult, CommandKind
from triton_agent.prompts import build_prompt


class OpenHandsRunnerTests(unittest.TestCase):
    def test_run_rejects_interactive_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = OpenHandsRunner()

            result = runner.run(self._request(workspace, interact=True))

            self.assertEqual(result.return_code, 1)
            self.assertIn("does not support --interact", result.stderr)

    def test_run_requires_llm_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = OpenHandsRunner()

            with patch(
                "triton_agent.backends.openhands._supports_openhands_runtime",
                return_value=True,
            ):
                with patch.dict(os.environ, {"LLM_MODEL": "gpt-5.4-mini"}, clear=True):
                    result = runner.run(self._request(workspace))

            self.assertEqual(result.return_code, 1)
            self.assertIn("LLM_API_KEY", result.stderr)

    def test_run_requires_python_312(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = OpenHandsRunner()

            with patch(
                "triton_agent.backends.openhands._supports_openhands_runtime",
                return_value=False,
            ):
                result = runner.run(self._request(workspace))

            self.assertEqual(result.return_code, 1)
            self.assertIn("Python 3.12", result.stderr)

    def test_run_requires_llm_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = OpenHandsRunner()

            with patch(
                "triton_agent.backends.openhands._supports_openhands_runtime",
                return_value=True,
            ):
                with patch.dict(os.environ, {"LLM_API_KEY": "secret"}, clear=True):
                    result = runner.run(self._request(workspace))

            self.assertEqual(result.return_code, 1)
            self.assertIn("LLM_MODEL", result.stderr)

    def test_run_returns_import_error_when_openhands_packages_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = OpenHandsRunner()
            (workspace / ".openhands" / "skills").mkdir(parents=True)

            with patch(
                "triton_agent.backends.openhands._supports_openhands_runtime",
                return_value=True,
            ):
                with patch.dict(
                    os.environ,
                    {
                        "LLM_API_KEY": "secret",
                        "LLM_MODEL": "gpt-5.4-mini",
                    },
                    clear=True,
                ):
                    with patch(
                        "triton_agent.backends.openhands._load_openhands_dependencies",
                        side_effect=OpenHandsSetupError(
                            "OpenHands backend requires the openhands-sdk and openhands-tools packages to be installed."
                        ),
                    ):
                        result = runner.run(self._request(workspace))

            self.assertEqual(result.return_code, 1)
            self.assertIn("openhands-sdk", result.stderr)

    def test_run_maps_successful_sdk_execution_to_agent_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = OpenHandsRunner()
            (workspace / ".openhands" / "skills").mkdir(parents=True)

            with patch(
                "triton_agent.backends.openhands._supports_openhands_runtime",
                return_value=True,
            ):
                with patch.dict(
                    os.environ,
                    {
                        "LLM_API_KEY": "secret",
                        "LLM_MODEL": "gpt-5.4-mini",
                        "LLM_BASE_URL": "https://example.invalid/v1/chat/completions",
                    },
                    clear=True,
                ):
                    with patch(
                        "triton_agent.backends.openhands._load_openhands_dependencies",
                        return_value=_fake_dependencies(),
                    ):
                        result = runner.run(self._request(workspace))

            self.assertEqual(result.return_code, 0)
            self.assertIn("assistant final", result.stdout)
            self.assertEqual(result.stderr, "")

    def test_show_output_writes_event_text_to_workspace_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = OpenHandsRunner()
            (workspace / ".openhands" / "skills").mkdir(parents=True)

            with patch(
                "triton_agent.backends.openhands._supports_openhands_runtime",
                return_value=True,
            ):
                with patch.dict(
                    os.environ,
                    {
                        "LLM_API_KEY": "secret",
                        "LLM_MODEL": "gpt-5.4-mini",
                    },
                    clear=True,
                ):
                    with patch(
                        "triton_agent.backends.openhands._load_openhands_dependencies",
                        return_value=_fake_dependencies(),
                    ):
                        result = runner.run(self._request(workspace, show_output=True), stdout=io.StringIO())

            self.assertEqual(result.return_code, 0)
            self.assertEqual(result.stdout, "")
            log_path = workspace / "triton-agent-logs" / "gen-test.show-output.log"
            self.assertTrue(log_path.exists())
            content = log_path.read_text(encoding="utf-8")
            self.assertNotIn("attempt=1", content)
            self.assertIn("assistant update", content)
            self.assertIn("assistant final", content)

    def test_run_normalizes_chat_completions_base_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = OpenHandsRunner()
            (workspace / ".openhands" / "skills").mkdir(parents=True)
            fake_dependencies = _fake_dependencies()

            with patch(
                "triton_agent.backends.openhands._supports_openhands_runtime",
                return_value=True,
            ):
                with patch.dict(
                    os.environ,
                    {
                        "LLM_API_KEY": "secret",
                        "LLM_MODEL": "gpt-5.4-mini",
                        "LLM_BASE_URL": "https://example.invalid/v1/chat/completions",
                    },
                    clear=True,
                ):
                    with patch(
                        "triton_agent.backends.openhands._load_openhands_dependencies",
                        return_value=fake_dependencies,
                    ):
                        result = runner.run(self._request(workspace))

            self.assertEqual(result.return_code, 0)
            self.assertEqual(
                fake_dependencies.state["last_llm_kwargs"]["base_url"],
                "https://example.invalid/v1",
            )

    def test_resume_uses_shared_optimize_resume_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = OpenHandsRunner()
            request = self._request(
                workspace,
                command_kind=CommandKind.OPTIMIZE,
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
            )

            with patch.object(runner, "run", return_value=AgentResult(0, "", "")) as mocked:
                runner.resume(request, "one round done")

            resumed_request = mocked.call_args.args[0]
            self.assertIn("Continue the existing optimize task", resumed_request.prompt)
            self.assertIn("This invocation owns rounds 1 through 5.", resumed_request.prompt)

    def test_run_does_not_inject_repo_agents_file_into_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = OpenHandsRunner()
            (workspace / ".openhands" / "skills").mkdir(parents=True)
            observed: dict[str, object] = {}

            def _load_project_skills(*, work_dir: str) -> list[object]:
                observed["agents_exists_during_run"] = (Path(work_dir) / "AGENTS.md").exists()
                return ["project-rule"]

            fake_dependencies = _fake_dependencies()
            fake_dependencies.load_project_skills = _load_project_skills

            with patch(
                "triton_agent.backends.openhands._supports_openhands_runtime",
                return_value=True,
            ):
                with patch.dict(
                    os.environ,
                    {
                        "LLM_API_KEY": "secret",
                        "LLM_MODEL": "gpt-5.4-mini",
                    },
                    clear=True,
                ):
                    with patch(
                        "triton_agent.backends.openhands._load_openhands_dependencies",
                        return_value=fake_dependencies,
                    ):
                        result = runner.run(self._request(workspace))

            self.assertEqual(result.return_code, 0)
            self.assertFalse(observed["agents_exists_during_run"])
            self.assertFalse((workspace / "AGENTS.md").exists())

    def test_run_requires_project_skills_loader_to_include_staged_openhands_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = OpenHandsRunner()
            skill_file = workspace / ".openhands" / "skills" / "demo" / "SKILL.md"
            skill_file.parent.mkdir(parents=True)
            skill_file.write_text("# demo\n", encoding="utf-8")
            fake_dependencies = _fake_dependencies()
            fake_dependencies.load_project_skills = lambda work_dir: [
                types.SimpleNamespace(name="agents", source=str(Path(work_dir) / "AGENTS.md"))
            ]

            with patch(
                "triton_agent.backends.openhands._supports_openhands_runtime",
                return_value=True,
            ):
                with patch.dict(
                    os.environ,
                    {
                        "LLM_API_KEY": "secret",
                        "LLM_MODEL": "gpt-5.4-mini",
                    },
                    clear=True,
                ):
                    with patch(
                        "triton_agent.backends.openhands._load_openhands_dependencies",
                        return_value=fake_dependencies,
                    ):
                        result = runner.run(self._request(workspace))

            self.assertEqual(result.return_code, 1)
            self.assertIn("did not include staged skills", result.stderr)
            self.assertIn(str(skill_file), result.stderr)

    def test_run_logs_actual_sdk_command_and_prompt_in_verbose_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runner = OpenHandsRunner()
            (workspace / ".openhands" / "skills").mkdir(parents=True)
            stderr = io.StringIO()

            with patch(
                "triton_agent.backends.openhands._supports_openhands_runtime",
                return_value=True,
            ):
                with patch.dict(
                    os.environ,
                    {
                        "LLM_API_KEY": "secret",
                        "LLM_MODEL": "gpt-5.4-mini",
                    },
                    clear=True,
                ):
                    with patch(
                        "triton_agent.backends.openhands._load_openhands_dependencies",
                        return_value=_fake_dependencies(),
                    ):
                        result = runner.run(
                            self._request(
                                workspace,
                                prompt="first line\nsecond line",
                                verbose=True,
                            ),
                            stderr=stderr,
                        )

            self.assertEqual(result.return_code, 0)
            self.assertIn("[command]", stderr.getvalue())
            self.assertIn("openhands sdk", stderr.getvalue())
            self.assertIn("prompt:", stderr.getvalue())
            self.assertIn("  first line", stderr.getvalue())
            self.assertIn("  second line", stderr.getvalue())
            self.assertNotIn("  sdk", stderr.getvalue())
            self.assertNotIn("[command] command:", stderr.getvalue())
            self.assertNotIn("[command]   first line", stderr.getvalue())
            self.assertNotIn("[command]   second line", stderr.getvalue())

    def _request(
        self,
        workspace: Path,
        *,
        command_kind: CommandKind = CommandKind.GEN_TEST,
        interact: bool = False,
        verbose: bool = False,
        show_output: bool = False,
        prompt: str = "Prompt body",
    ) -> AgentRequest:
        return AgentRequest(
            command_kind=command_kind,
            input_path=workspace / "op.py",
            operator_path=workspace / "op.py",
            output_path=workspace / "test_op.py",
            test_mode=None,
            bench_mode=None,
            interact=interact,
            verbose=verbose,
            stream_output=show_output,
            force_overwrite=False,
            agent_name="openhands",
            skill_name="ascend-npu-gen-test",
            prompt=prompt,
            workdir=workspace,
        )


def _fake_dependencies() -> types.SimpleNamespace:
    class FakeSecretStr:
        def __init__(self, value: str) -> None:
            self.value = value

    class FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            state["last_llm_kwargs"] = kwargs

    class FakeTool:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeAgentContext:
        def __init__(self, skills: list[object]) -> None:
            self.skills = skills

    class FakeAgent:
        def __init__(
            self,
            *,
            llm: object,
            tools: list[object],
            agent_context: object,
            condenser: object,
        ) -> None:
            self.llm = llm
            self.tools = tools
            self.agent_context = agent_context
            self.condenser = condenser

    class FakeCondenser:
        def __init__(self, *, llm: object) -> None:
            self.llm = llm

    class FakeNeverConfirm:
        pass

    class FakeEvent:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeConversation:
        def __init__(
            self,
            *,
            agent: object,
            workspace: str,
            callbacks: list[Callable[[object], None]],
        ) -> None:
            self.agent = agent
            self.workspace = workspace
            self.callbacks = callbacks
            self.prompt: Optional[str] = None
            self.confirmation_policy: Optional[object] = None

        def set_confirmation_policy(self, policy: object) -> None:
            self.confirmation_policy = policy

        def send_message(self, prompt: str) -> None:
            self.prompt = prompt

        def run(self) -> None:
            for callback in self.callbacks:
                callback(FakeEvent("assistant update"))
            for callback in self.callbacks:
                callback(FakeEvent("assistant final"))

    class FakeTerminalTool:
        name = "terminal"

    class FakeFileEditorTool:
        name = "file_editor"

    class FakeTaskTrackerTool:
        name = "task_tracker"

    state: dict[str, object] = {}

    return types.SimpleNamespace(
        SecretStr=FakeSecretStr,
        LLM=FakeLLM,
        Tool=FakeTool,
        AgentContext=FakeAgentContext,
        Agent=FakeAgent,
        Conversation=FakeConversation,
        LLMSummarizingCondenser=FakeCondenser,
        load_project_skills=lambda work_dir: ["project-rule", work_dir],
        NeverConfirm=FakeNeverConfirm,
        TerminalTool=FakeTerminalTool,
        FileEditorTool=FakeFileEditorTool,
        TaskTrackerTool=FakeTaskTrackerTool,
        state=state,
    )


if __name__ == "__main__":
    unittest.main()
