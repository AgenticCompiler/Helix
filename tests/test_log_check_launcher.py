import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.log_check.batch import run_log_check_batch
from triton_agent.log_check.log_check_launcher import build_log_check_prompt, run_log_check
from triton_agent.models import AgentRequest, AgentResult
from triton_agent.show_output_log import show_output_log_path
from triton_agent.skills import SkillLinkSet


def _dummy_resolve_staged_skills(*args, **kwargs):
    return None, None


class _DummySkillLinkManager:
    def prepare_skills(self, agent_name, workdir, *, skill_names=None, skill_sources=None):
        return SkillLinkSet(created_paths=[])

    def describe_prepare(self, links):
        return []

    def describe_cleanup(self, links):
        return []

    def cleanup(self, links):
        return []


class LogCheckLauncherTests(unittest.TestCase):
    def test_log_check_prompt_uses_staged_patterns_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "LayerNorm"
            target.mkdir()

            prompt = build_log_check_prompt(target_path=target, agent_name="codex")
            staged_patterns = ".codex/skills/triton-npu-optimize-knowledge/references/patterns"

            self.assertIn(staged_patterns, prompt)

    def test_run_log_check_uses_target_workspace_as_agent_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "LayerNorm"
            target.mkdir()
            resolved_target = target.resolve()

            captured: dict[str, AgentRequest] = {}

            class DummyRunner:
                def run(self, request: AgentRequest) -> AgentResult:
                    captured["request"] = request
                    (target / "log_check_result.md").write_text("summary:\noverall: PASS\n", encoding="utf-8")
                    return AgentResult(return_code=0, stdout="", stderr="")

            with patch(
                "triton_agent.log_check.log_check_launcher.resolve_staged_skills",
                side_effect=_dummy_resolve_staged_skills,
            ), patch(
                "triton_agent.log_check.log_check_launcher.SkillLinkManager",
                return_value=_DummySkillLinkManager(),
            ), patch(
                "triton_agent.log_check.log_check_launcher.create_runner",
                return_value=DummyRunner(),
            ):
                exit_code = run_log_check(target_path=target, agent_name="opencode")

            self.assertEqual(exit_code, 0)
            request = captured["request"]
            self.assertEqual(request.workdir, resolved_target)
            self.assertEqual(
                show_output_log_path(request),
                resolved_target / "triton-agent-logs" / "log-check.show-output.log",
            )

    def test_run_log_check_batch_uses_each_workspace_as_agent_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "LayerNorm"
            workspace.mkdir()
            (workspace / "opt-note.md").write_text("history\n", encoding="utf-8")

            captured: dict[str, AgentRequest] = {}

            class DummyRunner:
                def run(self, request: AgentRequest) -> AgentResult:
                    captured["request"] = request
                    (request.workdir / "log_check_result.md").write_text(
                        "summary:\noverall: PASS\n",
                        encoding="utf-8",
                    )
                    return AgentResult(return_code=0, stdout="", stderr="")

            with patch(
                "triton_agent.log_check.log_check_launcher.resolve_staged_skills",
                side_effect=_dummy_resolve_staged_skills,
            ), patch(
                "triton_agent.log_check.log_check_launcher.SkillLinkManager",
                return_value=_DummySkillLinkManager(),
            ), patch(
                "triton_agent.log_check.log_check_launcher.create_runner",
                return_value=DummyRunner(),
            ):
                exit_code = run_log_check_batch(root, stdout=StringIO())

            self.assertEqual(exit_code, 0)
            request = captured["request"]
            self.assertEqual(request.workdir, workspace.resolve())
            self.assertTrue((root / "log_check_summary.md").exists())
