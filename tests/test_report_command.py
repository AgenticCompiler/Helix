from __future__ import annotations

import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.cli import build_parser
from triton_agent.commands.report import handle_report
from triton_agent.report.workspace import generate_workspace_report
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


class _DummyRunner:
    def __init__(self, captured: dict[str, AgentRequest], workspace: Path) -> None:
        self._captured = captured
        self._workspace = workspace

    def run(self, request: AgentRequest) -> AgentResult:
        self._captured["request"] = request
        (self._workspace / "report.md").write_text("report\n", encoding="utf-8")
        return AgentResult(return_code=0, stdout="", stderr="")


class ReportCommandTests(unittest.TestCase):
    def test_handle_report_uses_run_id_scoped_show_output_path(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            args = parser.parse_args(["report", "-i", str(workspace), "--show-output"])
            captured: dict[str, AgentRequest] = {}

            with patch(
                "triton_agent.commands.report.resolve_staged_skills",
                side_effect=_dummy_resolve_staged_skills,
            ), patch(
                "triton_agent.commands.report.SkillLinkManager",
                return_value=_DummySkillLinkManager(),
            ), patch(
                "triton_agent.commands.report.create_runner",
                return_value=_DummyRunner(captured, workspace),
            ):
                exit_code = handle_report(parser, args)

            self.assertEqual(exit_code, 0)
            request = captured["request"]
            self.assertTrue(request.run_id.startswith("report-"))
            self.assertEqual(request.workdir, workspace)
            self.assertEqual(
                show_output_log_path(request),
                workspace / "triton-agent-logs" / request.run_id / "show-output.log",
            )

    def test_generate_workspace_report_uses_run_id_scoped_show_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            captured: dict[str, AgentRequest] = {}

            with patch(
                "triton_agent.report.workspace.resolve_staged_skills",
                side_effect=_dummy_resolve_staged_skills,
            ), patch(
                "triton_agent.report.workspace.SkillLinkManager",
                return_value=_DummySkillLinkManager(),
            ), patch(
                "triton_agent.report.workspace.create_runner",
                return_value=_DummyRunner(captured, workspace),
            ):
                ok, message = generate_workspace_report(workspace, "codex", show_output=True)

            self.assertTrue(ok, message)
            request = captured["request"]
            self.assertTrue(request.run_id.startswith("report-"))
            self.assertEqual(
                show_output_log_path(request),
                workspace / "triton-agent-logs" / request.run_id / "show-output.log",
            )

    def test_handle_report_failure_with_show_output_uses_log_hint(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            args = parser.parse_args(["report", "-i", str(workspace), "--show-output"])
            stderr = StringIO()
            captured: dict[str, AgentRequest] = {}

            class FailingRunner:
                def run(self, request: AgentRequest) -> AgentResult:
                    captured["request"] = request
                    return AgentResult(return_code=1, stdout="", stderr="")

            with patch(
                "triton_agent.commands.report.resolve_staged_skills",
                side_effect=_dummy_resolve_staged_skills,
            ), patch(
                "triton_agent.commands.report.SkillLinkManager",
                return_value=_DummySkillLinkManager(),
            ), patch(
                "triton_agent.commands.report.create_runner",
                return_value=FailingRunner(),
            ), patch("sys.stderr", stderr):
                exit_code = handle_report(parser, args)

            self.assertEqual(exit_code, 1)
            request = captured["request"]
            self.assertIn(
                f"see show-output log: {show_output_log_path(request)}",
                stderr.getvalue(),
            )

    def test_generate_workspace_report_failure_with_show_output_uses_log_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            captured: dict[str, AgentRequest] = {}

            class FailingRunner:
                def run(self, request: AgentRequest) -> AgentResult:
                    captured["request"] = request
                    return AgentResult(return_code=1, stdout="", stderr="")

            with patch(
                "triton_agent.report.workspace.resolve_staged_skills",
                side_effect=_dummy_resolve_staged_skills,
            ), patch(
                "triton_agent.report.workspace.SkillLinkManager",
                return_value=_DummySkillLinkManager(),
            ), patch(
                "triton_agent.report.workspace.create_runner",
                return_value=FailingRunner(),
            ):
                ok, message = generate_workspace_report(workspace, "codex", show_output=True)

            self.assertFalse(ok)
            request = captured["request"]
            self.assertEqual(
                message,
                f"agent execution failed; see show-output log: {show_output_log_path(request)}",
            )
