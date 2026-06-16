import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.cli import build_parser
from triton_agent.commands.report import handle_report
from triton_agent.report.workspace import build_hardware_info_text
from triton_agent.commands.report_batch import handle_report_batch
from triton_agent.models import AgentResult


class ReportCommandHandlerTests(unittest.TestCase):
    def test_build_hardware_info_text_omits_target_chip(self) -> None:
        with patch("triton_agent.hardware_info.capture_hardware_info", return_value={}):
            text = build_hardware_info_text()

        self.assertEqual(text, "")

    def test_build_hardware_info_text_queries_hardware_without_target_chip_argument(self) -> None:
        with patch(
            "triton_agent.hardware_info.capture_hardware_info",
            return_value={"chip_name": "Ascend 910B"},
        ) as mocked:
            text = build_hardware_info_text()

        mocked.assert_called_once_with()
        self.assertIn("- chip_name: Ascend 910B", text)

    def test_handle_report_includes_hardware_info_in_prompt(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            args = parser.parse_args(["report", "-i", str(workspace)])
            captured: dict[str, str] = {}

            def _fake_run(request):
                captured["prompt"] = request.prompt
                return AgentResult(return_code=0, stdout="", stderr="")

            fake_runner = SimpleNamespace(run=_fake_run)

            with patch("triton_agent.commands.report.build_prompt", return_value="Prompt body"):
                with patch(
                    "triton_agent.hardware_info.capture_hardware_info",
                    return_value={
                        "chip_name": "Ascend 910B",
                        "cann_version": "8.1.RC1",
                        "driver_version": "24.1",
                    },
                ):
                    with patch("triton_agent.commands.report.create_runner", return_value=fake_runner):
                        with patch("triton_agent.commands.report.resolve_staged_skills", return_value=((), {})):
                            with patch(
                                "triton_agent.commands.report.SkillLinkManager.prepare_skills",
                                return_value=(),
                            ):
                                with patch(
                                    "triton_agent.commands.report.SkillLinkManager.cleanup",
                                    return_value=[],
                                ):
                                    with patch("triton_agent.commands.report.render_result"):
                                        exit_code = handle_report(parser, args)

        self.assertEqual(exit_code, 0)
        self.assertIn("Hardware environment information:", captured["prompt"])
        self.assertIn("- chip_name: Ascend 910B", captured["prompt"])
        self.assertIn("- cann_version: 8.1.RC1", captured["prompt"])
        self.assertIn("- driver_version: 24.1", captured["prompt"])

    def test_report_command_does_not_accept_target_chip(self) -> None:
        parser = build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["report", "-i", "workspace", "--target-chip", "A3"])

    def test_report_batch_command_does_not_accept_target_chip(self) -> None:
        parser = build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["report-batch", "-i", "workspaces", "--target-chip", "A3"])

    def test_handle_report_batch_uses_workspace_reports_without_target_chip(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "case0"
            workspace.mkdir()
            args = parser.parse_args(["report-batch", "-i", str(root), "--report-workers", "1"])

            with patch(
                "triton_agent.commands.report_batch.write_report_batch_state",
                return_value=root / "report-batch-state.json",
            ):
                with patch(
                    "triton_agent.commands.report_batch.render_report_batch_file",
                    return_value=root / "report-batch.md",
                ):
                    with patch(
                        "triton_agent.commands.report_batch._discover_workspaces",
                        return_value=[workspace],
                    ):
                        with patch(
                            "triton_agent.commands.report_batch.generate_workspace_report",
                            return_value=(True, "ok"),
                        ) as mocked:
                            exit_code = handle_report_batch(parser, args)

        self.assertEqual(exit_code, 0)
        mocked.assert_called_once_with(workspace, "codex", True)
