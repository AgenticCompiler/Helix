import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.cli import build_parser
from triton_agent.commands.optimize import handle_optimize, optimize_run_options_from_args
from triton_agent.models import AgentResult


class OptimizeCommandHandlerTests(unittest.TestCase):
    def test_handle_optimize_rejects_openhands_interactive_mode(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            operator = Path(tmp) / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            args = parser.parse_args(
                [
                    "optimize",
                    "-i",
                    str(operator),
                    "--agent",
                    "openhands",
                    "--interact",
                ]
            )

            with self.assertRaises(SystemExit) as exc:
                handle_optimize(parser, args)

            self.assertEqual(exc.exception.code, 2)

    def test_handle_optimize_allows_interactive_mode(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            operator = Path(tmp) / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            args = parser.parse_args(
                [
                    "optimize",
                    "-i",
                    str(operator),
                    "--interact",
                ]
            )

            fake_result = AgentResult(return_code=0, stdout="", stderr="")

            with patch("triton_agent.commands.optimize.run_optimize_request", return_value=fake_result) as run_mock:
                exit_code = handle_optimize(parser, args)

            self.assertEqual(exit_code, 0)
            run_mock.assert_called_once()

    def test_optimize_interactive_mode_forces_long_round_batch(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "optimize",
                "-i",
                "kernel.py",
                "--interact",
                "--enable-report",
            ]
        )

        options = optimize_run_options_from_args(args)

        self.assertTrue(options.interact)
        self.assertEqual(options.round_batch_size, 99)
        self.assertFalse(options.report)

    def test_optimize_run_options_maps_compiler_source_analysis(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "optimize",
                "-i",
                "kernel.py",
                "--enable-compiler-source-analysis",
            ]
        )

        options = optimize_run_options_from_args(args)

        self.assertEqual(options.compiler_source_analysis, "auto")
        self.assertFalse(hasattr(options, "compiler_source_path"))

    def test_optimize_run_options_maps_cann_ext_api(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "optimize",
                "-i",
                "kernel.py",
                "--enable-cann-ext-api",
            ]
        )

        options = optimize_run_options_from_args(args)

        self.assertTrue(options.enable_cann_ext_api)

    def test_optimize_run_options_maps_agent_hooks(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--enable-agent-hook"])

        options = optimize_run_options_from_args(args)

        self.assertTrue(options.enable_agent_hooks)

    def test_optimize_run_options_maps_log_tools(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--log-tool"])

        options = optimize_run_options_from_args(args)

        self.assertTrue(options.log_tools)

    def test_optimize_run_options_disables_report_by_default(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py"])

        options = optimize_run_options_from_args(args)

        self.assertFalse(options.report)

    def test_optimize_run_options_enables_report_when_requested(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--enable-report"])

        options = optimize_run_options_from_args(args)

        self.assertTrue(options.report)

    def test_handle_optimize_rejects_cann_ext_api_without_a5(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            operator = Path(tmp) / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            args = parser.parse_args(
                [
                    "optimize",
                    "-i",
                    str(operator),
                    "--target-chip",
                    "A3",
                    "--enable-cann-ext-api",
                ]
            )

            with self.assertRaises(SystemExit) as exc:
                handle_optimize(parser, args)

            self.assertEqual(exc.exception.code, 2)

    def test_handle_optimize_directory_input_uses_workspace_as_workdir(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            args = parser.parse_args(["optimize", "-i", str(workspace), "--resume", "fresh"])
            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            captured: dict[str, Path] = {}

            def _fake_run(request):
                captured["input_path"] = request.input_path
                captured["workdir"] = request.workdir
                return fake_result

            with patch("triton_agent.commands.optimize.run_optimize_request", side_effect=_fake_run):
                exit_code = handle_optimize(parser, args)

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["input_path"], operator.resolve())
            self.assertEqual(captured["workdir"], workspace.resolve())

    def test_handle_optimize_auto_upload_on_success(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            args = parser.parse_args(["optimize", "-i", str(workspace), "--resume", "fresh"])
            fake_result = AgentResult(return_code=0, stdout="", stderr="")

            with patch("triton_agent.commands.optimize.run_optimize_request", return_value=fake_result):
                with patch("triton_agent.commands.optimize.upload_optimize_workspace") as mock_upload:
                    with patch("triton_agent.commands.optimize.generate_workspace_report") as mock_report:
                        exit_code = handle_optimize(parser, args)

            self.assertEqual(exit_code, 0)
            mock_upload.assert_called_once()
            mock_report.assert_not_called()

    def test_handle_optimize_auto_report_when_enabled(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            args = parser.parse_args(["optimize", "-i", str(workspace), "--resume", "fresh", "--enable-report"])
            fake_result = AgentResult(return_code=0, stdout="", stderr="")

            with patch("triton_agent.commands.optimize.run_optimize_request", return_value=fake_result):
                with patch("triton_agent.commands.optimize.upload_optimize_workspace") as mock_upload:
                    with patch(
                        "triton_agent.commands.optimize.generate_workspace_report",
                        return_value=(True, "report.md"),
                    ) as mock_report:
                        exit_code = handle_optimize(parser, args)

            self.assertEqual(exit_code, 0)
            mock_upload.assert_called_once()
            mock_report.assert_called_once_with(
                workspace=workspace.resolve(),
                agent_name="codex",
                show_output=True,
            )

    def test_handle_optimize_no_auto_upload_on_failure(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            args = parser.parse_args(["optimize", "-i", str(workspace), "--resume", "fresh"])
            fake_result = AgentResult(return_code=1, stdout="", stderr="error")

            with patch("triton_agent.commands.optimize.run_optimize_request", return_value=fake_result):
                with patch("triton_agent.commands.optimize.upload_optimize_workspace") as mock_upload:
                    exit_code = handle_optimize(parser, args)

            self.assertEqual(exit_code, 1)
            mock_upload.assert_not_called()

    def test_handle_optimize_no_auto_upload_when_disabled(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            args = parser.parse_args(
                ["optimize", "-i", str(workspace), "--resume", "fresh", "--no-upload"]
            )

            fake_result = AgentResult(return_code=0, stdout="", stderr="")

            with patch("triton_agent.commands.optimize.run_optimize_request", return_value=fake_result):
                with patch("triton_agent.commands.optimize.upload_optimize_workspace") as mock_upload:
                    exit_code = handle_optimize(parser, args)

            self.assertEqual(exit_code, 0)
            mock_upload.assert_not_called()

    def test_upload_optimize_rejects_missing_input(self) -> None:
        from triton_agent.commands.upload_optimize import handle_upload_optimize
        parser = build_parser()
        args = parser.parse_args(["upload-optimize", "-i", "/nonexistent/path"])
        with self.assertRaises(SystemExit) as exc:
            handle_upload_optimize(parser, args)
        self.assertEqual(exc.exception.code, 2)


class OptimizeUploadCommandHandlerTests(unittest.TestCase):
    def test_upload_optimize_success_calls_workflow(self) -> None:
        from triton_agent.commands.upload_optimize import handle_upload_optimize
        from triton_agent.optimize_upload.models import UploadResponse

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "kernel.py").write_text("code", encoding="utf-8")
            (ws / "baseline").mkdir()
            (ws / "baseline" / "state.json").write_text("{}", encoding="utf-8")

            expected_response = UploadResponse(
                ok=True,
                upload_uid="abc123",
                upload_timestamp="20260526T141530Z",
                workspace_name=ws.name,
                workspace_slug=ws.name,
                stored_path="/store/test.tar.gz",
            )

            with patch(
                "triton_agent.commands.upload_optimize.upload_optimize_workspace",
                return_value=expected_response,
            ) as mock_upload:
                parser = build_parser()
                args = parser.parse_args(["upload-optimize", "-i", str(ws)])
                exit_code = handle_upload_optimize(parser, args)
                self.assertEqual(exit_code, 0)
                mock_upload.assert_called_once()

    def test_upload_optimize_failure_returns_nonzero(self) -> None:
        from triton_agent.commands.upload_optimize import handle_upload_optimize

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "kernel.py").write_text("code", encoding="utf-8")
            (ws / "baseline").mkdir()
            (ws / "baseline" / "state.json").write_text("{}", encoding="utf-8")

            with patch(
                "triton_agent.commands.upload_optimize.upload_optimize_workspace",
                side_effect=ValueError("upload failed"),
            ) as mock_upload:
                parser = build_parser()
                args = parser.parse_args(["upload-optimize", "-i", str(ws)])
                exit_code = handle_upload_optimize(parser, args)
                self.assertNotEqual(exit_code, 0)
                mock_upload.assert_called_once()

    def test_upload_optimize_rejects_missing_input(self) -> None:
        from triton_agent.commands.upload_optimize import handle_upload_optimize

        parser = build_parser()
        args = parser.parse_args(["upload-optimize", "-i", "/nonexistent/path"])
        with self.assertRaises(SystemExit) as exc:
            handle_upload_optimize(parser, args)
        self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
