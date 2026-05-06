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


if __name__ == "__main__":
    unittest.main()
