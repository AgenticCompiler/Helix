import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.cli import build_parser
from triton_agent.commands.optimize import handle_optimize, optimize_run_options_from_args


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
                "--compiler-source-path",
                "/tmp/AscendNPU-IR",
            ]
        )

        options = optimize_run_options_from_args(args)

        self.assertEqual(options.compiler_source_analysis, "auto")
        self.assertEqual(options.compiler_source_path, "/tmp/AscendNPU-IR")

    def test_handle_optimize_rejects_compiler_source_path_without_enable_flag(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            operator = Path(tmp) / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            args = parser.parse_args(
                [
                    "optimize",
                    "-i",
                    str(operator),
                    "--compiler-source-path",
                    "/tmp/AscendNPU-IR",
                ]
            )

            with self.assertRaises(SystemExit) as exc:
                with redirect_stderr(StringIO()) as stderr:
                    handle_optimize(parser, args)

            self.assertEqual(exc.exception.code, 2)
            self.assertIn(
                "--compiler-source-path requires --enable-compiler-source-analysis",
                stderr.getvalue(),
            )


if __name__ == "__main__":
    unittest.main()
