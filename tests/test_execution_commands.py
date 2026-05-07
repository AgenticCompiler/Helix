import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.cli import build_parser
from triton_agent.commands.execution import handle_run_bench, handle_run_test
from triton_agent.models import AgentResult


class ExecutionCommandHandlerTests(unittest.TestCase):
    def test_handle_run_test_reads_mode_from_metadata_when_flag_missing(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "differential_test_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")

            args = parser.parse_args(
                [
                    "run-test",
                    "--test-file",
                    str(test_file),
                    "--operator-file",
                    str(operator),
                ]
            )
            fake_result = AgentResult(return_code=0, stdout="", stderr="")

            with patch(
                "triton_agent.commands.execution.run_local_test",
                return_value=(fake_result, None),
            ) as mocked:
                exit_code = handle_run_test(parser, args)

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(
                test_file.resolve(),
                operator.resolve(),
                "differential",
            )

    def test_handle_run_bench_prints_perf_file(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            perf_file = root / "kernel_perf.txt"
            operator.write_text("print('x')", encoding="utf-8")
            bench_file.write_text("# bench-mode: standalone\nprint('bench')", encoding="utf-8")

            args = parser.parse_args(
                [
                    "run-bench",
                    "--bench-file",
                    str(bench_file),
                    "--operator-file",
                    str(operator),
                ]
            )
            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            stdout = StringIO()

            with patch(
                "triton_agent.commands.execution.run_local_bench",
                return_value=(fake_result, perf_file),
            ):
                with redirect_stdout(stdout):
                    exit_code = handle_run_bench(parser, args)

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                stdout.getvalue(),
                (
                    f"Perf file: {perf_file}\n"
                    "Hint: use `compare-perf` to inspect this perf artifact instead of reading it directly.\n"
                ),
            )

    def test_handle_run_bench_prints_failure_output(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            bench_file.write_text("# bench-mode: standalone\nprint('bench')", encoding="utf-8")

            args = parser.parse_args(
                [
                    "run-bench",
                    "--bench-file",
                    str(bench_file),
                    "--operator-file",
                    str(operator),
                ]
            )
            fake_result = AgentResult(return_code=1, stdout="raw stdout\n", stderr="raw stderr\n")
            stdout = StringIO()
            stderr = StringIO()

            with patch(
                "triton_agent.commands.execution.run_local_bench",
                return_value=(fake_result, None),
            ):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exit_code = handle_run_bench(parser, args)

            self.assertEqual(exit_code, 1)
            self.assertIn("raw stdout", stdout.getvalue())
            self.assertIn("raw stderr", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
