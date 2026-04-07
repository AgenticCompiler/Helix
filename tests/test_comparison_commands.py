import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.cli import build_parser
from triton_agent.commands.comparison import handle_compare_perf, handle_compare_result


class ComparisonCommandHandlerTests(unittest.TestCase):
    def test_handle_compare_result_dispatches_remote_comparison(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oracle = root / "oracle.pt"
            new = root / "new.pt"
            oracle.write_text("oracle", encoding="utf-8")
            new.write_text("new", encoding="utf-8")
            args = parser.parse_args(
                [
                    "compare-result",
                    "--oracle-result",
                    str(oracle),
                    "--new-result",
                    str(new),
                    "--remote",
                    "alice@example.com",
                ]
            )

            with patch(
                "triton_agent.commands.comparison.compare_remote_result_files",
                return_value=0,
            ) as mocked:
                exit_code = handle_compare_result(parser, args)

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once()

    def test_handle_compare_perf_dispatches_local_comparison(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_perf.txt"
            compare = root / "candidate_perf.txt"
            baseline.write_text("latency-a: 10\n", encoding="utf-8")
            compare.write_text("latency-a: 11\n", encoding="utf-8")
            args = parser.parse_args(
                [
                    "compare-perf",
                    "--baseline",
                    str(baseline),
                    "--compare",
                    str(compare),
                ]
            )

            with patch(
                "triton_agent.commands.comparison.compare_perf_files",
                return_value=0,
            ) as mocked:
                exit_code = handle_compare_perf(parser, args)

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(baseline.resolve(), compare.resolve())


if __name__ == "__main__":
    unittest.main()
