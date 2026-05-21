import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.cli import build_parser
import triton_agent.commands.comparison as comparison_module
from triton_agent.commands.comparison import compare_perf_files, handle_compare_perf, handle_compare_result
from tests.run_skill_test_utils import load_compare_result_module, load_perf_artifacts_module


class ComparisonCommandHandlerTests(unittest.TestCase):
    def test_package_bridge_module_is_removed(self) -> None:
        self.assertIsNone(importlib.util.find_spec("triton_agent.comparison"))

    def test_load_compare_result_reuses_compare_module(self) -> None:
        self.assertIs(comparison_module._load_compare_result(), load_compare_result_module())

    def test_load_compare_perf_reuses_perf_artifacts_module(self) -> None:
        self.assertIs(comparison_module._load_compare_perf(), load_perf_artifacts_module())

    def test_compare_perf_files_runs_via_skill_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_perf.txt"
            compare = root / "candidate_perf.txt"
            baseline.write_text("latency-a: 10\n", encoding="utf-8")
            compare.write_text("latency-a: 8\n", encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = compare_perf_files(baseline, compare)

            self.assertEqual(exit_code, 0)
            self.assertIn("latency-a", stdout.getvalue())

    def test_compare_perf_files_forwards_skip_latency_errors_flag(self) -> None:
        module = comparison_module._load_compare_perf()
        baseline = Path("/tmp/baseline_perf.txt")
        compare = Path("/tmp/candidate_perf.txt")

        with patch.object(module, "compare_perf_files", return_value=1) as mocked:
            exit_code = comparison_module.compare_perf_files(
                baseline,
                compare,
                skip_latency_errors=True,
            )

        self.assertEqual(exit_code, 1)
        mocked.assert_called_once_with(
            baseline,
            compare,
            skip_latency_errors=True,
            metric_source="auto",
        )

    def test_compare_perf_files_forwards_metric_source_flag(self) -> None:
        module = comparison_module._load_compare_perf()
        baseline = Path("/tmp/baseline_perf.txt")
        compare = Path("/tmp/candidate_perf.txt")

        with patch.object(module, "compare_perf_files", return_value=1) as mocked:
            exit_code = comparison_module.compare_perf_files(
                baseline,
                compare,
                metric_source="total-op",
            )

        self.assertEqual(exit_code, 1)
        mocked.assert_called_once_with(
            baseline,
            compare,
            skip_latency_errors=False,
            metric_source="total-op",
        )

    def test_compare_perf_files_forwards_metric_source_all_flag(self) -> None:
        module = comparison_module._load_compare_perf()
        baseline = Path("/tmp/baseline_perf.txt")
        compare = Path("/tmp/candidate_perf.txt")

        with patch.object(module, "compare_perf_files", return_value=1) as mocked:
            exit_code = comparison_module.compare_perf_files(
                baseline,
                compare,
                metric_source="all",
            )

        self.assertEqual(exit_code, 1)
        mocked.assert_called_once_with(
            baseline,
            compare,
            skip_latency_errors=False,
            metric_source="all",
        )

    def test_compare_result_files_runs_via_skill_wrapper(self) -> None:
        module = comparison_module._load_compare_result()
        oracle = Path("/tmp/oracle.pt")
        new = Path("/tmp/new.pt")

        with patch.object(module, "compare_result_files", return_value=0) as mocked:
            exit_code = comparison_module.compare_result_files(oracle, new, "balanced")

        self.assertEqual(exit_code, 0)
        mocked.assert_called_once_with(oracle, new, "balanced")

    def test_compare_remote_result_files_runs_via_skill_wrapper(self) -> None:
        module = comparison_module._load_compare_result()
        oracle = Path("/tmp/oracle.pt")
        new = Path("/tmp/new.pt")

        with patch.object(module, "compare_remote_result_files", return_value=0) as mocked:
            exit_code = comparison_module.compare_remote_result_files(
                oracle,
                new,
                "balanced",
                "alice@example.com",
                "/tmp/remote-workdir",
                verbose=True,
                stderr=sys.stderr,
            )

        self.assertEqual(exit_code, 0)
        mocked.assert_called_once_with(
            oracle,
            new,
            "balanced",
            "alice@example.com",
            "/tmp/remote-workdir",
            verbose=True,
            stderr=sys.stderr,
        )

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
            mocked.assert_called_once_with(
                baseline.resolve(),
                compare.resolve(),
                skip_latency_errors=False,
                metric_source="auto",
            )

    def test_handle_compare_perf_forwards_skip_latency_errors_flag(self) -> None:
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
                    "--skip-latency-errors",
                ]
            )

            with patch(
                "triton_agent.commands.comparison.compare_perf_files",
                return_value=0,
            ) as mocked:
                exit_code = handle_compare_perf(parser, args)

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(
                baseline.resolve(),
                compare.resolve(),
                skip_latency_errors=True,
                metric_source="auto",
            )

    def test_handle_compare_perf_forwards_metric_source_flag(self) -> None:
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
                    "--metric-source",
                    "kernel",
                ]
            )

            with patch(
                "triton_agent.commands.comparison.compare_perf_files",
                return_value=0,
            ) as mocked:
                exit_code = handle_compare_perf(parser, args)

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(
                baseline.resolve(),
                compare.resolve(),
                skip_latency_errors=False,
                metric_source="kernel",
            )

    def test_handle_compare_perf_forwards_metric_source_all_flag(self) -> None:
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
                    "--metric-source",
                    "all",
                ]
            )

            with patch(
                "triton_agent.commands.comparison.compare_perf_files",
                return_value=0,
            ) as mocked:
                exit_code = handle_compare_perf(parser, args)

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(
                baseline.resolve(),
                compare.resolve(),
                skip_latency_errors=False,
                metric_source="all",
            )


if __name__ == "__main__":
    unittest.main()
