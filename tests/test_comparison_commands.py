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


class JsonlPerfArtifactParserTests(unittest.TestCase):
    """Tests for JSONL perf artifact parsing via the shared compatibility layer."""

    def _write_jsonl(self, path: Path, lines: list[str]) -> None:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_parse_jsonl_successful_kernel_latency_case(self) -> None:
        module = load_perf_artifacts_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            perf = root / "perf.txt"
            self._write_jsonl(
                perf,
                [
                    '{"case_label":"case-a","kernel_names":["KernelA"],"kernel_source":"metadata","kernel_avg_time_us":12.5,"total_op_avg_time_us":19.75,"error_message":null,"case_wall_clock_seconds":0.48}',
                ],
            )
            entries = module.parse_perf_file(perf)
            self.assertIn("latency-case-a", entries)
            self.assertEqual(entries["latency-case-a"], 12.5)

    def test_parse_jsonl_total_op_only_fallback_case(self) -> None:
        module = load_perf_artifacts_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            perf = root / "perf.txt"
            self._write_jsonl(
                perf,
                [
                    '{"case_label":"case-b","kernel_names":["KernelB"],"kernel_source":"metadata","kernel_avg_time_us":null,"total_op_avg_time_us":30.0,"error_message":"no resolved kernels matched","case_wall_clock_seconds":0.5}',
                ],
            )
            entries = module.parse_perf_file(perf)
            self.assertIn("latency-case-b", entries)
            self.assertEqual(entries["latency-case-b"], 30.0)

    def test_parse_jsonl_failed_case_with_error_message(self) -> None:
        module = load_perf_artifacts_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            perf = root / "perf.txt"
            self._write_jsonl(
                perf,
                [
                    '{"case_label":"case-c","kernel_names":[],"kernel_source":"metadata","kernel_avg_time_us":null,"total_op_avg_time_us":null,"error_message":"benchmark crashed","case_wall_clock_seconds":null}',
                ],
            )
            with self.assertRaises(ValueError) as ctx:
                module.parse_perf_file(perf)
            self.assertIn("case-c", str(ctx.exception))

    def test_parse_jsonl_multiple_cases(self) -> None:
        module = load_perf_artifacts_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            perf = root / "perf.txt"
            self._write_jsonl(
                perf,
                [
                    '{"case_label":"1","kernel_names":["KA"],"kernel_source":"metadata","kernel_avg_time_us":10.0,"total_op_avg_time_us":15.0,"error_message":null,"case_wall_clock_seconds":0.1}',
                    '{"case_label":"2","kernel_names":["KB"],"kernel_source":"metadata","kernel_avg_time_us":20.0,"total_op_avg_time_us":25.0,"error_message":null,"case_wall_clock_seconds":0.2}',
                ],
            )
            entries = module.parse_perf_file(perf)
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries["latency-1"], 10.0)
            self.assertEqual(entries["latency-2"], 20.0)

    def test_parse_jsonl_rejects_duplicate_case_labels(self) -> None:
        module = load_perf_artifacts_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            perf = root / "perf.txt"
            self._write_jsonl(
                perf,
                [
                    '{"case_label":"dup","kernel_names":[],"kernel_source":"m","kernel_avg_time_us":1.0,"total_op_avg_time_us":1.0,"error_message":null,"case_wall_clock_seconds":0.1}',
                    '{"case_label":"dup","kernel_names":[],"kernel_source":"m","kernel_avg_time_us":2.0,"total_op_avg_time_us":2.0,"error_message":null,"case_wall_clock_seconds":0.2}',
                ],
            )
            with self.assertRaises(ValueError) as ctx:
                module.parse_perf_file(perf)
            self.assertIn("duplicates", str(ctx.exception))

    def test_cross_format_comparison_equivalent_results(self) -> None:
        module = load_perf_artifacts_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "legacy_perf.txt"
            legacy.write_text(
                "latency-case-x: 5.0\n"
                "# raw-op-statistic-case-x: " + '{"ops":[{"op_type":"K","avg_time_us":5.0}]}\n',
                encoding="utf-8",
            )
            jsonl = root / "jsonl_perf.txt"
            jsonl.write_text(
                '{"case_label":"case-x","kernel_names":["K"],"kernel_source":"metadata","kernel_avg_time_us":5.0,"total_op_avg_time_us":5.0,"error_message":null,"case_wall_clock_seconds":0.1}\n',
                encoding="utf-8",
            )

            legacy_entries = module.parse_perf_file(legacy)
            jsonl_entries = module.parse_perf_file(jsonl)

            self.assertEqual(
                legacy_entries["latency-case-x"],
                jsonl_entries["latency-case-x"],
            )

    def test_compare_perf_files_accepts_legacy_msprof_baseline_against_numeric_jsonl_case_labels(
        self,
    ) -> None:
        module = load_perf_artifacts_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_perf.txt"
            baseline.write_text(
                "latency-case-1: 10.0\n"
                '# raw-op-statistic-case-1: {"ops":[{"op_type":"K","avg_time_us":10.0}]}\n',
                encoding="utf-8",
            )
            compare = root / "compare_perf.txt"
            compare.write_text(
                '{"case_label":"1","kernel_names":["K"],"kernel_source":"metadata","kernel_avg_time_us":8.0,"total_op_avg_time_us":8.0,"error_message":null,"case_wall_clock_seconds":0.1}\n',
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = module.compare_perf_files(baseline, compare)

            self.assertEqual(exit_code, 0)
            self.assertIn("latency-case-1", stdout.getvalue())

    def test_legacy_parser_still_works_for_text_format(self) -> None:
        module = load_perf_artifacts_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            perf = root / "perf.txt"
            perf.write_text("latency-case-z: 7.5\n", encoding="utf-8")
            entries = module.parse_perf_file(perf)
            self.assertEqual(entries["latency-case-z"], 7.5)

    # --- Regression: Bug 1 — metric_source=total-op must use total_op_avg_time_us ---

    def test_jsonl_metric_source_total_op_uses_total_op_when_kernel_present(self) -> None:
        module = load_perf_artifacts_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            perf = root / "perf.txt"
            perf.write_text(
                '{"case_label":"a","kernel_names":["K"],"kernel_source":"m","kernel_avg_time_us":10.0,"total_op_avg_time_us":20.0,"error_message":null,"case_wall_clock_seconds":0.1}\n',
                encoding="utf-8",
            )
            entries = module.parse_perf_file_for_metric_source(perf, metric_source="total-op")
            self.assertEqual(entries["latency-a"], 20.0)

    def test_jsonl_metric_source_total_op_raises_when_total_op_missing(self) -> None:
        module = load_perf_artifacts_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            perf = root / "perf.txt"
            perf.write_text(
                '{"case_label":"a","kernel_names":["K"],"kernel_source":"m","kernel_avg_time_us":10.0,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":0.1}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                module.parse_perf_file_for_metric_source(perf, metric_source="total-op")
            self.assertIn("total-op", str(ctx.exception))

    # --- Regression: Bug 2 — required parse under auto must preserve baseline comparison_mode ---

    def test_jsonl_required_parse_preserves_baseline_total_op_mode_under_auto(self) -> None:
        module = load_perf_artifacts_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_perf.txt"
            baseline.write_text(
                '{"case_label":"a","kernel_names":["K"],"kernel_source":"m","kernel_avg_time_us":null,"total_op_avg_time_us":30.0,"error_message":"no resolved kernels matched","case_wall_clock_seconds":0.0}\n',
                encoding="utf-8",
            )
            compare = root / "compare_perf.txt"
            compare.write_text(
                '{"case_label":"a","kernel_names":["K"],"kernel_source":"m","kernel_avg_time_us":10.0,"total_op_avg_time_us":20.0,"error_message":null,"case_wall_clock_seconds":0.0}\n',
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = module.compare_perf_files(baseline, compare, metric_source="auto")

            self.assertEqual(exit_code, 0)
            output = stdout.getvalue()
            self.assertIn("latency-a: baseline=NA (total-op=30.0)", output)
            self.assertIn("compare=total-op=20.0", output)

    # --- Regression: Bug 3 — required parse must ignore unrelated extra cases ---

    def test_jsonl_required_parse_ignores_unrelated_extra_cases(self) -> None:
        module = load_perf_artifacts_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            perf = root / "perf.txt"
            perf.write_text(
                '{"case_label":"wanted","kernel_names":["K"],"kernel_source":"m","kernel_avg_time_us":5.0,"total_op_avg_time_us":5.0,"error_message":null,"case_wall_clock_seconds":0.1}\n'
                '{"case_label":"extra","kernel_names":[],"kernel_source":"m","kernel_avg_time_us":null,"total_op_avg_time_us":null,"error_message":"real failure","case_wall_clock_seconds":null}\n',
                encoding="utf-8",
            )
            entries = module.parse_required_perf_file(perf, {"latency-wanted"})
            self.assertEqual(entries["latency-wanted"], 5.0)


    def test_jsonl_required_parse_reports_missing_ids_when_all_cases_unrelated(self) -> None:
        module = load_perf_artifacts_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            perf = root / "perf.txt"
            perf.write_text(
                '{"case_label":"unrelated","kernel_names":["K"],"kernel_source":"m","kernel_avg_time_us":1.0,"total_op_avg_time_us":1.0,"error_message":null,"case_wall_clock_seconds":0.1}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                module.parse_required_perf_file(perf, {"latency-wanted"})
            self.assertIn("is missing required latency ids", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
