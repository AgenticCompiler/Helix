import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.run_skill_test_utils import load_bench_runner_module, make_skill_result


class LocalBenchRunnerTests(unittest.TestCase):
    def test_parse_bench_metadata_reads_kernel_name(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            bench_file = Path(tmp) / "bench_abs.py"
            bench_file.write_text(
                "# bench-mode: msprof\n# api-name: abs_\n# kernel: abs_kernel\nprint('x')\n",
                encoding="utf-8",
            )

            metadata = module.parse_bench_metadata(bench_file)

            self.assertEqual(metadata["kernel"], "abs_kernel")
            self.assertEqual(metadata["api-name"], "abs_")

    def test_run_local_bench_standalone_saves_operator_filename_perf_file(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_abs.py"
            operator_file = root / "abs.py"
            bench_file.write_text("# kernel: abs_kernel\n", encoding="utf-8")
            operator_file.write_text("def abs_():\n    pass\n", encoding="utf-8")

            fake_result = make_skill_result(0, "latency-a: 1.0\nlatency-b: 2.0\n", "")
            with patch.object(module, "run_streaming_process", return_value=fake_result) as mocked:
                result, perf_path = module.run_local_bench(
                    bench_file,
                    operator_file,
                    "standalone",
                )

            self.assertEqual(result["return_code"], 0)
            if perf_path is None:
                self.fail("expected standalone perf path")
            self.assertEqual(perf_path, root / "abs_perf.txt")
            self.assertEqual(perf_path.read_text(encoding="utf-8"), "latency-a: 1.0\nlatency-b: 2.0\n")
            mocked.assert_called_once()

    def test_run_local_bench_standalone_uses_operator_filename_for_any_operator(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_abs.py"
            operator_file = root / "opt_abs.py"
            bench_file.write_text("# kernel: abs_kernel\n", encoding="utf-8")
            operator_file.write_text("def abs_():\n    pass\n", encoding="utf-8")

            fake_result = make_skill_result(0, "latency-a: 1.0\n", "")
            with patch.object(module, "run_streaming_process", return_value=fake_result):
                _, perf_path = module.run_local_bench(
                    bench_file,
                    operator_file,
                    "standalone",
                )

            self.assertEqual(perf_path, root / "opt_abs_perf.txt")

    def test_run_local_bench_msprof_queries_case_count_and_runs_each_case(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_abs.py"
            operator_file = root / "abs.py"
            bench_file.write_text(
                "# bench-mode: msprof\n# api-name: abs_\n# kernel: abs_kernel\n",
                encoding="utf-8",
            )
            operator_file.write_text("def abs_():\n    pass\n", encoding="utf-8")

            results = [
                make_skill_result(0, "2\n", ""),
                make_skill_result(0, "Task Duration(us): 10.5\n", ""),
                make_skill_result(0, "Task Duration(us): 11.5\n", ""),
            ]
            with patch.object(module, "run_buffered_process", return_value=results[0]), patch.object(
                module,
                "run_streaming_process",
                side_effect=results[1:],
            ) as mocked:
                result, perf_path = module.run_local_bench(
                    bench_file,
                    operator_file,
                    "msprof",
                )

            self.assertEqual(result["return_code"], 0)
            if perf_path is None:
                self.fail("expected msprof perf path")
            self.assertEqual(perf_path, root / "abs_perf.txt")
            self.assertEqual(
                perf_path.read_text(encoding="utf-8"),
                "latency-case-1: 10.5\nlatency-case-2: 11.5\n",
            )
            self.assertEqual(mocked.call_count, 2)
            case_command = mocked.call_args_list[1].args[0]
            self.assertEqual(case_command[:3], ["msprof", "op", "--kernel-name=abs_kernel"])
            self.assertIn("--bench", case_command)

    def test_run_local_bench_msprof_accepts_zero_duration_output(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_abs.py"
            operator_file = root / "abs.py"
            bench_file.write_text(
                "# bench-mode: msprof\n# api-name: abs_\n# kernel: abs_kernel\n",
                encoding="utf-8",
            )
            operator_file.write_text("def abs_():\n    pass\n", encoding="utf-8")

            results = [
                make_skill_result(0, "1\n", ""),
                make_skill_result(0, "Task Duration(us): 0.000000\n", ""),
            ]
            with patch.object(module, "run_buffered_process", return_value=results[0]), patch.object(
                module,
                "run_streaming_process",
                side_effect=results[1:],
            ):
                result, perf_path = module.run_local_bench(
                    bench_file,
                    operator_file,
                    "msprof",
                )

            self.assertEqual(result["return_code"], 0)
            if perf_path is None:
                self.fail("expected msprof perf path")
            self.assertEqual(
                perf_path.read_text(encoding="utf-8"),
                "latency-case-1: 0.000000\n",
            )

    def test_run_local_bench_msprof_requires_kernel_metadata(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_abs.py"
            operator_file = root / "abs.py"
            bench_file.write_text("# bench-mode: msprof\n", encoding="utf-8")
            operator_file.write_text("def abs_():\n    pass\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                module.run_local_bench(
                    bench_file,
                    operator_file,
                    "msprof",
                )

    def test_compare_perf_files_reports_per_case_deltas(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_perf.txt"
            compare = root / "compare_perf.txt"
            baseline.write_text("latency-a: 10\nlatency-b: 20\n", encoding="utf-8")
            compare.write_text("latency-a: 11\nlatency-b: 18\n", encoding="utf-8")

            stdout_path = Path(tmp) / "stdout.txt"
            original_stdout = sys.stdout
            try:
                with stdout_path.open("w", encoding="utf-8") as handle:
                    sys.stdout = handle
                    return_code = module.compare_perf_files(baseline, compare)
            finally:
                sys.stdout = original_stdout

            output = stdout_path.read_text(encoding="utf-8")
            self.assertEqual(return_code, 0)
            self.assertIn("latency-a", output)
            self.assertIn("baseline=10.0", output)
            self.assertIn("compare=11.0", output)
            self.assertIn("delta=10.00%", output)

    def test_compare_perf_files_fails_when_case_ids_do_not_match(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_perf.txt"
            compare = root / "compare_perf.txt"
            baseline.write_text("latency-a: 10\n", encoding="utf-8")
            compare.write_text("latency-b: 11\n", encoding="utf-8")

            stdout_path = Path(tmp) / "stdout.txt"
            original_stdout = sys.stdout
            try:
                with stdout_path.open("w", encoding="utf-8") as handle:
                    sys.stdout = handle
                    return_code = module.compare_perf_files(baseline, compare)
            finally:
                sys.stdout = original_stdout

            output = stdout_path.read_text(encoding="utf-8")
            self.assertEqual(return_code, 1)
            self.assertIn("FAIL", output)
            self.assertIn("latency-a", output)
            self.assertIn("missing required latency ids", output)

    def test_compare_perf_files_ignores_extra_compare_fields(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_perf.txt"
            compare = root / "compare_perf.txt"
            baseline.write_text("latency-a: 10\nlatency-b: 20\n", encoding="utf-8")
            compare.write_text(
                "latency-a: 9\nmean_ms: 14.5\nlatency-b: 18\nnotes: candidate\n",
                encoding="utf-8",
            )

            stdout_path = Path(tmp) / "stdout.txt"
            original_stdout = sys.stdout
            try:
                with stdout_path.open("w", encoding="utf-8") as handle:
                    sys.stdout = handle
                    return_code = module.compare_perf_files(baseline, compare)
            finally:
                sys.stdout = original_stdout

            output = stdout_path.read_text(encoding="utf-8")
            self.assertEqual(return_code, 0)
            self.assertIn("PASS: compared 2 latency entries", output)
            self.assertIn("latency-a", output)
            self.assertIn("latency-b", output)


if __name__ == "__main__":
    unittest.main()
