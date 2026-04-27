import os
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
                "# bench-mode: msprof\n# api-name: abs_\n# kernel: OpB\n",
                encoding="utf-8",
            )
            operator_file.write_text("def abs_():\n    pass\n", encoding="utf-8")

            created_output_dirs: list[Path] = []

            def _fake_streaming(command, workdir, stall_timeout_seconds):
                self.assertEqual(workdir, str(root))
                self.assertEqual(stall_timeout_seconds, 900)
                self.assertEqual(command[0], "msprof")
                self.assertTrue(command[1].startswith("--output="))
                output_dir = Path(command[1].split("=", 1)[1])
                created_output_dirs.append(output_dir)
                case_idx = int(command[-1])
                csv_path = output_dir / f"op_statistic_20260424{case_idx}.csv"
                csv_path.write_text(
                    "\n".join(
                        [
                            "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                            f"0,OpA,AI_CORE,1,{10 + case_idx},1,{case_idx * 1.5},2,50",
                            f"0,OpB,AI_VECTOR_CORE,1,{20 + case_idx},2,{case_idx * 2.5},3,50",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return make_skill_result(0, f"profile {case_idx}\n", "")

            with patch.object(module, "run_buffered_process", return_value=make_skill_result(0, "2\n", "")), patch.object(
                module,
                "run_streaming_process",
                side_effect=_fake_streaming,
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
                (
                    'latency-case-1: 2.5\n'
                    '# raw-op-statistic-case-1: {"ops":[{"op_type":"OpA","avg_time_us":1.5},{"op_type":"OpB","avg_time_us":2.5}]}\n'
                    'latency-case-2: 5.0\n'
                    '# raw-op-statistic-case-2: {"ops":[{"op_type":"OpA","avg_time_us":3.0},{"op_type":"OpB","avg_time_us":5.0}]}\n'
                ),
            )
            self.assertEqual(mocked.call_count, 2)
            case_command = mocked.call_args_list[1].args[0]
            self.assertEqual(case_command[0], "msprof")
            self.assertTrue(case_command[1].startswith("--output="))
            self.assertEqual(case_command[2:4], [sys.executable, "bench_abs.py"])
            self.assertIn("--bench", case_command)
            self.assertTrue(created_output_dirs)
            self.assertTrue(all(not path.exists() for path in created_output_dirs))

    def test_run_local_bench_msprof_sums_multiple_declared_kernels(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_abs.py"
            operator_file = root / "abs.py"
            bench_file.write_text(
                "# bench-mode: msprof\n# api-name: abs_\n# kernels: OpA, OpB\n",
                encoding="utf-8",
            )
            operator_file.write_text("def abs_():\n    pass\n", encoding="utf-8")

            def _fake_streaming(command, workdir, stall_timeout_seconds):
                self.assertEqual(workdir, str(root))
                self.assertEqual(stall_timeout_seconds, 900)
                self.assertEqual(command[0], "msprof")
                self.assertTrue(command[1].startswith("--output="))
                output_dir = Path(command[1].split("=", 1)[1])
                (output_dir / "op_statistic_1.csv").write_text(
                    "\n".join(
                        [
                            "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                            "0,OpA,AI_CORE,1,11,1,1.5,2,50",
                            "0,OpB,AI_VECTOR_CORE,1,21,2,2.5,3,50",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return make_skill_result(0, "profile 1\n", "")

            with patch.object(module, "run_buffered_process", return_value=make_skill_result(0, "1\n", "")), patch.object(
                module,
                "run_streaming_process",
                side_effect=_fake_streaming,
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
                (
                    'latency-case-1: 4.0\n'
                    '# raw-op-statistic-case-1: {"ops":[{"op_type":"OpA","avg_time_us":1.5},{"op_type":"OpB","avg_time_us":2.5}]}\n'
                ),
            )

    def test_run_local_bench_msprof_accepts_zero_duration_output(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_abs.py"
            operator_file = root / "abs.py"
            bench_file.write_text(
                "# bench-mode: msprof\n# api-name: abs_\n# kernel: Zero\n",
                encoding="utf-8",
            )
            operator_file.write_text("def abs_():\n    pass\n", encoding="utf-8")

            def _fake_streaming(command, workdir, stall_timeout_seconds):
                self.assertEqual(command[0], "msprof")
                self.assertTrue(command[1].startswith("--output="))
                output_dir = Path(command[1].split("=", 1)[1])
                (output_dir / "op_statistic_1.csv").write_text(
                    "\n".join(
                        [
                            "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                            "0,Zero,AI_CORE,1,0,0,0.000000,0,100",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return make_skill_result(0, "", "")

            with patch.object(module, "run_buffered_process", return_value=make_skill_result(0, "1\n", "")), patch.object(
                module,
                "run_streaming_process",
                side_effect=_fake_streaming,
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
                'latency-case-1: 0.0\n# raw-op-statistic-case-1: {"ops":[{"op_type":"Zero","avg_time_us":0.0}]}\n',
            )

    def test_run_local_bench_msprof_keeps_artifacts_under_configured_output_root(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_abs.py"
            operator_file = root / "abs.py"
            keep_root = root / "kept-msprof"
            bench_file.write_text(
                "# bench-mode: msprof\n# api-name: abs_\n# kernel: KeepMe\n",
                encoding="utf-8",
            )
            operator_file.write_text("def abs_():\n    pass\n", encoding="utf-8")

            created_output_dirs: list[Path] = []

            def _fake_streaming(command, workdir, stall_timeout_seconds):
                self.assertEqual(command[0], "msprof")
                self.assertTrue(command[1].startswith("--output="))
                output_dir = Path(command[1].split("=", 1)[1])
                created_output_dirs.append(output_dir)
                (output_dir / "op_statistic_1.csv").write_text(
                    "\n".join(
                        [
                            "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                            "0,KeepMe,AI_CORE,1,11,1,4.5,6,100",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return make_skill_result(0, "", "")

            with patch.dict(os.environ, {"TRITON_AGENT_MSPROF_OUTPUT_DIR": str(keep_root)}, clear=False), patch.object(
                module,
                "run_buffered_process",
                return_value=make_skill_result(0, "1\n", ""),
            ), patch.object(
                module,
                "run_streaming_process",
                side_effect=_fake_streaming,
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
                'latency-case-1: 4.5\n# raw-op-statistic-case-1: {"ops":[{"op_type":"KeepMe","avg_time_us":4.5}]}\n',
            )
            self.assertTrue(keep_root.exists())
            self.assertTrue(created_output_dirs)
            self.assertTrue(all(path.exists() for path in created_output_dirs))
            self.assertTrue(all(keep_root in path.parents for path in created_output_dirs))
            self.assertTrue(all((path / "op_statistic_1.csv").exists() for path in created_output_dirs))

    def test_run_local_bench_msprof_kept_case_directories_ignore_permissive_umask(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_abs.py"
            operator_file = root / "abs.py"
            keep_root = root / "kept-msprof"
            bench_file.write_text(
                "# bench-mode: msprof\n# api-name: abs_\n# kernel: KeepMe\n",
                encoding="utf-8",
            )
            operator_file.write_text("def abs_():\n    pass\n", encoding="utf-8")

            created_output_dirs: list[Path] = []

            def _fake_streaming(command, workdir, stall_timeout_seconds):
                output_dir = Path(command[1].split("=", 1)[1])
                created_output_dirs.append(output_dir)
                (output_dir / "op_statistic_1.csv").write_text(
                    "\n".join(
                        [
                            "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                            "0,KeepMe,AI_CORE,1,11,1,4.5,6,100",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return make_skill_result(0, "", "")

            original_umask = os.umask(0o002)
            try:
                with patch.dict(os.environ, {"TRITON_AGENT_MSPROF_OUTPUT_DIR": str(keep_root)}, clear=False), patch.object(
                    module,
                    "run_buffered_process",
                    return_value=make_skill_result(0, "1\n", ""),
                ), patch.object(
                    module,
                    "run_streaming_process",
                    side_effect=_fake_streaming,
                ):
                    result, perf_path = module.run_local_bench(
                        bench_file,
                        operator_file,
                        "msprof",
                    )
            finally:
                os.umask(original_umask)

            self.assertEqual(result["return_code"], 0)
            if perf_path is None:
                self.fail("expected msprof perf path")
            self.assertTrue(created_output_dirs)
            self.assertEqual(created_output_dirs[0].stat().st_mode & 0o777, 0o700)

    def test_run_local_bench_msprof_fails_when_statistic_csv_is_missing(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_abs.py"
            operator_file = root / "abs.py"
            bench_file.write_text("# bench-mode: msprof\n# kernel: OpB\n", encoding="utf-8")
            operator_file.write_text("def abs_():\n    pass\n", encoding="utf-8")

            with patch.object(module, "run_buffered_process", return_value=make_skill_result(0, "1\n", "")), patch.object(
                module,
                "run_streaming_process",
                return_value=make_skill_result(0, "", ""),
            ):
                with self.assertRaises(FileNotFoundError):
                    module.run_local_bench(
                        bench_file,
                        operator_file,
                        "msprof",
                    )

    def test_run_local_bench_msprof_records_na_when_kernel_row_is_missing(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_abs.py"
            operator_file = root / "abs.py"
            bench_file.write_text("# bench-mode: msprof\n# kernel: MissingKernel\n", encoding="utf-8")
            operator_file.write_text("def abs_():\n    pass\n", encoding="utf-8")

            def _fake_streaming(command, workdir, stall_timeout_seconds):
                output_dir = Path(command[1].split("=", 1)[1])
                (output_dir / "op_statistic_1.csv").write_text(
                    "\n".join(
                        [
                            "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                            "0,OpA,AI_CORE,1,10,1,1.5,2,50",
                            "0,OpB,AI_VECTOR_CORE,1,20,2,2.5,3,50",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return make_skill_result(0, "", "")

            with patch.object(module, "run_buffered_process", return_value=make_skill_result(0, "1\n", "")), patch.object(
                module,
                "run_streaming_process",
                side_effect=_fake_streaming,
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
                (
                    'latency-case-1: NA\n'
                    '# raw-op-statistic-case-1: {"ops":[{"op_type":"OpA","avg_time_us":1.5},{"op_type":"OpB","avg_time_us":2.5}]}\n'
                ),
            )

    def test_compare_perf_files_reports_per_case_deltas(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_perf.txt"
            compare = root / "compare_perf.txt"
            baseline.write_text("latency-a: 10\nlatency-b: 20\n", encoding="utf-8")
            compare.write_text("latency-a: 8\nlatency-b: 10\n", encoding="utf-8")

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
            self.assertIn("baseline=10", output)
            self.assertIn("compare=8", output)
            self.assertIn("delta=-20.00%", output)
            self.assertIn("Avg improvement: +35.0%", output)
            self.assertIn("Geomean speedup: 1.58x", output)
            self.assertIn("Total speedup: 1.67x", output)
            self.assertIn("Metric source: kernel", output)

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

    def test_compare_perf_files_ignores_comment_lines_in_both_inputs(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_perf.txt"
            compare = root / "compare_perf.txt"
            baseline.write_text(
                "latency-case-1: 10\n# raw-op-statistic-case-1: {\"ops\":[{\"op_type\":\"Kernel\",\"avg_time_us\":10.0}]}\n",
                encoding="utf-8",
            )
            compare.write_text(
                "latency-case-1: 8\n# raw-op-statistic-case-1: {\"ops\":[{\"op_type\":\"Kernel\",\"avg_time_us\":8.0}]}\n",
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
            self.assertIn("latency-case-1", output)
            self.assertIn("delta=-20.00%", output)

    def test_compare_perf_files_falls_back_to_total_op_when_baseline_latency_is_na(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_perf.txt"
            compare = root / "compare_perf.txt"
            baseline.write_text(
                (
                    'latency-case-1: NA\n'
                    '# raw-op-statistic-case-1: {"ops":[{"op_type":"OpA","avg_time_us":4.0},{"op_type":"OpB","avg_time_us":6.0}]}\n'
                ),
                encoding="utf-8",
            )
            compare.write_text(
                (
                    'latency-case-1: 3.0\n'
                    '# raw-op-statistic-case-1: {"ops":[{"op_type":"OpA","avg_time_us":2.5},{"op_type":"OpB","avg_time_us":5.0}]}\n'
                ),
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
            self.assertIn("baseline=NA (total-op=10.0)", output)
            self.assertIn("compare=total-op=7.5", output)
            self.assertIn("delta=-25.00%", output)
            self.assertIn("Metric source: total-op", output)

    def test_compare_perf_files_reports_mixed_metric_source_when_cases_mix_kernel_and_total_op(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_perf.txt"
            compare = root / "compare_perf.txt"
            baseline.write_text(
                (
                    "latency-case-1: 10\n"
                    "latency-case-2: NA\n"
                    '# raw-op-statistic-case-2: {"ops":[{"op_type":"OpA","avg_time_us":4.0},{"op_type":"OpB","avg_time_us":6.0}]}\n'
                ),
                encoding="utf-8",
            )
            compare.write_text(
                (
                    "latency-case-1: 8\n"
                    "latency-case-2: 3.0\n"
                    '# raw-op-statistic-case-2: {"ops":[{"op_type":"OpA","avg_time_us":2.5},{"op_type":"OpB","avg_time_us":5.0}]}\n'
                ),
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
            self.assertIn("Metric source: mixed (kernel + total-op fallback)", output)

    def test_compare_perf_files_preserves_original_display_precision(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_perf.txt"
            compare = root / "compare_perf.txt"
            baseline.write_text("latency-case-1: 0.0038\n", encoding="utf-8")
            compare.write_text("latency-case-1: 0.0254\n", encoding="utf-8")

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
            self.assertIn("baseline=0.0038", output)
            self.assertIn("compare=0.0254", output)


if __name__ == "__main__":
    unittest.main()
