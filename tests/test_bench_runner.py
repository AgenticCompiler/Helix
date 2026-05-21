import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from tests.run_skill_test_utils import load_bench_runner_module, make_skill_result


class LocalBenchRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._module = load_bench_runner_module()
        cls._monotonic_patcher = patch.object(cls._module.time, "monotonic", return_value=0.0)
        cls._monotonic_patcher.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._monotonic_patcher.stop()

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

    def test_parse_npu_devices_returns_none_when_unset(self) -> None:
        module = load_bench_runner_module()

        self.assertIsNone(module.parse_npu_devices(None))

    def test_parse_npu_devices_trims_whitespace_and_expands_ranges(self) -> None:
        module = load_bench_runner_module()

        self.assertEqual(
            module.parse_npu_devices(" 0, 2-4 , 7 "),
            ("0", "2", "3", "4", "7"),
        )

    def test_parse_npu_devices_rejects_empty_entries(self) -> None:
        module = load_bench_runner_module()

        with self.assertRaisesRegex(ValueError, "--npu-devices"):
            module.parse_npu_devices("0,,1")

    def test_parse_npu_devices_rejects_duplicates(self) -> None:
        module = load_bench_runner_module()

        with self.assertRaisesRegex(ValueError, "duplicate"):
            module.parse_npu_devices("0,1,0")

    def test_parse_npu_devices_rejects_descending_ranges(self) -> None:
        module = load_bench_runner_module()

        with self.assertRaisesRegex(ValueError, "range"):
            module.parse_npu_devices("5-3")

    def test_parse_npu_devices_rejects_malformed_ranges(self) -> None:
        module = load_bench_runner_module()

        with self.assertRaisesRegex(ValueError, "range"):
            module.parse_npu_devices("1-3-5")

    def test_run_local_bench_standalone_delegates_to_hook_runtime(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_abs.py"
            operator_file = root / "abs.py"
            bench_file.write_text("# bench-mode: standalone\n# kernel: abs_kernel\n", encoding="utf-8")
            operator_file.write_text("def abs_():\n    pass\n", encoding="utf-8")

            fake_result = make_skill_result(0, "bench stdout\n", "")
            perf_file = root / "abs_perf.txt"
            perf_file.write_text("latency-case-a: 1.0\n", encoding="utf-8")
            with patch.object(
                module,
                "run_local_standalone_bench",
                create=True,
                return_value=(fake_result, perf_file),
            ) as helper, patch.object(module, "run_streaming_process") as streaming:
                result, resolved_perf = module.run_local_bench(
                    bench_file,
                    operator_file,
                    "standalone",
                )

            self.assertEqual(result["return_code"], 0)
            self.assertEqual(resolved_perf, perf_file)
            helper.assert_called_once_with(bench_file, operator_file, verbose=False)
            streaming.assert_not_called()

    def test_run_local_bench_standalone_preserves_helper_perf_path(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_abs.py"
            operator_file = root / "opt_abs.py"
            bench_file.write_text("# bench-mode: standalone\n# kernel: abs_kernel\n", encoding="utf-8")
            operator_file.write_text("def abs_():\n    pass\n", encoding="utf-8")

            fake_result = make_skill_result(0, "bench stdout\n", "")
            perf_file = root / "opt_abs_perf.txt"
            with patch.object(
                module,
                "run_local_standalone_bench",
                create=True,
                return_value=(fake_result, perf_file),
            ):
                _, perf_path = module.run_local_bench(
                    bench_file,
                    operator_file,
                    "standalone",
                )

            self.assertEqual(perf_path, root / "opt_abs_perf.txt")

    def test_run_local_bench_standalone_runs_in_bench_workdir_and_cleans_extra_info(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_dir = root / "generated"
            bench_dir.mkdir()
            bench_file = bench_dir / "bench_abs.py"
            operator_file = root / "abs.py"
            extra_info = bench_dir / "extra-info"
            extra_info.mkdir()
            bench_file.write_text("# bench-mode: standalone\n# kernel: abs_kernel\n", encoding="utf-8")
            operator_file.write_text("def abs_():\n    pass\n", encoding="utf-8")

            observed_cwds: list[Path] = []
            fake_result = make_skill_result(0, "", "")
            perf_file = root / "abs_perf.txt"

            def _fake_helper(passed_bench: Path, passed_operator: Path, *, verbose: bool = False):
                del passed_bench, passed_operator, verbose
                observed_cwds.append(Path.cwd())
                return fake_result, perf_file

            with patch.object(
                module,
                "run_local_standalone_bench",
                create=True,
                side_effect=_fake_helper,
            ):
                result, resolved_perf = module.run_local_bench(
                    bench_file,
                    operator_file,
                    "standalone",
                )

            self.assertEqual(result["return_code"], 0)
            self.assertEqual(resolved_perf, perf_file)
            self.assertEqual(observed_cwds, [bench_dir.resolve()])

    def test_run_local_bench_msprof_delegates_via_facade_helper(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_abs.py"
            operator_file = root / "abs.py"
            bench_file.write_text("# bench-mode: msprof\n# kernel: OpA\n", encoding="utf-8")
            operator_file.write_text("def abs_():\n    pass\n", encoding="utf-8")

            fake_result = make_skill_result(0, "", "")
            perf_file = root / "abs_perf.txt"

            with patch.object(
                module,
                "_run_local_bench_msprof",
                return_value=(fake_result, perf_file),
            ) as helper:
                result, resolved_perf = module.run_local_bench(
                    bench_file,
                    operator_file,
                    "msprof",
                )

            self.assertEqual(result, fake_result)
            self.assertEqual(resolved_perf, perf_file)
            helper.assert_called_once_with(bench_file, operator_file, verbose=False)

    def test_run_local_bench_standalone_parallel_uses_isolated_case_workdirs_and_device_envs(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
            bench_file.write_text("# bench-mode: standalone\n# kernel: KernelA\n", encoding="utf-8")
            operator_file.write_text("def build_api():\n    return None\n", encoding="utf-8")

            observed: list[tuple[Path, Optional[str], str, list[str]]] = []

            def _fake_buffered_process(command, workdir, stall_timeout_seconds, extra_env=None):
                del stall_timeout_seconds
                workdir_path = Path(workdir)
                case_id = command[-1]
                observed.append(
                    (
                        workdir_path,
                        None if extra_env is None else extra_env.get("ASCEND_RT_VISIBLE_DEVICES"),
                        case_id,
                        command,
                    )
                )
                self.assertEqual(command[0:2], [module.local_python_executable(), "-c"])
                self.assertIn("run_one_standalone_case_record", command[2])
                self.assertEqual(command[3:], [bench_file.name, operator_file.name, case_id])
                self.assertNotEqual(workdir_path, root)
                self.assertTrue((workdir_path / bench_file.name).exists())
                self.assertTrue((workdir_path / operator_file.name).exists())
                avg = 1.0 if case_id == "case-a" else 2.0
                return make_skill_result(
                    0,
                    (
                        '{"case_label":"'
                        + case_id
                        + '","kernel_names":["KernelA"],"kernel_source":"metadata","metrics":{"kernel_avg_time_us":'
                        + str(avg)
                        + ',"ops":[{"op_type":"KernelA","avg_time_us":'
                        + str(avg)
                        + '}]},"error_message":null,"elapsed_seconds":0.0}\n'
                    ),
                    "",
                )

            with patch.object(
                module,
                "_load_standalone_runtime_module",
            ) as load_runtime:
                load_runtime.return_value = type(
                    "_FakeRuntime",
                    (),
                    {
                        "load_standalone_bench_cases": staticmethod(
                            lambda *_args, **_kwargs: (
                                [
                                    type("_Case", (), {"case_id": "case-a"})(),
                                    type("_Case", (), {"case_id": "case-b"})(),
                                ],
                                type("_Resolution", (), {"kernel_names": ["KernelA"], "kernel_source": "metadata"})(),
                            )
                        ),
                        "runtime_support_paths": staticmethod(lambda: []),
                    },
                )()
                with patch.object(
                    module,
                    "run_buffered_process",
                    side_effect=_fake_buffered_process,
                ):
                    result, perf_path = module.run_local_bench(
                        bench_file,
                        operator_file,
                        "standalone",
                        npu_devices="0,2",
                    )

            self.assertEqual(result["return_code"], 0)
            self.assertEqual(len(observed), 2)
            self.assertEqual(len({item[0] for item in observed}), 2)
            self.assertEqual({item[1] for item in observed}, {"0", "2"})
            if perf_path is None:
                self.fail("expected standalone perf path")
            perf_text = perf_path.read_text(encoding="utf-8")
            self.assertLess(perf_text.index("latency-case-a"), perf_text.index("latency-case-b"))

    def test_run_remote_bench_standalone_uses_module_helper_files(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_abs.py"
            operator_file = root / "abs.py"
            perf_file = root / "abs_perf.txt"
            bench_file.write_text("# bench-mode: standalone\n# kernel: abs_kernel\n", encoding="utf-8")
            operator_file.write_text("def abs_():\n    pass\n", encoding="utf-8")

            with patch.object(
                module,
                "create_remote_workspace",
                return_value=("spec", "/tmp/remote-bench"),
            ), patch.object(module, "copy_file_to_remote") as copy_to_remote, patch.object(
                module,
                "run_remote_command_streaming",
                return_value=make_skill_result(0, "", ""),
            ) as remote_run, patch.object(
                module,
                "copy_file_from_remote",
            ) as copy_back, patch.object(module, "cleanup_remote_workspace") as cleanup:
                result, resolved_perf, remote_workspace = module.run_remote_bench(
                    bench_file,
                    operator_file,
                    "standalone",
                    "alice@example.com",
                    None,
                    keep_remote_workdir=True,
                )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(resolved_perf, perf_file)
        self.assertEqual(remote_workspace, "/tmp/remote-bench")
        copy_targets = [call.args[2].rsplit("/", 1)[-1] for call in copy_to_remote.call_args_list]
        self.assertIn("bench_abs.py", copy_targets)
        self.assertIn("abs.py", copy_targets)
        self.assertIn("standalone_bench_runtime.py", copy_targets)
        self.assertIn("bench_contract.py", copy_targets)
        self.assertIn("perf_artifacts.py", copy_targets)
        remote_command = remote_run.call_args.args[2]
        self.assertEqual(remote_command[0:2], ["python3", "-c"])
        self.assertIn("run_local_standalone_bench", remote_command[2])
        self.assertEqual(remote_command[3:], ["bench_abs.py", "abs.py", "abs_perf.txt"])
        copy_back.assert_called_once_with(
            "spec",
            "/tmp/remote-bench/abs_perf.txt",
            perf_file,
            verbose=False,
            stderr=None,
        )
        cleanup.assert_not_called()

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

            def _fake_streaming(command, workdir, stall_timeout_seconds, stdout=None, **kwargs):
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
                    '# elapsed-seconds-case-1: 0.000000\n'
                    '# raw-op-statistic-case-1: {"ops":[{"op_type":"OpA","avg_time_us":1.5},{"op_type":"OpB","avg_time_us":2.5}]}\n'
                    '# resolved-kernels-case-1: OpB\n'
                    '# kernel-source-case-1: metadata\n'
                    'latency-case-2: 5.0\n'
                    '# elapsed-seconds-case-2: 0.000000\n'
                    '# raw-op-statistic-case-2: {"ops":[{"op_type":"OpA","avg_time_us":3.0},{"op_type":"OpB","avg_time_us":5.0}]}\n'
                    '# resolved-kernels-case-2: OpB\n'
                    '# kernel-source-case-2: metadata\n'
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

    def test_run_local_bench_msprof_parallel_uses_isolated_case_workdirs_and_device_envs(self) -> None:
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

            observed_workdirs: list[Path] = []
            observed_devices: list[Optional[str]] = []

            def _fake_streaming(command, workdir, stall_timeout_seconds, stdout=None, **kwargs):
                del stall_timeout_seconds, stdout
                self.assertEqual(command[0], "msprof")
                workdir_path = Path(workdir)
                observed_workdirs.append(workdir_path)
                observed_devices.append((kwargs.get("extra_env") or {}).get("ASCEND_RT_VISIBLE_DEVICES"))
                self.assertNotEqual(workdir_path, root)
                self.assertTrue((workdir_path / "bench_abs.py").exists())
                self.assertTrue((workdir_path / "abs.py").exists())
                output_dir = Path(command[1].split("=", 1)[1])
                case_idx = int(command[-1])
                if case_idx == 1:
                    __import__("time").sleep(0.05)
                (output_dir / f"op_statistic_{case_idx}.csv").write_text(
                    "\n".join(
                        [
                            "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                            f"0,OpB,AI_CORE,1,{case_idx * 10},1,{case_idx * 2.5},3,100",
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
            ):
                result, perf_path = module.run_local_bench(
                    bench_file,
                    operator_file,
                    "msprof",
                    npu_devices="0,2",
                )

            self.assertEqual(result["return_code"], 0)
            self.assertEqual(len(observed_workdirs), 2)
            self.assertEqual(len({path for path in observed_workdirs}), 2)
            self.assertEqual(set(observed_devices), {"0", "2"})
            if perf_path is None:
                self.fail("expected msprof perf path")
            perf_text = perf_path.read_text(encoding="utf-8")
            self.assertLess(perf_text.index("latency-case-1"), perf_text.index("latency-case-2"))

    def test_run_local_bench_msprof_parallel_stages_discovered_case_json_files(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_all_cases.py"
            operator_file = root / "opt_kernel.py"
            discovered_json = root / "5_MoeInitRouting.json"
            bench_file.write_text(
                "# bench-mode: msprof\n# api-name: kernel\n# kernel: KernelB\n",
                encoding="utf-8",
            )
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")
            discovered_json.write_text('{"cases":[1]}\n', encoding="utf-8")

            observed_workdirs: list[Path] = []

            def _fake_streaming(command, workdir, stall_timeout_seconds, stdout=None, **kwargs):
                del stall_timeout_seconds, stdout, kwargs
                self.assertEqual(command[0], "msprof")
                workdir_path = Path(workdir)
                observed_workdirs.append(workdir_path)
                self.assertTrue((workdir_path / bench_file.name).exists())
                self.assertTrue((workdir_path / operator_file.name).exists())
                self.assertTrue((workdir_path / discovered_json.name).exists())
                output_dir = Path(command[1].split("=", 1)[1])
                (output_dir / "op_statistic_1.csv").write_text(
                    "\n".join(
                        [
                            "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                            "0,KernelB,AI_CORE,1,20,2,5.0,7,100",
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
                    npu_devices="1",
                )

            self.assertEqual(result["return_code"], 0)
            self.assertEqual(len(observed_workdirs), 1)
            if perf_path is None:
                self.fail("expected msprof perf path")
            self.assertIn("latency-case-1: 5.0", perf_path.read_text(encoding="utf-8"))

    def test_run_local_bench_msprof_parallel_preserves_relative_operator_layout(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_all_cases.py"
            operator_dir = root / "opt-round-13"
            operator_dir.mkdir()
            operator_file = operator_dir / "opt_kernel.py"
            discovered_json = root / "5_MoeInitRouting.json"
            bench_file.write_text(
                "# bench-mode: msprof\n# api-name: kernel\n# kernel: KernelB\n",
                encoding="utf-8",
            )
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")
            discovered_json.write_text('{"cases":[1]}\n', encoding="utf-8")

            observed_workdirs: list[Path] = []

            def _fake_streaming(command, workdir, stall_timeout_seconds, stdout=None, **kwargs):
                del stall_timeout_seconds, stdout, kwargs
                workdir_path = Path(workdir)
                observed_workdirs.append(workdir_path)
                self.assertEqual(workdir_path.name, root.name)
                self.assertTrue((workdir_path / "bench_all_cases.py").exists())
                self.assertTrue((workdir_path / "5_MoeInitRouting.json").exists())
                self.assertTrue((workdir_path / "opt-round-13" / "opt_kernel.py").exists())
                self.assertEqual(
                    command[3:6],
                    ["bench_all_cases.py", "--operator-file", "opt-round-13/opt_kernel.py"],
                )
                output_dir = Path(command[1].split("=", 1)[1])
                (output_dir / "op_statistic_1.csv").write_text(
                    "\n".join(
                        [
                            "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                            "0,KernelB,AI_CORE,1,20,2,5.0,7,100",
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
                    npu_devices="1",
                )

            self.assertEqual(result["return_code"], 0)
            self.assertEqual(len(observed_workdirs), 1)
            if perf_path is None:
                self.fail("expected msprof perf path")

    def test_run_local_bench_msprof_cleans_extra_info_in_bench_workdir(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_abs.py"
            operator_file = root / "abs.py"
            extra_info = root / "extra-info"
            extra_info.mkdir()
            bench_file.write_text(
                "# bench-mode: msprof\n# api-name: abs_\n# kernel: OpB\n",
                encoding="utf-8",
            )
            operator_file.write_text("def abs_():\n    pass\n", encoding="utf-8")

            def _fake_streaming(command, workdir, stall_timeout_seconds, stdout=None, **kwargs):
                del stall_timeout_seconds, stdout, kwargs
                self.assertEqual(workdir, str(root))
                output_dir = Path(command[1].split("=", 1)[1])
                (output_dir / "op_statistic_1.csv").write_text(
                    "\n".join(
                        [
                            "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                            "0,OpB,AI_CORE,1,20,2,5.0,7,100",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                extra_info = root / "extra-info"
                if not extra_info.exists():
                    extra_info.mkdir()
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
            self.assertIsNotNone(perf_path)
            self.assertFalse(extra_info.exists())

    def test_run_local_bench_preserves_non_directory_extra_info_entry(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_abs.py"
            operator_file = root / "abs.py"
            extra_info = root / "extra-info"
            extra_info.write_text("keep me\n", encoding="utf-8")
            bench_file.write_text(
                "# bench-mode: msprof\n# api-name: abs_\n# kernel: OpB\n",
                encoding="utf-8",
            )
            operator_file.write_text("def abs_():\n    pass\n", encoding="utf-8")

            def _fake_streaming(command, workdir, stall_timeout_seconds, stdout=None, **kwargs):
                del workdir, stall_timeout_seconds, stdout, kwargs
                output_dir = Path(command[1].split("=", 1)[1])
                (output_dir / "op_statistic_1.csv").write_text(
                    "\n".join(
                        [
                            "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                            "0,OpB,AI_CORE,1,20,2,5.0,7,100",
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
            self.assertIsNotNone(perf_path)
            self.assertTrue(extra_info.is_file())

    def test_run_local_bench_msprof_suppresses_live_stream_output(self) -> None:
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

            def _fake_streaming(command, workdir, stall_timeout_seconds, stdout=None, **kwargs):
                del workdir, stall_timeout_seconds, kwargs
                print("noisy live output", file=stdout or sys.stdout, end="")
                output_dir = Path(command[1].split("=", 1)[1])
                (output_dir / "op_statistic_1.csv").write_text(
                    "\n".join(
                        [
                            "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                            "0,OpB,AI_CORE,1,20,2,5.0,7,100",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return make_skill_result(0, "profile stdout\n", "")

            stdout = StringIO()
            with patch.object(module, "run_buffered_process", return_value=make_skill_result(0, "1\n", "")), patch.object(
                module,
                "run_streaming_process",
                side_effect=_fake_streaming,
            ), redirect_stdout(stdout):
                result, perf_path = module.run_local_bench(
                    bench_file,
                    operator_file,
                    "msprof",
                )

            self.assertEqual(result["return_code"], 0)
            self.assertIsNotNone(perf_path)
            self.assertEqual(stdout.getvalue(), "")

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

            def _fake_streaming(command, workdir, stall_timeout_seconds, stdout=None, **kwargs):
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
                    '# elapsed-seconds-case-1: 0.000000\n'
                    '# raw-op-statistic-case-1: {"ops":[{"op_type":"OpA","avg_time_us":1.5},{"op_type":"OpB","avg_time_us":2.5}]}\n'
                    '# resolved-kernels-case-1: OpA,OpB\n'
                    '# kernel-source-case-1: metadata\n'
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

            def _fake_streaming(command, workdir, stall_timeout_seconds, stdout=None, **kwargs):
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
                (
                    'latency-case-1: 0.0\n'
                    '# elapsed-seconds-case-1: 0.000000\n'
                    '# raw-op-statistic-case-1: {"ops":[{"op_type":"Zero","avg_time_us":0.0}]}\n'
                    '# resolved-kernels-case-1: Zero\n'
                    '# kernel-source-case-1: metadata\n'
                ),
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

            def _fake_streaming(command, workdir, stall_timeout_seconds, stdout=None, **kwargs):
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

            with patch.dict(os.environ, {"TRITON_AGENT_BENCH_PROFILE_OUTPUT_DIR": str(keep_root)}, clear=False), patch.object(
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
                (
                    'latency-case-1: 4.5\n'
                    '# elapsed-seconds-case-1: 0.000000\n'
                    '# raw-op-statistic-case-1: {"ops":[{"op_type":"KeepMe","avg_time_us":4.5}]}\n'
                    '# resolved-kernels-case-1: KeepMe\n'
                    '# kernel-source-case-1: metadata\n'
                ),
            )
            self.assertTrue(keep_root.exists())
            self.assertTrue(created_output_dirs)
            self.assertTrue(all(path.exists() for path in created_output_dirs))
            self.assertTrue(all(keep_root in path.parents for path in created_output_dirs))
            self.assertTrue(all((path / "op_statistic_1.csv").exists() for path in created_output_dirs))

    def test_run_local_bench_msprof_continues_after_failed_case_and_persists_perf(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_abs.py"
            operator_file = root / "abs.py"
            bench_file.write_text(
                "# bench-mode: msprof\n# api-name: abs_\n# kernel: KernelB\n",
                encoding="utf-8",
            )
            operator_file.write_text("def abs_():\n    pass\n", encoding="utf-8")

            def _fake_streaming(command, workdir, stall_timeout_seconds, stdout=None, **kwargs):
                self.assertEqual(workdir, str(root))
                self.assertEqual(stall_timeout_seconds, 900)
                case_idx = int(command[-1])
                output_dir = Path(command[1].split("=", 1)[1])
                if case_idx == 1:
                    return make_skill_result(1, "", "case one failed\n")
                (output_dir / "op_statistic_2.csv").write_text(
                    "\n".join(
                        [
                            "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                            "0,KernelB,AI_CORE,1,20,2,5.0,7,100",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return make_skill_result(0, "profile 2\n", "")

            with patch.object(module, "run_buffered_process", return_value=make_skill_result(0, "2\n", "")), patch.object(
                module,
                "run_streaming_process",
                side_effect=_fake_streaming,
            ):
                result, perf_path = module.run_local_bench(
                    bench_file,
                    operator_file,
                    "msprof",
                )

            self.assertEqual(result["return_code"], 1)
            if perf_path is None:
                self.fail("expected msprof perf path even when one case fails")
            self.assertEqual(
                perf_path.read_text(encoding="utf-8"),
                (
                    "latency-case-1: NA\n"
                    "# elapsed-seconds-case-1: 0.000000\n"
                    "# latency-error-case-1: msprof command failed with return code 1\n"
                    "# resolved-kernels-case-1: KernelB\n"
                    "# kernel-source-case-1: metadata\n"
                    "latency-case-2: 5.0\n"
                    "# elapsed-seconds-case-2: 0.000000\n"
                    '# raw-op-statistic-case-2: {"ops":[{"op_type":"KernelB","avg_time_us":5.0}]}\n'
                    "# resolved-kernels-case-2: KernelB\n"
                    "# kernel-source-case-2: metadata\n"
                ),
            )

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

            def _fake_streaming(command, workdir, stall_timeout_seconds, stdout=None, **kwargs):
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
                with patch.dict(os.environ, {"TRITON_AGENT_BENCH_PROFILE_OUTPUT_DIR": str(keep_root)}, clear=False), patch.object(
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
            if os.name != "nt":
                self.assertEqual(created_output_dirs[0].stat().st_mode & 0o777, 0o700)

    def test_run_local_bench_msprof_persists_statistic_csv_error_in_perf_file(self) -> None:
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
                result, perf_path = module.run_local_bench(
                    bench_file,
                    operator_file,
                    "msprof",
                )

            self.assertEqual(result["return_code"], 1)
            if perf_path is None:
                self.fail("expected msprof perf path for csv parse failure")
            text = perf_path.read_text(encoding="utf-8")
            self.assertIn("latency-case-1: NA\n", text)
            self.assertIn("# elapsed-seconds-case-1: 0.000000\n", text)
            self.assertIn("# latency-error-case-1: No op_statistic_*.csv found under", text)
            self.assertIn("# resolved-kernels-case-1: OpB\n", text)
            self.assertIn("# kernel-source-case-1: metadata\n", text)

    def test_run_local_bench_msprof_records_na_when_kernel_row_is_missing(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_abs.py"
            operator_file = root / "abs.py"
            bench_file.write_text("# bench-mode: msprof\n# kernel: MissingKernel\n", encoding="utf-8")
            operator_file.write_text("def abs_():\n    pass\n", encoding="utf-8")

            def _fake_streaming(command, workdir, stall_timeout_seconds, stdout=None, **kwargs):
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
                    '# elapsed-seconds-case-1: 0.000000\n'
                    '# raw-op-statistic-case-1: {"ops":[{"op_type":"OpA","avg_time_us":1.5},{"op_type":"OpB","avg_time_us":2.5}]}\n'
                    '# latency-error-case-1: no resolved kernels matched op_statistic csv\n'
                    '# resolved-kernels-case-1: MissingKernel\n'
                    '# kernel-source-case-1: metadata\n'
                ),
            )

    def test_run_local_bench_msprof_elapsed_seconds_in_perf_output_success(self) -> None:
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

            def _fake_streaming(command, workdir, stall_timeout_seconds, stdout=None, **kwargs):
                output_dir = Path(command[1].split("=", 1)[1])
                csv_path = output_dir / "op_statistic_1.csv"
                csv_path.write_text(
                    "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)\n"
                    "0,OpB,AI_CORE,1,10,1,3.0,2,100\n",
                    encoding="utf-8",
                )
                return make_skill_result(0, "", "")

            self._monotonic_patcher.stop()
            try:
                with patch.object(module, "run_buffered_process", return_value=make_skill_result(0, "1\n", "")), patch.object(
                    module,
                    "run_streaming_process",
                    side_effect=_fake_streaming,
                ), patch.object(module.time, "monotonic", side_effect=[0.0, 1.5]):
                    result, perf_path = module.run_local_bench(
                        bench_file,
                        operator_file,
                        "msprof",
                    )
            finally:
                self._monotonic_patcher.start()

            self.assertEqual(result["return_code"], 0)
            if perf_path is None:
                self.fail("expected msprof perf path")
            perf_text = perf_path.read_text(encoding="utf-8")
            self.assertIn("latency-case-1: 3.0\n", perf_text)
            self.assertIn("# elapsed-seconds-case-1: 1.500000\n", perf_text)

    def test_run_local_bench_msprof_elapsed_seconds_in_perf_output_failure(self) -> None:
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

            self._monotonic_patcher.stop()
            try:
                with patch.object(module, "run_buffered_process", return_value=make_skill_result(0, "1\n", "")), patch.object(
                    module,
                    "run_streaming_process",
                    return_value=make_skill_result(1, "", "command failed"),
                ), patch.object(module.time, "monotonic", side_effect=[0.0, 2.5]):
                    result, perf_path = module.run_local_bench(
                        bench_file,
                        operator_file,
                        "msprof",
                    )
            finally:
                self._monotonic_patcher.start()

            self.assertEqual(result["return_code"], 1)
            if perf_path is None:
                self.fail("expected msprof perf path for failed case")
            perf_text = perf_path.read_text(encoding="utf-8")
            self.assertIn("latency-case-1: NA\n", perf_text)
            self.assertIn("# elapsed-seconds-case-1: 2.500000\n", perf_text)

    def test_resolve_bench_kernel_names_unions_metadata_and_operator_kernels(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_abs.py"
            operator_file = root / "abs.py"
            bench_file.write_text(
                "# bench-mode: msprof\n# kernels: MetaKernel\n",
                encoding="utf-8",
            )
            operator_file.write_text(
                "\n".join(
                    [
                        "import triton",
                        "",
                        "@triton.jit",
                        "def MetaKernel(x):",
                        "    return x",
                        "",
                        "@triton.jit()",
                        "def NewKernel(x):",
                        "    return x",
                        "",
                        "def helper():",
                        "    return 1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                module.resolve_bench_kernel_names(bench_file, operator_file),
                ["MetaKernel", "NewKernel"],
            )

    def test_resolve_bench_kernel_names_rejects_malformed_operator_source(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_abs.py"
            operator_file = root / "abs.py"
            bench_file.write_text(
                "# bench-mode: msprof\n# kernels: MetaKernel\n",
                encoding="utf-8",
            )
            operator_file.write_text("def broken(:\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Failed to parse operator file for Triton kernels"):
                module.resolve_bench_kernel_names(bench_file, operator_file)

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

    def test_compare_perf_files_fails_by_default_on_case_execution_error_marker(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_perf.txt"
            compare = root / "compare_perf.txt"
            baseline.write_text(
                (
                    "latency-case-1: NA\n"
                    "# latency-error-case-1: msprof command failed with return code 1\n"
                    "# resolved-kernels-case-1: Kernel\n"
                    "# kernel-source-case-1: metadata\n"
                    "latency-case-2: 10\n"
                ),
                encoding="utf-8",
            )
            compare.write_text("latency-case-1: 8\nlatency-case-2: 8\n", encoding="utf-8")

            stdout_path = root / "stdout.txt"
            original_stdout = sys.stdout
            try:
                with stdout_path.open("w", encoding="utf-8") as handle:
                    sys.stdout = handle
                    return_code = module.compare_perf_files(baseline, compare)
            finally:
                sys.stdout = original_stdout

            output = stdout_path.read_text(encoding="utf-8")
            self.assertEqual(return_code, 1)
            self.assertIn("cannot compare 'latency-case-1'", output)
            self.assertNotIn("Perf comparison:", output)

    def test_compare_perf_files_skips_case_execution_error_when_requested(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_perf.txt"
            compare = root / "compare_perf.txt"
            baseline.write_text(
                (
                    "latency-case-1: NA\n"
                    "# latency-error-case-1: msprof command failed with return code 1\n"
                    "# resolved-kernels-case-1: Kernel\n"
                    "# kernel-source-case-1: metadata\n"
                    "latency-case-2: 10\n"
                ),
                encoding="utf-8",
            )
            compare.write_text("latency-case-1: 8\nlatency-case-2: 8\n", encoding="utf-8")

            stdout_path = root / "stdout.txt"
            original_stdout = sys.stdout
            try:
                with stdout_path.open("w", encoding="utf-8") as handle:
                    sys.stdout = handle
                    return_code = module.compare_perf_files(
                        baseline, compare, skip_latency_errors=True
                    )
            finally:
                sys.stdout = original_stdout

            output = stdout_path.read_text(encoding="utf-8")
            self.assertEqual(return_code, 1)
            self.assertIn("Perf comparison:", output)
            self.assertIn("latency-case-2: baseline=10, compare=8, delta=-20.00%", output)
            self.assertIn("Avg improvement: +20.0%", output)
            self.assertIn("Geomean speedup: 1.25x", output)
            self.assertIn("Total speedup: 1.25x", output)
            self.assertIn("Metric source: kernel", output)
            self.assertIn("FAIL: skipped 1 latency entries due to latency errors", output)
            self.assertIn("latency-case-1", output)
            self.assertIn("latency-error-case-1", output)

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

    def test_compare_perf_files_kernel_mode_fails_when_kernel_is_missing(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_perf.txt"
            compare = root / "compare_perf.txt"
            baseline.write_text(
                (
                    "latency-case-1: NA\n"
                    '# raw-op-statistic-case-1: {"ops":[{"op_type":"OpA","avg_time_us":4.0},{"op_type":"OpB","avg_time_us":6.0}]}\n'
                ),
                encoding="utf-8",
            )
            compare.write_text(
                (
                    "latency-case-1: 3.0\n"
                    '# raw-op-statistic-case-1: {"ops":[{"op_type":"OpA","avg_time_us":2.5},{"op_type":"OpB","avg_time_us":5.0}]}\n'
                ),
                encoding="utf-8",
            )

            stdout_path = root / "stdout.txt"
            original_stdout = sys.stdout
            try:
                with stdout_path.open("w", encoding="utf-8") as handle:
                    sys.stdout = handle
                    return_code = module.compare_perf_files(
                        baseline, compare, metric_source="kernel"
                    )
            finally:
                sys.stdout = original_stdout

            output = stdout_path.read_text(encoding="utf-8")
            self.assertEqual(return_code, 1)
            self.assertIn("requires kernel latency", output)

    def test_compare_perf_files_total_op_mode_uses_raw_totals_even_when_kernel_exists(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_perf.txt"
            compare = root / "compare_perf.txt"
            baseline.write_text(
                (
                    "latency-case-1: 10\n"
                    '# raw-op-statistic-case-1: {"ops":[{"op_type":"OpA","avg_time_us":4.0},{"op_type":"OpB","avg_time_us":6.0}]}\n'
                ),
                encoding="utf-8",
            )
            compare.write_text(
                (
                    "latency-case-1: 3.0\n"
                    '# raw-op-statistic-case-1: {"ops":[{"op_type":"OpA","avg_time_us":2.5},{"op_type":"OpB","avg_time_us":5.0}]}\n'
                ),
                encoding="utf-8",
            )

            stdout_path = root / "stdout.txt"
            original_stdout = sys.stdout
            try:
                with stdout_path.open("w", encoding="utf-8") as handle:
                    sys.stdout = handle
                    return_code = module.compare_perf_files(
                        baseline, compare, metric_source="total-op"
                    )
            finally:
                sys.stdout = original_stdout

            output = stdout_path.read_text(encoding="utf-8")
            self.assertEqual(return_code, 0)
            self.assertIn("baseline=total-op=10.0", output)
            self.assertIn("compare=total-op=7.5", output)
            self.assertIn("Metric source: total-op", output)

    def test_compare_perf_files_total_op_mode_fails_when_raw_totals_are_missing(self) -> None:
        module = load_bench_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_perf.txt"
            compare = root / "compare_perf.txt"
            baseline.write_text("latency-case-1: 10\n", encoding="utf-8")
            compare.write_text("latency-case-1: 8\n", encoding="utf-8")

            stdout_path = root / "stdout.txt"
            original_stdout = sys.stdout
            try:
                with stdout_path.open("w", encoding="utf-8") as handle:
                    sys.stdout = handle
                    return_code = module.compare_perf_files(
                        baseline, compare, metric_source="total-op"
                    )
            finally:
                sys.stdout = original_stdout

            output = stdout_path.read_text(encoding="utf-8")
            self.assertEqual(return_code, 1)
            self.assertIn("requires '# raw-op-statistic-case-1: ...'", output)

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
