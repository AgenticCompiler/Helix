import sys
import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

from tests.run_skill_test_utils import load_simulator_runner_module, make_skill_result


class SimulatorRunnerTests(unittest.TestCase):
    def test_run_local_simulator_defaults_single_case_and_single_kernel(self) -> None:
        module = load_simulator_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            bench_file.write_text("# bench-mode: torch-npu-profiler\n# kernel: KernelA\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            with patch.object(module, "_load_bench_runtime_module") as load_runtime, patch.object(
                module,
                "resolve_bench_kernel_resolution",
                return_value=type("_Resolution", (), {"kernel_names": ["KernelA"], "kernel_source": "metadata"})(),
            ), patch.object(
                module,
                "run_streaming_process",
                return_value=make_skill_result(0, "stdout\n", ""),
            ) as mocked:
                load_runtime.return_value = type(
                    "_FakeRuntime",
                    (),
                    {
                        "load_bench_cases": staticmethod(
                            lambda *_args, **_kwargs: ([type("_Case", (), {"case_id": "only-case"})()], None)
                        ),
                        "select_bench_case": staticmethod(lambda cases, _case_id: cases[0]),
                    },
                )()
                result = module.run_local_simulator(bench_file, operator_file, case_id=None, kernel_name=None)

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(
            mocked.call_args.args[0],
            [
                "msprof",
                "op",
                "simulator",
                "--soc-version=Ascend950PR_9599",
                "--kernel-name=KernelA",
                sys.executable,
                str(module._bench_runtime_script_path()),
                "run-one",
                "--bench-file",
                "bench_kernel.py",
                "--operator-file",
                "kernel.py",
                "--case-id",
                "only-case",
            ],
        )

    def test_run_local_simulator_reads_soc_version_from_environment(self) -> None:
        module = load_simulator_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            bench_file.write_text("# bench-mode: torch-npu-profiler\n# kernel: KernelA\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            with patch.dict(os.environ, {"HELIX_SIMULATOR_SOC_VERSION": "Ascend910B_TEST"}, clear=False):
                with patch.object(module, "_load_bench_runtime_module") as load_runtime, patch.object(
                    module,
                    "resolve_bench_kernel_resolution",
                    return_value=type("_Resolution", (), {"kernel_names": ["KernelA"], "kernel_source": "metadata"})(),
                ), patch.object(
                    module,
                    "run_streaming_process",
                    return_value=make_skill_result(0, "stdout\n", ""),
                ) as mocked:
                    load_runtime.return_value = type(
                        "_FakeRuntime",
                        (),
                        {
                            "load_bench_cases": staticmethod(
                                lambda *_args, **_kwargs: ([type("_Case", (), {"case_id": "only-case"})()], None)
                            ),
                            "select_bench_case": staticmethod(lambda cases, _case_id: cases[0]),
                        },
                    )()
                    result = module.run_local_simulator(bench_file, operator_file, case_id=None, kernel_name=None)

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(
            mocked.call_args.args[0][:5],
            [
                "msprof",
                "op",
                "simulator",
                "--soc-version=Ascend910B_TEST",
                "--kernel-name=KernelA",
            ],
        )

    def test_run_local_simulator_requires_case_id_when_multiple_cases_exist(self) -> None:
        module = load_simulator_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            bench_file.write_text("# bench-mode: msprof\n# kernels: KernelA\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            with patch.object(module, "_load_bench_runtime_module") as load_runtime:
                load_runtime.return_value = type(
                    "_FakeRuntime",
                    (),
                    {
                        "load_bench_cases": staticmethod(
                            lambda *_args, **_kwargs: (
                                [
                                    type("_Case", (), {"case_id": "case-a"})(),
                                    type("_Case", (), {"case_id": "case-b"})(),
                                ],
                                None,
                            )
                        ),
                        "select_bench_case": staticmethod(
                            lambda _cases, _case_id: (_ for _ in ()).throw(
                                ValueError(
                                    "Benchmark profiling requires --case-id when multiple cases exist. "
                                    "Available case ids: case-a, case-b"
                                )
                            )
                        ),
                    },
                )()
                with self.assertRaisesRegex(ValueError, "requires --case-id when multiple cases exist"):
                    module.run_local_simulator(bench_file, operator_file, case_id=None, kernel_name="KernelA")

    def test_run_local_simulator_requires_kernel_name_when_multiple_kernels_resolve(self) -> None:
        module = load_simulator_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            bench_file.write_text("# bench-mode: msprof\n# kernels: KernelA, KernelB\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            with patch.object(module, "_load_bench_runtime_module") as load_runtime, patch.object(
                module,
                "resolve_bench_kernel_resolution",
                return_value=type("_Resolution", (), {"kernel_names": ["KernelA", "KernelB"], "kernel_source": "metadata"})(),
            ):
                load_runtime.return_value = type(
                    "_FakeRuntime",
                    (),
                    {
                        "load_bench_cases": staticmethod(
                            lambda *_args, **_kwargs: ([type("_Case", (), {"case_id": "only-case"})()], None)
                        ),
                        "select_bench_case": staticmethod(lambda cases, _case_id: cases[0]),
                    },
                )()
                with self.assertRaisesRegex(ValueError, "requires --kernel-name when multiple kernels resolve"):
                    module.run_local_simulator(bench_file, operator_file, case_id=None, kernel_name=None)

    def test_run_local_simulator_rejects_unknown_kernel_name(self) -> None:
        module = load_simulator_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            bench_file.write_text("# bench-mode: msprof\n# kernels: KernelA\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            with patch.object(module, "_load_bench_runtime_module") as load_runtime, patch.object(
                module,
                "resolve_bench_kernel_resolution",
                return_value=type("_Resolution", (), {"kernel_names": ["KernelA"], "kernel_source": "metadata"})(),
            ):
                load_runtime.return_value = type(
                    "_FakeRuntime",
                    (),
                    {
                        "load_bench_cases": staticmethod(
                            lambda *_args, **_kwargs: ([type("_Case", (), {"case_id": "only-case"})()], None)
                        ),
                        "select_bench_case": staticmethod(lambda cases, _case_id: cases[0]),
                    },
                )()
                with self.assertRaisesRegex(ValueError, "Unknown simulator kernel 'KernelB'"):
                    module.run_local_simulator(bench_file, operator_file, case_id=None, kernel_name="KernelB")


if __name__ == "__main__":
    unittest.main()
