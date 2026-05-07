import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.run_skill_test_utils import (
    load_standalone_bench_runtime_module,
    make_skill_result,
)


class StandaloneBenchRuntimeTests(unittest.TestCase):
    def test_load_standalone_bench_cases_builds_hooked_cases(self) -> None:
        module = load_standalone_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
            bench_file.write_text(
                """# bench-mode: standalone
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA, KernelB

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_standalone_bench_cases(operator_api):
    prepared = {"token": "bound"}
    def run_case():
        operator_api("case-a", prepared)
    return [{"id": "case-a", "fn": run_case, "warmup": 3, "repeats": 7}]
""",
                encoding="utf-8",
            )
            operator_file.write_text(
                """def build_api():
    def operator_api(name, prepared):
        return name, prepared["token"]
    return operator_api
""",
                encoding="utf-8",
            )

            cases, resolution = module.load_standalone_bench_cases(bench_file, operator_file)

        self.assertEqual([case.case_id for case in cases], ["case-a"])
        self.assertEqual(cases[0].warmup, 3)
        self.assertEqual(cases[0].repeats, 7)
        self.assertEqual(resolution.kernel_names, ["KernelA", "KernelB"])

    def test_load_standalone_bench_cases_rejects_missing_hooks_and_duplicate_ids(self) -> None:
        module = load_standalone_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
            operator_file.write_text("def build_api():\n    return lambda *_args, **_kwargs: None\n", encoding="utf-8")

            bench_file.write_text("def build_standalone_bench_cases(operator_api):\n    return []\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required hook 'build_operator_api'"):
                module.load_standalone_bench_cases(bench_file, operator_file)

            bench_file.write_text(
                """def build_operator_api(operator_module):
    return operator_module.build_api()

def build_standalone_bench_cases(operator_api):
    def run_case():
        operator_api()
    return [{"id": "case-a", "fn": run_case}, {"id": "case-a", "fn": run_case}]
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate standalone benchmark case id: case-a"):
                module.load_standalone_bench_cases(bench_file, operator_file)

    def test_run_local_standalone_bench_writes_msprof_shaped_perf_lines(self) -> None:
        module = load_standalone_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
            bench_file.write_text(
                """# bench-mode: standalone
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA, KernelB

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_standalone_bench_cases(operator_api):
    def run_case():
        operator_api("case-a")
    return [{"id": "case-a", "fn": run_case}]
""",
                encoding="utf-8",
            )
            operator_file.write_text(
                """def build_api():
    return lambda *_args, **_kwargs: None
""",
                encoding="utf-8",
            )

            with patch.object(
                module,
                "_profile_case_with_profiler",
                return_value=(
                    {
                        "kernel_avg_time_us": 11.0,
                        "ops": [
                            {"op_type": "KernelA", "avg_time_us": 5.0},
                            {"op_type": "KernelB", "avg_time_us": 6.0},
                        ],
                    },
                    None,
                ),
            ):
                result, perf_path = module.run_local_standalone_bench(bench_file, operator_file)
                perf_text = perf_path.read_text(encoding="utf-8")

        self.assertEqual(result, make_skill_result(0, "", ""))
        if perf_path is None:
            self.fail("expected standalone perf path")
        self.assertEqual(
            perf_text,
            (
                'latency-case-a: 11.0\n'
                '# raw-op-statistic-case-a: {"ops":[{"op_type":"KernelA","avg_time_us":5.0},{"op_type":"KernelB","avg_time_us":6.0}]}\n'
                '# resolved-kernels-case-a: KernelA,KernelB\n'
                '# kernel-source-case-a: metadata\n'
            ),
        )

    def test_run_local_standalone_bench_keeps_profiler_artifacts_under_configured_output_root(self) -> None:
        module = load_standalone_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
            keep_root = root / "kept-profile"
            bench_file.write_text(
                """# bench-mode: standalone
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_standalone_bench_cases(operator_api):
    def run_case_a():
        operator_api("case-a")
    def run_case_b():
        operator_api("case-b")
    return [{"id": "case-a", "fn": run_case_a}, {"id": "case-b", "fn": run_case_b}]
""",
                encoding="utf-8",
            )
            operator_file.write_text(
                """def build_api():
    return lambda *_args, **_kwargs: None
""",
                encoding="utf-8",
            )

            created_output_dirs: list[Path] = []

            def _fake_profile_case(case, resolution, profile_root):
                del resolution
                created_output_dirs.append(profile_root)
                profile_root.mkdir(parents=True, exist_ok=True)
                (profile_root / f"{case.case_id}.txt").write_text("kept\n", encoding="utf-8")
                return (
                    {
                        "kernel_avg_time_us": 7.5,
                        "ops": [{"op_type": "KernelA", "avg_time_us": 7.5}],
                    },
                    None,
                )

            with patch.dict(
                os.environ,
                {"TRITON_AGENT_BENCH_PROFILE_OUTPUT_DIR": str(keep_root)},
                clear=False,
            ), patch.object(
                module,
                "_profile_case_with_profiler",
                side_effect=_fake_profile_case,
            ):
                result, perf_path = module.run_local_standalone_bench(bench_file, operator_file)
            self.assertEqual(result, make_skill_result(0, "", ""))
            self.assertEqual(perf_path, root / "operator_case_perf.txt")
            self.assertTrue(keep_root.exists())
            self.assertEqual(len(created_output_dirs), 2)
            self.assertTrue(all(path.exists() for path in created_output_dirs))
            self.assertTrue(all(keep_root in path.parents for path in created_output_dirs))
            self.assertTrue((created_output_dirs[0] / "case-a.txt").exists())
            self.assertTrue((created_output_dirs[1] / "case-b.txt").exists())
            self.assertEqual(sorted(path.name for path in created_output_dirs), ["case-case-a", "case-case-b"])


if __name__ == "__main__":
    unittest.main()
