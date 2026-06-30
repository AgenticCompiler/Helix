import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

from tests.run_skill_test_utils import (
    load_bench_runtime_module,
    make_skill_result,
)


class StandaloneBenchRuntimeTests(unittest.TestCase):
    def test_load_bench_cases_bootstraps_torch_before_user_module_exec(self) -> None:
        module = load_bench_runtime_module()
        import_events: list[str] = []

        def fake_import(name: str, package: Optional[str] = None):
            import_events.append(name)
            if name == "torch":
                return SimpleNamespace(npu=SimpleNamespace())
            if name == "torch_npu":
                return SimpleNamespace()
            return original_import(name, package)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
            bench_file.write_text(
                """# bench-mode: torch-npu-profiler
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_bench_cases():
    return [{"id": "case-a"}]

def build_bench_case_fn(operator_api, case):
    return lambda: operator_api(case["id"])
""",
                encoding="utf-8",
            )
            operator_file.write_text(
                """def build_api():
    return lambda *_args, **_kwargs: None
""",
                encoding="utf-8",
            )
            original_import = module.importlib.import_module
            with patch.object(module.importlib, "import_module", side_effect=fake_import):
                cases, _resolution = module.load_bench_cases(bench_file, operator_file)

        self.assertEqual([case.case_id for case in cases], ["case-a"])
        self.assertGreaterEqual(import_events[:2], ["torch", "torch_npu"])

    def test_load_bench_cases_builds_callables_without_eager_execution(self) -> None:
        module = load_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
            trace_file = root / "trace.txt"
            bench_file.write_text(
                f"""# bench-mode: torch-npu-profiler
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA, KernelB

TRACE_PATH = r"{trace_file}"

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_bench_cases():
    with open(TRACE_PATH, "a", encoding="utf-8") as handle:
        handle.write("declare\\n")
    return [{{"id": "case-a", "shape": (16,), "warmup": 3, "repeats": 7}}]

def build_bench_case_fn(operator_api, case):
    with open(TRACE_PATH, "a", encoding="utf-8") as handle:
        handle.write(f"build:{{case['id']}}\\n")
    def _run():
        with open(TRACE_PATH, "a", encoding="utf-8") as handle:
            handle.write(f"run:{{case['id']}}\\n")
        operator_api(case["id"], case["shape"])
    return _run
""",
                encoding="utf-8",
            )
            operator_file.write_text(
                """def build_api():
    def operator_api(case_id, shape):
        return case_id, shape
    return operator_api
""",
                encoding="utf-8",
            )

            cases, resolution = module.load_bench_cases(bench_file, operator_file)

            self.assertEqual([case.case_id for case in cases], ["case-a"])
            self.assertEqual(cases[0].warmup, 3)
            self.assertEqual(cases[0].repeats, 7)
            self.assertEqual(cases[0].case_data["id"], "case-a")
            self.assertEqual(resolution.kernel_names, ["KernelA", "KernelB"])
            self.assertEqual(trace_file.read_text(encoding="utf-8"), "declare\nbuild:case-a\n")

            cases[0].fn()

            self.assertEqual(
                trace_file.read_text(encoding="utf-8"),
                "declare\nbuild:case-a\nrun:case-a\n",
            )

    def test_load_bench_cases_rejects_missing_hooks_duplicate_ids_and_empty_case_lists(self) -> None:
        module = load_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
            operator_file.write_text("def build_api():\n    return lambda *_args, **_kwargs: None\n", encoding="utf-8")

            bench_file.write_text("def build_bench_cases():\n    return []\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required hook 'build_operator_api'"):
                module.load_bench_cases(bench_file, operator_file)

            bench_file.write_text(
                """def build_operator_api(operator_module):
    return operator_module.build_api()

def build_bench_cases():
    return []

def build_bench_case_fn(operator_api, case):
    return lambda: operator_api(case["id"])
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "build_bench_cases' returned no cases"):
                module.load_bench_cases(bench_file, operator_file)

            bench_file.write_text(
                """def build_operator_api(operator_module):
    return operator_module.build_api()

def build_bench_cases():
    return [{"id": "case-a"}, {"id": "case-a"}]

def build_bench_case_fn(operator_api, case):
    return lambda: operator_api(case["id"])
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate benchmark case id: case-a"):
                module.load_bench_cases(bench_file, operator_file)

    def test_load_bench_cases_rejects_non_callable_case_builders_and_invalid_case_selection(self) -> None:
        module = load_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
            operator_file.write_text("def build_api():\n    return lambda *_args, **_kwargs: None\n", encoding="utf-8")

            bench_file.write_text(
                """# bench-mode: torch-npu-profiler
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_bench_cases():
    return [{"id": "case-a"}]

def build_bench_case_fn(operator_api, case):
    return object()
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must return a callable"):
                module.load_bench_cases(bench_file, operator_file)

            bench_file.write_text(
                """# bench-mode: torch-npu-profiler
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_bench_cases():
    return [{"id": "case-a"}, {"id": "case-b"}]

def build_bench_case_fn(operator_api, case):
    return lambda: operator_api(case["id"])
""",
                encoding="utf-8",
            )
            cases, _resolution = module.load_bench_cases(bench_file, operator_file)

            selected = module.select_bench_case(cases, "case-b")
            self.assertEqual(selected.case_id, "case-b")

            with self.assertRaisesRegex(ValueError, "Available case ids: case-a, case-b"):
                module.select_bench_case(cases, "case-missing")

            with self.assertRaisesRegex(ValueError, "requires --case-id when multiple cases exist"):
                module.select_bench_case(cases, None)

            single_case = [cases[0]]
            self.assertEqual(module.select_bench_case(single_case, None).case_id, "case-a")

    def test_load_bench_cases_rejects_legacy_standalone_hook_contract(self) -> None:
        module = load_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
            bench_file.write_text(
                """# bench-mode: torch-npu-profiler
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

            with self.assertRaisesRegex(ValueError, "missing required hook 'build_bench_cases'"):
                module.load_bench_cases(bench_file, operator_file)

    def test_profile_all_bench_cases_writes_msprof_shaped_perf_lines(self) -> None:
        module = load_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
            bench_file.write_text(
                """# bench-mode: torch-npu-profiler
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA, KernelB

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_bench_cases():
    return [{"id": "case-a"}]

def build_bench_case_fn(operator_api, case):
    def run_case():
        operator_api(case["id"])
    return run_case
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
            ), patch.object(module.time, "monotonic", side_effect=[0.0, 1.5]):
                result, perf_path = module.profile_all_bench_cases(bench_file, operator_file)
                perf_text = perf_path.read_text(encoding="utf-8")

        self.assertEqual(result, make_skill_result(0, "", ""))
        self.assertEqual(
            perf_text,
            (
                '{"case_label":"case-a","kernel_names":["KernelA","KernelB"],"kernel_source":"metadata","kernel_avg_time_us":11.0,"ops":[{"op_type":"KernelA","avg_time_us":5.0},{"op_type":"KernelB","avg_time_us":6.0}],"total_op_avg_time_us":11.0,"error_message":null,"case_wall_clock_seconds":1.5,"bench_mode":"torch-npu-profiler"}\n'
            ),
        )

    def test_profile_all_bench_cases_case_wall_clock_seconds_on_failure(self) -> None:
        module = load_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
            bench_file.write_text(
                """# bench-mode: torch-npu-profiler
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_bench_cases():
    return [{"id": "case-a"}]

def build_bench_case_fn(operator_api, case):
    def run_case():
        operator_api(case["id"])
    return run_case
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
                return_value=(None, "profiling failed"),
            ), patch.object(module.time, "monotonic", side_effect=[0.0, 2.5]):
                result, perf_path = module.profile_all_bench_cases(bench_file, operator_file)
                perf_text = perf_path.read_text(encoding="utf-8")

        self.assertEqual(result, make_skill_result(1, "", "case-a: profiling failed"))
        self.assertIn('"kernel_avg_time_us":null', perf_text)
        self.assertIn('"case_wall_clock_seconds":2.5', perf_text)
        self.assertIn('"error_message":"profiling failed"', perf_text)
        self.assertIn('"case_label":"case-a"', perf_text)
        self.assertIn('"kernel_names":["KernelA"]', perf_text)
        self.assertIn('"kernel_source":"metadata"', perf_text)

    def test_profile_all_bench_cases_keeps_profiler_artifacts_under_configured_output_root(self) -> None:
        module = load_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
            keep_root = root / "kept-profile"
            bench_file.write_text(
                """# bench-mode: torch-npu-profiler
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_bench_cases():
    return [{"id": "case-a"}, {"id": "case-b"}]

def build_bench_case_fn(operator_api, case):
    def run_case():
        operator_api(case["id"])
    return run_case
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

            def _fake_profile_case(case, resolution, profile_root, *, verbose=False):
                del resolution, verbose
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
                {"TRITON_AGENT_BENCH_OUTPUT_DIR": str(keep_root)},
                clear=False,
            ), patch.object(
                module,
                "_profile_case_with_profiler",
                side_effect=_fake_profile_case,
            ):
                result, perf_path = module.profile_all_bench_cases(bench_file, operator_file)
            self.assertEqual(result, make_skill_result(0, "", ""))
            self.assertEqual(perf_path, root / "operator_case_perf.txt")
            self.assertTrue(keep_root.exists())
            self.assertEqual(len(created_output_dirs), 2)
            self.assertTrue(all(path.exists() for path in created_output_dirs))
            self.assertTrue(all(keep_root.resolve() in path.resolve().parents for path in created_output_dirs))
            self.assertTrue((created_output_dirs[0] / "case-a.txt").exists())
            self.assertTrue((created_output_dirs[1] / "case-b.txt").exists())
            self.assertEqual(sorted(path.name for path in created_output_dirs), ["case-case-a", "case-case-b"])

    def test_profile_all_bench_cases_resolves_relative_profile_output_root_to_absolute_path(self) -> None:
        module = load_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
            keep_root = root / "relative-keep-root"
            bench_file.write_text(
                """# bench-mode: torch-npu-profiler
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_bench_cases():
    return [{"id": "case-a"}]

def build_bench_case_fn(operator_api, case):
    def run_case():
        operator_api(case["id"])
    return run_case
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

            def _fake_profile_case(case, resolution, profile_root, *, verbose=False):
                del case, resolution, verbose
                created_output_dirs.append(profile_root)
                profile_root.mkdir(parents=True, exist_ok=True)
                return (
                    {
                        "kernel_avg_time_us": 7.5,
                        "ops": [{"op_type": "KernelA", "avg_time_us": 7.5}],
                    },
                    None,
                )

            original_cwd = Path.cwd()
            try:
                os.chdir(root)
                with patch.dict(
                    os.environ,
                    {"TRITON_AGENT_BENCH_OUTPUT_DIR": "./relative-keep-root"},
                    clear=False,
                ), patch.object(
                    module,
                    "_profile_case_with_profiler",
                    side_effect=_fake_profile_case,
                ):
                    result, perf_path = module.profile_all_bench_cases(bench_file, operator_file)
            finally:
                os.chdir(original_cwd)

            self.assertEqual(result, make_skill_result(0, "", ""))
            self.assertEqual(perf_path, root / "operator_case_perf.txt")
            self.assertTrue(keep_root.exists())
            self.assertEqual(len(created_output_dirs), 1)
            self.assertTrue(created_output_dirs[0].is_absolute())
            self.assertTrue(keep_root.resolve() in created_output_dirs[0].resolve().parents)

    def test_profile_all_bench_cases_cleans_extra_info_after_each_case(self) -> None:
        module = load_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
            extra_info = root / "extra-info"
            extra_info.mkdir()
            bench_file.write_text(
                """# bench-mode: torch-npu-profiler
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_bench_cases():
    return [{"id": "case-a"}, {"id": "case-b"}]

def build_bench_case_fn(operator_api, case):
    def run_case():
        operator_api(case["id"])
    return run_case
""",
                encoding="utf-8",
            )
            operator_file.write_text(
                """def build_api():
    return lambda *_args, **_kwargs: None
""",
                encoding="utf-8",
            )

            seen_after_cleanup: list[bool] = []

            def _fake_profile_case(case, resolution, profile_root, *, verbose=False):
                del resolution, profile_root, verbose
                seen_after_cleanup.append(extra_info.exists())
                if case.case_id == "case-a":
                    if not extra_info.exists():
                        extra_info.mkdir()
                return (
                    {
                        "kernel_avg_time_us": 7.5,
                        "ops": [{"op_type": "KernelA", "avg_time_us": 7.5}],
                    },
                    None,
                )

            with patch.object(
                module,
                "_profile_case_with_profiler",
                side_effect=_fake_profile_case,
            ):
                result, perf_path = module.profile_all_bench_cases(bench_file, operator_file)

            self.assertEqual(result, make_skill_result(0, "", ""))
            self.assertEqual(perf_path, root / "operator_case_perf.txt")
            self.assertEqual(seen_after_cleanup, [True, False])
            self.assertFalse(extra_info.exists())

    def test_profile_all_bench_cases_reuses_single_case_helper(self) -> None:
        module = load_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
            bench_file.write_text(
                """# bench-mode: torch-npu-profiler
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_bench_cases():
    return [{"id": "case-a"}, {"id": "case-b"}]

def build_bench_case_fn(operator_api, case):
    def run_case():
        operator_api(case["id"])
    return run_case
""",
                encoding="utf-8",
            )
            operator_file.write_text(
                """def build_api():
    return lambda *_args, **_kwargs: None
""",
                encoding="utf-8",
            )

            observed_case_ids: list[str] = []

            def _fake_run_case(case, resolution, preserved_run_dir, cleanup_workdir, *, verbose=False):
                del resolution, preserved_run_dir, cleanup_workdir, verbose
                observed_case_ids.append(case.case_id)
                return module.PerfCaseRecord(
                    case_label=case.case_id,
                    kernel_names=["KernelA"],
                    kernel_source="metadata",
                    metrics={
                        "kernel_avg_time_us": 7.5,
                        "ops": [{"op_type": "KernelA", "avg_time_us": 7.5}],
                    },
                    case_wall_clock_seconds=1.0,
                )

            with patch.object(module, "_run_bench_case", side_effect=_fake_run_case):
                result, perf_path = module.profile_all_bench_cases(bench_file, operator_file)

            self.assertEqual(result, make_skill_result(0, "", ""))
            self.assertEqual(observed_case_ids, ["case-a", "case-b"])
            self.assertEqual(perf_path, root / "operator_case_perf.txt")

    def test_profile_bench_case_returns_selected_case_record(self) -> None:
        module = load_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
            bench_file.write_text(
                """# bench-mode: torch-npu-profiler
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_bench_cases():
    return [{"id": "case-a"}, {"id": "case-b"}]

def build_bench_case_fn(operator_api, case):
    def run_case():
        operator_api(case["id"])
    return run_case
""",
                encoding="utf-8",
            )
            operator_file.write_text(
                """def build_api():
    return lambda *_args, **_kwargs: None
""",
                encoding="utf-8",
            )

            def _fake_run_case(case, resolution, preserved_run_dir, cleanup_workdir, *, verbose=False):
                del resolution, preserved_run_dir, cleanup_workdir, verbose
                return module.PerfCaseRecord(
                    case_label=case.case_id,
                    kernel_names=["KernelA"],
                    kernel_source="metadata",
                    metrics={
                        "kernel_avg_time_us": 3.5,
                        "ops": [{"op_type": "KernelA", "avg_time_us": 3.5}],
                    },
                    case_wall_clock_seconds=2.0,
                )

            with patch.object(module, "_run_bench_case", side_effect=_fake_run_case):
                record = module.profile_bench_case(bench_file, operator_file, "case-b")

            self.assertEqual(record.case_label, "case-b")
            self.assertEqual(record.metrics["kernel_avg_time_us"], 3.5)

    def test_read_profiler_metrics_prefers_kernel_details_and_uses_step_totals(self) -> None:
        module = load_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            profile_root = Path(tmp)
            (profile_root / "operator_details.csv").write_text(
                "\n".join(
                    [
                        "Name,Device Self Duration(us)",
                        "aclnnMul,999",
                        "aclnnInplaceCopy,111",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (profile_root / "kernel_details.csv").write_text(
                "\n".join(
                    [
                        "Step Id,Name,Duration(us),Wait Time(us),Block Dim",
                        "6,KernelA_kernel,5.0,0,3",
                        "6,HelperKernel,3.0,0,3",
                        "7,KernelA_kernel,8.0,0,3",
                        "7,HelperKernel,4.0,0,3",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            metrics = module._read_profiler_metrics(
                profile_root,
                active_count=2,
                kernel_names=["KernelA"],
            )

        self.assertEqual(
            metrics["ops"],
            [
                {"op_type": "KernelA_kernel", "avg_time_us": 6.5},
                {"op_type": "HelperKernel", "avg_time_us": 3.5},
            ],
        )
        self.assertEqual(metrics["kernel_avg_time_us"], 6.5)
        self.assertEqual(metrics["total_op_avg_time_us"], 10.0)

    def test_read_profiler_metrics_ignores_operator_details_without_kernel_view_csv(self) -> None:
        module = load_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            profile_root = Path(tmp)
            (profile_root / "operator_details.csv").write_text(
                "\n".join(
                    [
                        "Name,Device Self Duration(us)",
                        "aclnnMul,10",
                        "aclnnInplaceCopy,2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "No kernel_details.csv or op_statistic.csv found"):
                module._read_profiler_metrics(
                    profile_root,
                    active_count=5,
                    kernel_names=["KernelZero"],
                )

    def test_read_profiler_metrics_falls_back_to_op_statistic_when_operator_details_is_missing(self) -> None:
        module = load_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            profile_root = Path(tmp)
            (profile_root / "op_statistic.csv").write_text(
                "\n".join(
                    [
                        "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                        "0,KernelA,AI_CORE,5,20,1,4.0,6,80",
                        "0,KernelB,AI_VECTOR_CORE,5,5,0.5,1.0,2,20",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            metrics = module._read_profiler_metrics(
                profile_root,
                active_count=5,
                kernel_names=["KernelB"],
            )

        self.assertEqual(
            metrics["ops"],
            [
                {"op_type": "KernelA", "avg_time_us": 4.0},
                {"op_type": "KernelB", "avg_time_us": 1.0},
            ],
        )
        self.assertEqual(metrics["kernel_avg_time_us"], 1.0)
        self.assertEqual(metrics["total_op_avg_time_us"], 5.0)

    def test_read_profiler_metrics_falls_back_to_kernel_details_when_operator_details_is_missing(self) -> None:
        module = load_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            profile_root = Path(tmp)
            (profile_root / "kernel_details.csv").write_text(
                "\n".join(
                    [
                        "Step Id,Name,Duration(us),Wait Time(us),Block Dim",
                        "6,KernelA,9.0,0,3",
                        "7,KernelA,6.0,0,3",
                        "8,KernelB,3.0,0,1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            metrics = module._read_profiler_metrics(
                profile_root,
                active_count=3,
                kernel_names=["KernelA"],
            )

        self.assertEqual(
            metrics["ops"],
            [
                {"op_type": "KernelA", "avg_time_us": 5.0},
                {"op_type": "KernelB", "avg_time_us": 1.0},
            ],
        )
        self.assertEqual(metrics["kernel_avg_time_us"], 5.0)
        self.assertEqual(metrics["total_op_avg_time_us"], 6.0)

    def test_read_profiler_metrics_falls_back_to_kernel_details_when_operator_details_total_is_zero(self) -> None:
        module = load_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            profile_root = Path(tmp)
            (profile_root / "operator_details.csv").write_text(
                "\n".join(
                    [
                        "Name,Device Self Duration(us)",
                        "Wrapper,0",
                        "KernelZero,0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (profile_root / "kernel_details.csv").write_text(
                "\n".join(
                    [
                        "Step Id,Name,Duration(us),Wait Time(us),Block Dim",
                        "6,KernelFromKernelDetails,5.0,0,3",
                        "7,KernelFromKernelDetails,5.0,0,3",
                        "8,KernelFromKernelDetails,5.0,0,3",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (profile_root / "op_statistic.csv").write_text(
                "\n".join(
                    [
                        "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                        "0,KernelFromOpStatistic,AI_CORE,5,25,2,5.0,7,100",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            metrics = module._read_profiler_metrics(
                profile_root,
                active_count=3,
                kernel_names=["KernelFromKernelDetails"],
            )

        self.assertEqual(
            metrics["ops"],
            [
                {"op_type": "KernelFromKernelDetails", "avg_time_us": 5.0},
            ],
        )
        self.assertEqual(metrics["kernel_avg_time_us"], 5.0)
        self.assertEqual(metrics["total_op_avg_time_us"], 5.0)

    def test_read_profiler_metrics_falls_back_to_op_statistic_with_kernel_suffix_alias(self) -> None:
        module = load_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            profile_root = Path(tmp)
            (profile_root / "op_statistic.csv").write_text(
                "\n".join(
                    [
                        "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                        "0,KernelA_kernel,AI_CORE,2,13,5,6.5,8,65",
                        "0,HelperKernel,AI_VECTOR_CORE,2,7,3,3.5,4,35",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            metrics = module._read_profiler_metrics(
                profile_root,
                active_count=2,
                kernel_names=["KernelA"],
            )

        self.assertEqual(
            metrics["ops"],
            [
                {"op_type": "KernelA_kernel", "avg_time_us": 6.5},
                {"op_type": "HelperKernel", "avg_time_us": 3.5},
            ],
        )
        self.assertEqual(metrics["kernel_avg_time_us"], 6.5)
        self.assertEqual(metrics["total_op_avg_time_us"], 10.0)

    def test_read_profiler_metrics_op_statistic_fallback_prefers_active_count_over_count_proxy(self) -> None:
        module = load_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            profile_root = Path(tmp)
            (profile_root / "op_statistic.csv").write_text(
                "\n".join(
                    [
                        "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                        "0,engram_hash_kernel_kernel,MIX_AIC,45,9227.36,204.78,205.052,205.34,56.884",
                        "0,BroadcastTo,AI_VECTOR_CORE,180,1913.52,5.72,10.63,16.4,11.796",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            metrics = module._read_profiler_metrics(
                profile_root,
                active_count=50,
                kernel_names=["engram_hash_kernel"],
            )

        self.assertAlmostEqual(metrics["kernel_avg_time_us"], 184.5472, places=6)
        self.assertEqual(
            metrics["ops"],
            [
                {"op_type": "engram_hash_kernel_kernel", "avg_time_us": 184.5472},
                {"op_type": "BroadcastTo", "avg_time_us": 38.2704},
            ],
        )
        self.assertAlmostEqual(metrics["total_op_avg_time_us"], 222.8176, places=6)

    def test_profile_case_with_profiler_suppresses_profiler_output(self) -> None:
        module = load_bench_runtime_module()

        class _FakeProfilerContext:
            def __init__(self, profile_root: Path, on_trace_ready):
                self.profile_root = profile_root
                self.on_trace_ready = on_trace_ready

            def __enter__(self):
                print("profile enter stdout")
                print("profile enter stderr", file=sys.stderr)
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                self.profile_root.mkdir(parents=True, exist_ok=True)
                csv_path = self.profile_root / "kernel_details.csv"
                csv_path.write_text(
                    "\n".join(
                        [
                            "Name,Duration(us),Wait Time(us),Block Dim",
                            "KernelA,4.0,0,1",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                if callable(self.on_trace_ready):
                    self.on_trace_ready()
                print("profile exit stdout")
                print("profile exit stderr", file=sys.stderr)
                return False

            def step(self):
                print("profile step stdout")
                print("profile step stderr", file=sys.stderr)

        class _FakeProfilerApi:
            profile_root: Optional[Path] = None

            class _ExperimentalConfig:
                def __init__(self, **kwargs):
                    del kwargs

            class ProfilerLevel:
                Level1 = object()

            class ProfilerActivity:
                NPU = object()
                CPU = object()

            @staticmethod
            def schedule(**kwargs):
                return kwargs

            @staticmethod
            def tensorboard_trace_handler(profile_root: str):
                _FakeProfilerApi.profile_root = Path(profile_root)

                def _handler():
                    Path(profile_root).mkdir(parents=True, exist_ok=True)

                return _handler

            @staticmethod
            def profile(**kwargs):
                profile_root = _FakeProfilerApi.profile_root
                if profile_root is None:
                    raise AssertionError("expected tensorboard_trace_handler to set profile_root")
                return _FakeProfilerContext(profile_root, kwargs["on_trace_ready"])

        fake_torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))
        fake_torch_npu = SimpleNamespace(profiler=_FakeProfilerApi())
        case = module.BenchCase(
            case_id="case-a",
            fn=lambda: (print("case stdout"), print("case stderr", file=sys.stderr)),
            warmup=0,
            repeats=1,
            case_data={"id": "case-a"},
        )
        resolution = module.KernelResolution(kernel_names=["KernelA"], kernel_source="metadata")

        stdout = StringIO()
        stderr = StringIO()
        with patch.dict(
            "sys.modules",
            {"torch": fake_torch, "torch_npu": fake_torch_npu},
            clear=False,
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            metrics, error_message = module._profile_case_with_profiler(
                case,
                resolution,
                Path(tempfile.mkdtemp()) / "profile",
            )

        self.assertIsNotNone(metrics)
        self.assertIsNone(error_message)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")

    def test_profile_case_with_profiler_suppresses_fd_level_output(self) -> None:
        module = load_bench_runtime_module()

        class _FakeProfilerContext:
            def __init__(self, profile_root: Path):
                self.profile_root = profile_root

            def __enter__(self):
                os.write(1, b"fd stdout before\n")
                os.write(2, b"fd stderr before\n")
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                self.profile_root.mkdir(parents=True, exist_ok=True)
                (self.profile_root / "kernel_details.csv").write_text(
                    "\n".join(
                        [
                            "Name,Duration(us),Wait Time(us),Block Dim",
                            "KernelA,4.0,0,1",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                os.write(1, b"fd stdout after\n")
                os.write(2, b"fd stderr after\n")
                return False

            def step(self):
                os.write(1, b"fd stdout step\n")
                os.write(2, b"fd stderr step\n")

        class _FakeProfilerApi:
            profile_root: Optional[Path] = None

            class _ExperimentalConfig:
                def __init__(self, **kwargs):
                    del kwargs

            class ProfilerLevel:
                Level1 = object()

            class ProfilerActivity:
                NPU = object()
                CPU = object()

            @staticmethod
            def schedule(**kwargs):
                return kwargs

            @staticmethod
            def tensorboard_trace_handler(profile_root: str):
                _FakeProfilerApi.profile_root = Path(profile_root)

                def _handler():
                    Path(profile_root).mkdir(parents=True, exist_ok=True)

                return _handler

            @staticmethod
            def profile(**kwargs):
                profile_root = _FakeProfilerApi.profile_root
                if profile_root is None:
                    raise AssertionError("expected tensorboard_trace_handler to set profile_root")
                return _FakeProfilerContext(profile_root)

        fake_torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))
        fake_torch_npu = SimpleNamespace(profiler=_FakeProfilerApi())
        case = module.BenchCase(
            case_id="case-a",
            fn=lambda: (
                os.write(1, b"case fd stdout\n"),
                os.write(2, b"case fd stderr\n"),
            ),
            warmup=0,
            repeats=1,
            case_data={"id": "case-a"},
        )
        resolution = module.KernelResolution(kernel_names=["KernelA"], kernel_source="metadata")

        stdout = StringIO()
        stderr = StringIO()
        with patch.dict(
            "sys.modules",
            {"torch": fake_torch, "torch_npu": fake_torch_npu},
            clear=False,
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            metrics, error_message = module._profile_case_with_profiler(
                case,
                resolution,
                Path(tempfile.mkdtemp()) / "profile",
            )

        self.assertIsNotNone(metrics)
        self.assertIsNone(error_message)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")

    def test_profile_case_with_profiler_preserves_active_iterations_after_warmup(self) -> None:
        module = load_bench_runtime_module()
        per_iteration_us = 4.0

        class _FakeProfilerAction:
            NONE = "NONE"
            WARMUP = "WARMUP"
            RECORD = "RECORD"
            RECORD_AND_SAVE = "RECORD_AND_SAVE"

        class _FakeSchedule:
            def __init__(
                self,
                *,
                wait: int,
                warmup: int,
                active: int,
                repeat: int = 0,
                skip_first: int = 0,
                skip_first_wait: int = 0,
            ) -> None:
                self.wait = wait
                self.warmup = warmup
                self.active = active
                self.repeat = repeat
                self.skip_first = skip_first
                self.skip_first_wait = skip_first_wait

            def __call__(self, step: int) -> str:
                if step < self.skip_first:
                    return _FakeProfilerAction.NONE
                step -= self.skip_first
                if self.skip_first_wait != 0:
                    step += self.wait
                num_steps = self.wait + self.warmup + self.active
                if self.repeat > 0 and step / num_steps >= self.repeat:
                    return _FakeProfilerAction.NONE
                mod_step = step % num_steps
                if mod_step < self.wait:
                    return _FakeProfilerAction.NONE
                if mod_step < self.wait + self.warmup:
                    return _FakeProfilerAction.WARMUP
                if mod_step < num_steps - 1:
                    return _FakeProfilerAction.RECORD
                return _FakeProfilerAction.RECORD_AND_SAVE

        class _FakeProfilerContext:
            def __init__(self, profile_root: Path, schedule_fn, on_trace_ready):
                self.profile_root = profile_root
                self.schedule_fn = schedule_fn
                self.on_trace_ready = on_trace_ready
                self.step_num = 0
                self.current_action = self.schedule_fn(self.step_num)
                self.recorded_iterations = 0

            def __enter__(self):
                _FakeProfilerApi.current_context = self
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                _FakeProfilerApi.current_context = None
                self.profile_root.mkdir(parents=True, exist_ok=True)
                (self.profile_root / "kernel_details.csv").write_text(
                    "\n".join(
                        [
                            "Name,Duration(us),Wait Time(us),Block Dim",
                            f"KernelA,{self.recorded_iterations * per_iteration_us},0,1",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                if callable(self.on_trace_ready):
                    self.on_trace_ready()
                return False

            def step(self):
                self.step_num += 1
                self.current_action = self.schedule_fn(self.step_num)

        class _FakeProfilerApi:
            profile_root: Optional[Path] = None
            current_context: Optional[_FakeProfilerContext] = None

            class _ExperimentalConfig:
                def __init__(self, **kwargs):
                    del kwargs

            class ProfilerLevel:
                Level1 = object()

            class ProfilerActivity:
                NPU = object()
                CPU = object()

            @staticmethod
            def schedule(**kwargs):
                return _FakeSchedule(**kwargs)

            @staticmethod
            def tensorboard_trace_handler(profile_root: str):
                _FakeProfilerApi.profile_root = Path(profile_root)

                def _handler():
                    Path(profile_root).mkdir(parents=True, exist_ok=True)

                return _handler

            @staticmethod
            def profile(**kwargs):
                profile_root = _FakeProfilerApi.profile_root
                if profile_root is None:
                    raise AssertionError("expected tensorboard_trace_handler to set profile_root")
                return _FakeProfilerContext(
                    profile_root,
                    kwargs["schedule"],
                    kwargs["on_trace_ready"],
                )

        def _run_case() -> None:
            ctx = _FakeProfilerApi.current_context
            if ctx is None:
                return
            if ctx.current_action in (
                _FakeProfilerAction.RECORD,
                _FakeProfilerAction.RECORD_AND_SAVE,
            ):
                ctx.recorded_iterations += 1

        fake_torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))
        fake_torch_npu = SimpleNamespace(profiler=_FakeProfilerApi())
        case = module.BenchCase(
            case_id="case-a",
            fn=_run_case,
            warmup=1,
            repeats=3,
            case_data={"id": "case-a"},
        )
        resolution = module.KernelResolution(kernel_names=["KernelA"], kernel_source="metadata")

        with patch.dict(
            "sys.modules",
            {"torch": fake_torch, "torch_npu": fake_torch_npu},
            clear=False,
        ):
            metrics, error_message = module._profile_case_with_profiler(
                case,
                resolution,
                Path(tempfile.mkdtemp()) / "profile",
            )

        self.assertIsNone(error_message)
        self.assertIsNotNone(metrics)
        self.assertEqual(
            metrics["ops"],
            [{"op_type": "KernelA", "avg_time_us": per_iteration_us}],
        )
        self.assertEqual(metrics["kernel_avg_time_us"], per_iteration_us)

    # ------------------------------------------------------------------
    # perf-counter timing functions
    # ------------------------------------------------------------------

    def test_time_case_iterations_returns_per_iteration_average_us(self) -> None:
        module = load_bench_runtime_module()
        call_count = 0

        def fake_fn() -> None:
            nonlocal call_count
            call_count += 1

        metrics = module._time_case_iterations(
            fn=fake_fn,
            warmup=2,
            repeats=3,
        )
        self.assertEqual(call_count, 5)  # 2 warmup + 3 measurement
        self.assertIsInstance(metrics["kernel_avg_time_us"], float)
        self.assertGreater(metrics["kernel_avg_time_us"], 0.0)
        self.assertEqual(metrics["ops"], [])

    def test_time_all_bench_cases_produces_jsonl_with_perf_counter_bench_mode(self) -> None:
        module = load_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
            bench_file.write_text(
                """# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_bench_cases():
    return [{"id": "case-a"}]

def build_bench_case_fn(operator_api, case):
    def run_case():
        operator_api(case["id"])
    return run_case
""",
                encoding="utf-8",
            )
            operator_file.write_text(
                "def build_api():\n    return lambda *_args, **_kwargs: None\n",
                encoding="utf-8",
            )

            result, perf_path = module.time_all_bench_cases(
                bench_file, operator_file, bench_mode="perf-counter",
            )
            perf_text = perf_path.read_text(encoding="utf-8")

        self.assertEqual(result["return_code"], 0)
        self.assertIn('"bench_mode":"perf-counter"', perf_text)
        self.assertIn('"kernel_avg_time_us":', perf_text)
        self.assertIn('"ops":[]', perf_text)
        self.assertIn('"case_wall_clock_seconds":', perf_text)
        self.assertNotIn('"bench_mode":null', perf_text)

    def test_time_all_bench_cases_sets_triton_always_compile(self) -> None:
        module = load_bench_runtime_module()
        saved = os.environ.get("TRITON_ALWAYS_COMPILE")

        try:
            if "TRITON_ALWAYS_COMPILE" in os.environ:
                del os.environ["TRITON_ALWAYS_COMPILE"]

            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                bench_file = root / "bench_case.py"
                operator_file = root / "operator_case.py"
                bench_file.write_text(
                    """# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_bench_cases():
    return [{"id": "case-a", "warmup": 0, "repeats": 2}]

def build_bench_case_fn(operator_api, case):
    def run_case():
        pass
    return run_case
""",
                    encoding="utf-8",
                )
                operator_file.write_text(
                    "def build_api():\n    return lambda: None\n",
                    encoding="utf-8",
                )

                observed_value: list[Optional[str]] = []

                _original_time_iterations = module._time_case_iterations

                def _wrap_time_iterations(
                    fn, warmup, repeats,
                ):
                    observed_value.append(
                        os.environ.get("TRITON_ALWAYS_COMPILE"),
                    )
                    return _original_time_iterations(
                        fn=fn, warmup=warmup, repeats=repeats,
                    )

                with patch.object(
                    module,
                    "_time_case_iterations",
                    side_effect=_wrap_time_iterations,
                ):
                    module.time_all_bench_cases(
                        bench_file, operator_file,
                        bench_mode="perf-counter",
                    )

            self.assertEqual(observed_value, ["1"])
            self.assertNotIn("TRITON_ALWAYS_COMPILE", os.environ)
        finally:
            if saved is not None:
                os.environ["TRITON_ALWAYS_COMPILE"] = saved
            elif "TRITON_ALWAYS_COMPILE" in os.environ:
                del os.environ["TRITON_ALWAYS_COMPILE"]

    def test_time_all_bench_cases_records_error_on_case_fn_exception(self) -> None:
        module = load_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
            bench_file.write_text(
                """# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_bench_cases():
    return [{"id": "case-fail", "warmup": 0, "repeats": 2}]

def build_bench_case_fn(operator_api, case):
    def run_case():
        raise RuntimeError("boom")
    return run_case
""",
                encoding="utf-8",
            )
            operator_file.write_text(
                "def build_api():\n    return lambda: None\n",
                encoding="utf-8",
            )

            result, perf_path = module.time_all_bench_cases(
                bench_file, operator_file, bench_mode="perf-counter",
            )
            perf_text = perf_path.read_text(encoding="utf-8")

        self.assertEqual(result["return_code"], 1)
        self.assertIn('"error_message":"RuntimeError: boom"', perf_text)
        self.assertIn('"case_wall_clock_seconds":', perf_text)
        self.assertIn('"bench_mode":"perf-counter"', perf_text)
        self.assertIn('"case_label":"case-fail"', perf_text)


class ExecuteBenchCaseIterationsTests(unittest.TestCase):
    def _run(self, *, iterations: Optional[int]) -> int:
        module = load_bench_runtime_module()
        call_count = 0

        def _fn() -> None:
            nonlocal call_count
            call_count += 1

        fake_case = SimpleNamespace(case_id="case-a", fn=_fn)
        with patch.object(module, "load_bench_cases", return_value=([fake_case], object())), patch.object(
            module, "select_bench_case", return_value=fake_case
        ), patch.object(module, "_synchronize"):
            kwargs = {} if iterations is None else {"iterations": iterations}
            result = module.execute_bench_case(Path("bench.py"), Path("op.py"), "case-a", **kwargs)
        self.assertEqual(result["return_code"], 0)
        return call_count

    def test_defaults_to_single_invocation(self) -> None:
        self.assertEqual(self._run(iterations=None), 1)

    def test_runs_requested_iteration_count(self) -> None:
        self.assertEqual(self._run(iterations=55), 55)


if __name__ == "__main__":
    unittest.main()
