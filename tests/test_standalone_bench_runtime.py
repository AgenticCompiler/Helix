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
            ), patch.object(module.time, "monotonic", side_effect=[0.0, 1.5]):
                result, perf_path = module.run_local_standalone_bench(bench_file, operator_file)
                perf_text = perf_path.read_text(encoding="utf-8")

        self.assertEqual(result, make_skill_result(0, "", ""))
        if perf_path is None:
            self.fail("expected standalone perf path")
        self.assertEqual(
            perf_text,
            (
                '{"case_label":"case-a","kernel_names":["KernelA","KernelB"],"kernel_source":"metadata","kernel_avg_time_us":11.0,"ops":[{"op_type":"KernelA","avg_time_us":5.0},{"op_type":"KernelB","avg_time_us":6.0}],"total_op_avg_time_us":11.0,"error_message":null,"case_wall_clock_seconds":1.5}\n'
            ),
        )

    def test_run_local_standalone_bench_case_wall_clock_seconds_on_failure(self) -> None:
        module = load_standalone_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
            bench_file.write_text(
                """# bench-mode: standalone
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

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
                return_value=(None, "profiling failed"),
            ), patch.object(module.time, "monotonic", side_effect=[0.0, 2.5]):
                result, perf_path = module.run_local_standalone_bench(bench_file, operator_file)
                perf_text = perf_path.read_text(encoding="utf-8")

        self.assertEqual(result, make_skill_result(1, "", "case-a: profiling failed"))
        if perf_path is None:
            self.fail("expected standalone perf path")
        self.assertIn('"kernel_avg_time_us":null', perf_text)
        self.assertIn('"case_wall_clock_seconds":2.5', perf_text)
        self.assertIn('"error_message":"profiling failed"', perf_text)
        self.assertIn('"case_label":"case-a"', perf_text)
        self.assertIn('"kernel_names":["KernelA"]', perf_text)
        self.assertIn('"kernel_source":"metadata"', perf_text)

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
                result, perf_path = module.run_local_standalone_bench(bench_file, operator_file)
            self.assertEqual(result, make_skill_result(0, "", ""))
            self.assertEqual(perf_path, root / "operator_case_perf.txt")
            self.assertTrue(keep_root.exists())
            self.assertEqual(len(created_output_dirs), 2)
            self.assertTrue(all(path.exists() for path in created_output_dirs))
            self.assertTrue(all(keep_root.resolve() in path.resolve().parents for path in created_output_dirs))
            self.assertTrue((created_output_dirs[0] / "case-a.txt").exists())
            self.assertTrue((created_output_dirs[1] / "case-b.txt").exists())
            self.assertEqual(sorted(path.name for path in created_output_dirs), ["case-case-a", "case-case-b"])

    def test_run_local_standalone_bench_resolves_relative_profile_output_root_to_absolute_path(self) -> None:
        module = load_standalone_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
            keep_root = root / "relative-keep-root"
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
    return [{"id": "case-a", "fn": run_case_a}]
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
                    result, perf_path = module.run_local_standalone_bench(bench_file, operator_file)
            finally:
                os.chdir(original_cwd)

            self.assertEqual(result, make_skill_result(0, "", ""))
            self.assertEqual(perf_path, root / "operator_case_perf.txt")
            self.assertTrue(keep_root.exists())
            self.assertEqual(len(created_output_dirs), 1)
            self.assertTrue(created_output_dirs[0].is_absolute())
            self.assertTrue(keep_root.resolve() in created_output_dirs[0].resolve().parents)

    def test_run_local_standalone_bench_cleans_extra_info_after_each_case(self) -> None:
        module = load_standalone_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
            extra_info = root / "extra-info"
            extra_info.mkdir()
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
                result, perf_path = module.run_local_standalone_bench(bench_file, operator_file)

            self.assertEqual(result, make_skill_result(0, "", ""))
            self.assertEqual(perf_path, root / "operator_case_perf.txt")
            self.assertEqual(seen_after_cleanup, [True, False])
            self.assertFalse(extra_info.exists())

    def test_run_local_standalone_bench_reuses_single_case_helper(self) -> None:
        module = load_standalone_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
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

            with patch.object(module, "_run_standalone_case", side_effect=_fake_run_case):
                result, perf_path = module.run_local_standalone_bench(bench_file, operator_file)

            self.assertEqual(result, make_skill_result(0, "", ""))
            self.assertEqual(observed_case_ids, ["case-a", "case-b"])
            self.assertEqual(perf_path, root / "operator_case_perf.txt")

    def test_run_one_standalone_case_record_returns_selected_case_record(self) -> None:
        module = load_standalone_bench_runtime_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
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

            with patch.object(module, "_run_standalone_case", side_effect=_fake_run_case):
                record = module.run_one_standalone_case_record(bench_file, operator_file, "case-b")

            self.assertEqual(record.case_label, "case-b")
            self.assertEqual(record.metrics["kernel_avg_time_us"], 3.5)

    def test_profile_case_with_profiler_suppresses_profiler_output(self) -> None:
        module = load_standalone_bench_runtime_module()

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
                csv_path = self.profile_root / "operator_details.csv"
                csv_path.write_text(
                    "\n".join(
                        [
                            "Name,Device Self Duration(us),Count",
                            "KernelA,4.0,1",
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
        case = module.StandaloneBenchCase(
            case_id="case-a",
            fn=lambda: (print("case stdout"), print("case stderr", file=sys.stderr)),
            warmup=0,
            repeats=1,
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
        module = load_standalone_bench_runtime_module()

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
                (self.profile_root / "operator_details.csv").write_text(
                    "\n".join(
                        [
                            "Name,Device Self Duration(us),Count",
                            "KernelA,4.0,1",
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
        case = module.StandaloneBenchCase(
            case_id="case-a",
            fn=lambda: (
                os.write(1, b"case fd stdout\n"),
                os.write(2, b"case fd stderr\n"),
            ),
            warmup=0,
            repeats=1,
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


if __name__ == "__main__":
    unittest.main()
