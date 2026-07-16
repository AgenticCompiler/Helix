from __future__ import annotations

import os
import subprocess
import shutil
import sys
import tempfile
import unittest
import importlib.util
import json
from io import StringIO
from pathlib import Path
from typing import Optional, TextIO, get_type_hints
from unittest.mock import patch

_TRITON_ROUND_OPERATOR = """\
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=128)
    return out
"""

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUN_EVAL_SCRIPT_DIR = _REPO_ROOT / "skills" / "common" / "ascend-npu-run-eval" / "scripts"
_OPTIMIZE_STATE_SCRIPT = (
    _REPO_ROOT / "skills" / "common" / "ascend-npu-optimize-state" / "scripts" / "cli.py"
)
if str(_RUN_EVAL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_RUN_EVAL_SCRIPT_DIR))

_REMOTE_TARGET_ENV = "HELIX_REMOTE"
_REMOTE_WORKDIR_ENV = "HELIX_REMOTE_WORKDIR"


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class SkillCommandScriptTests(unittest.TestCase):
    def test_optimize_state_submit_round_resolve_min_speedup_reads_env_without_args(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-optimize-state"
            / "scripts"
            / "state_manage"
            / "submit_round.py"
        )
        spec = importlib.util.spec_from_file_location("optimize_submit_round_module", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        before = list(sys.path)
        try:
            sys.path.insert(0, str(script.parents[1]))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = before

        with patch.dict(os.environ, {"HELIX_OPTIMIZE_MIN_SPEEDUP": "1.20"}, clear=False):
            self.assertEqual(module._resolve_min_speedup(), 1.2)

    def test_loading_run_command_does_not_mutate_sys_path(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "run_test_command.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test_sys_path", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")

        before = list(sys.path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertEqual(sys.path, before)

    def test_run_bench_parser_accepts_npu_devices_flag(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        args = module.build_parser().parse_args(
            [
                "run-bench",
                "--bench-file",
                "bench_abs.py",
                "--operator-file",
                "opt_abs.py",
                "--npu-devices",
                "0,1,4-5",
            ]
        )

        self.assertEqual(args.npu_devices, "0,1,4-5")

    def test_run_bench_parser_accepts_baseline_operator_file(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test_baseline_flag", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        args = module.build_parser().parse_args(
            [
                "run-bench",
                "--bench-file",
                "bench_abs.py",
                "--operator-file",
                "opt_abs.py",
                "--baseline-operator-file",
                "baseline_abs.py",
            ]
        )

        self.assertEqual(args.baseline_operator_file, "baseline_abs.py")

    def test_run_bench_parser_accepts_compare_perf_options(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test_bench_compare_flags", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        args = module.build_parser().parse_args(
            [
                "run-bench",
                "--bench-file",
                "bench_abs.py",
                "--operator-file",
                "opt_abs.py",
                "--baseline-operator-file",
                "baseline_abs.py",
                "--skip-latency-errors",
                "--metric-source",
                "all",
            ]
        )

        self.assertTrue(args.skip_latency_errors)
        self.assertEqual(args.metric_source, "all")

    def test_script_run_bench_threads_npu_devices_to_local_runner(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            perf_file = root / "kernel_perf.txt"
            operator.write_text("print('x')\n", encoding="utf-8")
            bench_file.write_text("# bench-mode: msprof\nprint('bench')\n", encoding="utf-8")

            observed_args: list[object] = []

            def fake_run_local_bench(
                bench_path: Path,
                operator_path: Path,
                bench_mode: str,
                npu_devices: Optional[str] = None,
                **kwargs: object,
            ) -> tuple[dict[str, object], Path]:
                observed_args.extend([bench_path, operator_path, bench_mode, npu_devices])
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    perf_file,
                )

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.object(
                    module,
                    "_load_bench_functions",
                    return_value=(
                        lambda _path: {"bench-mode": "msprof"},
                        fake_run_local_bench,
                        lambda *_args, **_kwargs: None,
                    ),
                ):
                    exit_code = module.main(
                        [
                            "run-bench",
                            "--bench-file",
                            str(bench_file),
                            "--operator-file",
                            str(operator),
                            "--npu-devices",
                            "0,2",
                        ]
                    )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            observed_args,
            [
                bench_file.resolve(),
                operator.resolve(),
                "torch-npu-profiler",
                "0,2",
            ],
        )
        self.assertEqual(
            stdout.getvalue(),
            (
                f"Perf file: {perf_file}\n"
                "Hint: use `compare-perf` to inspect this perf artifact instead of reading it directly.\n"
            ),
        )
        self.assertEqual(stderr.getvalue(), "")

    def test_script_run_bench_threads_output_to_local_runner(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test_output", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            perf_file = root / "custom_perf.txt"
            operator.write_text("print('x')\n", encoding="utf-8")
            bench_file.write_text("# bench-mode: msprof\nprint('bench')\n", encoding="utf-8")

            observed: list[object] = []

            def fake_run_local_bench(
                bench_path: Path,
                operator_path: Path,
                bench_mode: str,
                npu_devices: Optional[str] = None,
                **kwargs: object,
            ) -> tuple[dict[str, object], Path]:
                observed.extend([bench_path, operator_path, bench_mode, npu_devices, kwargs.get("output")])
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    perf_file,
                )

            with patch.object(
                module,
                "_load_bench_functions",
                return_value=(
                    lambda _path: {"bench-mode": "msprof"},
                    fake_run_local_bench,
                    lambda *_args, **_kwargs: None,
                ),
            ):
                exit_code = module.main(
                    [
                        "run-bench",
                        "--bench-file",
                        str(bench_file),
                        "--operator-file",
                        str(operator),
                        "--output",
                        str(perf_file),
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            observed,
            [bench_file.resolve(), operator.resolve(), "torch-npu-profiler", None, str(perf_file)],
        )

    def test_script_run_bench_with_baseline_operator_auto_compares(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test_baseline_compare", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.py"
            operator = root / "opt_kernel.py"
            bench_file = root / "bench_kernel.py"
            baseline_perf = root / "baseline_perf.txt"
            candidate_perf = root / "opt_kernel_perf.txt"
            baseline.write_text("print('baseline')\n", encoding="utf-8")
            operator.write_text("print('x')\n", encoding="utf-8")
            bench_file.write_text("# bench-mode: msprof\nprint('bench')\n", encoding="utf-8")
            baseline_perf.write_text('{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":1.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n', encoding="utf-8")

            observed: list[object] = []
            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr

            def fake_run_local_bench(
                bench_path: Path,
                operator_path: Path,
                bench_mode: str,
                npu_devices: Optional[str] = None,
                **kwargs: object,
            ) -> tuple[dict[str, object], Path]:
                observed.extend([bench_path, operator_path, bench_mode, npu_devices, kwargs.get("output")])
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    candidate_perf,
                )

            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.object(
                    module,
                    "_load_bench_functions",
                    return_value=(
                        lambda _path: {"bench-mode": "msprof"},
                        fake_run_local_bench,
                        lambda *_args, **_kwargs: None,
                    ),
                ):
                    with patch.object(module, "_load_compare_perf_function", return_value=lambda baseline_path, new_path, **_kwargs: 0):
                        exit_code = module.main(
                            [
                                "run-bench",
                                "--bench-file",
                                str(bench_file),
                                "--operator-file",
                                str(operator),
                                "--baseline-operator-file",
                                str(baseline),
                            ]
                        )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            observed,
            [bench_file.resolve(), operator.resolve(), "torch-npu-profiler", None, None],
        )
        self.assertEqual(
            stdout.getvalue(),
            (
                f"Baseline perf file: {baseline_perf.resolve()}\n"
                f"Perf file: {candidate_perf}\n"
            ),
        )

    def test_script_run_bench_remote_prints_both_kept_workspaces_when_baseline_is_generated(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test_remote_baseline_compare", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.py"
            operator = root / "opt_kernel.py"
            bench_file = root / "bench_kernel.py"
            baseline_perf = root / "baseline_perf.txt"
            candidate_perf = root / "opt_kernel_perf.txt"
            baseline.write_text("print('baseline')\n", encoding="utf-8")
            operator.write_text("print('x')\n", encoding="utf-8")
            bench_file.write_text("# bench-mode: msprof\nprint('bench')\n", encoding="utf-8")

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr

            def fake_run_remote_bench(
                bench_path: Path,
                operator_path: Path,
                bench_mode: str,
                remote: str,
                remote_workdir: Optional[str],
                npu_devices: Optional[str] = None,
                keep_remote_workdir: bool = False,
                verbose: bool = False,
                stderr: Optional[object] = None,
                **kwargs: object,
            ) -> tuple[dict[str, object], Path, str]:
                del bench_path, bench_mode, remote, remote_workdir, npu_devices, keep_remote_workdir, verbose, stderr, kwargs
                if operator_path == baseline.resolve():
                    return (
                        {
                            "return_code": 0,
                            "stdout": "",
                            "stderr": "",
                            "stalled": False,
                            "session_id": None,
                        },
                        baseline_perf,
                        "/tmp/baseline-ws",
                    )
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    candidate_perf,
                    "/tmp/candidate-ws",
                )

            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.object(
                    module,
                    "_load_bench_functions",
                    return_value=(
                        lambda _path: {"bench-mode": "msprof"},
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("local runner should not be used")),
                        fake_run_remote_bench,
                    ),
                ):
                    with patch.object(module, "_load_compare_perf_function", return_value=lambda baseline_path, new_path, **_kwargs: 0):
                        exit_code = module.main(
                            [
                                "run-bench",
                                "--bench-file",
                                str(bench_file),
                                "--operator-file",
                                str(operator),
                                "--baseline-operator-file",
                                str(baseline),
                                "--remote",
                                "alice@example.com",
                                "--keep-remote-workdir",
                            ]
                        )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            stdout.getvalue(),
            (
                f"Baseline perf file: {baseline_perf}\n"
                "Remote workspace: /tmp/baseline-ws\n"
                "Remote workspace: /tmp/candidate-ws\n"
                f"Perf file: {candidate_perf}\n"
            ),
        )

    def test_script_run_bench_remote_failure_with_perf_prints_baseline_workspace_once(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test_remote_baseline_failure", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.py"
            operator = root / "opt_kernel.py"
            bench_file = root / "bench_kernel.py"
            baseline_perf = root / "baseline_perf.txt"
            baseline.write_text("print('baseline')\n", encoding="utf-8")
            operator.write_text("print('x')\n", encoding="utf-8")
            bench_file.write_text("# bench-mode: msprof\nprint('bench')\n", encoding="utf-8")

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr

            def fake_run_remote_bench(
                bench_path: Path,
                operator_path: Path,
                bench_mode: str,
                remote: str,
                remote_workdir: Optional[str],
                npu_devices: Optional[str] = None,
                keep_remote_workdir: bool = False,
                verbose: bool = False,
                stderr: Optional[object] = None,
                **kwargs: object,
            ) -> tuple[dict[str, object], Path, str]:
                del bench_path, operator_path, bench_mode, remote, remote_workdir, npu_devices, keep_remote_workdir, verbose, stderr, kwargs
                return (
                    {
                        "return_code": 1,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    baseline_perf,
                    "/tmp/baseline-ws",
                )

            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.object(
                    module,
                    "_load_bench_functions",
                    return_value=(
                        lambda _path: {"bench-mode": "msprof"},
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("local runner should not be used")),
                        fake_run_remote_bench,
                    ),
                ):
                    exit_code = module.main(
                        [
                            "run-bench",
                            "--bench-file",
                            str(bench_file),
                            "--operator-file",
                            str(operator),
                            "--baseline-operator-file",
                            str(baseline),
                            "--remote",
                            "alice@example.com",
                            "--keep-remote-workdir",
                        ]
                    )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue().count("Remote workspace: /tmp/baseline-ws\n"), 1)

    def test_script_run_bench_forwards_compare_perf_options(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test_compare_flags", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.py"
            operator = root / "opt_kernel.py"
            bench_file = root / "bench_kernel.py"
            baseline_perf = root / "baseline_perf.txt"
            candidate_perf = root / "opt_kernel_perf.txt"
            baseline.write_text("print('baseline')\n", encoding="utf-8")
            operator.write_text("print('x')\n", encoding="utf-8")
            bench_file.write_text("# bench-mode: msprof\nprint('bench')\n", encoding="utf-8")
            baseline_perf.write_text('{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":1.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n', encoding="utf-8")

            observed_compare: list[object] = []

            def fake_compare_perf(
                baseline_path: Path,
                compare_path: Path,
                *,
                skip_latency_errors: bool = False,
                metric_source: str = "auto",
            ) -> int:
                observed_compare.extend([baseline_path, compare_path, skip_latency_errors, metric_source])
                return 0

            with patch.object(
                module,
                "_load_bench_functions",
                return_value=(
                    lambda _path: {"bench-mode": "msprof"},
                    lambda *_args, **_kwargs: (
                        {
                            "return_code": 0,
                            "stdout": "",
                            "stderr": "",
                            "stalled": False,
                            "session_id": None,
                        },
                        candidate_perf,
                    ),
                    lambda *_args, **_kwargs: None,
                ),
            ):
                with patch.object(module, "_load_compare_perf_function", return_value=fake_compare_perf):
                    exit_code = module.main(
                        [
                            "run-bench",
                            "--bench-file",
                            str(bench_file),
                            "--operator-file",
                            str(operator),
                            "--baseline-operator-file",
                            str(baseline),
                            "--skip-latency-errors",
                            "--metric-source",
                            "total-op",
                        ]
                    )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            observed_compare,
            [baseline_perf.resolve(), candidate_perf, True, "total-op"],
        )

    def test_script_run_bench_uses_remote_env_when_flag_missing(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test_remote_bench", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            bench_file.write_text("# bench-mode: torch-npu-profiler\nprint('bench')\n", encoding="utf-8")

            observed: list[object] = []

            def fake_run_remote_bench(
                bench_path: Path,
                operator_path: Path,
                bench_mode: str,
                remote: str,
                remote_workdir: Optional[str],
                npu_devices: Optional[str] = None,
                keep_remote_workdir: bool = False,
                verbose: bool = False,
                stderr: Optional[object] = None,
                **kwargs: object,
            ) -> tuple[dict[str, object], None, str]:
                del keep_remote_workdir, verbose, stderr
                observed.extend([bench_path, operator_path, bench_mode, remote, remote_workdir, npu_devices])
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    None,
                    "/tmp/helix-123",
                )

            with patch.dict(
                os.environ,
                {
                    _REMOTE_TARGET_ENV: "alice@example.com",
                    _REMOTE_WORKDIR_ENV: "/tmp/helix",
                },
                clear=False,
            ):
                with patch.object(
                    module,
                    "_load_bench_functions",
                    return_value=(
                        lambda _path: {"bench-mode": "torch-npu-profiler"},
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("local runner should not be used")),
                        fake_run_remote_bench,
                    ),
                ):
                    exit_code = module.main(
                        [
                            "run-bench",
                            "--bench-file",
                            str(bench_file),
                            "--operator-file",
                            str(operator),
                        ]
                    )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            observed,
            [
                bench_file.resolve(),
                operator.resolve(),
                "torch-npu-profiler",
                "alice@example.com",
                "/tmp/helix",
                None,
            ],
        )

    def test_compare_perf_parser_accepts_skip_latency_errors_flag(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        args = module.build_parser().parse_args(
            [
                "compare-perf",
                "--baseline",
                "baseline_perf.txt",
                "--compare",
                "candidate_perf.txt",
                "--skip-latency-errors",
            ]
        )

        self.assertTrue(args.skip_latency_errors)

    def test_compare_perf_parser_accepts_metric_source_flag(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        args = module.build_parser().parse_args(
            [
                "compare-perf",
                "--baseline",
                "baseline_perf.txt",
                "--compare",
                "candidate_perf.txt",
                "--metric-source",
                "kernel",
            ]
        )

        self.assertEqual(args.metric_source, "kernel")

    def test_run_bench_parser_accepts_metric_source_short_alias(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_metric_source_short_alias_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        args = module.build_parser().parse_args(
            [
                "run-bench",
                "--bench-file",
                "bench_kernel.py",
                "--operator-file",
                "opt_kernel.py",
                "-m",
                "all",
            ]
        )

        self.assertEqual(args.metric_source, "all")

    def test_compare_perf_parser_accepts_metric_source_short_alias(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_compare_perf_short_alias_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        args = module.build_parser().parse_args(
            [
                "compare-perf",
                "--baseline",
                "baseline_perf.txt",
                "--compare",
                "candidate_perf.txt",
                "-m",
                "kernel",
            ]
        )

        self.assertEqual(args.metric_source, "kernel")

    def test_compare_perf_parser_accepts_metric_source_all_flag(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        args = module.build_parser().parse_args(
            [
                "compare-perf",
                "--baseline",
                "baseline_perf.txt",
                "--compare",
                "candidate_perf.txt",
                "--metric-source",
                "all",
            ]
        )

        self.assertEqual(args.metric_source, "all")

    @unittest.skipIf(shutil.which("bash") is None, "requires bash")
    def test_skill_script_pyright_wrapper_requires_exactly_one_target(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = "scripts/run-skill-script-pyright.sh"
        completed = subprocess.run(
            ["bash", script],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("<skill-script.py>", completed.stderr)
        self.assertNotIn("[<skill-script.py> ...]", completed.stderr)

    @unittest.skipIf(shutil.which("bash") is None, "requires bash")
    def test_skill_script_pyright_wrapper_rejects_multiple_targets(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = "scripts/run-skill-script-pyright.sh"
        completed = subprocess.run(
            [
                "bash",
                script,
                "skills/common/ascend-npu-run-eval/scripts/bench_runner.py",
                "skills/common/ascend-npu-run-eval/scripts/profile_runner.py",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("usage:", completed.stderr)

    def test_render_result_accepts_skill_result_payload(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        stdout = StringIO()
        stderr = StringIO()
        try:
            sys.stdout = stdout
            sys.stderr = stderr
            module._render_result(
                {
                    "return_code": 0,
                    "stdout": "skill stdout\n",
                    "stderr": "skill stderr\n",
                    "stalled": False,
                    "session_id": None,
                },
                skip_stdout=False,
            )
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

        self.assertEqual(stdout.getvalue(), "skill stdout\n")
        self.assertEqual(stderr.getvalue(), "skill stderr\n")

    def test_load_profile_functions_restores_sys_path_after_import(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test_profile_path", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        before = list(sys.path)
        run_local_profile_bench, run_remote_profile_bench = module._load_profile_functions()

        self.assertEqual(sys.path, before)
        self.assertTrue(callable(run_local_profile_bench))
        self.assertTrue(callable(run_remote_profile_bench))

    def test_script_run_test_baseline_preserves_differential_archive(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "differential_test_kernel.py"
            archive = root / "kernel_result.pt"
            operator.write_text("print('x')\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")

            def fake_run_local_test(
                test_path: Path,
                operator_path: Path,
                test_mode: str,
                verbose: bool = False,
                **_kwargs: object,
            ) -> tuple[dict[str, object], Path]:
                self.assertEqual(test_path, test_file.resolve())
                self.assertEqual(operator_path, operator.resolve())
                self.assertEqual(test_mode, "differential")
                self.assertFalse(verbose)
                archive.write_text("payload\n", encoding="utf-8")
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    archive,
                )

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.object(
                    module,
                    "_load_test_functions",
                    return_value=(
                        lambda _path: {"test-mode": "differential"},
                        fake_run_local_test,
                        lambda *_args, **_kwargs: None,
                    ),
                ):
                    exit_code = module.main(
                        [
                            "run-test-baseline",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator),
                            "--test-mode",
                            "differential",
                        ]
                    )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
            archive_exists_after_run = archive.exists()

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            stdout.getvalue(),
            (
                "Return code: 0\n"
                f"Archived result: {archive}\n"
            ),
        )
        self.assertTrue(archive_exists_after_run)
        self.assertEqual(stderr.getvalue(), "")

    def test_script_run_test_forces_blocks_parallel_to_zero_and_restores_env(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "test_kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            test_file.write_text("# test-mode: standalone\nprint('test')\n", encoding="utf-8")

            observed_env_values: list[Optional[str]] = []

            def fake_run_local_test(
                test_path: Path,
                operator_path: Path,
                test_mode: str,
                verbose: bool = False,
                **_kwargs: object,
            ) -> tuple[dict[str, object], None]:
                self.assertEqual(test_path, test_file.resolve())
                self.assertEqual(operator_path, operator.resolve())
                self.assertEqual(test_mode, "standalone")
                self.assertFalse(verbose)
                observed_env_values.append(os.environ.get("TRITON_ALL_BLOCKS_PARALLEL"))
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    None,
                )

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.dict(
                    os.environ,
                    {"TRITON_ALL_BLOCKS_PARALLEL": "1"},
                    clear=False,
                ):
                    with patch.object(
                        module,
                        "_load_test_functions",
                        return_value=(
                            lambda _path: {"test-mode": "standalone"},
                            fake_run_local_test,
                            lambda *_args, **_kwargs: None,
                        ),
                    ):
                        exit_code = module.main(
                            [
                                "run-test-baseline",
                                "--test-file",
                                str(test_file),
                                "--operator-file",
                                str(operator),
                            ]
                        )
                    restored_value = os.environ.get("TRITON_ALL_BLOCKS_PARALLEL")
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        self.assertEqual(exit_code, 0)
        self.assertEqual(observed_env_values, ["0"])
        self.assertEqual(restored_value, "1")

    def test_script_run_test_convert_guards_blocks_parallel_env(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test_convert_guard", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "triton_kernel.py"
            test_file = root / "test_kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            test_file.write_text("# test-mode: standalone\nprint('test')\n", encoding="utf-8")

            observed_env_values: list[Optional[str]] = []

            def fake_run_local_test(
                test_path: Path,
                operator_path: Path,
                test_mode: str,
                verbose: bool = False,
                **_kwargs: object,
            ) -> tuple[dict[str, object], None]:
                self.assertEqual(test_path, test_file.resolve())
                self.assertEqual(operator_path, operator.resolve())
                self.assertEqual(test_mode, "standalone")
                self.assertFalse(verbose)
                observed_env_values.append(os.environ.get("TRITON_ALL_BLOCKS_PARALLEL"))
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    None,
                )

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.dict(
                    os.environ,
                    {"TRITON_ALL_BLOCKS_PARALLEL": "1"},
                    clear=False,
                ):
                    with patch.object(
                        module,
                        "_load_test_functions",
                        return_value=(
                            lambda _path: {"test-mode": "standalone"},
                            fake_run_local_test,
                            lambda *_args, **_kwargs: None,
                        ),
                    ):
                        exit_code = module.main(
                            [
                                "run-test-convert",
                                "--test-file",
                                str(test_file),
                                "--operator-file",
                                str(operator),
                            ]
                        )
                    restored_value = os.environ.get("TRITON_ALL_BLOCKS_PARALLEL")
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        self.assertEqual(exit_code, 0)
        self.assertEqual(observed_env_values, ["0"])
        self.assertEqual(restored_value, "1")

    def test_script_run_test_optimize_deletes_pt_files_when_run_test_cleanup_policy_enabled(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test_pt_cleanup", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "test_kernel.py"
            archived_result = root / "kernel_result.pt"
            operator.write_text("print('x')\n", encoding="utf-8")
            test_file.write_text("# test-mode: standalone\nprint('test')\n", encoding="utf-8")
            archived_result.write_text("payload\n", encoding="utf-8")

            def fake_run_local_test(
                test_path: Path,
                operator_path: Path,
                test_mode: str,
                verbose: bool = False,
                **_kwargs: object,
            ) -> tuple[dict[str, object], Path]:
                self.assertEqual(test_path, test_file.resolve())
                self.assertEqual(operator_path, operator.resolve())
                self.assertEqual(test_mode, "standalone")
                self.assertFalse(verbose)
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    archived_result,
                )

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.dict(
                    os.environ,
                    {"HELIX_OPTIMIZE_DELETE_PT_FILES": "run-test"},
                    clear=False,
                ), patch.object(
                    module,
                    "_load_test_functions",
                    return_value=(
                        lambda _path: {"test-mode": "standalone"},
                        fake_run_local_test,
                        lambda *_args, **_kwargs: None,
                    ),
                ):
                    exit_code = module.main(
                        [
                            "run-test-optimize",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator),
                        ]
                    )
                self.assertEqual(exit_code, 0)
                self.assertFalse(archived_result.exists())
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

    def test_script_run_test_convert_preserves_pt_files_when_run_test_cleanup_policy_enabled(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test_convert_pt_cleanup", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "triton_kernel.py"
            test_file = root / "differential_test_kernel.py"
            archived_result = root / "triton_kernel_result.pt"
            baseline_result = root / "kernel_result.pt"
            operator.write_text("print('x')\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")
            archived_result.write_text("payload\n", encoding="utf-8")
            baseline_result.write_text("baseline\n", encoding="utf-8")

            def fake_run_local_test(
                test_path: Path,
                operator_path: Path,
                test_mode: str,
                verbose: bool = False,
                **_kwargs: object,
            ) -> tuple[dict[str, object], Path]:
                self.assertEqual(test_path, test_file.resolve())
                self.assertEqual(operator_path, operator.resolve())
                self.assertEqual(test_mode, "differential")
                self.assertFalse(verbose)
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    archived_result,
                )

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.dict(
                    os.environ,
                    {"HELIX_OPTIMIZE_DELETE_PT_FILES": "run-test"},
                    clear=False,
                ), patch.object(
                    module,
                    "_load_test_functions",
                    return_value=(
                        lambda _path: {"test-mode": "differential"},
                        fake_run_local_test,
                        lambda *_args, **_kwargs: None,
                    ),
                ), patch.object(
                    module,
                    "_load_compare_result_functions",
                    return_value=(lambda *_args, **_kwargs: 0, lambda *_args, **_kwargs: 0),
                ):
                    exit_code = module.main(
                        [
                            "run-test-convert",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator),
                            "--ref-result",
                            str(baseline_result),
                        ]
                    )
                self.assertEqual(exit_code, 0)
                self.assertTrue(archived_result.exists())
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

    def test_script_run_test_baseline_preserves_pt_files_when_run_test_cleanup_policy_enabled(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test_pt_cleanup_hint", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "differential_test_kernel.py"
            archived_result = root / "kernel_result.pt"
            operator.write_text("print('x')\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")
            archived_result.write_text("payload\n", encoding="utf-8")

            def fake_run_local_test(
                test_path: Path,
                operator_path: Path,
                test_mode: str,
                verbose: bool = False,
                **_kwargs: object,
            ) -> tuple[dict[str, object], Path]:
                self.assertEqual(test_path, test_file.resolve())
                self.assertEqual(operator_path, operator.resolve())
                self.assertEqual(test_mode, "differential")
                self.assertFalse(verbose)
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    archived_result,
                )

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.dict(
                    os.environ,
                    {"HELIX_OPTIMIZE_DELETE_PT_FILES": "run-test"},
                    clear=False,
                ), patch.object(
                    module,
                    "_load_test_functions",
                    return_value=(
                        lambda _path: {"test-mode": "differential"},
                        fake_run_local_test,
                        lambda *_args, **_kwargs: None,
                    ),
                ):
                    exit_code = module.main(
                        [
                            "run-test-baseline",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator),
                        ]
                    )
                self.assertEqual(exit_code, 0)
                self.assertTrue(archived_result.exists())
                self.assertEqual(
                    stdout.getvalue(),
                    f"Return code: 0\nArchived result: {archived_result}\n",
                )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

    def test_script_run_test_rejects_removed_oracle_result_flag(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "differential_test_kernel.py"
            oracle = root / "oracle_result.pt"
            operator.write_text("print('x')\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")
            oracle.write_text("oracle\n", encoding="utf-8")

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with self.assertRaises(SystemExit) as exc:
                    module.main(
                        [
                            "run-test-optimize",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator),
                            "--test-mode",
                            "differential",
                            "--oracle-result",
                            str(oracle),
                        ]
                    )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        self.assertEqual(exc.exception.code, 2)
        self.assertIn("--oracle-result", stderr.getvalue())

    def test_script_run_test_auto_compares_when_ref_result_is_provided(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "differential_test_kernel.py"
            archive = root / "kernel_result.pt"
            baseline_result = root / "baseline_result.pt"
            operator.write_text("print('x')\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")
            baseline_result.write_text("baseline\n", encoding="utf-8")
            compare_calls: list[tuple[Path, Path, object]] = []

            def fake_run_local_test(
                test_path: Path,
                operator_path: Path,
                test_mode: str,
                verbose: bool = False,
                **_kwargs: object,
            ) -> tuple[dict[str, object], Path]:
                self.assertEqual(test_path, test_file.resolve())
                self.assertEqual(operator_path, operator.resolve())
                self.assertEqual(test_mode, "differential")
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    archive,
                )

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.object(
                    module,
                    "_load_test_functions",
                    return_value=(
                        lambda _path: {"test-mode": "differential"},
                        fake_run_local_test,
                        lambda *_args, **_kwargs: None,
                    ),
                ):
                    with patch.object(
                        module,
                        "_load_compare_result_functions",
                        return_value=(
                            lambda baseline_path, new_path, **kwargs: (
                                compare_calls.append(
                                    (baseline_path, new_path, kwargs.get("accuracy_mode"))
                                )
                                or 0
                                if baseline_path == baseline_result.resolve()
                                and new_path == archive
                                else 2
                            ),
                            lambda *_args, **_kwargs: 0,
                        ),
                    ):
                        exit_code = module.main(
                            [
                                "run-test-optimize",
                                "--test-file",
                                str(test_file),
                                "--operator-file",
                                str(operator),
                                "--ref-result",
                                str(baseline_result),
                                "--accuracy-mode",
                                "dtype-close",
                            ]
                        )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        self.assertEqual(exit_code, 0)
        self.assertEqual(compare_calls, [(baseline_result.resolve(), archive, "dtype-close")])
        self.assertEqual(stdout.getvalue(), f"Return code: 0\nArchived result: {archive}\n")
        self.assertEqual(stderr.getvalue(), "")

    def test_run_test_parser_prefers_ref_flag_names_with_legacy_aliases(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_ref_flag_parser_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        ref_args = module.build_parser().parse_args(
            [
                "run-test-optimize",
                "--test-file",
                "differential_test_kernel.py",
                "--operator-file",
                "kernel.py",
                "--ref-result",
                "ref_result.pt",
                "--ref-operator-file",
                "ref_kernel.py",
            ]
        )
        alias_args = module.build_parser().parse_args(
            [
                "run-test-optimize",
                "--test-file",
                "differential_test_kernel.py",
                "--operator-file",
                "kernel.py",
                "--baseline-result",
                "baseline_result.pt",
                "--baseline-operator-file",
                "baseline_kernel.py",
            ]
        )

        self.assertEqual(ref_args.ref_result, "ref_result.pt")
        self.assertEqual(ref_args.ref_operator_file, "ref_kernel.py")
        self.assertEqual(alias_args.ref_result, "baseline_result.pt")
        self.assertEqual(alias_args.ref_operator_file, "baseline_kernel.py")

    def test_script_run_test_optimize_appends_active_round_timing_events(self) -> None:
        script = _REPO_ROOT / "skills" / "common" / "ascend-npu-run-eval" / "scripts" / "cli.py"
        spec = importlib.util.spec_from_file_location("run_command_test_round_timing_run_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            round_dir = workspace / "opt-round-3"
            round_dir.mkdir()
            (workspace / ".helix").mkdir()
            (workspace / ".helix" / "state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "optimize-20260706-123456-abcdef",
                        "phase": "round_active",
                        "current_round": 3,
                        "baseline": {"status": "passed", "submitted_at": "2026-07-06T12:34:56Z"},
                        "rounds": {
                            "3": {
                                "status": "active",
                                "round_dir": "opt-round-3",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            test_file = workspace / "differential_test_kernel.py"
            operator_file = round_dir / "opt_kernel.py"
            timing_path = workspace / ".helix" / "round-timings" / "opt-round-3.jsonl"
            test_file.write_text("# test-mode: standalone\nprint('test')\n", encoding="utf-8")
            operator_file.write_text("print('kernel')\n", encoding="utf-8")

            def fake_run_local_test(
                test_path: Path,
                candidate_operator_path: Path,
                test_mode: str,
                *,
                verbose: bool = False,
                **_kwargs: object,
            ) -> tuple[dict[str, object], Path | None]:
                self.assertEqual(test_path, test_file.resolve())
                self.assertEqual(candidate_operator_path, operator_file.resolve())
                self.assertEqual(test_mode, "standalone")
                self.assertFalse(verbose)
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    None,
                )

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.object(
                    module,
                    "_load_test_functions",
                    return_value=(
                        lambda _path: {"test-mode": "standalone"},
                        fake_run_local_test,
                        lambda *_args, **_kwargs: None,
                    ),
                ):
                    exit_code = module.main(
                        [
                            "run-test-optimize",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator_file),
                        ]
                    )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

            timing_events = _load_jsonl(timing_path)

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [event["event"] for event in timing_events],
            ["run_test_start", "run_test_end"],
        )
        self.assertEqual(timing_events[0]["round"], "opt-round-3")
        self.assertEqual(timing_events[0]["command"], "run-test-optimize")
        self.assertEqual(timing_events[1]["return_code"], 0)

    def test_script_run_test_convert_does_not_append_active_round_timing_events(self) -> None:
        script = _REPO_ROOT / "skills" / "common" / "ascend-npu-run-eval" / "scripts" / "cli.py"
        spec = importlib.util.spec_from_file_location("run_command_test_convert_round_timing", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            round_dir = workspace / "opt-round-2"
            round_dir.mkdir()
            (workspace / ".helix").mkdir()
            (workspace / ".helix" / "state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "optimize-20260707-123456-abcdef",
                        "phase": "round_active",
                        "current_round": 2,
                        "baseline": {"status": "passed", "submitted_at": "2026-07-07T12:34:56Z"},
                        "rounds": {
                            "2": {
                                "status": "active",
                                "round_dir": "opt-round-2",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            test_file = workspace / "differential_test_kernel.py"
            operator_file = round_dir / "triton_kernel.py"
            timing_path = workspace / ".helix" / "round-timings" / "opt-round-2.jsonl"
            baseline_result = workspace / "kernel_result.pt"
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")
            operator_file.write_text("print('kernel')\n", encoding="utf-8")
            baseline_result.write_text("baseline\n", encoding="utf-8")

            def fake_run_local_test(
                test_path: Path,
                candidate_operator_path: Path,
                test_mode: str,
                *,
                verbose: bool = False,
                **_kwargs: object,
            ) -> tuple[dict[str, object], Path]:
                self.assertEqual(test_path, test_file.resolve())
                self.assertEqual(candidate_operator_path, operator_file.resolve())
                self.assertEqual(test_mode, "differential")
                self.assertFalse(verbose)
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    workspace / "triton_kernel_result.pt",
                )

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.object(
                    module,
                    "_load_test_functions",
                    return_value=(
                        lambda _path: {"test-mode": "differential"},
                        fake_run_local_test,
                        lambda *_args, **_kwargs: None,
                    ),
                ), patch.object(
                    module,
                    "_load_compare_result_functions",
                    return_value=(lambda *_args, **_kwargs: 0, lambda *_args, **_kwargs: 0),
                ):
                    exit_code = module.main(
                        [
                            "run-test-convert",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator_file),
                            "--ref-result",
                            str(baseline_result),
                        ]
                    )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

            self.assertFalse(timing_path.exists())

        self.assertEqual(exit_code, 0)

    def test_script_run_bench_appends_active_round_timing_events(self) -> None:
        script = _REPO_ROOT / "skills" / "common" / "ascend-npu-run-eval" / "scripts" / "cli.py"
        spec = importlib.util.spec_from_file_location("run_command_test_round_timing_run_bench", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            round_dir = workspace / "opt-round-4"
            round_dir.mkdir()
            (workspace / ".helix").mkdir()
            (workspace / ".helix" / "state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "optimize-20260706-123456-abcdef",
                        "phase": "round_active",
                        "current_round": 4,
                        "baseline": {"status": "passed", "submitted_at": "2026-07-06T12:34:56Z"},
                        "rounds": {
                            "4": {
                                "status": "active",
                                "round_dir": "opt-round-4",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            bench_file = workspace / "bench_kernel.py"
            operator_file = round_dir / "opt_kernel.py"
            perf_path = round_dir / "opt_kernel_perf.txt"
            timing_path = workspace / ".helix" / "round-timings" / "opt-round-4.jsonl"
            bench_file.write_text("# bench-mode: torch-npu-profiler\nprint('bench')\n", encoding="utf-8")
            operator_file.write_text("print('kernel')\n", encoding="utf-8")
            perf_path.write_text('{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":1.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n', encoding="utf-8")

            def fake_run_local_bench(
                bench_path: Path,
                candidate_operator_path: Path,
                bench_mode: str,
                npu_devices: str | None = None,
                verbose: bool = False,
                output: str | None = None,
            ) -> tuple[dict[str, object], Path | None]:
                self.assertEqual(bench_path, bench_file.resolve())
                self.assertEqual(candidate_operator_path, operator_file.resolve())
                self.assertEqual(bench_mode, "torch-npu-profiler")
                self.assertIsNone(npu_devices)
                self.assertIsNone(output)
                self.assertFalse(verbose)
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    perf_path,
                )

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.object(
                    module,
                    "_load_bench_functions",
                    return_value=(
                        lambda _path: {"bench-mode": "torch-npu-profiler"},
                        fake_run_local_bench,
                        lambda *_args, **_kwargs: None,
                    ),
                ):
                    exit_code = module.main(
                        [
                            "run-bench",
                            "--bench-file",
                            str(bench_file),
                            "--operator-file",
                            str(operator_file),
                        ]
                    )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

            timing_events = _load_jsonl(timing_path)

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [event["event"] for event in timing_events],
            ["run_bench_start", "run_bench_end"],
        )
        self.assertEqual(timing_events[0]["round"], "opt-round-4")
        self.assertEqual(timing_events[0]["command"], "run-bench")
        self.assertEqual(timing_events[1]["return_code"], 0)

    def test_script_run_test_uses_existing_derived_baseline_result(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_operator = root / "baseline.py"
            operator = root / "kernel.py"
            test_file = root / "differential_test_kernel.py"
            archive = root / "kernel_result.pt"
            derived_baseline_result = root / "baseline_result.pt"
            baseline_operator.write_text("print('baseline')\n", encoding="utf-8")
            operator.write_text("print('x')\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")
            derived_baseline_result.write_text("baseline\n", encoding="utf-8")

            observed_calls: list[tuple[str, Path, Path, str]] = []

            def fake_run_local_test(
                test_path: Path,
                operator_path: Path,
                test_mode: str,
                verbose: bool = False,
                **_kwargs: object,
            ) -> tuple[dict[str, object], Path]:
                observed_calls.append(("local", test_path, operator_path, test_mode))
                self.assertEqual(test_path, test_file.resolve())
                self.assertEqual(operator_path, operator.resolve())
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    archive,
                )

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.object(
                    module,
                    "_load_test_functions",
                    return_value=(
                        lambda _path: {"test-mode": "differential"},
                        fake_run_local_test,
                        lambda *_args, **_kwargs: None,
                    ),
                ):
                    with patch.object(
                        module,
                        "_load_compare_result_functions",
                        return_value=(
                            lambda baseline_path, new_path, **_kwargs: (
                                0
                                if baseline_path == derived_baseline_result.resolve()
                                and new_path == archive
                                else 2
                            ),
                            lambda *_args, **_kwargs: 0,
                        ),
                    ):
                        exit_code = module.main(
                            [
                                "run-test-optimize",
                                "--test-file",
                                str(test_file),
                                "--operator-file",
                                str(operator),
                                "--ref-operator-file",
                                str(baseline_operator),
                            ]
                        )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        self.assertEqual(exit_code, 0)
        self.assertEqual(observed_calls, [("local", test_file.resolve(), operator.resolve(), "differential")])
        self.assertEqual(stdout.getvalue(), f"Return code: 0\nArchived result: {archive}\n")
        self.assertEqual(stderr.getvalue(), "")

    def test_script_run_test_case_id_uses_existing_derived_baseline_case_without_archiving(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test_case_payload", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_operator = root / "baseline.py"
            operator = root / "kernel.py"
            test_file = root / "differential_test_kernel.py"
            derived_baseline_result = root / "baseline_result.pt"
            baseline_operator.write_text("print('baseline')\n", encoding="utf-8")
            operator.write_text("print('x')\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")
            derived_baseline_result.write_text("baseline\n", encoding="utf-8")

            observed_calls: list[tuple[Path, Path, str | None]] = []
            baseline_payload = {
                "compute": True,
                "cases": [{"id": "case-b", "inputs": ("b",), "result": "BASE"}],
            }
            candidate_payload = {
                "compute": True,
                "cases": [{"id": "case-b", "inputs": ("b",), "result": "OPT"}],
            }

            def fake_run_local_test_case_payload(
                test_path: Path,
                operator_path: Path,
                *,
                case_id: str,
                verbose: bool = False,
                **_kwargs: object,
            ) -> tuple[dict[str, object], object]:
                del verbose
                observed_calls.append((test_path, operator_path, case_id))
                self.assertEqual(test_path, test_file.resolve())
                self.assertEqual(operator_path, operator.resolve())
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    candidate_payload,
                )

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.object(
                    module,
                    "_load_test_functions",
                    return_value=(
                        lambda _path: {"test-mode": "differential"},
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("full-run local test helper should not be used in case-id mode")
                        ),
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("full-run remote test helper should not be used in local case-id mode")
                        ),
                    ),
                ), patch.object(
                    module,
                    "_load_test_payload_functions",
                    return_value=(
                        fake_run_local_test_case_payload,
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("remote case payload helper should not be used in local mode")
                        ),
                    ),
                ), patch.object(
                    module,
                    "_load_compare_result_payload_functions",
                    return_value=(
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("explicit ref-result loader should not be used for derived payload reuse")
                        ),
                        lambda result_path, case_id: (
                            baseline_payload
                            if result_path == derived_baseline_result.resolve() and case_id == "case-b"
                            else (_ for _ in ()).throw(AssertionError("unexpected baseline payload request"))
                        ),
                        lambda ref_payload, new_payload, **_kwargs: (
                            0 if ref_payload == baseline_payload and new_payload == candidate_payload else 2
                        ),
                    ),
                ):
                    exit_code = module.main(
                        [
                            "run-test-optimize",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator),
                            "--ref-operator-file",
                            str(baseline_operator),
                            "--case-id",
                            "case-b",
                        ]
                    )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        self.assertEqual(exit_code, 0)
        self.assertEqual(observed_calls, [(test_file.resolve(), operator.resolve(), "case-b")])
        self.assertEqual(stdout.getvalue(), "Return code: 0\n")
        self.assertEqual(stderr.getvalue(), "")

    def test_script_run_test_auto_runs_baseline_when_derived_result_missing(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_operator = root / "baseline.py"
            operator = root / "kernel.py"
            test_file = root / "differential_test_kernel.py"
            baseline_archive = root / "baseline_result.pt"
            baseline_operator.write_text("print('baseline')\n", encoding="utf-8")
            operator.write_text("print('x')\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")
            baseline_archive.write_text("stale baseline\n", encoding="utf-8")

            observed_calls: list[tuple[Path, Path, str]] = []
            baseline_payload = {
                "compute": True,
                "cases": [{"id": "case-b", "inputs": ("b",), "result": "BASE"}],
            }
            candidate_payload = {
                "compute": True,
                "cases": [{"id": "case-b", "inputs": ("b",), "result": "OPT"}],
            }

            def fake_run_local_test_case_payload(
                test_path: Path,
                operator_path: Path,
                *,
                case_id: str,
                verbose: bool = False,
                **_kwargs: object,
            ) -> tuple[dict[str, object], object]:
                del verbose
                observed_calls.append((test_path, operator_path, case_id))
                if operator_path == baseline_operator.resolve():
                    return (
                        {
                            "return_code": 0,
                            "stdout": "",
                            "stderr": "",
                            "stalled": False,
                            "session_id": None,
                        },
                        baseline_payload,
                    )
                self.assertEqual(operator_path, operator.resolve())
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    candidate_payload,
                )

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.object(
                    module,
                    "_load_test_functions",
                    return_value=(
                        lambda _path: {"test-mode": "differential"},
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("full-run local test helper should not be used in case-id mode")
                        ),
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("full-run remote test helper should not be used in local case-id mode")
                        ),
                    ),
                ), patch.object(
                    module,
                    "_load_test_payload_functions",
                    return_value=(
                        fake_run_local_test_case_payload,
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("remote case payload helper should not be used in local mode")
                        ),
                    ),
                ), patch.object(
                    module,
                    "_load_compare_result_payload_functions",
                    return_value=(
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            ValueError("missing case")
                        ),
                        lambda result_path, case_id: (
                            None
                            if result_path == baseline_archive.resolve() and case_id == "case-b"
                            else (_ for _ in ()).throw(AssertionError("unexpected derived payload lookup"))
                        ),
                        lambda ref_payload, new_payload, **_kwargs: (
                            0 if ref_payload == baseline_payload and new_payload == candidate_payload else 2
                        ),
                    ),
                ):
                    exit_code = module.main(
                        [
                            "run-test-optimize",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator),
                            "--ref-operator-file",
                            str(baseline_operator),
                            "--case-id",
                            "case-b",
                        ]
                    )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            observed_calls,
            [
                (test_file.resolve(), baseline_operator.resolve(), "case-b"),
                (test_file.resolve(), operator.resolve(), "case-b"),
            ],
        )
        self.assertEqual(stdout.getvalue(), "Return code: 0\nReturn code: 0\n")
        self.assertEqual(stderr.getvalue(), "")

    def test_run_test_baseline_parser_accepts_test_flags(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        args = module.build_parser().parse_args(
            [
                "run-test-baseline",
                "--test-file",
                "test_kernel.py",
                "--operator-file",
                "kernel.py",
                "--test-mode",
                "standalone",
                "--case-id",
                "case-a",
            ]
        )

        self.assertEqual(args.command, "run-test-baseline")
        self.assertEqual(args.test_file, "test_kernel.py")
        self.assertEqual(args.operator_file, "kernel.py")
        self.assertEqual(args.test_mode, "standalone")
        self.assertEqual(args.case_id, "case-a")

    def test_script_run_test_baseline_rejects_case_id_in_standalone_mode(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test_case_id_guard", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "test_kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            test_file.write_text("# test-mode: standalone\nprint('test')\n", encoding="utf-8")

            stderr = StringIO()
            original_stderr = sys.stderr
            try:
                sys.stderr = stderr
                with self.assertRaises(SystemExit) as exc:
                    module.main(
                        [
                            "run-test-baseline",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator),
                            "--case-id",
                            "case-a",
                        ]
                    )
            finally:
                sys.stderr = original_stderr

        self.assertEqual(exc.exception.code, 2)
        self.assertIn("run-test-baseline standalone mode does not accept --case-id", stderr.getvalue())

    def test_run_test_convert_parser_accepts_reference_flags(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        args = module.build_parser().parse_args(
            [
                "run-test-convert",
                "--test-file",
                "differential_test_kernel.py",
                "--operator-file",
                "triton_kernel.py",
                "--ref-operator-file",
                "kernel.py",
                "--test-mode",
                "differential",
            ]
        )

        self.assertEqual(args.command, "run-test-convert")
        self.assertEqual(args.ref_operator_file, "kernel.py")
        self.assertEqual(args.test_mode, "differential")

    def test_script_run_test_optimize_requires_baseline_source_in_differential_mode(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "opt_kernel.py"
            test_file = root / "differential_test_kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")

            stderr = StringIO()
            original_stderr = sys.stderr
            try:
                sys.stderr = stderr
                with self.assertRaises(SystemExit) as exc:
                    module.main(
                        [
                            "run-test-optimize",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator),
                            "--test-mode",
                            "differential",
                        ]
                    )
            finally:
                sys.stderr = original_stderr

        self.assertEqual(exc.exception.code, 2)
        self.assertIn("requires exactly one of --ref-result or --ref-operator-file", stderr.getvalue())

    def test_script_run_test_convert_requires_reference_input_in_differential_mode(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test_convert_requires_ref", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "triton_kernel.py"
            test_file = root / "differential_test_kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")

            stderr = StringIO()
            original_stderr = sys.stderr
            try:
                sys.stderr = stderr
                with self.assertRaises(SystemExit) as exc:
                    module.main(
                        [
                            "run-test-convert",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator),
                        ]
                    )
            finally:
                sys.stderr = original_stderr

        self.assertEqual(exc.exception.code, 2)
        self.assertIn(
            "run-test-convert differential mode requires exactly one of --ref-result or --ref-operator-file",
            stderr.getvalue(),
        )

    def test_script_run_test_convert_rejects_reference_inputs_in_standalone_mode(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test_convert_standalone", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "triton_kernel.py"
            test_file = root / "test_kernel.py"
            baseline_result = root / "kernel_result.pt"
            operator.write_text("print('x')\n", encoding="utf-8")
            test_file.write_text("# test-mode: standalone\nprint('test')\n", encoding="utf-8")
            baseline_result.write_text("baseline\n", encoding="utf-8")

            stderr = StringIO()
            original_stderr = sys.stderr
            try:
                sys.stderr = stderr
                with self.assertRaises(SystemExit) as exc:
                    module.main(
                        [
                            "run-test-convert",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator),
                            "--ref-result",
                            str(baseline_result),
                        ]
                    )
            finally:
                sys.stderr = original_stderr

        self.assertEqual(exc.exception.code, 2)
        self.assertIn("run-test-convert standalone mode does not accept --ref-result", stderr.getvalue())

    def test_script_run_test_optimize_requires_baseline_source_for_differential_metadata(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "opt_kernel.py"
            test_file = root / "differential_test_kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")

            stderr = StringIO()
            original_stderr = sys.stderr
            try:
                sys.stderr = stderr
                with self.assertRaises(SystemExit) as exc:
                    module.main(
                        [
                            "run-test-optimize",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator),
                        ]
                    )
            finally:
                sys.stderr = original_stderr

        self.assertEqual(exc.exception.code, 2)
        self.assertIn("requires exactly one of --ref-result or --ref-operator-file", stderr.getvalue())

    def test_script_run_test_optimize_rejects_both_ref_result_and_operator_file(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "opt_kernel.py"
            baseline_operator = root / "kernel.py"
            test_file = root / "differential_test_kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            baseline_operator.write_text("print('baseline')\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")
            baseline_result = root / "kernel_result.pt"
            baseline_result.write_text("baseline\n", encoding="utf-8")

            stderr = StringIO()
            original_stderr = sys.stderr
            try:
                sys.stderr = stderr
                with self.assertRaises(SystemExit) as exc:
                    module.main(
                        [
                            "run-test-optimize",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator),
                            "--ref-result",
                            str(baseline_result),
                            "--ref-operator-file",
                            str(baseline_operator),
                        ]
                    )
            finally:
                sys.stderr = original_stderr

        self.assertEqual(exc.exception.code, 2)
        self.assertIn("requires exactly one of --ref-result or --ref-operator-file", stderr.getvalue())

    def test_script_run_test_optimize_auto_compares_when_ref_result_is_provided(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "opt_kernel.py"
            test_file = root / "differential_test_kernel.py"
            archive = root / "opt_kernel_result.pt"
            baseline_result = root / "kernel_result.pt"
            operator.write_text("print('x')\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")
            baseline_result.write_text("baseline\n", encoding="utf-8")

            def fake_run_local_test(
                test_path: Path,
                operator_path: Path,
                test_mode: str,
                verbose: bool = False,
                **_kwargs: object,
            ) -> tuple[dict[str, object], Path]:
                self.assertEqual(test_path, test_file.resolve())
                self.assertEqual(operator_path, operator.resolve())
                self.assertEqual(test_mode, "differential")
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    archive,
                )

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.object(
                    module,
                    "_load_test_functions",
                    return_value=(
                        lambda _path: {"test-mode": "differential"},
                        fake_run_local_test,
                        lambda *_args, **_kwargs: None,
                    ),
                ):
                    with patch.object(
                    module,
                    "_load_compare_result_functions",
                    return_value=(
                            lambda baseline_path, new_path, **_kwargs: (
                                0
                                if baseline_path == baseline_result.resolve()
                                and new_path == archive
                                else 2
                            ),
                            lambda *_args, **_kwargs: 0,
                        ),
                    ):
                        exit_code = module.main(
                            [
                                "run-test-optimize",
                                "--test-file",
                                str(test_file),
                                "--operator-file",
                                str(operator),
                                "--ref-result",
                                str(baseline_result),
                            ]
                        )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), f"Return code: 0\nArchived result: {archive}\n")
        self.assertEqual(stderr.getvalue(), "")

    def test_script_run_test_convert_auto_compares_when_ref_result_is_provided(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test_convert_compare", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "triton_kernel.py"
            test_file = root / "differential_test_kernel.py"
            archive = root / "triton_kernel_result.pt"
            baseline_result = root / "kernel_result.pt"
            operator.write_text("print('x')\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")
            baseline_result.write_text("baseline\n", encoding="utf-8")

            def fake_run_local_test(
                test_path: Path,
                operator_path: Path,
                test_mode: str,
                verbose: bool = False,
                **_kwargs: object,
            ) -> tuple[dict[str, object], Path]:
                self.assertEqual(test_path, test_file.resolve())
                self.assertEqual(operator_path, operator.resolve())
                self.assertEqual(test_mode, "differential")
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    archive,
                )

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.object(
                    module,
                    "_load_test_functions",
                    return_value=(
                        lambda _path: {"test-mode": "differential"},
                        fake_run_local_test,
                        lambda *_args, **_kwargs: None,
                    ),
                ), patch.object(
                    module,
                    "_load_compare_result_functions",
                    return_value=(
                        lambda baseline_path, new_path, **_kwargs: (
                            0
                            if baseline_path == baseline_result.resolve() and new_path == archive
                            else 2
                        ),
                        lambda *_args, **_kwargs: 0,
                    ),
                ):
                    exit_code = module.main(
                        [
                            "run-test-convert",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator),
                            "--ref-result",
                            str(baseline_result),
                        ]
                    )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), f"Return code: 0\nArchived result: {archive}\n")
        self.assertEqual(stderr.getvalue(), "")

    def test_script_run_test_optimize_compares_remote_operators_without_pt_transfer(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "opt_kernel.py"
            test_file = root / "differential_test_kernel.py"
            baseline_operator = root / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            baseline_operator.write_text("print('baseline')\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")

            def fake_run_remote_differential_comparison(
                test_path: Path,
                ref_operator_path: Path,
                operator_path: Path,
                remote: str,
                remote_workdir: str | None,
                *,
                case_id: str | None = None,
                accuracy_mode: str | None = None,
                keep_remote_workdir: bool = False,
                verbose: bool = False,
                stderr: TextIO | None = None,
            ) -> tuple[dict[str, object], str]:
                self.assertEqual(test_path, test_file.resolve())
                self.assertEqual(ref_operator_path, baseline_operator.resolve())
                self.assertEqual(operator_path, operator.resolve())
                self.assertEqual(remote, "alice@example.com")
                self.assertEqual(remote_workdir, "/tmp/helix")
                self.assertIsNone(case_id)
                self.assertEqual(accuracy_mode, "dtype-close")
                self.assertFalse(keep_remote_workdir)
                self.assertFalse(verbose)
                self.assertIs(stderr, sys.stderr)
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    "/tmp/helix-123",
                )

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.object(
                    module,
                    "_load_remote_differential_comparison_function",
                    return_value=fake_run_remote_differential_comparison,
                ):
                    exit_code = module.main(
                        [
                            "run-test-optimize",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator),
                            "--ref-operator-file",
                            str(baseline_operator),
                            "--accuracy-mode",
                            "dtype-close",
                            "--remote",
                            "alice@example.com",
                            "--remote-workdir",
                            "/tmp/helix",
                        ]
                    )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "Return code: 0\n")
        self.assertEqual(stderr.getvalue(), "")

    def test_script_run_test_optimize_uses_existing_derived_baseline_result(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_operator = root / "kernel.py"
            operator = root / "opt_kernel.py"
            test_file = root / "differential_test_kernel.py"
            archive = root / "opt_kernel_result.pt"
            derived_baseline_result = root / "kernel_result.pt"
            baseline_operator.write_text("print('baseline')\n", encoding="utf-8")
            operator.write_text("print('x')\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")
            derived_baseline_result.write_text("baseline\n", encoding="utf-8")

            observed_calls: list[tuple[str, Path, Path, str]] = []

            def fake_run_local_test(
                test_path: Path,
                operator_path: Path,
                test_mode: str,
                verbose: bool = False,
                **_kwargs: object,
            ) -> tuple[dict[str, object], Path]:
                observed_calls.append(("local", test_path, operator_path, test_mode))
                self.assertEqual(test_path, test_file.resolve())
                self.assertEqual(operator_path, operator.resolve())
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    archive,
                )

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.object(
                    module,
                    "_load_test_functions",
                    return_value=(
                        lambda _path: {"test-mode": "differential"},
                        fake_run_local_test,
                        lambda *_args, **_kwargs: None,
                    ),
                ):
                    with patch.object(
                        module,
                        "_load_compare_result_functions",
                        return_value=(
                            lambda baseline_path, new_path, **_kwargs: (
                                0
                                if baseline_path == derived_baseline_result.resolve()
                                and new_path == archive
                                else 2
                            ),
                            lambda *_args, **_kwargs: 0,
                        ),
                    ):
                        exit_code = module.main(
                            [
                                "run-test-optimize",
                                "--test-file",
                                str(test_file),
                                "--operator-file",
                                str(operator),
                                "--ref-operator-file",
                                str(baseline_operator),
                            ]
                        )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        self.assertEqual(exit_code, 0)
        self.assertEqual(observed_calls, [("local", test_file.resolve(), operator.resolve(), "differential")])
        self.assertEqual(stdout.getvalue(), f"Return code: 0\nArchived result: {archive}\n")
        self.assertEqual(stderr.getvalue(), "")

    def test_script_run_test_optimize_auto_runs_baseline_when_derived_result_missing(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_operator = root / "kernel.py"
            operator = root / "opt_kernel.py"
            test_file = root / "differential_test_kernel.py"
            baseline_archive = root / "kernel_result.pt"
            optimize_archive = root / "opt_kernel_result.pt"
            baseline_operator.write_text("print('baseline')\n", encoding="utf-8")
            operator.write_text("print('opt')\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")

            observed_calls: list[tuple[Path, Path, str]] = []

            def fake_run_local_test(
                test_path: Path,
                operator_path: Path,
                test_mode: str,
                verbose: bool = False,
                **_kwargs: object,
            ) -> tuple[dict[str, object], Path]:
                observed_calls.append((test_path, operator_path, test_mode))
                if operator_path == baseline_operator.resolve():
                    baseline_archive.write_text("baseline\n", encoding="utf-8")
                    return (
                        {
                            "return_code": 0,
                            "stdout": "",
                            "stderr": "",
                            "stalled": False,
                            "session_id": None,
                        },
                        baseline_archive,
                    )
                self.assertEqual(operator_path, operator.resolve())
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    optimize_archive,
                )

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.object(
                    module,
                    "_load_test_functions",
                    return_value=(
                        lambda _path: {"test-mode": "differential"},
                        fake_run_local_test,
                        lambda *_args, **_kwargs: None,
                    ),
                ):
                    with patch.object(
                        module,
                        "_load_compare_result_functions",
                        return_value=(
                            lambda baseline_path, new_path, **_kwargs: (
                                0
                                if baseline_path == baseline_archive.resolve()
                                and new_path == optimize_archive
                                else 2
                            ),
                            lambda *_args, **_kwargs: 0,
                        ),
                    ):
                        exit_code = module.main(
                            [
                                "run-test-optimize",
                                "--test-file",
                                str(test_file),
                                "--operator-file",
                                str(operator),
                                "--baseline-operator-file",
                                str(baseline_operator),
                            ]
                        )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            observed_calls,
            [
                (test_file.resolve(), baseline_operator.resolve(), "differential"),
                (test_file.resolve(), operator.resolve(), "differential"),
            ],
        )
        self.assertIn(f"Archived result: {baseline_archive}\n", stdout.getvalue())
        self.assertIn(f"Archived result: {optimize_archive}\n", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")

    def test_script_run_test_threads_verbose_to_local_runner(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "test_kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            test_file.write_text("# test-mode: standalone\nprint('test')\n", encoding="utf-8")

            def fake_run_local_test(
                test_path: Path,
                operator_path: Path,
                test_mode: str,
                verbose: bool = False,
                **_kwargs: object,
            ) -> tuple[dict[str, object], None]:
                self.assertEqual(test_path, test_file.resolve())
                self.assertEqual(operator_path, operator.resolve())
                self.assertEqual(test_mode, "standalone")
                self.assertTrue(verbose)
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    None,
                )

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.object(
                    module,
                    "_load_test_functions",
                    return_value=(
                        lambda _path: {"test-mode": "standalone"},
                        fake_run_local_test,
                        lambda *_args, **_kwargs: None,
                    ),
                ):
                    exit_code = module.main(
                        [
                            "run-test-baseline",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator),
                            "--verbose",
                        ]
                    )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "Return code: 0\n")
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")

    def test_script_runs_cli_help_without_installed_entrypoint(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("cli.py", completed.stdout)
        self.assertNotIn("usage: helix", completed.stdout)
        self.assertNotRegex(completed.stdout, r"(?<![-\w])run-test(?![-\w])")
        self.assertIn("run-test-baseline", completed.stdout)
        self.assertIn("run-test-optimize", completed.stdout)
        self.assertNotIn("compare-result", completed.stdout)
        self.assertIn("compare-perf", completed.stdout)
        self.assertIn("profile-bench", completed.stdout)
        self.assertNotIn("usage: cli.py optimize", completed.stdout)
        self.assertNotIn("gen-test", completed.stdout)

    def test_script_resolves_real_repo_root_when_called_through_symlink(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        source_skills = repo_root / "skills"
        source_script = source_skills / "common" / "ascend-npu-run-eval" / "scripts" / "cli.py"

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            symlinked_skills = workspace / "skills"
            try:
                symlinked_skills.symlink_to(source_skills, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlinks are unavailable: {exc}")
            symlinked_script = symlinked_skills / "common" / "ascend-npu-run-eval" / "scripts" / "cli.py"

            completed = subprocess.run(
                [sys.executable, str(symlinked_script), "--help"],
                capture_output=True,
                text=True,
                cwd=workspace,
                check=False,
            )

        self.assertTrue(source_script.exists())
        self.assertEqual(completed.returncode, 0)
        self.assertNotRegex(completed.stdout, r"(?<![-\w])run-test(?![-\w])")
        self.assertNotIn("compare-result", completed.stdout)

    def test_script_rejects_removed_run_test_command(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        completed = subprocess.run(
            [sys.executable, str(script), "run-test", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("invalid choice", completed.stderr)
        self.assertIn("run-test", completed.stderr)
        self.assertEqual(completed.stdout, "")

    def test_script_rejects_removed_compare_result_command(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        completed = subprocess.run(
            [sys.executable, str(script), "compare-result", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("invalid choice", completed.stderr)
        self.assertIn("compare-result", completed.stderr)
        self.assertEqual(completed.stdout, "")

    def test_script_exposes_run_test_optimize_help(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        completed = subprocess.run(
            [sys.executable, str(script), "run-test-optimize", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("usage: cli.py run-test-optimize", completed.stdout)
        self.assertNotIn("--oracle-result", completed.stdout)
        self.assertIn("--baseline-result", completed.stdout)
        self.assertIn("--baseline-operator-file", completed.stdout)
        self.assertIn("--case-id", completed.stdout)
        self.assertIn("--test-mode", completed.stdout)

    def test_script_exposes_run_test_convert_help(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        completed = subprocess.run(
            [sys.executable, str(script), "run-test-convert", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("usage: cli.py run-test-convert", completed.stdout)
        self.assertIn("--ref-result", completed.stdout)
        self.assertIn("--ref-operator-file", completed.stdout)
        self.assertIn("--baseline-operator-file", completed.stdout)

    def test_script_exposes_profile_bench_help(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        completed = subprocess.run(
            [sys.executable, str(script), "profile-bench", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("--bench-file", completed.stdout)
        self.assertIn("--operator-file", completed.stdout)
        self.assertIn("--case-id", completed.stdout)
        self.assertNotIn("  --bench ", completed.stdout)
        self.assertNotIn("--kernel-name", completed.stdout)
        self.assertIn("--target-op", completed.stdout)
        self.assertIn("--keep-remote-workdir", completed.stdout)

    def test_script_profile_bench_prints_profile_report_hint(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test_profile_hint", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            profile_dir = root / "PROF_000001"
            operator.write_text("print('x')\n", encoding="utf-8")
            bench_file.write_text("# bench-mode: torch-npu-profiler\nprint('bench')\n", encoding="utf-8")

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.object(
                    module,
                    "_load_profile_functions",
                    return_value=(
                        lambda *_args, **_kwargs: (
                            {
                                "return_code": 0,
                                "stdout": "",
                                "stderr": "",
                                "stalled": False,
                                "session_id": None,
                            },
                            profile_dir,
                        ),
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("remote runner should not be used")),
                    ),
                ), patch.object(module, "_build_profile_report", return_value="profile summary"):
                    exit_code = module.main(
                        [
                            "profile-bench",
                            "--bench-file",
                            str(bench_file),
                            "--operator-file",
                            str(operator),
                        ]
                    )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            stdout.getvalue(),
            (
                "Return code: 0\n"
                f"Profile directory: {profile_dir}\n"
                "profile summary\n"
                f"Hint: rerun the bundled `profile-report` helper for this `--profile-dir {profile_dir}` "
                "if you need the summary again; if that is not enough, inspect the raw files in this profile directory directly.\n"
            ),
        )
        self.assertEqual(stderr.getvalue(), "")

    def test_script_profile_bench_preserves_prof_artifacts_after_report(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test_profile_cleanup", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            profile_dir = root / "PROF_000001"
            stale_profile_dir = root / "PROF_000000"
            profile_dir.mkdir()
            stale_profile_dir.mkdir()
            operator.write_text("print('x')\n", encoding="utf-8")
            bench_file.write_text("# bench-mode: torch-npu-profiler\nprint('bench')\n", encoding="utf-8")

            def build_profile_report(profile_path: Path, *_args: object, **_kwargs: object) -> str:
                self.assertEqual(profile_path, profile_dir)
                self.assertTrue(profile_dir.exists())
                return "profile summary"

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.object(
                    module,
                    "_load_profile_functions",
                    return_value=(
                        lambda *_args, **_kwargs: (
                            {
                                "return_code": 0,
                                "stdout": "",
                                "stderr": "",
                                "stalled": False,
                                "session_id": None,
                            },
                            profile_dir,
                        ),
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("remote runner should not be used")),
                    ),
                ), patch.object(module, "_build_profile_report", side_effect=build_profile_report):
                    exit_code = module.main(
                        [
                            "profile-bench",
                            "--bench-file",
                            str(bench_file),
                            "--operator-file",
                            str(operator),
                        ]
                    )
                self.assertEqual(exit_code, 0)
                self.assertTrue(profile_dir.exists())
                self.assertTrue(stale_profile_dir.exists())
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

    def test_script_exposes_run_bench_help(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        completed = subprocess.run(
            [sys.executable, str(script), "run-bench", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("--bench-file", completed.stdout)
        self.assertIn("--operator-file", completed.stdout)
        self.assertIn("--output", completed.stdout)
        self.assertIn("--baseline-operator-file", completed.stdout)
        self.assertIn("--skip-latency-errors", completed.stdout)
        self.assertIn("--metric-source", completed.stdout)

    def test_load_compare_perf_function_reuses_perf_artifacts_implementation(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_compare_perf_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        compare_perf = module._load_compare_perf_function()

        self.assertEqual(compare_perf.__name__, "compare_perf_files")
        self.assertEqual(compare_perf.__module__, "perf_artifacts")

    def test_load_compare_result_functions_reuse_compare_payload_implementation(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "cli.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_compare_result_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        compare_result, compare_remote_result = module._load_compare_result_functions()

        self.assertEqual(compare_result.__name__, "compare_result_files")
        self.assertEqual(compare_result.__module__, "compare_result")
        self.assertEqual(compare_remote_result.__name__, "compare_remote_result_files")
        self.assertEqual(compare_remote_result.__module__, "compare_result")

    def test_compare_remote_result_protocol_uses_textio_stderr(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "run_test_command.py"
        )
        spec = importlib.util.spec_from_file_location("run_test_command_compare_result_protocol_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        hints = get_type_hints(module.CompareRemoteResultFn.__call__)

        self.assertEqual(hints["stderr"], Optional[TextIO])

    def test_optimize_state_submit_baseline_help_runs_without_installed_entrypoint(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        env["PYTHONPATH"] = src_dir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        completed = subprocess.run(
            [sys.executable, str(script), "submit-baseline", "--help"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("cli.py submit-baseline", completed.stdout)
        self.assertIn("--baseline-dir", completed.stdout)
        self.assertNotIn("submit-round", completed.stdout)

    def test_optimize_state_submit_round_help_runs_without_installed_entrypoint(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        env["PYTHONPATH"] = src_dir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        completed = subprocess.run(
            [sys.executable, str(script), "submit-round", "--help"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("cli.py submit-round", completed.stdout)
        self.assertIn("--round-dir", completed.stdout)
        self.assertNotIn("submit-baseline", completed.stdout)

    def test_optimize_state_start_round_help_runs_without_installed_entrypoint(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        env["PYTHONPATH"] = src_dir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        completed = subprocess.run(
            [sys.executable, str(script), "start-round", "--help"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("cli.py start-round", completed.stdout)
        self.assertIn("--round-dir", completed.stdout)
        self.assertIn("--round-strategy", completed.stdout)
        self.assertIn("--analysis-policy", completed.stdout)
        self.assertIn("--reason", completed.stdout)

    def test_optimize_state_set_current_round_state_help_runs_without_installed_entrypoint(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        env["PYTHONPATH"] = src_dir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        completed = subprocess.run(
            [sys.executable, str(script), "set-current-round-state", "--help"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("cli.py set-current-round-state", completed.stdout)
        self.assertIn("--round-strategy", completed.stdout)
        self.assertIn("--analysis-policy", completed.stdout)
        self.assertIn("--reason", completed.stdout)

    def test_optimize_state_start_round_returns_json_hint_when_workflow_state_missing(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        env["PYTHONPATH"] = src_dir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            round_dir = workspace / "opt-round-1"
            round_dir.mkdir()
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "start-round",
                    "--round-dir",
                    str(round_dir),
                    "--round-strategy",
                    "exploration",
                    "--analysis-policy",
                    "pattern_entry",
                    "--reason",
                    "Need to narrow the first promising direction.",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=workspace,
                env=env,
            )

        self.assertEqual(completed.returncode, 1)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "fail")
        self.assertIn("workflow state", payload["issues"][0])
        self.assertNotIn(".helix/state.json", payload["issues"][0])
        self.assertIn("ascend-npu-optimize-state", payload["guideline"])
        self.assertIn("submit-baseline", payload["guideline"])
        self.assertIn("start-round", payload["guideline"])
        self.assertNotIn(".helix/state.json", payload["guideline"])
        self.assertIn("hard_rules", payload)
        self.assertIn("Only one optimize round may be active at a time.", payload["hard_rules"])
        self.assertEqual(completed.stderr, "")

    def test_optimize_state_start_round_returns_json_hint_when_baseline_is_pending(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        env["PYTHONPATH"] = src_dir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            round_dir = workspace / "opt-round-1"
            round_dir.mkdir()
            (workspace / ".helix").mkdir()
            (workspace / ".helix" / "state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "optimize-20260623-123456-abcdef",
                        "phase": "baseline",
                        "current_round": None,
                        "baseline": {"status": "pending", "submitted_at": None},
                        "rounds": {},
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "start-round",
                    "--round-dir",
                    str(round_dir),
                    "--round-strategy",
                    "exploration",
                    "--analysis-policy",
                    "pattern_entry",
                    "--reason",
                    "Need to narrow the first promising direction.",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=workspace,
                env=env,
            )

        self.assertEqual(completed.returncode, 1)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "fail")
        self.assertIn("ascend-npu-optimize-state", payload["guideline"])
        self.assertIn("submit-baseline", payload["guideline"])
        self.assertIn("hard_rules", payload)
        self.assertEqual(completed.stderr, "")

    def test_optimize_state_start_round_success_returns_strategy_state(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        env["PYTHONPATH"] = src_dir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            round_dir = workspace / "opt-round-1"
            round_dir.mkdir()
            (workspace / ".helix").mkdir()
            (workspace / ".helix" / "state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "optimize-20260623-123456-abcdef",
                        "phase": "awaiting_round_start",
                        "current_round": None,
                        "baseline": {"status": "passed", "submitted_at": "2026-06-23T12:34:56Z"},
                        "rounds": {},
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "start-round",
                    "--round-dir",
                    str(round_dir),
                    "--round-strategy",
                    "exploration",
                    "--analysis-policy",
                    "pattern_entry",
                    "--reason",
                    "Need to narrow the first promising direction.",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=workspace,
                env=env,
            )
            attempts_text = (round_dir / "attempts.md").read_text(encoding="utf-8")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "pass")
        self.assertEqual(payload["round"], "opt-round-1")
        self.assertEqual(payload["round_strategy"], "exploration")
        self.assertEqual(payload["analysis_policy"], "pattern_entry")
        self.assertEqual(payload["reason"], "Need to narrow the first promising direction.")
        self.assertIn("hard_rules", payload)
        self.assertIn(
            "Do not use agents or subagents to advance multiple rounds in parallel while the current round is still in flight.",
            payload["hard_rules"],
        )
        self.assertIn(
            "Treat each round as one code-changing optimization attempt followed by canonical validation.",
            payload["hard_rules"],
        )
        self.assertIn(
            "After the first canonical `run-bench` plus `compare-perf` conclusion for a round, do not keep editing that round.",
            payload["hard_rules"],
        )
        self.assertIn("## State Update", attempts_text)
        self.assertIn("Source: start-round", attempts_text)
        self.assertEqual(completed.stderr, "")

    def test_optimize_state_start_round_creates_missing_round_directory(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        env["PYTHONPATH"] = src_dir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            round_dir = workspace / "opt-round-9"
            (workspace / ".helix").mkdir()
            (workspace / ".helix" / "state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "optimize-20260630-123456-abcdef",
                        "phase": "awaiting_round_start",
                        "source_operator": "kernel.py",
                        "current_round": None,
                        "baseline": {"status": "passed", "submitted_at": "2026-06-30T12:34:56Z"},
                        "rounds": {},
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "start-round",
                    "--round-dir",
                    str(round_dir),
                    "--round-strategy",
                    "exploration",
                    "--analysis-policy",
                    "pattern_entry",
                    "--reason",
                    "Create the next round directory when it is missing.",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=workspace,
                env=env,
            )
            round_dir_exists = round_dir.is_dir()
            attempts_exists = (round_dir / "attempts.md").exists()

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(round_dir_exists)
        self.assertTrue(attempts_exists)

    def test_optimize_state_set_current_round_state_success_updates_active_round(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        env["PYTHONPATH"] = src_dir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            round_dir = workspace / "opt-round-2"
            round_dir.mkdir()
            (workspace / ".helix").mkdir()
            (workspace / ".helix" / "state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "optimize-20260627-123456-abcdef",
                        "phase": "round_active",
                        "source_operator": "kernel.py",
                        "current_round": 2,
                        "baseline": {"status": "passed", "submitted_at": "2026-06-27T12:34:56Z"},
                        "rounds": {
                            "2": {
                                "status": "active",
                                "round_dir": "opt-round-2",
                                "started_at": "2026-06-27T12:40:00Z",
                                "ended_at": None,
                                "strategy_state": {
                                    "round_strategy": "exploration",
                                    "analysis_policy": "pattern_entry",
                                    "reason": "Start from pattern triage.",
                                    "updated_at": "2026-06-27T12:40:00Z",
                                    "updated_by": "start-round",
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "set-current-round-state",
                    "--round-strategy",
                    "structural_change",
                    "--analysis-policy",
                    "profile_required",
                    "--reason",
                    "Profiler evidence is now required before the main rewrite.",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=workspace,
                env=env,
            )
            state_payload = json.loads((workspace / ".helix" / "state.json").read_text(encoding="utf-8"))
            attempts_text = (round_dir / "attempts.md").read_text(encoding="utf-8")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "pass")
        self.assertEqual(payload["round"], "opt-round-2")
        self.assertEqual(payload["round_strategy"], "structural_change")
        self.assertEqual(payload["analysis_policy"], "profile_required")
        self.assertEqual(payload["previous_round_strategy"], "exploration")
        self.assertEqual(payload["previous_analysis_policy"], "pattern_entry")
        self.assertEqual(payload["reason"], "Profiler evidence is now required before the main rewrite.")
        self.assertEqual(
            state_payload["rounds"]["2"]["strategy_state"]["round_strategy"],
            "structural_change",
        )
        self.assertIn("Source: set-current-round-state", attempts_text)
        self.assertEqual(completed.stderr, "")

    def test_optimize_state_set_current_round_state_finds_workspace_from_round_subdirectory(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        env["PYTHONPATH"] = src_dir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            round_dir = workspace / "opt-round-5"
            round_dir.mkdir()
            nested_cwd = round_dir / "notes"
            nested_cwd.mkdir()
            (workspace / ".helix").mkdir()
            (workspace / ".helix" / "state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "optimize-20260627-123456-abcdef",
                        "phase": "round_active",
                        "source_operator": "kernel.py",
                        "current_round": 5,
                        "baseline": {"status": "passed", "submitted_at": "2026-06-27T12:34:56Z"},
                        "rounds": {
                            "5": {
                                "status": "active",
                                "round_dir": "opt-round-5",
                                "started_at": "2026-06-27T12:40:00Z",
                                "ended_at": None,
                                "strategy_state": {
                                    "round_strategy": "exploration",
                                    "analysis_policy": "pattern_entry",
                                    "reason": "Start from pattern triage.",
                                    "updated_at": "2026-06-27T12:40:00Z",
                                    "updated_by": "start-round",
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "set-current-round-state",
                    "--analysis-policy",
                    "profile_required",
                    "--reason",
                    "Need profiler evidence before the next edit.",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=nested_cwd,
                env=env,
            )
            state_payload = json.loads((workspace / ".helix" / "state.json").read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            state_payload["rounds"]["5"]["strategy_state"]["analysis_policy"],
            "profile_required",
        )

    def test_optimize_state_start_round_rejects_invalid_strategy_in_argparse(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        env["PYTHONPATH"] = src_dir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            round_dir = workspace / "opt-round-1"
            round_dir.mkdir()
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "start-round",
                    "--round-dir",
                    str(round_dir),
                    "--round-strategy",
                    "unknown_strategy",
                    "--analysis-policy",
                    "pattern_entry",
                    "--reason",
                    "Need to narrow the first promising direction.",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=workspace,
                env=env,
            )

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, "")
        self.assertIn("invalid choice", completed.stderr)

    def test_optimize_state_set_current_round_state_rejects_invalid_policy_in_argparse(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        env["PYTHONPATH"] = src_dir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "set-current-round-state",
                    "--analysis-policy",
                    "unknown_policy",
                    "--reason",
                    "Need to narrow the first promising direction.",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=workspace,
                env=env,
            )

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, "")
        self.assertIn("invalid choice", completed.stderr)

    def test_optimize_state_set_current_round_state_returns_json_hint_when_no_active_round(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        env["PYTHONPATH"] = src_dir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / ".helix").mkdir()
            (workspace / ".helix" / "state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "optimize-20260627-123456-abcdef",
                        "phase": "awaiting_round_start",
                        "source_operator": "kernel.py",
                        "current_round": None,
                        "baseline": {"status": "passed", "submitted_at": "2026-06-27T12:34:56Z"},
                        "rounds": {},
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "set-current-round-state",
                    "--round-strategy",
                    "focused_tuning",
                    "--reason",
                    "Need to deepen analysis before the next edit.",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=workspace,
                env=env,
            )

        self.assertEqual(completed.returncode, 1)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "fail")
        self.assertIn("No optimize round is currently active", payload["guideline"])
        self.assertEqual(completed.stderr, "")

    def test_optimize_state_set_current_round_state_returns_json_hint_when_workflow_state_is_missing(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        env["PYTHONPATH"] = src_dir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "set-current-round-state",
                    "--round-strategy",
                    "focused_tuning",
                    "--reason",
                    "Need to repair missing workflow state before changing round strategy.",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=workspace,
                env=env,
            )

        self.assertEqual(completed.returncode, 1)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "fail")
        self.assertIn("workflow state", payload["issues"][0])
        self.assertNotIn(".helix/state.json", payload["issues"][0])
        self.assertIn("ascend-npu-optimize-state", payload["guideline"])
        self.assertIn("submit-baseline", payload["guideline"])
        self.assertIn("start-round", payload["guideline"])
        self.assertIn("set-current-round-state", payload["guideline"])
        self.assertNotIn(".helix/state.json", payload["guideline"])
        self.assertEqual(completed.stderr, "")

    def test_optimize_state_set_current_round_state_returns_json_hint_for_noop_update(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        env["PYTHONPATH"] = src_dir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            round_dir = workspace / "opt-round-3"
            round_dir.mkdir()
            (workspace / ".helix").mkdir()
            (workspace / ".helix" / "state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "optimize-20260627-123456-abcdef",
                        "phase": "round_active",
                        "source_operator": "kernel.py",
                        "current_round": 3,
                        "baseline": {"status": "passed", "submitted_at": "2026-06-27T12:34:56Z"},
                        "rounds": {
                            "3": {
                                "status": "active",
                                "round_dir": "opt-round-3",
                                "started_at": "2026-06-27T12:40:00Z",
                                "ended_at": None,
                                "strategy_state": {
                                    "round_strategy": "focused_tuning",
                                    "analysis_policy": "ir_required",
                                    "reason": "IR is already required for this round.",
                                    "updated_at": "2026-06-27T12:40:00Z",
                                    "updated_by": "start-round",
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "set-current-round-state",
                    "--round-strategy",
                    "focused_tuning",
                    "--analysis-policy",
                    "ir_required",
                    "--reason",
                    "same state",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=workspace,
                env=env,
            )

        self.assertEqual(completed.returncode, 1)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "fail")
        self.assertIn("no-op", payload["issues"][0])
        self.assertEqual(completed.stderr, "")

    def test_optimize_state_set_current_round_state_returns_json_hint_for_policy_rollback(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        env["PYTHONPATH"] = src_dir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            round_dir = workspace / "opt-round-4"
            round_dir.mkdir()
            (workspace / ".helix").mkdir()
            (workspace / ".helix" / "state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "optimize-20260627-123456-abcdef",
                        "phase": "round_active",
                        "current_round": 4,
                        "baseline": {"status": "passed", "submitted_at": "2026-06-27T12:34:56Z"},
                        "rounds": {
                            "4": {
                                "status": "active",
                                "round_dir": "opt-round-4",
                                "started_at": "2026-06-27T12:40:00Z",
                                "ended_at": None,
                                "strategy_state": {
                                    "round_strategy": "focused_tuning",
                                    "analysis_policy": "ir_required",
                                    "reason": "IR is already required for this round.",
                                    "updated_at": "2026-06-27T12:40:00Z",
                                    "updated_by": "start-round",
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "set-current-round-state",
                    "--analysis-policy",
                    "profile_required",
                    "--reason",
                    "Trying to reduce evidence depth should fail.",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=workspace,
                env=env,
            )

        self.assertEqual(completed.returncode, 1)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "fail")
        self.assertIn("shallower", payload["issues"][0])
        self.assertEqual(completed.stderr, "")

    def test_optimize_state_submit_baseline_supports_runtime_without_pt_cleanup_module(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            workspace = tmpdir / "workspace"
            runtime_root = tmpdir / "runtime"
            optimize_dir = runtime_root / "helix" / "optimize"
            workspace.mkdir()
            (workspace / "baseline").mkdir()
            optimize_dir.mkdir(parents=True)

            (workspace / "baseline" / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "prepared",
                        "source_operator": "kernel.py",
                        "baseline_operator": "baseline/kernel.py",
                        "test_file": "differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "bench_kernel.py",
                        "bench_mode": "torch-npu-profiler",
                        "perf_artifact": "baseline/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (workspace / "baseline" / "perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
            (workspace / "baseline" / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            (workspace / "baseline" / "test_result.pt").write_text("stub\n", encoding="utf-8")

            (runtime_root / "helix" / "__init__.py").write_text("", encoding="utf-8")
            (optimize_dir / "__init__.py").write_text("", encoding="utf-8")
            (optimize_dir / "naming.py").write_text(
                "from pathlib import Path\n\n"
                "def expected_round_operator_name(workspace: Path) -> str:\n"
                "    return 'opt_kernel.py'\n\n"
                "def expected_round_perf_name(workspace: Path) -> str:\n"
                "    return 'opt_kernel_perf.txt'\n\n"
                "def resolve_round_operator_file(round_dir: Path):\n"
                "    return None\n\n"
                "def resolve_round_perf_file(round_dir: Path):\n"
                "    return None\n",
                encoding="utf-8",
            )

            env = os.environ.copy()
            script_dir = str(script.parent)
            runtime_path = str(runtime_root)
            pythonpath_entries = [runtime_path, script_dir]
            if env.get("PYTHONPATH"):
                pythonpath_entries.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = ":".join(pythonpath_entries)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "submit-baseline",
                    "--baseline-dir",
                    str(workspace / "baseline"),
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=workspace,
                env=env,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["status"], "pass")
            self.assertIn("guideline", payload)
            self.assertNotIn("summary", payload)
            self.assertEqual(completed.stderr, "")
            self.assertTrue((workspace / "baseline" / "test_result.pt").exists())

    def test_optimize_state_submit_baseline_updates_workflow_state_when_present(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        script_dir = str(script.parent)
        env["PYTHONPATH"] = ":".join(
            entry for entry in (src_dir, script_dir, env.get("PYTHONPATH", "")) if entry
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            baseline_dir = workspace / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "prepared",
                        "source_operator": "kernel.py",
                        "baseline_operator": "baseline/kernel.py",
                        "test_file": "differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "bench_kernel.py",
                        "bench_mode": "torch-npu-profiler",
                        "perf_artifact": "baseline/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            (workspace / ".helix").mkdir()
            (workspace / ".helix" / "state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "optimize-20260622-123456-abcdef",
                        "phase": "baseline",
                        "source_operator": "kernel.py",
                        "current_round": None,
                        "baseline": {"status": "pending", "submitted_at": None},
                        "rounds": {},
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "submit-baseline",
                    "--baseline-dir",
                    str(baseline_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=workspace,
                env=env,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            state_payload = json.loads((workspace / ".helix" / "state.json").read_text(encoding="utf-8"))

        self.assertEqual(state_payload["phase"], "awaiting_round_start")
        self.assertEqual(state_payload["baseline"]["status"], "passed")

    def test_optimize_state_submit_baseline_bootstraps_missing_workflow_state(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        script_dir = str(script.parent)
        env["PYTHONPATH"] = ":".join(
            entry for entry in (src_dir, script_dir, env.get("PYTHONPATH", "")) if entry
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            baseline_dir = workspace / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "prepared",
                        "source_operator": "kernel.py",
                        "baseline_operator": "baseline/kernel.py",
                        "test_file": "differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "bench_kernel.py",
                        "bench_mode": "torch-npu-profiler",
                        "perf_artifact": "baseline/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "submit-baseline",
                    "--baseline-dir",
                    str(baseline_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=workspace,
                env=env,
            )
            state_payload = json.loads((workspace / ".helix" / "state.json").read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(state_payload["phase"], "awaiting_round_start")
        self.assertEqual(state_payload["baseline"]["status"], "passed")
        self.assertNotIn("source_operator", state_payload)

    def test_optimize_state_submit_baseline_returns_json_hint_when_workflow_state_is_invalid(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        script_dir = str(script.parent)
        env["PYTHONPATH"] = ":".join(
            entry for entry in (src_dir, script_dir, env.get("PYTHONPATH", "")) if entry
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            baseline_dir = workspace / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "prepared",
                        "source_operator": "kernel.py",
                        "baseline_operator": "baseline/kernel.py",
                        "test_file": "differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "bench_kernel.py",
                        "bench_mode": "torch-npu-profiler",
                        "perf_artifact": "baseline/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            (workspace / ".helix").mkdir()
            (workspace / ".helix" / "state.json").write_text("{", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "submit-baseline",
                    "--baseline-dir",
                    str(baseline_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=workspace,
                env=env,
            )

        self.assertEqual(completed.returncode, 1)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "fail")
        self.assertIn("restart the optimize session", payload["guideline"])
        self.assertEqual(completed.stderr, "")

    def test_optimize_state_submit_round_outputs_json_only_with_guideline_and_next_option(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        script_dir = str(script.parent)
        env["PYTHONPATH"] = ":".join(
            entry for entry in (src_dir, script_dir, env.get("PYTHONPATH", "")) if entry
        )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            baseline_dir = workdir / "baseline"
            round_dir = workdir / "opt-round-4"
            baseline_dir.mkdir()
            round_dir.mkdir()

            (workdir / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            (workdir / "opt-note.md").write_text("## Round\n", encoding="utf-8")
            (baseline_dir / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "prepared",
                        "source_operator": "kernel.py",
                        "baseline_operator": "baseline/kernel.py",
                        "test_file": "differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "bench_kernel.py",
                        "bench_mode": "torch-npu-profiler",
                        "perf_artifact": "baseline/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text('{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":1.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n', encoding="utf-8")
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            (round_dir / "opt_kernel.py").write_text(_TRITON_ROUND_OPERATOR, encoding="utf-8")
            (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
            (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
            (round_dir / "opt_kernel_perf.txt").write_text('{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":0.9,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n', encoding="utf-8")
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-4",
                        "parent_round": "round-3",
                        "hypothesis": "vectorize loads",
                        "evidence_sources": ["benchmark"],
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "perf_artifact": "opt_kernel_perf.txt",
                        "comparison_target_path": "baseline/perf.txt",
                        "effective_metric_source": "kernel",
                        "summary_path": "summary.md",
                        "opt_note_updated": True
                    }
                ),
                encoding="utf-8",
            )
            (workdir / ".helix").mkdir()
            (workdir / ".helix" / "state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "optimize-20260630-123456-abcdef",
                        "phase": "round_active",
                        "source_operator": "kernel.py",
                        "current_round": 4,
                        "baseline": {"status": "passed", "submitted_at": "2026-06-30T12:34:56Z"},
                        "rounds": {
                            "4": {
                                "status": "active",
                                "round_dir": "opt-round-4",
                                "started_at": "2026-06-30T12:40:00Z",
                                "ended_at": None,
                                "strategy_state": {
                                    "round_strategy": "exploration",
                                    "analysis_policy": "pattern_entry",
                                    "reason": "Start from pattern triage.",
                                    "updated_at": "2026-06-30T12:40:00Z",
                                    "updated_by": "start-round",
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "submit-round",
                    "--round-dir",
                    str(round_dir),
                    "--current-round",
                    "4",
                    "--final-round",
                    "25",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=workdir,
                env=env,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["status"], "pass")
            self.assertEqual(payload["next_option"], "opt-round-5")
            self.assertIn("guideline", payload)
            self.assertNotIn("summary", payload)
            self.assertIn("Round 4/25 in the current worker batch is complete.", payload["guideline"])
            self.assertIn(
                "Use the staged `ascend-npu-optimize-state` skill's `start-round` subcommand to open opt-round-5 before beginning the next round.",
                payload["guideline"],
            )
            self.assertEqual(completed.stderr, "")

    def test_optimize_state_submit_round_uses_injected_min_speedup_target(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        script_dir = str(script.parent)
        env["PYTHONPATH"] = ":".join(
            entry for entry in (src_dir, script_dir, env.get("PYTHONPATH", "")) if entry
        )
        env["HELIX_OPTIMIZE_MIN_SPEEDUP"] = "1.20"

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            baseline_dir = workdir / "baseline"
            round_dir = workdir / "opt-round-4"
            baseline_dir.mkdir()
            round_dir.mkdir()

            (workdir / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            (workdir / "opt-note.md").write_text("## Round\n", encoding="utf-8")
            (baseline_dir / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "prepared",
                        "source_operator": "kernel.py",
                        "baseline_operator": "baseline/kernel.py",
                        "test_file": "differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "bench_kernel.py",
                        "bench_mode": "torch-npu-profiler",
                        "perf_artifact": "baseline/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text('{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":1.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n', encoding="utf-8")
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            (round_dir / "opt_kernel.py").write_text(_TRITON_ROUND_OPERATOR, encoding="utf-8")
            (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
            (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
            (round_dir / "opt_kernel_perf.txt").write_text('{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":0.8,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n', encoding="utf-8")
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-4",
                        "parent_round": "round-3",
                        "hypothesis": "vectorize loads",
                        "evidence_sources": ["benchmark"],
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "perf_artifact": "opt_kernel_perf.txt",
                        "comparison_target_path": "baseline/perf.txt",
                        "effective_metric_source": "kernel",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                    }
                ),
                encoding="utf-8",
            )
            (workdir / ".helix").mkdir()
            (workdir / ".helix" / "state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "optimize-20260630-123456-abcdef",
                        "phase": "round_active",
                        "source_operator": "kernel.py",
                        "current_round": 4,
                        "baseline": {"status": "passed", "submitted_at": "2026-06-30T12:34:56Z"},
                        "rounds": {
                            "4": {
                                "status": "active",
                                "round_dir": "opt-round-4",
                                "started_at": "2026-06-30T12:40:00Z",
                                "ended_at": None,
                                "strategy_state": {
                                    "round_strategy": "exploration",
                                    "analysis_policy": "pattern_entry",
                                    "reason": "Start from pattern triage.",
                                    "updated_at": "2026-06-30T12:40:00Z",
                                    "updated_by": "start-round",
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "submit-round",
                    "--round-dir",
                    str(round_dir),
                    "--current-round",
                    "4",
                    "--final-round",
                    "25",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=workdir,
                env=env,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["status"], "pass")
            self.assertNotIn("next_option", payload)
            self.assertIn("Minimum speedup target satisfied", payload["guideline"])
            self.assertIn("1.25x", payload["guideline"])
            self.assertIn("Stop the optimize session immediately", payload["guideline"])
            self.assertEqual(completed.stderr, "")

    def test_optimize_state_submit_round_rejects_explicit_min_speedup_argument(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        script_dir = str(script.parent)
        env["PYTHONPATH"] = ":".join(
            entry for entry in (src_dir, script_dir, env.get("PYTHONPATH", "")) if entry
        )

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            round_dir = workdir / "opt-round-1"
            round_dir.mkdir()

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "submit-round",
                    "--round-dir",
                    str(round_dir),
                    "--min-speedup",
                    "1.20",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=workdir,
                env=env,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("unrecognized arguments: --min-speedup 1.20", completed.stderr)

    def test_optimize_state_submit_round_returns_json_hint_when_round_has_not_started(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        script_dir = str(script.parent)
        env["PYTHONPATH"] = ":".join(
            entry for entry in (src_dir, script_dir, env.get("PYTHONPATH", "")) if entry
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            baseline_dir = workspace / "baseline"
            round_dir = workspace / "opt-round-4"
            baseline_dir.mkdir()
            round_dir.mkdir()

            (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            (workspace / "opt-note.md").write_text("## Round\n", encoding="utf-8")
            (baseline_dir / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "prepared",
                        "source_operator": "kernel.py",
                        "baseline_operator": "baseline/kernel.py",
                        "test_file": "differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "bench_kernel.py",
                        "bench_mode": "torch-npu-profiler",
                        "perf_artifact": "baseline/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text('{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":1.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n', encoding="utf-8")
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            (round_dir / "opt_kernel.py").write_text(_TRITON_ROUND_OPERATOR, encoding="utf-8")
            (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
            (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
            (round_dir / "opt_kernel_perf.txt").write_text('{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":0.9,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n', encoding="utf-8")
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-4",
                        "parent_round": "round-3",
                        "hypothesis": "vectorize loads",
                        "evidence_sources": ["benchmark"],
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "perf_artifact": "opt_kernel_perf.txt",
                        "comparison_target_path": "baseline/perf.txt",
                        "effective_metric_source": "kernel",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                    }
                ),
                encoding="utf-8",
            )
            (round_dir / "attempts.md").write_text("# Round 4 Attempts\n", encoding="utf-8")
            (round_dir / "summary.md").write_text("# Round 4 Summary\n", encoding="utf-8")
            (round_dir / "opt_kernel_perf.txt").write_text("case0: 2.0\n", encoding="utf-8")
            (round_dir / "opt_kernel.py").write_text("import triton\nimport triton.language as tl\n\n@triton.jit\ndef kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):\n    pid = tl.program_id(0)\n    offs = pid * BLOCK + tl.arange(0, BLOCK)\n    x = tl.load(x_ptr + offs)\n    y = x * 2\n    tl.store(y_ptr + offs, y)\n\nkernel[(1,)](x, y, N, BLOCK=128)\n", encoding="utf-8")
            baseline_dir = workspace / "baseline"
            baseline_dir.mkdir(exist_ok=True)
            (baseline_dir / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "prepared",
                        "source_operator": "kernel.py",
                        "baseline_operator": "baseline/kernel.py",
                        "test_file": "differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "bench_kernel.py",
                        "bench_mode": "torch-npu-profiler",
                        "perf_artifact": "baseline/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            (workspace / ".helix").mkdir()
            (workspace / ".helix" / "state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "optimize-20260623-123456-abcdef",
                        "phase": "awaiting_round_start",
                        "source_operator": "kernel.py",
                        "current_round": None,
                        "baseline": {"status": "passed", "submitted_at": "2026-06-23T12:34:56Z"},
                        "rounds": {},
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "submit-round",
                    "--round-dir",
                    str(round_dir),
                    "--current-round",
                    "4",
                    "--final-round",
                    "25",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=workspace,
                env=env,
            )

        self.assertEqual(completed.returncode, 1)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "fail")
        self.assertIn("ascend-npu-optimize-state", payload["guideline"])
        self.assertIn("start-round", payload["guideline"])
        self.assertEqual(completed.stderr, "")

    def test_optimize_state_submit_round_returns_json_hint_when_workflow_state_is_missing(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        script_dir = str(script.parent)
        env["PYTHONPATH"] = ":".join(
            entry for entry in (src_dir, script_dir, env.get("PYTHONPATH", "")) if entry
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            baseline_dir = workspace / "baseline"
            round_dir = workspace / "opt-round-4"
            baseline_dir.mkdir()
            round_dir.mkdir()

            (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            (workspace / "opt-note.md").write_text("## Round\n", encoding="utf-8")
            (baseline_dir / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "prepared",
                        "source_operator": "kernel.py",
                        "baseline_operator": "baseline/kernel.py",
                        "test_file": "differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "bench_kernel.py",
                        "bench_mode": "torch-npu-profiler",
                        "perf_artifact": "baseline/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            (round_dir / "opt_kernel.py").write_text(_TRITON_ROUND_OPERATOR, encoding="utf-8")
            (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
            (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
            (round_dir / "opt_kernel_perf.txt").write_text('{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":0.9,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n', encoding="utf-8")
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-4",
                        "parent_round": "round-3",
                        "hypothesis": "vectorize loads",
                        "evidence_sources": ["benchmark"],
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "perf_artifact": "opt_kernel_perf.txt",
                        "comparison_target_path": "baseline/perf.txt",
                        "effective_metric_source": "kernel",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "submit-round",
                    "--round-dir",
                    str(round_dir),
                    "--current-round",
                    "4",
                    "--final-round",
                    "25",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=workspace,
                env=env,
            )

        self.assertEqual(completed.returncode, 1)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "fail")
        self.assertIn("workflow state", payload["issues"][0])
        self.assertNotIn(".helix/state.json", payload["issues"][0])
        self.assertIn("ascend-npu-optimize-state", payload["guideline"])
        self.assertIn("submit-baseline", payload["guideline"])
        self.assertIn("start-round", payload["guideline"])
        self.assertIn("submit-round", payload["guideline"])
        self.assertNotIn(".helix/state.json", payload["guideline"])
        self.assertEqual(completed.stderr, "")

    def test_optimize_state_submit_round_returns_structured_json_when_round_dir_is_missing(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        script_dir = str(script.parent)
        env["PYTHONPATH"] = ":".join(
            entry for entry in (src_dir, script_dir, env.get("PYTHONPATH", "")) if entry
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / ".helix").mkdir()
            (workspace / ".helix" / "state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "optimize-20260630-123456-abcdef",
                        "phase": "round_active",
                        "source_operator": "kernel.py",
                        "current_round": 7,
                        "baseline": {"status": "passed", "submitted_at": "2026-06-30T12:34:56Z"},
                        "rounds": {
                            "7": {
                                "status": "active",
                                "round_dir": "opt-round-7",
                                "started_at": "2026-06-30T12:40:00Z",
                                "ended_at": None,
                                "strategy_state": {
                                    "round_strategy": "exploration",
                                    "analysis_policy": "pattern_entry",
                                    "reason": "Start from pattern triage.",
                                    "updated_at": "2026-06-30T12:40:00Z",
                                    "updated_by": "start-round",
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            missing_round_dir = workspace / "opt-round-7"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "submit-round",
                    "--round-dir",
                    str(missing_round_dir),
                    "--current-round",
                    "7",
                    "--final-round",
                    "25",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=workspace,
                env=env,
            )

        self.assertEqual(completed.returncode, 1)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "fail")
        self.assertIn("missing round directory", payload["issues"][0])
        self.assertEqual(completed.stderr, "")

    def test_optimize_state_submit_round_updates_workflow_state_when_present(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        script_dir = str(script.parent)
        env["PYTHONPATH"] = ":".join(
            entry for entry in (src_dir, script_dir, env.get("PYTHONPATH", "")) if entry
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            baseline_dir = workspace / "baseline"
            round_dir = workspace / "opt-round-4"
            baseline_dir.mkdir()
            round_dir.mkdir()

            (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            (workspace / "opt-note.md").write_text("## Round\n", encoding="utf-8")
            (baseline_dir / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "prepared",
                        "source_operator": "kernel.py",
                        "baseline_operator": "baseline/kernel.py",
                        "test_file": "differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "bench_kernel.py",
                        "bench_mode": "torch-npu-profiler",
                        "perf_artifact": "baseline/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text('{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":1.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n', encoding="utf-8")
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            (round_dir / "opt_kernel.py").write_text(_TRITON_ROUND_OPERATOR, encoding="utf-8")
            (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
            (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
            (round_dir / "opt_kernel_perf.txt").write_text('{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":0.9,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n', encoding="utf-8")
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-4",
                        "parent_round": "round-3",
                        "hypothesis": "vectorize loads",
                        "evidence_sources": ["benchmark"],
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "perf_artifact": "opt_kernel_perf.txt",
                        "comparison_target_path": "baseline/perf.txt",
                        "effective_metric_source": "kernel",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                    }
                ),
                encoding="utf-8",
            )
            (workspace / ".helix").mkdir()
            (workspace / ".helix" / "state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "optimize-20260622-123456-abcdef",
                        "phase": "round_active",
                        "source_operator": "kernel.py",
                        "current_round": 4,
                        "baseline": {"status": "passed", "submitted_at": "2026-06-22T12:34:56Z"},
                        "rounds": {
                            "4": {
                                "status": "active",
                                "round_dir": "opt-round-4",
                                "started_at": "2026-06-22T12:40:00Z",
                                "ended_at": None,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "submit-round",
                    "--round-dir",
                    str(round_dir),
                    "--current-round",
                    "4",
                    "--final-round",
                    "25",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=workspace,
                env=env,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            state_payload = json.loads((workspace / ".helix" / "state.json").read_text(encoding="utf-8"))

        self.assertEqual(state_payload["phase"], "awaiting_round_start")
        self.assertIsNone(state_payload["current_round"])
        self.assertEqual(state_payload["rounds"]["4"]["status"], "passed")

    def test_optimize_state_submit_round_closes_rejected_terminal_round_in_workflow_state(self) -> None:
        script = _OPTIMIZE_STATE_SCRIPT
        env = os.environ.copy()
        src_dir = str(_REPO_ROOT / "src")
        script_dir = str(script.parent)
        env["PYTHONPATH"] = ":".join(
            entry for entry in (src_dir, script_dir, env.get("PYTHONPATH", "")) if entry
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            baseline_dir = workspace / "baseline"
            round_dir = workspace / "opt-round-4"
            baseline_dir.mkdir()
            round_dir.mkdir()

            (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            (workspace / "opt-note.md").write_text("## Round\n", encoding="utf-8")
            (baseline_dir / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "prepared",
                        "source_operator": "kernel.py",
                        "baseline_operator": "baseline/kernel.py",
                        "test_file": "differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "bench_kernel.py",
                        "bench_mode": "torch-npu-profiler",
                        "perf_artifact": "baseline/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text('{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":1.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n', encoding="utf-8")
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            (round_dir / "opt_kernel.py").write_text(_TRITON_ROUND_OPERATOR, encoding="utf-8")
            (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
            (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-4",
                        "parent_round": "round-3",
                        "hypothesis": "vectorize loads",
                        "evidence_sources": ["benchmark"],
                        "correctness_status": "failed",
                        "benchmark_status": "not_run",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                    }
                ),
                encoding="utf-8",
            )
            (workspace / ".helix").mkdir()
            (workspace / ".helix" / "state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "optimize-20260709-123456-abcdef",
                        "phase": "round_active",
                        "source_operator": "kernel.py",
                        "current_round": 4,
                        "baseline": {"status": "passed", "submitted_at": "2026-07-09T12:34:56Z"},
                        "rounds": {
                            "4": {
                                "status": "active",
                                "round_dir": "opt-round-4",
                                "started_at": "2026-07-09T12:40:00Z",
                                "ended_at": None,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "submit-round",
                    "--round-dir",
                    str(round_dir),
                    "--current-round",
                    "4",
                    "--final-round",
                    "25",
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=workspace,
                env=env,
            )
            payload = json.loads(completed.stdout)
            state_payload = json.loads((workspace / ".helix" / "state.json").read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 1)
        self.assertEqual(payload["status"], "fail")
        self.assertEqual(state_payload["phase"], "awaiting_round_start")
        self.assertIsNone(state_payload["current_round"])
        self.assertEqual(state_payload["rounds"]["4"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
