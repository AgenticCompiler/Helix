import sys
import tempfile
import unittest
from io import StringIO
import os
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.cli import build_parser
from triton_agent.commands.execution import handle_probe_bench, handle_run_bench, handle_run_simulator, handle_run_test
from triton_agent.models import AgentResult
from triton_agent.remote.env import remote_target_env_name, remote_workdir_env_name


class ExecutionCommandHandlerTests(unittest.TestCase):
    def test_handle_run_simulator_returns_child_exit_code(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            bench_file.write_text("# bench-mode: msprof\nprint('bench')", encoding="utf-8")

            args = parser.parse_args(
                [
                    "run-simulator",
                    "--bench-file",
                    str(bench_file),
                    "--operator-file",
                    str(operator),
                    "--case-id",
                    "case-a",
                    "--kernel-name",
                    "KernelA",
                ]
            )
            fake_result = AgentResult(return_code=7, stdout="sim out\n", stderr="")

            with patch(
                "triton_agent.commands.execution.run_local_simulator",
                return_value=fake_result,
            ) as mocked:
                exit_code = handle_run_simulator(parser, args)

            self.assertEqual(exit_code, 7)
            mocked.assert_called_once_with(
                bench_file.resolve(),
                operator.resolve(),
                case_id="case-a",
                kernel_name="KernelA",
            )

    def test_handle_run_simulator_prints_runtime_error(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            bench_file.write_text("# bench-mode: msprof\nprint('bench')", encoding="utf-8")

            args = parser.parse_args(
                [
                    "run-simulator",
                    "--bench-file",
                    str(bench_file),
                    "--operator-file",
                    str(operator),
                ]
            )
            stderr = StringIO()

            with patch(
                "triton_agent.commands.execution.run_local_simulator",
                side_effect=ValueError("case selection failed"),
            ):
                with redirect_stderr(stderr):
                    exit_code = handle_run_simulator(parser, args)

            self.assertEqual(exit_code, 1)
            self.assertIn("case selection failed", stderr.getvalue())

    def test_handle_run_test_reads_mode_from_metadata_when_flag_missing(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "differential_test_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")

            args = parser.parse_args(
                [
                    "run-test",
                    "--test-file",
                    str(test_file),
                    "--operator-file",
                    str(operator),
                ]
            )
            fake_result = AgentResult(return_code=0, stdout="", stderr="")

            with patch(
                "triton_agent.commands.execution.run_local_test",
                return_value=(fake_result, None),
            ) as mocked:
                exit_code = handle_run_test(parser, args)

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(
                test_file.resolve(),
                operator.resolve(),
                "differential",
                case_id=None,
                accuracy_mode="npu-contract",
                verbose=False,
            )

    def test_handle_run_test_auto_compares_differential_result_when_ref_result_provided(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "differential_test_kernel.py"
            ref_result = root / "ref_result.pt"
            archive = root / "kernel_result.pt"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")
            ref_result.write_text("baseline", encoding="utf-8")

            args = parser.parse_args(
                [
                    "run-test",
                    "--test-file",
                    str(test_file),
                    "--operator-file",
                    str(operator),
                    "--ref-result",
                    str(ref_result),
                    "--accuracy-mode",
                    "dtype-close",
                ]
            )
            fake_result = AgentResult(return_code=0, stdout="", stderr="")

            with patch(
                "triton_agent.commands.execution.run_local_test",
                return_value=(fake_result, archive),
            ) as run_mock:
                with patch(
                    "triton_agent.commands.execution.compare_result_files",
                    return_value=0,
                ) as compare_mock:
                    exit_code = handle_run_test(parser, args)

            self.assertEqual(exit_code, 0)
            run_mock.assert_called_once_with(
                test_file.resolve(),
                operator.resolve(),
                "differential",
                case_id=None,
                accuracy_mode="dtype-close",
                verbose=False,
            )
            compare_mock.assert_called_once_with(
                ref_result.resolve(),
                archive,
                accuracy_mode="dtype-close",
            )

    def test_handle_run_test_prints_hint_when_run_test_cleanup_does_not_delete_archive(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "differential_test_kernel.py"
            archive = root / "kernel_result.pt"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")
            archive.write_text("archive", encoding="utf-8")

            args = parser.parse_args(
                [
                    "run-test",
                    "--test-file",
                    str(test_file),
                    "--operator-file",
                    str(operator),
                ]
            )
            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            stdout = StringIO()

            with patch.dict(
                os.environ,
                {"TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES": "run-test"},
                clear=False,
            ), patch(
                "triton_agent.commands.execution.run_local_test",
                return_value=(fake_result, archive),
            ), patch(
                "triton_agent.commands.execution.cleanup_run_test_pt_files",
                return_value=[],
            ) as cleanup:
                with redirect_stdout(stdout):
                    exit_code = handle_run_test(parser, args)

            self.assertEqual(exit_code, 0)
            cleanup.assert_called_once_with((archive,))
            self.assertIn("Hint: use `compare-result`", stdout.getvalue())

    def test_handle_run_test_auto_compares_remote_differential_result_via_remote_helper(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "differential_test_kernel.py"
            ref_result = root / "ref_result.pt"
            archive = root / "kernel_result.pt"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")
            ref_result.write_text("baseline", encoding="utf-8")

            args = parser.parse_args(
                [
                    "run-test",
                    "--test-file",
                    str(test_file),
                    "--operator-file",
                    str(operator),
                    "--ref-result",
                    str(ref_result),
                    "--remote",
                    "alice@example.com",
                    "--remote-workdir",
                    "/tmp/triton-agent",
                ]
            )
            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            stdout = StringIO()

            with patch(
                "triton_agent.commands.execution.run_remote_test",
                return_value=(fake_result, archive, "/tmp/triton-agent-123"),
            ) as run_mock:
                with patch(
                    "triton_agent.commands.execution.compare_result_files",
                    side_effect=AssertionError("local comparison should not run for remote tests"),
                ):
                    with patch(
                        "triton_agent.commands.execution.compare_remote_result_files",
                        return_value=0,
                    ) as compare_remote_mock:
                        with redirect_stdout(stdout):
                            exit_code = handle_run_test(parser, args)

            self.assertEqual(exit_code, 0)
            run_mock.assert_called_once_with(
                test_file.resolve(),
                operator.resolve(),
                "differential",
                "alice@example.com",
                "/tmp/triton-agent",
                case_id=None,
                accuracy_mode="npu-contract",
                keep_remote_workdir=False,
                verbose=False,
                stderr=sys.stderr,
            )
            compare_remote_mock.assert_called_once_with(
                ref_result.resolve(),
                archive,
                "alice@example.com",
                "/tmp/triton-agent",
                accuracy_mode="npu-contract",
                verbose=False,
                stderr=sys.stderr,
            )
            self.assertEqual(
                stdout.getvalue(),
                f"Return code: 0\nArchived result: {archive}\n",
            )

    def test_handle_run_test_threads_verbose_to_local_runner(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "test_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: standalone\nprint('test')\n", encoding="utf-8")

            args = parser.parse_args(
                [
                    "run-test",
                    "--test-file",
                    str(test_file),
                    "--operator-file",
                    str(operator),
                    "--verbose",
                ]
            )
            fake_result = AgentResult(return_code=0, stdout="", stderr="")

            with patch(
                "triton_agent.commands.execution.run_local_test",
                return_value=(fake_result, None),
            ) as mocked:
                exit_code = handle_run_test(parser, args)

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(
                test_file.resolve(),
                operator.resolve(),
                "standalone",
                case_id=None,
                accuracy_mode="npu-contract",
                verbose=True,
            )

    def test_handle_run_test_rejects_case_id_in_standalone_mode(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "test_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: standalone\nprint('test')\n", encoding="utf-8")

            args = parser.parse_args(
                [
                    "run-test",
                    "--test-file",
                    str(test_file),
                    "--operator-file",
                    str(operator),
                    "--case-id",
                    "case-a",
                ]
            )
            stderr = StringIO()

            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exc:
                    handle_run_test(parser, args)

        self.assertEqual(exc.exception.code, 2)
        self.assertIn("run-test standalone mode does not accept --case-id", stderr.getvalue())

    def test_handle_run_test_reuses_case_id_for_reference_operator_auto_run(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_operator = root / "baseline.py"
            operator = root / "kernel.py"
            test_file = root / "differential_test_kernel.py"
            baseline_archive = root / "baseline_result.pt"
            archive = root / "kernel_result.pt"
            baseline_operator.write_text("print('baseline')", encoding="utf-8")
            operator.write_text("print('candidate')", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")

            args = parser.parse_args(
                [
                    "run-test",
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
            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            observed_calls: list[tuple[Path, Path, str, Optional[str]]] = []

            def fake_run_local_test(
                test_path: Path,
                operator_path: Path,
                test_mode: str,
                *,
                accuracy_mode: Optional[str] = None,
                verbose: bool = False,
                case_id: Optional[str] = None,
            ) -> tuple[AgentResult, Path]:
                del accuracy_mode, verbose
                observed_calls.append((test_path, operator_path, test_mode, case_id))
                if operator_path == baseline_operator.resolve():
                    return fake_result, baseline_archive
                return fake_result, archive

            with patch(
                "triton_agent.commands.execution.run_local_test",
                side_effect=fake_run_local_test,
            ), patch(
                "triton_agent.commands.execution.compare_result_files",
                return_value=0,
            ) as compare_mock:
                exit_code = handle_run_test(parser, args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            observed_calls,
            [
                (test_file.resolve(), baseline_operator.resolve(), "differential", "case-b"),
                (test_file.resolve(), operator.resolve(), "differential", "case-b"),
            ],
        )
        compare_mock.assert_called_once_with(
            baseline_archive.resolve(),
            archive,
            accuracy_mode="npu-contract",
        )

    def test_handle_run_test_uses_remote_env_when_flag_missing(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "test_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: standalone\nprint('test')\n", encoding="utf-8")

            args = parser.parse_args(
                [
                    "run-test",
                    "--test-file",
                    str(test_file),
                    "--operator-file",
                    str(operator),
                ]
            )
            fake_result = AgentResult(return_code=0, stdout="", stderr="")

            with patch.dict(
                "os.environ",
                {
                    remote_target_env_name(): "alice@example.com",
                    remote_workdir_env_name(): "/tmp/triton-agent",
                },
                clear=False,
            ):
                with patch(
                    "triton_agent.commands.execution.run_remote_test",
                    return_value=(fake_result, None, "/tmp/triton-agent-123"),
                ) as mocked:
                    exit_code = handle_run_test(parser, args)

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(
                test_file.resolve(),
                operator.resolve(),
                "standalone",
                "alice@example.com",
                "/tmp/triton-agent",
                case_id=None,
                accuracy_mode="npu-contract",
                keep_remote_workdir=False,
                verbose=False,
                stderr=sys.stderr,
            )

    def test_handle_run_bench_prints_perf_file(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            perf_file = root / "kernel_perf.txt"
            operator.write_text("print('x')", encoding="utf-8")
            bench_file.write_text("# bench-mode: torch-npu-profiler\nprint('bench')", encoding="utf-8")

            args = parser.parse_args(
                [
                    "run-bench",
                    "--bench-file",
                    str(bench_file),
                    "--operator-file",
                    str(operator),
                ]
            )
            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            stdout = StringIO()

            with patch(
                "triton_agent.commands.execution.run_local_bench",
                return_value=(fake_result, perf_file),
            ):
                with redirect_stdout(stdout):
                    exit_code = handle_run_bench(parser, args)

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                stdout.getvalue(),
                (
                    f"Perf file: {perf_file}\n"
                    "Hint: use `compare-perf` to inspect this perf artifact instead of reading it directly.\n"
                ),
            )

    def test_handle_run_bench_prints_failure_output(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            bench_file.write_text("# bench-mode: torch-npu-profiler\nprint('bench')", encoding="utf-8")

            args = parser.parse_args(
                [
                    "run-bench",
                    "--bench-file",
                    str(bench_file),
                    "--operator-file",
                    str(operator),
                ]
            )
            fake_result = AgentResult(return_code=1, stdout="raw stdout\n", stderr="raw stderr\n")
            stdout = StringIO()
            stderr = StringIO()

            with patch(
                "triton_agent.commands.execution.run_local_bench",
                return_value=(fake_result, None),
            ):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exit_code = handle_run_bench(parser, args)

            self.assertEqual(exit_code, 1)
            self.assertIn("raw stdout", stdout.getvalue())
            self.assertIn("raw stderr", stderr.getvalue())

    def test_handle_run_bench_uses_remote_env_when_flag_missing(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            bench_file.write_text("# bench-mode: torch-npu-profiler\nprint('bench')", encoding="utf-8")

            args = parser.parse_args(
                [
                    "run-bench",
                    "--bench-file",
                    str(bench_file),
                    "--operator-file",
                    str(operator),
                ]
            )
            fake_result = AgentResult(return_code=0, stdout="", stderr="")

            with patch.dict(
                "os.environ",
                {
                    remote_target_env_name(): "alice@example.com",
                    remote_workdir_env_name(): "/tmp/triton-agent",
                },
                clear=False,
            ):
                with patch(
                    "triton_agent.commands.execution.run_remote_bench",
                    return_value=(fake_result, None, "/tmp/triton-agent-123"),
                ) as mocked:
                    exit_code = handle_run_bench(parser, args)

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(
                bench_file.resolve(),
                operator.resolve(),
                "torch-npu-profiler",
                "alice@example.com",
                "/tmp/triton-agent",
                None,
                keep_remote_workdir=False,
                verbose=False,
                stderr=sys.stderr,
                output=None,
            )

    def test_handle_run_bench_reuses_existing_baseline_perf_and_auto_compares(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_kernel.py"
            operator = root / "opt_kernel.py"
            bench_file = root / "bench_kernel.py"
            baseline_perf = root / "baseline_kernel_perf.txt"
            candidate_perf = root / "opt_kernel_perf.txt"
            baseline.write_text("print('baseline')", encoding="utf-8")
            operator.write_text("print('opt')", encoding="utf-8")
            bench_file.write_text("# bench-mode: torch-npu-profiler\nprint('bench')", encoding="utf-8")
            baseline_perf.write_text("latency-a: 1.0\n", encoding="utf-8")

            args = parser.parse_args(
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
            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            stdout = StringIO()

            with patch(
                "triton_agent.commands.execution.run_local_bench",
                return_value=(fake_result, candidate_perf),
            ) as run_mock:
                with patch(
                    "triton_agent.commands.execution.compare_perf_files",
                    return_value=0,
                ) as compare_mock:
                    with redirect_stdout(stdout):
                        exit_code = handle_run_bench(parser, args)

            self.assertEqual(exit_code, 0)
            run_mock.assert_called_once_with(
                bench_file.resolve(),
                operator.resolve(),
                "torch-npu-profiler",
                None,
                verbose=False,
                output=None,
            )
            compare_mock.assert_called_once_with(
                baseline_perf.resolve(),
                candidate_perf,
                skip_latency_errors=False,
                metric_source="auto",
            )
            self.assertEqual(
                stdout.getvalue(),
                (
                    f"Baseline perf file: {baseline_perf.resolve()}\n"
                    f"Perf file: {candidate_perf}\n"
                ),
            )

    def test_handle_run_bench_generates_missing_baseline_perf_before_compare(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_kernel.py"
            operator = root / "opt_kernel.py"
            bench_file = root / "bench_kernel.py"
            baseline_perf = root / "baseline_kernel_perf.txt"
            candidate_perf = root / "opt_kernel_perf.txt"
            baseline.write_text("print('baseline')", encoding="utf-8")
            operator.write_text("print('opt')", encoding="utf-8")
            bench_file.write_text("# bench-mode: torch-npu-profiler\nprint('bench')", encoding="utf-8")

            args = parser.parse_args(
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
            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            stdout = StringIO()

            with patch(
                "triton_agent.commands.execution.run_local_bench",
                side_effect=[(fake_result, baseline_perf), (fake_result, candidate_perf)],
            ) as run_mock:
                with patch(
                    "triton_agent.commands.execution.compare_perf_files",
                    return_value=0,
                ) as compare_mock:
                    with redirect_stdout(stdout):
                        exit_code = handle_run_bench(parser, args)

            self.assertEqual(exit_code, 0)
            self.assertEqual(run_mock.call_count, 2)
            self.assertEqual(
                run_mock.call_args_list[0].args,
                (
                    bench_file.resolve(),
                    baseline.resolve(),
                    "torch-npu-profiler",
                    None,
                ),
            )
            self.assertEqual(
                run_mock.call_args_list[1].args,
                (
                    bench_file.resolve(),
                    operator.resolve(),
                    "torch-npu-profiler",
                    None,
                ),
            )
            compare_mock.assert_called_once_with(
                baseline_perf,
                candidate_perf,
                skip_latency_errors=False,
                metric_source="auto",
            )
            self.assertEqual(
                stdout.getvalue(),
                (
                    f"Baseline perf file: {baseline_perf}\n"
                    f"Perf file: {candidate_perf}\n"
                ),
            )

    def test_handle_run_bench_remote_prints_both_kept_workspaces_when_baseline_is_generated(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_kernel.py"
            operator = root / "opt_kernel.py"
            bench_file = root / "bench_kernel.py"
            baseline_perf = root / "baseline_kernel_perf.txt"
            candidate_perf = root / "opt_kernel_perf.txt"
            baseline.write_text("print('baseline')", encoding="utf-8")
            operator.write_text("print('opt')", encoding="utf-8")
            bench_file.write_text("# bench-mode: torch-npu-profiler\nprint('bench')", encoding="utf-8")

            args = parser.parse_args(
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
            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            stdout = StringIO()

            with patch(
                "triton_agent.commands.execution.run_remote_bench",
                side_effect=[
                    (fake_result, baseline_perf, "/tmp/baseline-ws"),
                    (fake_result, candidate_perf, "/tmp/candidate-ws"),
                ],
            ):
                with patch(
                    "triton_agent.commands.execution.compare_perf_files",
                    return_value=0,
                ):
                    with redirect_stdout(stdout):
                        exit_code = handle_run_bench(parser, args)

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

    def test_handle_run_bench_remote_failure_with_perf_prints_baseline_workspace_once(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_kernel.py"
            operator = root / "opt_kernel.py"
            bench_file = root / "bench_kernel.py"
            baseline_perf = root / "baseline_kernel_perf.txt"
            baseline.write_text("print('baseline')", encoding="utf-8")
            operator.write_text("print('opt')", encoding="utf-8")
            bench_file.write_text("# bench-mode: torch-npu-profiler\nprint('bench')", encoding="utf-8")

            args = parser.parse_args(
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
            failing_result = AgentResult(return_code=1, stdout="", stderr="")
            stdout = StringIO()

            with patch(
                "triton_agent.commands.execution.run_remote_bench",
                return_value=(failing_result, baseline_perf, "/tmp/baseline-ws"),
            ):
                with redirect_stdout(stdout):
                    exit_code = handle_run_bench(parser, args)

            self.assertEqual(exit_code, 1)
            self.assertEqual(stdout.getvalue().count("Remote workspace: /tmp/baseline-ws\n"), 1)

    def test_handle_run_bench_forwards_compare_perf_options(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_kernel.py"
            operator = root / "opt_kernel.py"
            bench_file = root / "bench_kernel.py"
            baseline_perf = root / "baseline_kernel_perf.txt"
            candidate_perf = root / "opt_kernel_perf.txt"
            baseline.write_text("print('baseline')", encoding="utf-8")
            operator.write_text("print('opt')", encoding="utf-8")
            bench_file.write_text("# bench-mode: torch-npu-profiler\nprint('bench')", encoding="utf-8")
            baseline_perf.write_text("latency-a: 1.0\n", encoding="utf-8")

            args = parser.parse_args(
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
                    "kernel",
                ]
            )
            fake_result = AgentResult(return_code=0, stdout="", stderr="")

            with patch(
                "triton_agent.commands.execution.run_local_bench",
                return_value=(fake_result, candidate_perf),
            ):
                with patch(
                    "triton_agent.commands.execution.compare_perf_files",
                    return_value=0,
                ) as compare_mock:
                    exit_code = handle_run_bench(parser, args)

            self.assertEqual(exit_code, 0)
            compare_mock.assert_called_once_with(
                baseline_perf.resolve(),
                candidate_perf,
                skip_latency_errors=True,
                metric_source="kernel",
            )


class ProbeBenchHandlerTests(unittest.TestCase):
    def _write_inputs(self, tmp: str) -> tuple[Path, Path, Path]:
        root = Path(tmp)
        baseline = root / "baseline_kernel.py"
        operator = root / "opt_kernel.py"
        bench_file = root / "bench_kernel.py"
        baseline.write_text("print('baseline')", encoding="utf-8")
        operator.write_text("print('opt')", encoding="utf-8")
        bench_file.write_text("# bench-mode: torch-npu-profiler\nprint('bench')", encoding="utf-8")
        return bench_file, operator, baseline

    def test_handle_probe_bench_local_prints_classification_and_returns_zero(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            bench_file, operator, baseline = self._write_inputs(tmp)
            args = parser.parse_args(
                [
                    "probe-bench",
                    "--bench-file",
                    str(bench_file),
                    "--operator-file",
                    str(operator),
                    "--baseline-operator-file",
                    str(baseline),
                ]
            )
            fake_result = SimpleNamespace(
                return_code=0,
                default_lines=["Probe classification: likely_gain", "Summary: ok"],
                verbose_lines=[],
                warnings=[],
            )
            stdout = StringIO()
            with patch(
                "triton_agent.commands.execution.run_local_probe_bench",
                return_value=fake_result,
            ) as mocked:
                with redirect_stdout(stdout):
                    exit_code = handle_probe_bench(parser, args)
            self.assertEqual(exit_code, 0)
            self.assertIn("Probe classification: likely_gain", stdout.getvalue())
            mocked.assert_called_once_with(
                bench_file.resolve(),
                operator.resolve(),
                baseline.resolve(),
                "torch-npu-profiler",
                metric_source="auto",
                npu_devices=None,
                verbose=False,
            )

    def test_handle_probe_bench_missing_baseline_operator_file_errors(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            bench_file, operator, _ = self._write_inputs(tmp)
            args = parser.parse_args(
                [
                    "probe-bench",
                    "--bench-file",
                    str(bench_file),
                    "--operator-file",
                    str(operator),
                    "--baseline-operator-file",
                    str(bench_file / "missing.py"),
                ]
            )
            with self.assertRaises(SystemExit):
                handle_probe_bench(parser, args)

    def test_handle_probe_bench_verbose_prints_verbose_lines(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            bench_file, operator, baseline = self._write_inputs(tmp)
            args = parser.parse_args(
                [
                    "probe-bench",
                    "--bench-file",
                    str(bench_file),
                    "--operator-file",
                    str(operator),
                    "--baseline-operator-file",
                    str(baseline),
                    "--verbose",
                ]
            )
            fake_result = SimpleNamespace(
                return_code=0,
                default_lines=["Probe classification: inconclusive"],
                verbose_lines=["Baseline cache: hit", "Baseline probe perf: /tmp/x"],
                warnings=[],
            )
            stdout = StringIO()
            with patch(
                "triton_agent.commands.execution.run_local_probe_bench",
                return_value=fake_result,
            ):
                with redirect_stdout(stdout):
                    exit_code = handle_probe_bench(parser, args)
            self.assertEqual(exit_code, 0)
            output = stdout.getvalue()
            self.assertIn("Baseline cache: hit", output)
            self.assertIn("Baseline probe perf: /tmp/x", output)

    def test_handle_probe_bench_failure_returns_nonzero(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            bench_file, operator, baseline = self._write_inputs(tmp)
            args = parser.parse_args(
                [
                    "probe-bench",
                    "--bench-file",
                    str(bench_file),
                    "--operator-file",
                    str(operator),
                    "--baseline-operator-file",
                    str(baseline),
                ]
            )
            fake_result = SimpleNamespace(
                return_code=1,
                default_lines=["FAIL: baseline probe execution failed"],
                verbose_lines=[],
                warnings=[],
            )
            stdout = StringIO()
            with patch(
                "triton_agent.commands.execution.run_local_probe_bench",
                return_value=fake_result,
            ):
                with redirect_stdout(stdout):
                    exit_code = handle_probe_bench(parser, args)
            self.assertEqual(exit_code, 1)
            self.assertIn("FAIL: baseline probe execution failed", stdout.getvalue())

    def test_handle_probe_bench_remote_calls_remote_runner(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            bench_file, operator, baseline = self._write_inputs(tmp)
            args = parser.parse_args(
                [
                    "probe-bench",
                    "--bench-file",
                    str(bench_file),
                    "--operator-file",
                    str(operator),
                    "--baseline-operator-file",
                    str(baseline),
                    "--remote",
                    "user@host",
                    "--keep-remote-workdir",
                ]
            )
            fake_result = SimpleNamespace(
                return_code=0,
                default_lines=["Probe classification: likely_gain"],
                verbose_lines=[],
                warnings=[],
                remote_workspace="/tmp/remote-ws",
            )
            stdout = StringIO()
            with patch(
                "triton_agent.commands.execution.run_remote_probe_bench",
                return_value=fake_result,
            ) as mocked:
                with redirect_stdout(stdout):
                    exit_code = handle_probe_bench(parser, args)
            self.assertEqual(exit_code, 0)
            self.assertIn("Remote workspace: /tmp/remote-ws", stdout.getvalue())
            mocked.assert_called_once()
            self.assertEqual(mocked.call_args.args[4], "user@host")

    def test_handle_probe_bench_propagates_metric_source(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            bench_file, operator, baseline = self._write_inputs(tmp)
            args = parser.parse_args(
                [
                    "probe-bench",
                    "--bench-file",
                    str(bench_file),
                    "--operator-file",
                    str(operator),
                    "--baseline-operator-file",
                    str(baseline),
                    "--metric-source",
                    "total-op",
                ]
            )
            fake_result = SimpleNamespace(
                return_code=0,
                default_lines=["Probe classification: inconclusive"],
                verbose_lines=[],
                warnings=[],
            )
            with patch(
                "triton_agent.commands.execution.run_local_probe_bench",
                return_value=fake_result,
            ) as mocked:
                handle_probe_bench(parser, args)
            self.assertEqual(mocked.call_args.kwargs["metric_source"], "total-op")


if __name__ == "__main__":
    unittest.main()
