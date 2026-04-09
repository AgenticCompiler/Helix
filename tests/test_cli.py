import sys
import tempfile
import threading
import unittest
from io import StringIO
from pathlib import Path
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.cli import (
    _normalize_command_aliases,
    build_parser,
    main,
    prepare_generation_target,
    render_result,
)
from triton_agent.models import AgentResult
from triton_agent.models import CommandKind
from triton_agent.paths import (
    default_generated_output_path,
    resolve_execution_target,
)
from triton_agent.prompts import build_prompt
from triton_agent.result_normalization import normalize_agent_result


class CliParserTests(unittest.TestCase):
    def test_gen_eval_batch_maps_to_command_kind(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["gen-eval-batch", "-i", "kernels"])
        self.assertEqual(args.command, "gen-eval-batch")
        self.assertEqual(args.command_kind, CommandKind.GEN_EVAL_BATCH)
        self.assertEqual(args.max_concurrency, 2)
        self.assertEqual(args.test_mode, "differential")
        self.assertEqual(args.bench_mode, "standalone")
        self.assertEqual(args.agent, "codex")
        self.assertFalse(hasattr(args, "interact"))
        self.assertFalse(hasattr(args, "output"))

    def test_gen_eval_maps_to_command_kind(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["gen-eval", "-i", "kernel.py"])
        self.assertEqual(args.command, "gen-eval")
        self.assertEqual(args.command_kind, CommandKind.GEN_EVAL)
        self.assertEqual(args.test_mode, "differential")
        self.assertEqual(args.bench_mode, "standalone")
        self.assertEqual(args.agent, "codex")
        self.assertFalse(args.interact)

    def test_gen_test_maps_to_command_kind(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["gen-test", "-i", "kernel.py"])
        self.assertEqual(args.command, "gen-test")
        self.assertEqual(args.command_kind, CommandKind.GEN_TEST)
        self.assertEqual(args.agent, "codex")
        self.assertFalse(args.interact)

    def test_snake_case_aliases_map_to_same_command_kind(self) -> None:
        parser = build_parser()
        cases = [
            ("gen_eval_batch", CommandKind.GEN_EVAL_BATCH),
            ("gen_eval", CommandKind.GEN_EVAL),
            ("gen_test", CommandKind.GEN_TEST),
            ("run_test", CommandKind.RUN_TEST),
            ("gen_bench", CommandKind.GEN_BENCH),
            ("run_bench", CommandKind.RUN_BENCH),
            ("optimize_status", CommandKind.OPTIMIZE_STATUS),
            ("optimize_batch", CommandKind.OPTIMIZE_BATCH),
        ]

        for alias, expected_kind in cases:
            with self.subTest(alias=alias):
                argv = [alias, "-i", "kernel.py"]
                if expected_kind == CommandKind.RUN_TEST:
                    argv = [alias, "--test-file", "test_kernel.py", "--operator-file", "kernel.py"]
                elif expected_kind == CommandKind.RUN_BENCH:
                    argv = [alias, "--bench-file", "bench_kernel.py", "--operator-file", "kernel.py"]
                args = parser.parse_args(_normalize_command_aliases(argv))
                self.assertEqual(args.command_kind, expected_kind)

    def test_help_keeps_only_canonical_kebab_case_commands(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()
        self.assertIn("gen-eval-batch", help_text)
        self.assertIn("gen-eval", help_text)
        self.assertIn("gen-test", help_text)
        self.assertIn("run-test", help_text)
        self.assertIn("gen-bench", help_text)
        self.assertIn("run-bench", help_text)
        self.assertIn("compare-result", help_text)
        self.assertIn("compare-perf", help_text)
        self.assertIn("optimize-status", help_text)
        self.assertIn("optimize-batch", help_text)
        self.assertNotIn("gen_eval_batch", help_text)
        self.assertNotIn("gen_eval", help_text)
        self.assertNotIn("gen_test", help_text)
        self.assertNotIn("run_test", help_text)
        self.assertNotIn("gen_bench", help_text)
        self.assertNotIn("run_bench", help_text)
        self.assertNotIn("compare_result", help_text)
        self.assertNotIn("compare_perf", help_text)
        self.assertNotIn("optimize_status", help_text)
        self.assertNotIn("optimize_batch", help_text)

    def test_run_bench_has_common_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run-bench",
                "--bench-file",
                "bench_kernel.py",
                "--operator-file",
                "kernel.py",
                "-o",
                "out.txt",
            ]
        )
        self.assertEqual(args.command_kind, CommandKind.RUN_BENCH)
        self.assertEqual(args.output, "out.txt")
        self.assertFalse(hasattr(args, "interact"))
        self.assertFalse(hasattr(args, "agent"))

    def test_run_test_requires_test_and_operator_files(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["run-test", "--test-file", "test_kernel.py", "--operator-file", "kernel.py"]
        )
        self.assertEqual(args.command_kind, CommandKind.RUN_TEST)
        self.assertEqual(args.test_file, "test_kernel.py")
        self.assertEqual(args.operator_file, "kernel.py")
        self.assertFalse(hasattr(args, "agent"))

    def test_run_bench_requires_bench_and_operator_files(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["run-bench", "--bench-file", "bench_kernel.py", "--operator-file", "kernel.py"]
        )
        self.assertEqual(args.command_kind, CommandKind.RUN_BENCH)
        self.assertEqual(args.bench_file, "bench_kernel.py")
        self.assertEqual(args.operator_file, "kernel.py")

    def test_agent_commands_accept_pi_backend(self) -> None:
        parser = build_parser()
        gen_eval_batch_args = parser.parse_args(["gen-eval-batch", "-i", "kernels", "--agent", "pi"])
        gen_eval_args = parser.parse_args(["gen-eval", "-i", "kernel.py", "--agent", "pi"])
        gen_test_args = parser.parse_args(["gen-test", "-i", "kernel.py", "--agent", "pi"])
        gen_bench_args = parser.parse_args(["gen-bench", "-i", "kernel.py", "--agent", "pi"])
        optimize_args = parser.parse_args(["optimize", "-i", "kernel.py", "--agent", "pi"])
        self.assertEqual(gen_eval_batch_args.agent, "pi")
        self.assertEqual(gen_eval_args.agent, "pi")
        self.assertEqual(gen_test_args.agent, "pi")
        self.assertEqual(gen_bench_args.agent, "pi")
        self.assertEqual(optimize_args.agent, "pi")

    def test_agent_commands_accept_claude_backend(self) -> None:
        parser = build_parser()
        gen_eval_batch_args = parser.parse_args(
            ["gen-eval-batch", "-i", "kernels", "--agent", "claude"]
        )
        gen_eval_args = parser.parse_args(["gen-eval", "-i", "kernel.py", "--agent", "claude"])
        gen_test_args = parser.parse_args(["gen-test", "-i", "kernel.py", "--agent", "claude"])
        gen_bench_args = parser.parse_args(["gen-bench", "-i", "kernel.py", "--agent", "claude"])
        optimize_args = parser.parse_args(["optimize", "-i", "kernel.py", "--agent", "claude"])
        self.assertEqual(gen_eval_batch_args.agent, "claude")
        self.assertEqual(gen_eval_args.agent, "claude")
        self.assertEqual(gen_test_args.agent, "claude")
        self.assertEqual(gen_bench_args.agent, "claude")
        self.assertEqual(optimize_args.agent, "claude")

    def test_run_commands_accept_remote_options(self) -> None:
        parser = build_parser()
        test_args = parser.parse_args(
            [
                "run-test",
                "--test-file",
                "test_kernel.py",
                "--operator-file",
                "kernel.py",
                "--remote",
                "alice@example.com:2200",
                "--remote-workdir",
                "/tmp/runs",
            ]
        )
        bench_args = parser.parse_args(
            [
                "run-bench",
                "--bench-file",
                "bench_kernel.py",
                "--operator-file",
                "kernel.py",
                "--remote",
                "alice@example.com",
            ]
        )
        compare_args = parser.parse_args(
            [
                "compare-result",
                "--oracle-result",
                "oracle_result.pt",
                "--new-result",
                "new_result.pt",
                "--remote",
                "alice@example.com",
            ]
        )
        self.assertEqual(test_args.remote, "alice@example.com:2200")
        self.assertEqual(test_args.remote_workdir, "/tmp/runs")
        self.assertFalse(test_args.keep_remote_workdir)
        self.assertEqual(bench_args.remote, "alice@example.com")
        self.assertIsNone(bench_args.remote_workdir)
        self.assertFalse(bench_args.keep_remote_workdir)
        self.assertEqual(compare_args.remote, "alice@example.com")

    def test_run_commands_accept_keep_remote_workdir(self) -> None:
        parser = build_parser()
        test_args = parser.parse_args(
            [
                "run-test",
                "--test-file",
                "test_kernel.py",
                "--operator-file",
                "kernel.py",
                "--remote",
                "alice@example.com",
                "--keep-remote-workdir",
            ]
        )
        bench_args = parser.parse_args(
            [
                "run-bench",
                "--bench-file",
                "bench_kernel.py",
                "--operator-file",
                "kernel.py",
                "--remote",
                "alice@example.com",
                "--keep-remote-workdir",
            ]
        )
        self.assertTrue(test_args.keep_remote_workdir)
        self.assertTrue(bench_args.keep_remote_workdir)

    def test_agent_commands_accept_remote_options(self) -> None:
        parser = build_parser()
        gen_eval_batch_args = parser.parse_args(
            [
                "gen-eval-batch",
                "-i",
                "kernels",
                "--remote",
                "alice@example.com",
                "--remote-workdir",
                "/tmp/runs",
            ]
        )
        gen_eval_args = parser.parse_args(
            [
                "gen-eval",
                "-i",
                "kernel.py",
                "--remote",
                "alice@example.com:2200",
                "--remote-workdir",
                "/tmp/runs",
            ]
        )
        gen_test_args = parser.parse_args(
            [
                "gen-test",
                "-i",
                "kernel.py",
                "--remote",
                "alice@example.com:2200",
                "--remote-workdir",
                "/tmp/runs",
            ]
        )
        gen_bench_args = parser.parse_args(
            [
                "gen-bench",
                "-i",
                "kernel.py",
                "--remote",
                "alice@example.com",
            ]
        )
        optimize_args = parser.parse_args(
            [
                "optimize",
                "-i",
                "kernel.py",
                "--remote",
                "alice@example.com",
                "--remote-workdir",
                "/tmp/opt",
            ]
        )
        self.assertEqual(gen_eval_batch_args.remote, "alice@example.com")
        self.assertEqual(gen_eval_batch_args.remote_workdir, "/tmp/runs")
        self.assertEqual(gen_eval_args.remote, "alice@example.com:2200")
        self.assertEqual(gen_eval_args.remote_workdir, "/tmp/runs")
        self.assertEqual(gen_test_args.remote, "alice@example.com:2200")
        self.assertEqual(gen_test_args.remote_workdir, "/tmp/runs")
        self.assertEqual(gen_bench_args.remote, "alice@example.com")
        self.assertIsNone(gen_bench_args.remote_workdir)
        self.assertEqual(optimize_args.remote, "alice@example.com")
        self.assertEqual(optimize_args.remote_workdir, "/tmp/opt")

    def test_run_commands_reject_input_flag(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["run-test", "-i", "kernel.py"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["run-bench", "-i", "kernel.py"])

    def test_run_commands_reject_interact_flag(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "run-test",
                    "--test-file",
                    "test_kernel.py",
                    "--operator-file",
                    "kernel.py",
                    "--interact",
                ]
            )
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "run-bench",
                    "--bench-file",
                    "bench_kernel.py",
                    "--operator-file",
                    "kernel.py",
                    "--interact",
                ]
            )

    def test_compare_result_requires_oracle_and_new_paths(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "compare-result",
                "--oracle-result",
                "oracle_result.pt",
                "--new-result",
                "new_result.pt",
            ]
        )
        self.assertEqual(args.command_kind, CommandKind.COMPARE_RESULT)
        self.assertEqual(args.oracle_result, "oracle_result.pt")
        self.assertEqual(args.new_result, "new_result.pt")

    def test_compare_perf_requires_baseline_and_compare_paths(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "compare-perf",
                "--baseline",
                "baseline_perf.txt",
                "--compare",
                "candidate_perf.txt",
            ]
        )
        self.assertEqual(args.command_kind, CommandKind.COMPARE_PERF)
        self.assertEqual(args.baseline, "baseline_perf.txt")
        self.assertEqual(args.compare, "candidate_perf.txt")

    def test_verbose_option_is_available(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["gen-test", "-i", "kernel.py", "--verbose"])
        self.assertTrue(args.verbose)

    def test_show_output_option_is_available(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["gen-test", "-i", "kernel.py", "--show-output"])
        self.assertTrue(args.show_output)

    def test_force_overwrite_option_is_available_for_generators(self) -> None:
        parser = build_parser()
        gen_test_args = parser.parse_args(["gen-test", "-i", "kernel.py", "--force-overwrite"])
        gen_bench_args = parser.parse_args(["gen-bench", "-i", "kernel.py", "--force-overwrite"])
        self.assertTrue(gen_test_args.force_overwrite)
        self.assertTrue(gen_bench_args.force_overwrite)

    def test_test_mode_option_is_available_for_test_commands(self) -> None:
        parser = build_parser()
        eval_args = parser.parse_args(["gen-eval", "-i", "kernel.py", "--test-mode", "standalone"])
        gen_args = parser.parse_args(["gen-test", "-i", "kernel.py", "--test-mode", "standalone"])
        run_args = parser.parse_args(
            [
                "run-test",
                "--test-file",
                "test_kernel.py",
                "--operator-file",
                "kernel.py",
                "--test-mode",
                "differential",
            ]
        )
        self.assertEqual(eval_args.test_mode, "standalone")
        self.assertEqual(gen_args.test_mode, "standalone")
        self.assertEqual(run_args.test_mode, "differential")

    def test_test_commands_default_to_standalone_mode(self) -> None:
        parser = build_parser()
        gen_args = parser.parse_args(["gen-test", "-i", "kernel.py"])
        run_args = parser.parse_args(
            ["run-test", "--test-file", "test_kernel.py", "--operator-file", "kernel.py"]
        )
        self.assertEqual(gen_args.test_mode, "standalone")
        self.assertIsNone(run_args.test_mode)

    def test_gen_eval_defaults_to_differential_test_mode(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["gen-eval", "-i", "kernel.py"])
        self.assertEqual(args.test_mode, "differential")

    def test_gen_eval_batch_defaults_to_differential_test_mode(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["gen-eval-batch", "-i", "kernels"])
        self.assertEqual(args.test_mode, "differential")

    def test_bench_mode_option_is_available_for_bench_commands(self) -> None:
        parser = build_parser()
        eval_args = parser.parse_args(["gen-eval", "-i", "kernel.py", "--bench-mode", "msprof"])
        gen_args = parser.parse_args(["gen-bench", "-i", "kernel.py", "--bench-mode", "standalone"])
        run_args = parser.parse_args(
            [
                "run-bench",
                "--bench-file",
                "bench_kernel.py",
                "--operator-file",
                "kernel.py",
                "--bench-mode",
                "msprof",
            ]
        )
        self.assertEqual(eval_args.bench_mode, "msprof")
        self.assertEqual(gen_args.bench_mode, "standalone")
        self.assertEqual(run_args.bench_mode, "msprof")

    def test_bench_commands_default_to_standalone_mode(self) -> None:
        parser = build_parser()
        gen_args = parser.parse_args(["gen-bench", "-i", "kernel.py"])
        run_args = parser.parse_args(
            ["run-bench", "--bench-file", "bench_kernel.py", "--operator-file", "kernel.py"]
        )
        self.assertEqual(gen_args.bench_mode, "standalone")
        self.assertIsNone(run_args.bench_mode)

    def test_gen_eval_defaults_to_standalone_bench_mode(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["gen-eval", "-i", "kernel.py"])
        self.assertEqual(args.bench_mode, "standalone")

    def test_gen_eval_batch_defaults_to_standalone_bench_mode(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["gen-eval-batch", "-i", "kernels"])
        self.assertEqual(args.bench_mode, "standalone")

    def test_gen_eval_batch_accepts_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "gen-eval-batch",
                "-i",
                "kernels",
                "--agent",
                "pi",
                "--remote",
                "alice@example.com",
                "--remote-workdir",
                "/tmp/eval",
                "--test-mode",
                "standalone",
                "--bench-mode",
                "msprof",
                "--max-concurrency",
                "3",
                "--show-output",
            ]
        )
        self.assertEqual(args.agent, "pi")
        self.assertEqual(args.remote, "alice@example.com")
        self.assertEqual(args.remote_workdir, "/tmp/eval")
        self.assertEqual(args.test_mode, "standalone")
        self.assertEqual(args.bench_mode, "msprof")
        self.assertEqual(args.max_concurrency, 3)
        self.assertTrue(args.show_output)

    def test_optimize_command_supports_mode_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "optimize",
                "-i",
                "kernel.py",
                "--test-mode",
                "standalone",
                "--bench-mode",
                "msprof",
            ]
        )
        self.assertEqual(args.test_mode, "standalone")
        self.assertEqual(args.bench_mode, "msprof")

    def test_optimize_command_defers_mode_defaults_to_runtime(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py"])
        self.assertIsNone(args.test_mode)
        self.assertIsNone(args.bench_mode)

    def test_optimize_command_accepts_min_rounds(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--min-rounds", "3"])
        self.assertEqual(args.min_rounds, 3)

    def test_optimize_command_defaults_resume_to_auto(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py"])
        self.assertEqual(args.resume, "auto")

    def test_optimize_command_accepts_resume_modes(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--resume", "fresh"])
        self.assertEqual(args.resume, "fresh")

    def test_optimize_command_accepts_no_agent_session(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--no-agent-session"])
        self.assertTrue(args.no_agent_session)

    def test_optimize_batch_maps_to_command_kind(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize-batch", "-i", "kernels"])
        self.assertEqual(args.command_kind, CommandKind.OPTIMIZE_BATCH)
        self.assertEqual(args.max_concurrency, 2)
        self.assertEqual(args.agent, "codex")
        self.assertFalse(hasattr(args, "interact"))
        self.assertFalse(args.show_output)

    def test_optimize_status_maps_to_command_kind(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize-status", "-i", "kernels"])
        self.assertEqual(args.command_kind, CommandKind.OPTIMIZE_STATUS)
        self.assertTrue(args.verbose is False)
        self.assertFalse(hasattr(args, "agent"))
        self.assertFalse(hasattr(args, "interact"))
        self.assertFalse(hasattr(args, "remote"))
        self.assertFalse(hasattr(args, "output"))

    def test_optimize_batch_accepts_optimize_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "optimize-batch",
                "-i",
                "kernels",
                "--agent",
                "pi",
                "--remote",
                "alice@example.com",
                "--remote-workdir",
                "/tmp/opt",
                "--test-mode",
                "standalone",
                "--bench-mode",
                "msprof",
                "--min-rounds",
                "4",
                "--resume",
                "continue",
                "--no-agent-session",
                "--max-concurrency",
                "3",
                "--show-output",
            ]
        )
        self.assertEqual(args.agent, "pi")
        self.assertEqual(args.remote, "alice@example.com")
        self.assertEqual(args.remote_workdir, "/tmp/opt")
        self.assertEqual(args.test_mode, "standalone")
        self.assertEqual(args.bench_mode, "msprof")
        self.assertEqual(args.min_rounds, 4)
        self.assertEqual(args.resume, "continue")
        self.assertTrue(args.no_agent_session)
        self.assertEqual(args.max_concurrency, 3)
        self.assertTrue(args.show_output)

    def test_optimize_batch_defaults_resume_to_auto(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize-batch", "-i", "kernels"])
        self.assertEqual(args.resume, "auto")


class PathResolutionTests(unittest.TestCase):
    def test_main_optimize_status_rejects_missing_root(self) -> None:
        stderr = StringIO()
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as exc:
                main(["optimize-status", "-i", "/tmp/definitely-missing-triton-agent-root"])

        self.assertEqual(exc.exception.code, 2)
        self.assertIn("Input path does not exist", stderr.getvalue())

    def test_main_optimize_status_reports_empty_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stderr = StringIO()

            with redirect_stderr(stderr):
                exit_code = main(["optimize-status", "-i", str(root)])

            self.assertEqual(exit_code, 1)
            self.assertIn("No operator workspaces found under", stderr.getvalue())

    def test_main_optimize_status_reports_no_session_workspaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fresh").mkdir()
            stdout = StringIO()

            with redirect_stdout(stdout):
                exit_code = main(["optimize-status", "-i", str(root)])

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertIn("[NO-SESSION] fresh", rendered)
            self.assertIn("Summary: 0 ok, 0 warning, 1 no-session", rendered)

    def test_main_optimize_status_reports_numeric_best_and_logged_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "matmul"
            workspace.mkdir()
            (workspace / "kernel_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            (workspace / "opt-note.md").write_text(
                "\n".join(
                    [
                        "## Round 1",
                        "Best status: current best",
                        "## Round 2",
                        "Best status: validated branch",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            round_one = workspace / "opt-round-1"
            round_two = workspace / "opt-round-2"
            round_one.mkdir()
            round_two.mkdir()
            (round_one / "perf.txt").write_text(
                "latency-a: 8\nlatency-b: 18\n",
                encoding="utf-8",
            )
            (round_two / "perf.txt").write_text(
                "latency-a: 9\nlatency-b: 10\n",
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["optimize-status", "-i", str(root)])

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertIn("[OK] matmul", rendered)
            self.assertIn("Baseline mean: 15.000000", rendered)
            self.assertIn("Best mean: 9.500000", rendered)
            self.assertIn("Avg improvement: +30.0%", rendered)
            self.assertIn("Best round: round-2", rendered)
            self.assertIn("Logged best: round-1", rendered)
            self.assertIn("Warning: numeric best round differs from logged best round", rendered)

    def test_main_optimize_status_prefers_overall_summary_logged_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "matmul"
            workspace.mkdir()
            (workspace / "kernel_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            (workspace / "opt-note.md").write_text(
                "\n".join(
                    [
                        "## Round 1",
                        "Best status: current best",
                        "## Round 2",
                        "Best status: validated branch",
                        "",
                        "## Overall Summary",
                        "Final best round: round-2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            round_one = workspace / "opt-round-1"
            round_two = workspace / "opt-round-2"
            round_one.mkdir()
            round_two.mkdir()
            (round_one / "perf.txt").write_text(
                "latency-a: 8\nlatency-b: 18\n",
                encoding="utf-8",
            )
            (round_two / "perf.txt").write_text(
                "latency-a: 9\nlatency-b: 10\n",
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["optimize-status", "-i", str(root)])

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertIn("[OK] matmul", rendered)
            self.assertIn("Best round: round-2", rendered)
            self.assertIn("Logged best: round-2", rendered)
            self.assertIn(
                "Warning: overall summary best round differs from legacy current best marker",
                rendered,
            )

    def test_main_optimize_status_warns_when_perf_ids_do_not_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "layernorm"
            workspace.mkdir()
            (workspace / "kernel_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            round_one = workspace / "opt-round-1"
            round_one.mkdir()
            (round_one / "perf.txt").write_text(
                "latency-a: 8\nlatency-c: 18\n",
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["optimize-status", "-i", str(root)])

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertIn("[WARN] layernorm", rendered)
            self.assertIn("Best mean: unknown", rendered)
            self.assertIn("Avg improvement: unknown", rendered)
            self.assertIn("Warning: latency ids do not match for comparable perf data", rendered)
            self.assertIn("Summary: 0 ok, 1 warning, 0 no-session", rendered)

    def test_main_optimize_batch_auto_detects_operator_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            (first / "kernel.py").write_text("print('x')\n", encoding="utf-8")
            (first / "opt_kernel.py").write_text("", encoding="utf-8")
            (first / "__init__.py").write_text("", encoding="utf-8")
            (second / "matmul_impl.py").write_text("print('y')\n", encoding="utf-8")

            seen_inputs: list[Path] = []

            def _fake_run(request):
                seen_inputs.append(request.input_path)
                return AgentResult(return_code=0, stdout="", stderr="")

            with patch("triton_agent.optimize.batch.run_optimize_request", side_effect=_fake_run):
                exit_code = main(["optimize-batch", "-i", str(root), "--resume", "fresh"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                seen_inputs,
                [
                    (first / "kernel.py").resolve(),
                    (second / "matmul_impl.py").resolve(),
                ],
            )

    def test_main_optimize_batch_reports_workspace_selection_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = root / "bad"
            good = root / "good"
            bad.mkdir()
            good.mkdir()
            (bad / "a.py").write_text("print('a')\n", encoding="utf-8")
            (bad / "b.py").write_text("print('b')\n", encoding="utf-8")
            (good / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

            stdout = StringIO()
            seen_inputs: list[Path] = []

            def _fake_run(request):
                seen_inputs.append(request.input_path)
                return AgentResult(return_code=0, stdout="", stderr="")

            with patch("triton_agent.optimize.batch.run_optimize_request", side_effect=_fake_run):
                with redirect_stdout(stdout):
                    exit_code = main(["optimize-batch", "-i", str(root)])

            self.assertEqual(exit_code, 1)
            self.assertEqual(seen_inputs, [(good / "kernel.py").resolve()])
            self.assertIn("bad", stdout.getvalue())
            self.assertIn("found multiple candidate operator files", stdout.getvalue())
            self.assertIn("Summary: 1 succeeded, 1 failed", stdout.getvalue())

    def test_main_optimize_batch_honors_max_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("one", "two", "three"):
                workspace = root / name
                workspace.mkdir()
                (workspace / f"{name}.py").write_text("print('x')\n", encoding="utf-8")

            active = 0
            max_active = 0
            launched = 0
            lock = threading.Lock()
            first_pair_ready = threading.Event()
            release_gate = threading.Event()

            def _fake_run(_request):
                nonlocal active, max_active, launched
                with lock:
                    launched += 1
                    active += 1
                    max_active = max(max_active, active)
                    if launched >= 2:
                        first_pair_ready.set()
                first_pair_ready.wait(timeout=1)
                release_gate.wait(timeout=0.1)
                with lock:
                    active -= 1
                return AgentResult(return_code=0, stdout="", stderr="")

            with patch("triton_agent.optimize.batch.run_optimize_request", side_effect=_fake_run):
                exit_code = main(
                    ["optimize-batch", "-i", str(root), "--max-concurrency", "2"]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(max_active, 2)

    def test_main_optimize_batch_show_output_prefixes_workspace_streams(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("alpha", "beta"):
                workspace = root / name
                workspace.mkdir()
                (workspace / f"{name}.py").write_text("print('x')\n", encoding="utf-8")

            stdout = StringIO()

            def _fake_run(request, stdout=None, stderr=None):
                if stdout is not None:
                    stdout.write("round 1 start\n")
                if stderr is not None:
                    stderr.write("warn line\n")
                return AgentResult(return_code=0, stdout="round 1 start\n", stderr="warn line\n")

            with patch("triton_agent.optimize.batch.run_optimize_request", side_effect=_fake_run):
                with redirect_stdout(stdout):
                    exit_code = main(["optimize-batch", "-i", str(root), "--show-output"])

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertIn("[alpha] round 1 start", rendered)
            self.assertIn("[beta] round 1 start", rendered)
            self.assertIn("Summary: 2 succeeded, 0 failed", rendered)

    def test_main_optimize_batch_rejects_invalid_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stderr = StringIO()

            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exc:
                    main(["optimize-batch", "-i", str(root), "--max-concurrency", "0"])

            self.assertEqual(exc.exception.code, 2)
            self.assertIn("--max-concurrency must be at least 1", stderr.getvalue())

    def test_main_gen_eval_batch_auto_detects_operator_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            (first / "kernel.py").write_text("print('x')\n", encoding="utf-8")
            (first / "opt_kernel.py").write_text("", encoding="utf-8")
            (first / "__init__.py").write_text("", encoding="utf-8")
            (second / "matmul_impl.py").write_text("print('y')\n", encoding="utf-8")

            seen_inputs: list[Path] = []

            def _fake_run(request, stdout=None, stderr=None):
                seen_inputs.append(request.input_path)
                return AgentResult(return_code=0, stdout="", stderr="")

            with patch("triton_agent.generation_batch.run_generation_request", side_effect=_fake_run):
                exit_code = main(["gen-eval-batch", "-i", str(root)])

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                seen_inputs,
                [
                    (first / "kernel.py").resolve(),
                    (second / "matmul_impl.py").resolve(),
                ],
            )

    def test_main_gen_eval_batch_reports_workspace_selection_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = root / "bad"
            good = root / "good"
            bad.mkdir()
            good.mkdir()
            (bad / "a.py").write_text("print('a')\n", encoding="utf-8")
            (bad / "b.py").write_text("print('b')\n", encoding="utf-8")
            (good / "kernel.py").write_text("print('ok')\n", encoding="utf-8")

            stdout = StringIO()
            seen_inputs: list[Path] = []

            def _fake_run(request, stdout=None, stderr=None):
                seen_inputs.append(request.input_path)
                return AgentResult(return_code=0, stdout="", stderr="")

            with patch("triton_agent.generation_batch.run_generation_request", side_effect=_fake_run):
                with redirect_stdout(stdout):
                    exit_code = main(["gen-eval-batch", "-i", str(root)])

            self.assertEqual(exit_code, 1)
            self.assertEqual(seen_inputs, [(good / "kernel.py").resolve()])
            self.assertIn("bad", stdout.getvalue())
            self.assertIn("found multiple candidate operator files", stdout.getvalue())
            self.assertIn("Summary: 1 succeeded, 1 failed", stdout.getvalue())

    def test_main_gen_eval_batch_honors_max_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("one", "two", "three"):
                workspace = root / name
                workspace.mkdir()
                (workspace / f"{name}.py").write_text("print('x')\n", encoding="utf-8")

            active = 0
            max_active = 0
            launched = 0
            lock = threading.Lock()
            first_pair_ready = threading.Event()
            release_gate = threading.Event()

            def _fake_run(_request, stdout=None, stderr=None):
                nonlocal active, max_active, launched
                with lock:
                    launched += 1
                    active += 1
                    max_active = max(max_active, active)
                    if launched >= 2:
                        first_pair_ready.set()
                first_pair_ready.wait(timeout=1)
                release_gate.wait(timeout=0.1)
                with lock:
                    active -= 1
                return AgentResult(return_code=0, stdout="", stderr="")

            with patch("triton_agent.generation_batch.run_generation_request", side_effect=_fake_run):
                exit_code = main(["gen-eval-batch", "-i", str(root), "--max-concurrency", "2"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(max_active, 2)

    def test_main_gen_eval_batch_show_output_prefixes_workspace_streams(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("alpha", "beta"):
                workspace = root / name
                workspace.mkdir()
                (workspace / f"{name}.py").write_text("print('x')\n", encoding="utf-8")

            stdout = StringIO()

            def _fake_run(request, stdout=None, stderr=None):
                if stdout is not None:
                    stdout.write("repair start\n")
                if stderr is not None:
                    stderr.write("warn line\n")
                return AgentResult(return_code=0, stdout="repair start\n", stderr="warn line\n")

            with patch("triton_agent.generation_batch.run_generation_request", side_effect=_fake_run):
                with redirect_stdout(stdout):
                    exit_code = main(["gen-eval-batch", "-i", str(root), "--show-output"])

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertIn("[alpha] repair start", rendered)
            self.assertIn("[beta] repair start", rendered)
            self.assertIn("Summary: 2 succeeded, 0 failed", rendered)

    def test_main_gen_eval_batch_rejects_invalid_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stderr = StringIO()

            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exc:
                    main(["gen-eval-batch", "-i", str(root), "--max-concurrency", "0"])

            self.assertEqual(exc.exception.code, 2)
            self.assertIn("--max-concurrency must be at least 1", stderr.getvalue())

    def test_default_generated_paths_follow_convention(self) -> None:
        operator = Path("/tmp/add.py")
        self.assertEqual(
            default_generated_output_path(CommandKind.GEN_TEST, operator, test_mode="standalone"),
            Path("/tmp/test_add.py"),
        )
        self.assertEqual(
            default_generated_output_path(CommandKind.GEN_TEST, operator, test_mode="differential"),
            Path("/tmp/differential_test_add.py"),
        )
        self.assertEqual(
            default_generated_output_path(CommandKind.GEN_BENCH, operator),
            Path("/tmp/bench_add.py"),
        )
        self.assertEqual(
            default_generated_output_path(CommandKind.OPTIMIZE, operator),
            Path("/tmp/opt_add.py"),
        )

    def test_run_test_requires_generated_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                resolve_execution_target(CommandKind.RUN_TEST, operator)

    def test_run_bench_resolves_generated_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench = root / "bench_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            bench.write_text("print('bench')", encoding="utf-8")
            self.assertEqual(
                resolve_execution_target(CommandKind.RUN_BENCH, operator),
                bench,
            )

    def test_main_run_test_uses_explicit_files_and_harness_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            harness_dir = root / "generated"
            harness_dir.mkdir()
            operator = root / "kernel.py"
            test_file = harness_dir / "test_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: standalone\nprint('test')", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")

            with patch("triton_agent.commands.execution.run_local_test", return_value=(fake_result, None)) as mocked:
                exit_code = main(
                    [
                        "run-test",
                        "--test-file",
                        str(test_file),
                        "--operator-file",
                        str(operator),
                    ]
                )

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(
                test_file.resolve(),
                operator.resolve(),
                "standalone",
            )

    def test_main_run_test_reads_mode_from_metadata_when_flag_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "differential_test_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch("triton_agent.commands.execution.run_local_test", return_value=(fake_result, None)) as mocked:
                exit_code = main(
                    [
                        "run-test",
                        "--test-file",
                        str(test_file),
                        "--operator-file",
                        str(operator),
                    ]
                )

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(
                test_file.resolve(),
                operator.resolve(),
                "differential",
            )

    def test_run_test_wrapper_calls_loaded_skill_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "test_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: standalone\nprint('test')", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            runtime = SimpleNamespace(
                parse_test_metadata=lambda _path: {"test-mode": "standalone"},
                run_local_test=lambda *_args, **_kwargs: (fake_result, None),
            )

            with patch("triton_agent.execution.load_run_skill_module", return_value=runtime) as mocked_loader:
                exit_code = main(
                    [
                        "run-test",
                        "--test-file",
                        str(test_file),
                        "--operator-file",
                        str(operator),
                    ]
                )

            self.assertEqual(exit_code, 0)
            mocked_loader.assert_called_with("test_runner")

    def test_main_gen_test_differential_uses_differential_default_output_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")

            captured = {}

            def _fake_build_prompt(
                command_kind,
                input_path,
                operator_path,
                output_path,
                test_mode,
                bench_mode,
                force_overwrite,
                remote,
                remote_workdir,
                min_rounds,
                continue_optimize,
            ):
                captured["output_path"] = output_path
                captured["remote"] = remote
                captured["remote_workdir"] = remote_workdir
                captured["min_rounds"] = min_rounds
                captured["continue_optimize"] = continue_optimize
                return "Prompt body"

            def _fake_create_runner(_agent_name):
                class _Runner:
                    def run(self, request):
                        captured["request_output"] = request.output_path
                        return AgentResult(return_code=0, stdout="", stderr="")

                return _Runner()

            with patch("triton_agent.generation.build_prompt", side_effect=_fake_build_prompt):
                with patch("triton_agent.generation.create_runner", side_effect=_fake_create_runner):
                    with patch("triton_agent.generation.SkillLinkManager.prepare_skills", return_value=[]):
                        with patch("triton_agent.generation.SkillLinkManager.cleanup", return_value=[]):
                            exit_code = main(
                                [
                                    "gen-test",
                                    "-i",
                                    str(operator),
                                    "--test-mode",
                                    "differential",
                                ]
                            )

            self.assertEqual(exit_code, 0)
            expected_output = (root / "differential_test_kernel.py").resolve()
            self.assertEqual(captured["output_path"], expected_output)
            self.assertEqual(captured["request_output"], expected_output)

    def test_main_optimize_resume_continue_rejects_explicit_test_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")

            stderr = StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exc:
                    main(
                        [
                            "optimize",
                            "-i",
                            str(operator),
                            "--resume",
                            "continue",
                            "--test-mode",
                            "differential",
                        ]
                    )

            self.assertEqual(exc.exception.code, 2)
            self.assertIn(
                "--resume continue cannot be combined with --test-mode",
                stderr.getvalue(),
            )

    def test_main_optimize_resume_continue_rejects_explicit_bench_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")

            stderr = StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exc:
                    main(
                        [
                            "optimize",
                            "-i",
                            str(operator),
                            "--resume",
                            "continue",
                            "--bench-mode",
                            "standalone",
                        ]
                    )

            self.assertEqual(exc.exception.code, 2)
            self.assertIn(
                "--resume continue cannot be combined with --bench-mode",
                stderr.getvalue(),
            )

    def test_main_optimize_resume_continue_requires_opt_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")

            stderr = StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exc:
                    main(["optimize", "-i", str(operator), "--resume", "continue"])

            self.assertEqual(exc.exception.code, 2)
            self.assertIn(
                "resume continue requires existing opt-note.md",
                stderr.getvalue(),
            )

    def test_main_optimize_resume_auto_uses_fresh_for_no_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")

            captured = {}

            def _fake_build_prompt(
                command_kind,
                input_path,
                operator_path,
                output_path,
                test_mode,
                bench_mode,
                force_overwrite,
                remote,
                remote_workdir,
                min_rounds,
                resume_existing_session,
            ):
                captured["test_mode"] = test_mode
                captured["bench_mode"] = bench_mode
                captured["resume_existing_session"] = resume_existing_session
                return "Prompt body"

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch("triton_agent.optimize.runtime.build_prompt", side_effect=_fake_build_prompt):
                with patch(
                    "triton_agent.optimize.runtime.OptimizeSupervisor.run",
                    return_value=fake_result,
                ):
                    with patch("triton_agent.optimize.runtime.create_runner", return_value=object()):
                        with patch(
                            "triton_agent.optimize.runtime.SkillLinkManager.prepare_skills",
                            return_value=[],
                        ):
                            with patch(
                                "triton_agent.optimize.runtime.SkillLinkManager.cleanup",
                                return_value=[],
                            ):
                                exit_code = main(
                                    [
                                        "optimize",
                                        "-i",
                                        str(operator),
                                        "--resume",
                                        "auto",
                                        "--test-mode",
                                        "standalone",
                                        "--bench-mode",
                                        "msprof",
                                    ]
                                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["test_mode"], "standalone")
            self.assertEqual(captured["bench_mode"], "msprof")
            self.assertFalse(captured["resume_existing_session"])

    def test_main_optimize_resume_auto_rejects_partial_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            (root / "opt-note.md").write_text("history\n", encoding="utf-8")

            stderr = StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exc:
                    main(["optimize", "-i", str(operator), "--resume", "auto"])

            self.assertEqual(exc.exception.code, 2)
            self.assertIn(
                "resume auto found partial optimize state",
                stderr.getvalue(),
            )

    def test_main_optimize_resume_continue_rejects_multiple_test_harnesses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            (root / "opt-note.md").write_text("history\n", encoding="utf-8")
            (root / "opt-round-1").mkdir()
            (root / "test_kernel.py").write_text(
                "# test-mode: standalone\nprint('test')\n", encoding="utf-8"
            )
            (root / "differential_test_kernel.py").write_text(
                "# test-mode: differential\nprint('test')\n", encoding="utf-8"
            )
            (root / "bench_kernel.py").write_text(
                "# bench-mode: standalone\n# kernel: k\nprint('bench')\n", encoding="utf-8"
            )

            stderr = StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exc:
                    main(["optimize", "-i", str(operator), "--resume", "continue"])

            self.assertEqual(exc.exception.code, 2)
            self.assertIn("resume continue found multiple test harnesses", stderr.getvalue())

    def test_main_optimize_resume_fresh_rejects_existing_optimize_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            (root / "opt-note.md").write_text("history\n", encoding="utf-8")
            (root / "opt-round-1").mkdir()

            stderr = StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exc:
                    main(["optimize", "-i", str(operator), "--resume", "fresh"])

            self.assertEqual(exc.exception.code, 2)
            self.assertIn(
                "resume fresh refused because optimize artifacts already exist",
                stderr.getvalue(),
            )

    def test_main_optimize_resume_auto_uses_continue_for_resumable_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            (root / "opt-note.md").write_text("history\n", encoding="utf-8")
            (root / "opt-round-1").mkdir()
            (root / "differential_test_kernel.py").write_text(
                "# test-mode: differential\nprint('test')\n", encoding="utf-8"
            )
            (root / "bench_kernel.py").write_text(
                "# bench-mode: msprof\n# kernel: k\nprint('bench')\n", encoding="utf-8"
            )

            captured = {}

            def _fake_build_prompt(
                command_kind,
                input_path,
                operator_path,
                output_path,
                test_mode,
                bench_mode,
                force_overwrite,
                remote,
                remote_workdir,
                min_rounds,
                resume_existing_session,
            ):
                captured["test_mode"] = test_mode
                captured["bench_mode"] = bench_mode
                captured["resume_existing_session"] = resume_existing_session
                return "Prompt body"

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch("triton_agent.optimize.runtime.build_prompt", side_effect=_fake_build_prompt):
                with patch(
                    "triton_agent.optimize.runtime.OptimizeSupervisor.run",
                    return_value=fake_result,
                ) as mocked:
                    with patch("triton_agent.optimize.runtime.create_runner", return_value=object()):
                        with patch(
                            "triton_agent.optimize.runtime.SkillLinkManager.prepare_skills",
                            return_value=[],
                        ):
                            with patch(
                                "triton_agent.optimize.runtime.SkillLinkManager.cleanup",
                                return_value=[],
                            ):
                                exit_code = main(
                                    ["optimize", "-i", str(operator), "--resume", "auto"]
                                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["test_mode"], "differential")
            self.assertEqual(captured["bench_mode"], "msprof")
            self.assertTrue(captured["resume_existing_session"])
            request = mocked.call_args.args[1]
            self.assertEqual(request.test_mode, "differential")
            self.assertEqual(request.bench_mode, "msprof")

    def test_main_optimize_passes_no_agent_session_to_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch("triton_agent.optimize.runtime.build_prompt", return_value="Prompt body"):
                with patch(
                    "triton_agent.optimize.runtime.OptimizeSupervisor.run",
                    return_value=fake_result,
                ) as mocked:
                    with patch("triton_agent.optimize.runtime.create_runner", return_value=object()):
                        with patch(
                            "triton_agent.optimize.runtime.SkillLinkManager.prepare_skills",
                            return_value=[],
                        ):
                            with patch(
                                "triton_agent.optimize.runtime.SkillLinkManager.cleanup",
                                return_value=[],
                            ):
                                exit_code = main(
                                    ["optimize", "-i", str(operator), "--no-agent-session"]
                                )

            self.assertEqual(exit_code, 0)
            request = mocked.call_args.args[1]
            self.assertTrue(request.no_agent_session)

    def test_main_run_bench_reports_missing_bench_file_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            missing_bench = root / "bench_kernel.py"

            stderr = StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exc:
                    main(
                        [
                            "run-bench",
                            "--bench-file",
                            str(missing_bench),
                            "--operator-file",
                            str(operator),
                        ]
                    )

            self.assertEqual(exc.exception.code, 2)
            self.assertIn("Bench file path does not exist", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_main_run_test_executes_locally_and_prints_return_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "test_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: standalone\nprint('test')", encoding="utf-8")

            stdout = StringIO()
            stderr = StringIO()
            fake_result = AgentResult(return_code=0, stdout="test stdout\n", stderr="test stderr\n")

            with patch("triton_agent.commands.execution.run_local_test", return_value=(fake_result, None)) as mocked:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exit_code = main(
                        [
                            "run-test",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator),
                        ]
                    )

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once()
            self.assertIn("Return code: 0", stdout.getvalue())
            self.assertNotIn("test stdout", stdout.getvalue())
            self.assertIn("test stderr", stderr.getvalue())

    def test_main_run_test_reports_archived_differential_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "differential_test_kernel.py"
            archive = root / "kernel_result.pt"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')", encoding="utf-8")

            stdout = StringIO()
            fake_result = AgentResult(return_code=0, stdout="", stderr="")

            with patch("triton_agent.commands.execution.run_local_test", return_value=(fake_result, archive)):
                with redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "run-test",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator),
                            "--test-mode",
                            "differential",
                        ]
                    )

            self.assertEqual(exit_code, 0)
            self.assertIn(f"Archived result: {archive}", stdout.getvalue())

    def test_main_run_test_uses_remote_runner_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "test_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: standalone\nprint('test')", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch(
                "triton_agent.commands.execution.run_remote_test",
                return_value=(fake_result, None, "/tmp/triton-agent-abc"),
            ) as mocked:
                exit_code = main(
                    [
                        "run-test",
                        "--test-file",
                        str(test_file),
                        "--operator-file",
                        str(operator),
                        "--remote",
                        "alice@example.com:2200",
                        "--remote-workdir",
                        "/tmp/runs",
                    ]
                )

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(
                test_file.resolve(),
                operator.resolve(),
                "standalone",
                "alice@example.com:2200",
                "/tmp/runs",
                keep_remote_workdir=False,
                verbose=False,
                stderr=sys.stderr,
            )

    def test_main_run_test_prints_remote_workspace_when_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "test_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: standalone\nprint('test')", encoding="utf-8")

            stdout = StringIO()
            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch(
                "triton_agent.commands.execution.run_remote_test",
                return_value=(fake_result, None, "/tmp/triton-agent-keep"),
            ):
                with redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "run-test",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator),
                            "--remote",
                            "alice@example.com",
                            "--keep-remote-workdir",
                        ]
                    )

            self.assertEqual(exit_code, 0)
            self.assertIn("Remote workspace: /tmp/triton-agent-keep", stdout.getvalue())

    def test_main_run_bench_executes_locally_and_prints_perf_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            perf_file = root / "kernel_perf.txt"
            operator.write_text("print('x')", encoding="utf-8")
            bench_file.write_text("# bench-mode: standalone\nprint('bench')", encoding="utf-8")

            stdout = StringIO()
            stderr = StringIO()
            fake_result = AgentResult(return_code=0, stdout="latency-a: 1.0\n", stderr="bench stderr\n")

            with patch("triton_agent.commands.execution.run_local_bench", return_value=(fake_result, perf_file)) as mocked:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exit_code = main(
                        [
                            "run-bench",
                            "--bench-file",
                            str(bench_file),
                            "--operator-file",
                            str(operator),
                        ]
                    )

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(
                bench_file.resolve(),
                operator.resolve(),
                "standalone",
            )
            self.assertIn("Return code: 0", stdout.getvalue())
            self.assertIn(f"Perf file: {perf_file}", stdout.getvalue())
            self.assertNotIn("latency-a", stdout.getvalue())
            self.assertIn("bench stderr", stderr.getvalue())

    def test_main_run_bench_reads_mode_from_metadata_when_flag_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            bench_file.write_text("# bench-mode: msprof\n# kernel: k\nprint('bench')", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch("triton_agent.commands.execution.run_local_bench", return_value=(fake_result, None)) as mocked:
                exit_code = main(
                    [
                        "run-bench",
                        "--bench-file",
                        str(bench_file),
                        "--operator-file",
                        str(operator),
                    ]
                )

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(
                bench_file.resolve(),
                operator.resolve(),
                "msprof",
            )

    def test_run_bench_wrapper_calls_loaded_skill_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            bench_file.write_text("# bench-mode: standalone\nprint('bench')", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            runtime = SimpleNamespace(
                parse_bench_metadata=lambda _path: {"bench-mode": "standalone"},
                run_local_bench=lambda *_args, **_kwargs: (fake_result, None),
            )

            with patch("triton_agent.execution.load_run_skill_module", return_value=runtime) as mocked_loader:
                exit_code = main(
                    [
                        "run-bench",
                        "--bench-file",
                        str(bench_file),
                        "--operator-file",
                        str(operator),
                    ]
                )

            self.assertEqual(exit_code, 0)
            mocked_loader.assert_called_with("bench_runner")

    def test_main_run_bench_uses_remote_runner_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            bench_file.write_text("# bench-mode: standalone\nprint('bench')", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch(
                "triton_agent.commands.execution.run_remote_bench",
                return_value=(fake_result, None, "/tmp/triton-agent-bench"),
            ) as mocked:
                exit_code = main(
                    [
                        "run-bench",
                        "--bench-file",
                        str(bench_file),
                        "--operator-file",
                        str(operator),
                        "--remote",
                        "alice@example.com",
                    ]
                )

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(
                bench_file.resolve(),
                operator.resolve(),
                "standalone",
                "alice@example.com",
                None,
                keep_remote_workdir=False,
                verbose=False,
                stderr=sys.stderr,
            )

    def test_main_run_bench_prints_remote_workspace_when_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            bench_file.write_text("# bench-mode: standalone\nprint('bench')", encoding="utf-8")

            stdout = StringIO()
            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch(
                "triton_agent.commands.execution.run_remote_bench",
                return_value=(fake_result, None, "/tmp/triton-agent-keep-bench"),
            ):
                with redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "run-bench",
                            "--bench-file",
                            str(bench_file),
                            "--operator-file",
                            str(operator),
                            "--remote",
                            "alice@example.com",
                            "--keep-remote-workdir",
                        ]
                    )

            self.assertEqual(exit_code, 0)
            self.assertIn("Remote workspace: /tmp/triton-agent-keep-bench", stdout.getvalue())

    def test_main_run_bench_reports_missing_perf_artifact_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            bench_file.write_text("# bench-mode: standalone\nprint('bench')", encoding="utf-8")

            stderr = StringIO()
            with patch("triton_agent.commands.execution.run_local_bench", side_effect=FileNotFoundError("missing perf")):
                with redirect_stderr(stderr):
                    exit_code = main(
                        [
                            "run-bench",
                            "--bench-file",
                            str(bench_file),
                            "--operator-file",
                            str(operator),
                            "--bench-mode",
                            "msprof",
                        ]
                    )

            self.assertEqual(exit_code, 1)
            self.assertIn("missing perf", stderr.getvalue())

    def test_main_compare_result_uses_local_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oracle = root / "abs_result.pt"
            new = root / "opt_abs_result.pt"
            oracle.write_text("oracle", encoding="utf-8")
            new.write_text("new", encoding="utf-8")

            with patch("triton_agent.commands.comparison.compare_result_files", return_value=0) as mocked:
                exit_code = main(
                    [
                        "compare-result",
                        "--oracle-result",
                        str(oracle),
                        "--new-result",
                        str(new),
                    ]
                )

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(oracle.resolve(), new.resolve(), "balanced")

    def test_main_compare_result_uses_remote_comparison_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oracle = root / "abs_result.pt"
            new = root / "opt_abs_result.pt"
            oracle.write_text("oracle", encoding="utf-8")
            new.write_text("new", encoding="utf-8")

            with patch("triton_agent.commands.comparison.compare_remote_result_files", return_value=0) as mocked:
                exit_code = main(
                    [
                        "compare-result",
                        "--oracle-result",
                        str(oracle),
                        "--new-result",
                        str(new),
                        "--remote",
                        "alice@example.com:2200",
                    ]
                )

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(
                oracle.resolve(),
                new.resolve(),
                "balanced",
                "alice@example.com:2200",
                None,
                verbose=False,
                stderr=sys.stderr,
            )

    def test_main_compare_perf_uses_local_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline_perf.txt"
            compare = root / "candidate_perf.txt"
            baseline.write_text("latency-a: 10\n", encoding="utf-8")
            compare.write_text("latency-a: 11\n", encoding="utf-8")

            with patch("triton_agent.commands.comparison.compare_perf_files", return_value=0) as mocked:
                exit_code = main(
                    [
                        "compare-perf",
                        "--baseline",
                        str(baseline),
                        "--compare",
                        str(compare),
                    ]
                )

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(baseline.resolve(), compare.resolve())

    def test_main_run_test_reports_missing_operator_file_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_file = root / "test_kernel.py"
            test_file.write_text("# test-mode: standalone\nprint('test')", encoding="utf-8")
            missing_operator = root / "kernel.py"

            stderr = StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exc:
                    main(
                        [
                            "run-test",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(missing_operator),
                        ]
                    )

            self.assertEqual(exc.exception.code, 2)
            self.assertIn("Operator file path does not exist", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_generation_refuses_to_overwrite_existing_file_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "test_kernel.py"
            output.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_generation_target(CommandKind.GEN_TEST, output, force_overwrite=False)

    def test_generation_deletes_existing_file_when_overwrite_is_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "test_kernel.py"
            output.write_text("existing", encoding="utf-8")
            messages = prepare_generation_target(CommandKind.GEN_TEST, output, force_overwrite=True)
            self.assertFalse(output.exists())
            self.assertTrue(any("removed existing output file" in message for message in messages))

    def test_main_reports_existing_output_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            output = root / "test_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            output.write_text("existing", encoding="utf-8")

            stderr = StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exc:
                    main(["gen-test", "-i", str(operator), "-o", str(output)])

            self.assertEqual(exc.exception.code, 2)
            self.assertIn("Output file already exists", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())


class PromptTests(unittest.TestCase):
    def test_gen_eval_prompt_mentions_operator_repair_and_dual_outputs(self) -> None:
        prompt = build_prompt(
            CommandKind.GEN_EVAL,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
            force_overwrite=False,
        )
        self.assertIn("eval-gen", prompt)
        self.assertIn("Requested test output: /tmp/differential_test_op.py", prompt)
        self.assertIn("Requested benchmark output: /tmp/bench_op.py", prompt)
        self.assertIn("may edit the original operator file directly", prompt)
        self.assertIn("both generated artifacts must be executed", prompt)

    def test_prompt_mentions_skill_and_output(self) -> None:
        prompt = build_prompt(
            CommandKind.GEN_TEST,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/test_op.py"),
            test_mode=None,
            bench_mode=None,
            force_overwrite=False,
        )
        self.assertIn("test-gen", prompt)
        self.assertIn("/tmp/op.py", prompt)
        self.assertIn("/tmp/test_op.py", prompt)

    def test_prompt_mentions_force_overwrite(self) -> None:
        prompt = build_prompt(
            CommandKind.GEN_TEST,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/test_op.py"),
            test_mode=None,
            bench_mode=None,
            force_overwrite=True,
        )
        self.assertIn("overwrite", prompt.lower())

    def test_prompt_mentions_requested_test_mode(self) -> None:
        prompt = build_prompt(
            CommandKind.RUN_TEST,
            Path("/tmp/test_op.py"),
            Path("/tmp/op.py"),
            None,
            test_mode="differential",
            bench_mode=None,
            force_overwrite=False,
        )
        self.assertIn("Operator file: /tmp/op.py", prompt)
        self.assertIn("Test file: /tmp/test_op.py", prompt)
        self.assertIn("Requested test mode: differential", prompt)

    def test_prompt_mentions_requested_bench_mode(self) -> None:
        prompt = build_prompt(
            CommandKind.RUN_BENCH,
            Path("/tmp/bench_op.py"),
            Path("/tmp/op.py"),
            None,
            test_mode=None,
            bench_mode="msprof",
            force_overwrite=False,
        )
        self.assertIn("Operator file: /tmp/op.py", prompt)
        self.assertIn("Benchmark file: /tmp/bench_op.py", prompt)
        self.assertIn("Requested bench mode: msprof", prompt)

    def test_prompt_mentions_default_test_mode(self) -> None:
        prompt = build_prompt(
            CommandKind.GEN_TEST,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/test_op.py"),
            test_mode="standalone",
            bench_mode=None,
            force_overwrite=False,
        )
        self.assertIn("Requested test mode: standalone", prompt)

    def test_prompt_mentions_default_bench_mode(self) -> None:
        prompt = build_prompt(
            CommandKind.GEN_BENCH,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/bench_op.py"),
            test_mode=None,
            bench_mode="standalone",
            force_overwrite=False,
        )
        self.assertIn("Requested bench mode: standalone", prompt)

    def test_gen_test_prompt_requires_execute_and_autofix(self) -> None:
        prompt = build_prompt(
            CommandKind.GEN_TEST,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/test_op.py"),
            test_mode="standalone",
            bench_mode=None,
            force_overwrite=False,
        )
        self.assertIn("After generating the artifact, execute the generated test or benchmark case", prompt)
        self.assertIn("repair the generated artifact and retry automatically", prompt)

    def test_gen_bench_prompt_requires_execute_and_autofix(self) -> None:
        prompt = build_prompt(
            CommandKind.GEN_BENCH,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/bench_op.py"),
            test_mode=None,
            bench_mode="standalone",
            force_overwrite=False,
        )
        self.assertIn("After generating the artifact, execute the generated test or benchmark case", prompt)
        self.assertIn("repair the generated artifact and retry automatically", prompt)

    def test_prompt_mentions_remote_execution_context(self) -> None:
        prompt = build_prompt(
            CommandKind.GEN_TEST,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/test_op.py"),
            test_mode="standalone",
            bench_mode=None,
            force_overwrite=False,
            remote="alice@example.com:2200",
            remote_workdir="/tmp/triton-agent",
        )
        self.assertIn("Remote execution target: alice@example.com:2200", prompt)
        self.assertIn("Remote execution root: /tmp/triton-agent", prompt)
        self.assertIn("When you execute generated test cases or benchmark cases", prompt)
        self.assertIn("include the same `--remote` setting", prompt)

    def test_gen_eval_prompt_mentions_remote_execution_context(self) -> None:
        prompt = build_prompt(
            CommandKind.GEN_EVAL,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
            force_overwrite=False,
            remote="alice@example.com:2200",
            remote_workdir="/tmp/triton-agent",
        )
        self.assertIn("Remote execution target: alice@example.com:2200", prompt)
        self.assertIn("Remote execution root: /tmp/triton-agent", prompt)
        self.assertIn("both generated artifacts must be executed", prompt)
        self.assertIn("include the same `--remote` setting", prompt)

    def test_optimize_prompt_mentions_requested_modes(self) -> None:
        prompt = build_prompt(
            CommandKind.OPTIMIZE,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
            force_overwrite=False,
        )
        self.assertIn("Requested test mode: differential", prompt)
        self.assertIn("Requested bench mode: standalone", prompt)

    def test_optimize_prompt_mentions_min_rounds(self) -> None:
        prompt = build_prompt(
            CommandKind.OPTIMIZE,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
            force_overwrite=False,
            min_rounds=4,
        )
        self.assertIn("Complete at least 4 optimization rounds", prompt)

    def test_optimize_prompt_mentions_continue_mode_for_resolved_resume(self) -> None:
        prompt = build_prompt(
            CommandKind.OPTIMIZE,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
            force_overwrite=False,
            resume_existing_session=True,
        )
        self.assertIn("Continue the existing optimization session", prompt)
        self.assertIn("Read `opt-note.md`", prompt)


class OutputRenderingTests(unittest.TestCase):
    def test_render_result_skips_duplicate_stdout_when_show_output_enabled(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        result = AgentResult(return_code=0, stdout="streamed\n", stderr="")
        render_result(result, show_output=True, stdout=stdout, stderr=stderr)
        self.assertEqual(stdout.getvalue(), "")

    def test_render_result_prints_stdout_when_show_output_disabled(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        result = AgentResult(return_code=0, stdout="final\n", stderr="")
        render_result(result, show_output=False, stdout=stdout, stderr=stderr)
        self.assertEqual(stdout.getvalue(), "final\n")


class ResultNormalizationTests(unittest.TestCase):
    def test_invalid_skill_result_payload_raises_actionable_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required keys"):
            normalize_agent_result({"stdout": "", "stderr": ""})


if __name__ == "__main__":
    unittest.main()
