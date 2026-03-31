import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from contextlib import redirect_stderr
from unittest.mock import patch

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


class CliParserTests(unittest.TestCase):
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
            ("gen_test", CommandKind.GEN_TEST),
            ("run_test", CommandKind.RUN_TEST),
            ("gen_bench", CommandKind.GEN_BENCH),
            ("run_bench", CommandKind.RUN_BENCH),
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
        self.assertIn("gen-test", help_text)
        self.assertIn("run-test", help_text)
        self.assertIn("gen-bench", help_text)
        self.assertIn("run-bench", help_text)
        self.assertNotIn("gen_test", help_text)
        self.assertNotIn("run_test", help_text)
        self.assertNotIn("gen_bench", help_text)
        self.assertNotIn("run_bench", help_text)

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
                "--interact",
            ]
        )
        self.assertEqual(args.command_kind, CommandKind.RUN_BENCH)
        self.assertEqual(args.output, "out.txt")
        self.assertTrue(args.interact)

    def test_run_test_requires_test_and_operator_files(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["run-test", "--test-file", "test_kernel.py", "--operator-file", "kernel.py"]
        )
        self.assertEqual(args.command_kind, CommandKind.RUN_TEST)
        self.assertEqual(args.test_file, "test_kernel.py")
        self.assertEqual(args.operator_file, "kernel.py")

    def test_run_bench_requires_bench_and_operator_files(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["run-bench", "--bench-file", "bench_kernel.py", "--operator-file", "kernel.py"]
        )
        self.assertEqual(args.command_kind, CommandKind.RUN_BENCH)
        self.assertEqual(args.bench_file, "bench_kernel.py")
        self.assertEqual(args.operator_file, "kernel.py")

    def test_run_commands_reject_input_flag(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["run-test", "-i", "kernel.py"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["run-bench", "-i", "kernel.py"])

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
        self.assertEqual(gen_args.test_mode, "standalone")
        self.assertEqual(run_args.test_mode, "differential")

    def test_test_commands_default_to_standalone_mode(self) -> None:
        parser = build_parser()
        gen_args = parser.parse_args(["gen-test", "-i", "kernel.py"])
        run_args = parser.parse_args(
            ["run-test", "--test-file", "test_kernel.py", "--operator-file", "kernel.py"]
        )
        self.assertEqual(gen_args.test_mode, "standalone")
        self.assertEqual(run_args.test_mode, "standalone")

    def test_bench_mode_option_is_available_for_bench_commands(self) -> None:
        parser = build_parser()
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
        self.assertEqual(gen_args.bench_mode, "standalone")
        self.assertEqual(run_args.bench_mode, "msprof")

    def test_bench_commands_default_to_standalone_mode(self) -> None:
        parser = build_parser()
        gen_args = parser.parse_args(["gen-bench", "-i", "kernel.py"])
        run_args = parser.parse_args(
            ["run-bench", "--bench-file", "bench_kernel.py", "--operator-file", "kernel.py"]
        )
        self.assertEqual(gen_args.bench_mode, "standalone")
        self.assertEqual(run_args.bench_mode, "standalone")

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

    def test_optimize_command_defaults_to_optimize_modes(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py"])
        self.assertEqual(args.test_mode, "differential")
        self.assertEqual(args.bench_mode, "standalone")


class PathResolutionTests(unittest.TestCase):
    def test_default_generated_paths_follow_convention(self) -> None:
        operator = Path("/tmp/add.py")
        self.assertEqual(
            default_generated_output_path(CommandKind.GEN_TEST, operator),
            Path("/tmp/test_add.py"),
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
            test_file.write_text("print('test')", encoding="utf-8")

            captured = {}

            def _fake_build_prompt(
                command_kind, input_path, operator_path, output_path, test_mode, bench_mode, force_overwrite
            ):
                captured["prompt_input"] = input_path
                captured["prompt_operator"] = operator_path
                captured["prompt_output"] = output_path
                return "Prompt body"

            def _fake_create_runner(_agent_name):
                class _Runner:
                    def run(self, request):
                        captured["request"] = request
                        return AgentResult(return_code=0, stdout="", stderr="")

                return _Runner()

            with patch("triton_agent.cli.build_prompt", side_effect=_fake_build_prompt):
                with patch("triton_agent.cli.create_runner", side_effect=_fake_create_runner):
                    with patch("triton_agent.cli.SkillLinkManager.prepare_skills", return_value=[]):
                        with patch("triton_agent.cli.SkillLinkManager.cleanup", return_value=[]):
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
            self.assertEqual(captured["prompt_input"], test_file.resolve())
            self.assertEqual(captured["prompt_operator"], operator.resolve())
            self.assertEqual(captured["request"].input_path, test_file.resolve())
            self.assertEqual(captured["request"].operator_path, operator.resolve())
            self.assertEqual(captured["request"].workdir, harness_dir.resolve())

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

    def test_main_run_test_reports_missing_operator_file_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_file = root / "test_kernel.py"
            test_file.write_text("print('test')", encoding="utf-8")
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


if __name__ == "__main__":
    unittest.main()
