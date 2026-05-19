import argparse
import os
import sys
import tempfile
import threading
import unittest
from io import StringIO
from pathlib import Path
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from typing import Optional
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import triton_agent.cli as cli_module
import triton_agent.commands.optimize as optimize_commands
from triton_agent.cli import (
    _normalize_command_aliases,
    build_parser,
    main,
)
from triton_agent.commands.optimize import optimize_run_options_from_args
from triton_agent.generation.outputs import prepare_generation_target
from triton_agent.models import AgentResult
from triton_agent.models import CommandKind
from triton_agent.output import render_result
from triton_agent.paths import (
    default_generated_output_path,
    resolve_execution_target,
)
from triton_agent.prompts import (
    append_additional_user_instructions,
    build_optimize_resume_prompt,
    build_optimize_supervisor_prompt,
    build_optimize_unsupervised_prompt,
    build_optimize_worker_prompt,
    build_prompt,
)
from triton_agent.result_normalization import normalize_agent_result


class CliParserTests(unittest.TestCase):
    def test_cli_module_keeps_only_entrypoint_helpers(self) -> None:
        self.assertFalse(hasattr(cli_module, "prepare_generation_target"))
        self.assertFalse(hasattr(cli_module, "render_result"))
        self.assertFalse(hasattr(cli_module, "create_runner"))
        self.assertFalse(hasattr(cli_module, "run_local_test"))
        self.assertFalse(hasattr(cli_module, "run_remote_test"))
        self.assertFalse(hasattr(cli_module, "run_local_bench"))
        self.assertFalse(hasattr(cli_module, "run_remote_bench"))
        self.assertFalse(hasattr(cli_module, "compare_result_files"))
        self.assertFalse(hasattr(cli_module, "compare_remote_result_files"))
        self.assertFalse(hasattr(cli_module, "compare_perf_files"))
        self.assertFalse(hasattr(cli_module, "parse_perf_file"))

    def test_optimize_command_module_no_longer_owns_status_handler(self) -> None:
        self.assertFalse(hasattr(optimize_commands, "handle_optimize_status"))

    def test_command_definitions_cover_every_command_kind(self) -> None:
        self.assertEqual(set(cli_module._COMMAND_SPECS), set(CommandKind))

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

    def test_log_check_batch_accepts_result_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "log-check-batch",
                "-i",
                "kernels",
                "--check-result-file",
                "custom_check.txt",
                "--summary-file",
                "custom_summary.txt",
                "--max-concurrency",
                "4",
                "--verbose",
                "--show-output",
            ]
        )
        self.assertEqual(args.command, "log-check-batch")
        self.assertEqual(args.command_kind, CommandKind.LOG_CHECK_BATCH)
        self.assertEqual(args.check_result_file, "custom_check.txt")
        self.assertEqual(args.summary_file, "custom_summary.txt")
        self.assertEqual(args.max_concurrency, 4)
        self.assertTrue(args.verbose)
        self.assertTrue(args.show_output)

    def test_gen_eval_maps_to_command_kind(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["gen-eval", "-i", "kernel.py"])
        self.assertEqual(args.command, "gen-eval")
        self.assertEqual(args.command_kind, CommandKind.GEN_EVAL)
        self.assertEqual(args.test_mode, "differential")
        self.assertEqual(args.bench_mode, "standalone")
        self.assertEqual(args.agent, "codex")
        self.assertFalse(args.interact)

    def test_gen_eval_accepts_user_prompt(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["gen-eval", "-i", "kernel.py", "--prompt", "Avoid broad operator rewrites."]
        )
        self.assertEqual(args.prompt, "Avoid broad operator rewrites.")

        from triton_agent.commands.generation import generation_options_from_args
        from triton_agent.generation.models import GenerationOptions

        options = generation_options_from_args(args)
        self.assertIsInstance(options, GenerationOptions)
        self.assertEqual(options.prompt, "Avoid broad operator rewrites.")

    def test_convert_maps_to_command_kind(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["convert", "-i", "kernel.py"])
        self.assertEqual(args.command, "convert")
        self.assertEqual(args.command_kind, CommandKind.CONVERT)
        self.assertEqual(args.test_mode, "differential")
        self.assertFalse(hasattr(args, "bench_mode"))
        self.assertEqual(args.agent, "codex")
        self.assertFalse(args.interact)

    def test_convert_batch_maps_to_command_kind(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["convert-batch", "-i", "kernels"])
        self.assertEqual(args.command, "convert-batch")
        self.assertEqual(args.command_kind, CommandKind.CONVERT_BATCH)
        self.assertEqual(args.max_concurrency, 2)
        self.assertEqual(args.test_mode, "differential")
        self.assertEqual(args.agent, "codex")
        self.assertFalse(hasattr(args, "interact"))
        self.assertFalse(hasattr(args, "output"))

    def test_convert_command_accepts_user_prompt(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["convert", "-i", "kernel.py", "--prompt", "Keep the API shape."])
        self.assertEqual(args.prompt, "Keep the API shape.")
        from triton_agent.commands.convert import convert_options_from_args
        from triton_agent.convert.models import ConvertOptions

        options = convert_options_from_args(args)
        self.assertIsInstance(options, ConvertOptions)
        self.assertEqual(options.prompt, "Keep the API shape.")

    def test_convert_batch_accepts_user_prompt(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["convert-batch", "-i", "kernels", "--prompt", "Avoid numerics changes."]
        )
        self.assertEqual(args.prompt, "Avoid numerics changes.")
        from triton_agent.commands.convert import convert_options_from_args
        from triton_agent.convert.models import ConvertOptions

        options = convert_options_from_args(args)
        self.assertIsInstance(options, ConvertOptions)
        self.assertEqual(options.prompt, "Avoid numerics changes.")

    def test_convert_rejects_non_differential_test_mode(self) -> None:
        parser = build_parser()
        stderr = StringIO()
        with self.assertRaises(SystemExit) as exc, redirect_stderr(stderr):
            parser.parse_args(["convert", "-i", "kernel.py", "--test-mode", "standalone"])
        self.assertEqual(exc.exception.code, 2)
        self.assertIn("differential", stderr.getvalue())

    def test_gen_convert_is_no_longer_a_valid_command(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit) as exc:
            parser.parse_args(["gen-convert", "-i", "kernel.py"])
        self.assertEqual(exc.exception.code, 2)

    def test_gen_test_maps_to_command_kind(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["gen-test", "-i", "kernel.py"])
        self.assertEqual(args.command, "gen-test")
        self.assertEqual(args.command_kind, CommandKind.GEN_TEST)
        self.assertEqual(args.agent, "codex")
        self.assertFalse(args.interact)

    def test_gen_eval_batch_accepts_user_prompt(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["gen-eval-batch", "-i", "kernels", "--prompt", "Avoid changing numerics."]
        )
        self.assertEqual(args.prompt, "Avoid changing numerics.")

        from triton_agent.commands.generation import generation_options_from_args
        from triton_agent.generation.models import GenerationOptions

        options = generation_options_from_args(args)
        self.assertIsInstance(options, GenerationOptions)
        self.assertEqual(options.prompt, "Avoid changing numerics.")

    def test_gen_test_accepts_user_prompt(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["gen-test", "-i", "kernel.py", "--prompt", "Preserve helper names."])
        self.assertEqual(args.prompt, "Preserve helper names.")

        from triton_agent.commands.generation import generation_options_from_args
        from triton_agent.generation.models import GenerationOptions

        options = generation_options_from_args(args)
        self.assertIsInstance(options, GenerationOptions)
        self.assertEqual(options.prompt, "Preserve helper names.")

    def test_gen_bench_accepts_user_prompt(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["gen-bench", "-i", "kernel.py", "--prompt", "Keep benchmark shapes small."]
        )
        self.assertEqual(args.prompt, "Keep benchmark shapes small.")

        from triton_agent.commands.generation import generation_options_from_args
        from triton_agent.generation.models import GenerationOptions

        options = generation_options_from_args(args)
        self.assertIsInstance(options, GenerationOptions)
        self.assertEqual(options.prompt, "Keep benchmark shapes small.")

    def test_snake_case_aliases_map_to_same_command_kind(self) -> None:
        parser = build_parser()
        cases = [
            ("gen_eval_batch", CommandKind.GEN_EVAL_BATCH),
            ("gen_eval", CommandKind.GEN_EVAL),
            ("convert", CommandKind.CONVERT),
            ("convert_batch", CommandKind.CONVERT_BATCH),
            ("gen_test", CommandKind.GEN_TEST),
            ("run_test", CommandKind.RUN_TEST),
            ("gen_bench", CommandKind.GEN_BENCH),
            ("run_bench", CommandKind.RUN_BENCH),
            ("verify_batch", CommandKind.VERIFY_BATCH),
            ("optimize_batch", CommandKind.OPTIMIZE_BATCH),
        ]

        for alias, expected_kind in cases:
            with self.subTest(alias=alias):
                argv = [alias, "-i", "kernel.py"]
                if expected_kind == CommandKind.RUN_TEST:
                    argv = [alias, "--test-file", "test_kernel.py", "--operator-file", "kernel.py"]
                elif expected_kind == CommandKind.RUN_BENCH:
                    argv = [alias, "--bench-file", "bench_kernel.py", "--operator-file", "kernel.py"]
                elif expected_kind == CommandKind.VERIFY_BATCH:
                    argv = [alias, "-i", "workspace-root"]
                elif expected_kind == CommandKind.CONVERT_BATCH:
                    argv = [alias, "-i", "workspace-root"]
                args = parser.parse_args(_normalize_command_aliases(argv))
                self.assertEqual(args.command_kind, expected_kind)

    def test_help_keeps_only_canonical_convert_commands(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()
        self.assertIn("convert", help_text)
        self.assertIn("convert-batch", help_text)
        self.assertIn("gen-eval-batch", help_text)
        self.assertIn("gen-eval", help_text)
        self.assertIn("gen-test", help_text)
        self.assertIn("run-test", help_text)
        self.assertIn("gen-bench", help_text)
        self.assertIn("run-bench", help_text)
        self.assertIn("compare-result", help_text)
        self.assertIn("compare-perf", help_text)
        self.assertIn("status", help_text)
        self.assertIn("verify", help_text)
        self.assertIn("verify-batch", help_text)
        self.assertIn("optimize-batch", help_text)
        self.assertNotIn("gen-convert", help_text)
        self.assertNotIn("gen_eval_batch", help_text)
        self.assertNotIn("gen_eval", help_text)
        self.assertNotIn("gen_convert", help_text)
        self.assertNotIn("convert_batch", help_text)
        self.assertNotIn("gen_test", help_text)
        self.assertNotIn("run_test", help_text)
        self.assertNotIn("gen_bench", help_text)
        self.assertNotIn("run_bench", help_text)
        self.assertNotIn("compare_result", help_text)
        self.assertNotIn("compare_perf", help_text)
        self.assertNotIn("optimize-status", help_text)
        self.assertNotIn("optimize_status", help_text)
        self.assertNotIn("optimize-verify", help_text)
        self.assertNotIn("optimize-verify-batch", help_text)
        self.assertNotIn("optimize_batch", help_text)

    def test_top_level_help_groups_commands_and_examples(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()
        self.assertIn("Generate, run, verify, and optimize Triton NPU operator workflows.", help_text)
        self.assertIn("Command groups:", help_text)
        self.assertIn("Generation:", help_text)
        self.assertIn("Execution:", help_text)
        self.assertIn("Comparison:", help_text)
        self.assertIn("Status:", help_text)
        self.assertIn("Verification:", help_text)
        self.assertIn("Optimization:", help_text)
        self.assertIn("Conversion:", help_text)
        self.assertIn("convert", help_text)
        self.assertIn("convert-batch", help_text)
        self.assertIn("Examples:", help_text)
        self.assertIn("triton-agent gen-test -i kernel.py", help_text)
        self.assertIn("triton-agent optimize -i kernel.py --agent codex", help_text)
        self.assertIn("triton-agent verify -i .", help_text)
        self.assertIn("triton-agent status -i .", help_text)

    def test_top_level_help_lists_supported_environment_variables(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()
        self.assertIn("Environment variables:", help_text)
        self.assertIn("TRITON_AGENT_BATCH_NPU_DEVICES", help_text)
        self.assertIn("TRITON_AGENT_CODE_AGENT_MAX_RETRIES", help_text)
        self.assertIn("TRITON_AGENT_BENCH_PROFILE_OUTPUT_DIR", help_text)
        self.assertIn("TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES", help_text)
        self.assertIn("TRITON_AGENT_HOME", help_text)
        self.assertIn("TRITON_AGENT_STALL_TIMEOUT_SECONDS", help_text)
        self.assertIn("TRITON_AGENT_SSH_TIMEOUT_SECONDS", help_text)
        self.assertIn("TRITON_AGENT_SCP_TIMEOUT_SECONDS", help_text)
        self.assertIn("TRITON_AGENT_EVAL_TIMEOUT_SECONDS", help_text)
        self.assertIn("TRITON_AGENT_TEST_TIMEOUT_SECONDS", help_text)
        self.assertIn("TRITON_AGENT_BENCH_TIMEOUT_SECONDS", help_text)
        self.assertIn("TRITON_AGENT_PROFILE_TIMEOUT_SECONDS", help_text)
        self.assertIn("LLM_API_KEY", help_text)
        self.assertIn("LLM_MODEL", help_text)
        self.assertIn("LLM_BASE_URL", help_text)

    def test_subcommand_help_includes_command_description(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            with redirect_stdout(StringIO()) as stdout:
                parser.parse_args(["gen-test", "--help"])
        help_text = stdout.getvalue()
        self.assertIn("Generate a test harness for one operator file.", help_text)
        self.assertIn("--agent", help_text)
        self.assertIn("--test-mode", help_text)

    def test_verify_maps_to_command_kind(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["verify", "-i", "workspace"])
        self.assertEqual(args.command, "verify")
        self.assertEqual(args.command_kind, CommandKind.VERIFY)
        self.assertEqual(args.input, "workspace")
        self.assertEqual(args.phase, "all")
        self.assertIsNone(args.test_mode)
        self.assertIsNone(args.bench_mode)
        self.assertIsNone(args.remote)
        self.assertIsNone(args.remote_workdir)
        self.assertFalse(args.keep_remote_workdir)
        self.assertFalse(args.verbose)
        self.assertFalse(hasattr(args, "agent"))
        self.assertFalse(hasattr(args, "interact"))
        self.assertFalse(hasattr(args, "output"))
        self.assertFalse(hasattr(args, "show_output"))

    def test_verify_accepts_phase_and_remote_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "verify",
                "-i",
                "workspace",
                "--phase",
                "bench",
                "--remote",
                "alice@example.com",
                "--remote-workdir",
                "/tmp/triton-agent",
                "--keep-remote-workdir",
            ]
        )
        self.assertEqual(args.phase, "bench")
        self.assertEqual(args.remote, "alice@example.com")
        self.assertEqual(args.remote_workdir, "/tmp/triton-agent")
        self.assertTrue(args.keep_remote_workdir)

    def test_verify_batch_maps_to_command_kind(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["verify-batch", "-i", "workspace-root"])
        self.assertEqual(args.command, "verify-batch")
        self.assertEqual(args.command_kind, CommandKind.VERIFY_BATCH)
        self.assertEqual(args.input, "workspace-root")
        self.assertFalse(args.force_verify)
        self.assertFalse(args.verbose)
        self.assertFalse(hasattr(args, "agent"))
        self.assertFalse(hasattr(args, "interact"))
        self.assertFalse(hasattr(args, "output"))

    def test_verify_batch_accepts_force_verify(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["verify-batch", "-i", "workspace-root", "--force-verify"])
        self.assertEqual(args.command_kind, CommandKind.VERIFY_BATCH)
        self.assertTrue(args.force_verify)

    def test_verify_batch_accepts_remote_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "verify-batch",
                "-i",
                "workspace-root",
                "--remote",
                "alice@example.com",
                "--remote-workdir",
                "/tmp/triton-agent",
                "--keep-remote-workdir",
            ]
        )
        self.assertEqual(args.command_kind, CommandKind.VERIFY_BATCH)
        self.assertEqual(args.remote, "alice@example.com")
        self.assertEqual(args.remote_workdir, "/tmp/triton-agent")
        self.assertTrue(args.keep_remote_workdir)

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

    def test_agent_commands_accept_openhands_backend(self) -> None:
        parser = build_parser()
        gen_eval_batch_args = parser.parse_args(
            ["gen-eval-batch", "-i", "kernels", "--agent", "openhands"]
        )
        gen_eval_args = parser.parse_args(["gen-eval", "-i", "kernel.py", "--agent", "openhands"])
        gen_test_args = parser.parse_args(["gen-test", "-i", "kernel.py", "--agent", "openhands"])
        gen_bench_args = parser.parse_args(["gen-bench", "-i", "kernel.py", "--agent", "openhands"])
        optimize_args = parser.parse_args(["optimize", "-i", "kernel.py", "--agent", "openhands"])
        self.assertEqual(gen_eval_batch_args.agent, "openhands")
        self.assertEqual(gen_eval_args.agent, "openhands")
        self.assertEqual(gen_test_args.agent, "openhands")
        self.assertEqual(gen_bench_args.agent, "openhands")
        self.assertEqual(optimize_args.agent, "openhands")

    def test_agent_commands_accept_traecli_backend(self) -> None:
        parser = build_parser()
        gen_eval_batch_args = parser.parse_args(
            ["gen-eval-batch", "-i", "kernels", "--agent", "traecli"]
        )
        gen_eval_args = parser.parse_args(["gen-eval", "-i", "kernel.py", "--agent", "traecli"])
        gen_test_args = parser.parse_args(["gen-test", "-i", "kernel.py", "--agent", "traecli"])
        gen_bench_args = parser.parse_args(["gen-bench", "-i", "kernel.py", "--agent", "traecli"])
        optimize_args = parser.parse_args(["optimize", "-i", "kernel.py", "--agent", "traecli"])
        self.assertEqual(gen_eval_batch_args.agent, "traecli")
        self.assertEqual(gen_eval_args.agent, "traecli")
        self.assertEqual(gen_test_args.agent, "traecli")
        self.assertEqual(gen_bench_args.agent, "traecli")
        self.assertEqual(optimize_args.agent, "traecli")

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
        gen_eval_args = parser.parse_args(["gen-eval", "-i", "kernel.py", "--force-overwrite"])
        gen_test_args = parser.parse_args(["gen-test", "-i", "kernel.py", "--force-overwrite"])
        gen_bench_args = parser.parse_args(["gen-bench", "-i", "kernel.py", "--force-overwrite"])
        self.assertTrue(gen_eval_args.force_overwrite)
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

    def test_run_test_accepts_oracle_result_and_compare_level(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run-test",
                "--test-file",
                "differential_test_kernel.py",
                "--operator-file",
                "kernel.py",
                "--oracle-result",
                "oracle_result.pt",
                "--compare-level",
                "strict",
            ]
        )
        self.assertEqual(args.oracle_result, "oracle_result.pt")
        self.assertEqual(args.compare_level, "strict")

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

    def test_optimize_command_accepts_reset_optimize(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--reset-optimize"])
        self.assertTrue(args.reset_optimize)
        options = optimize_run_options_from_args(args)
        self.assertTrue(options.reset_optimize)

    def test_optimize_command_accepts_no_agent_session(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--no-agent-session"])
        self.assertTrue(args.no_agent_session)

    def test_optimize_command_rejects_require_analysis(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["optimize", "-i", "kernel.py", "--require-analysis"])

    def test_optimize_accepts_compiler_source_analysis_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "optimize",
                "-i",
                "kernel.py",
                "--enable-compiler-source-analysis",
            ]
        )

        self.assertTrue(args.enable_compiler_source_analysis)
        self.assertFalse(hasattr(args, "compiler_source_path"))

    def test_optimize_accepts_cann_ext_api_option(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "optimize",
                "-i",
                "kernel.py",
                "--enable-cann-ext-api",
            ]
        )

        self.assertTrue(args.enable_cann_ext_api)

    def test_optimize_accepts_agent_hooks_option(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--enable-agent-hooks"])

        self.assertTrue(args.enable_agent_hooks)
        options = optimize_run_options_from_args(args)
        self.assertTrue(options.enable_agent_hooks)

    def test_optimize_batch_accepts_compiler_source_analysis_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "optimize-batch",
                "-i",
                "operators",
                "--enable-compiler-source-analysis",
            ]
        )

        self.assertTrue(args.enable_compiler_source_analysis)
        self.assertFalse(hasattr(args, "compiler_source_path"))

    def test_optimize_batch_accepts_cann_ext_api_option(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "optimize-batch",
                "-i",
                "operators",
                "--enable-cann-ext-api",
            ]
        )

        self.assertTrue(args.enable_cann_ext_api)

    def test_optimize_command_defaults_optimize_knowledge_to_v1(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py"])
        self.assertEqual(args.optimize_knowledge, "v1")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.optimize_knowledge, "v1")

    def test_optimize_command_accepts_optimize_knowledge_v2(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["optimize", "-i", "kernel.py", "--optimize-knowledge", "v2"]
        )
        self.assertEqual(args.optimize_knowledge, "v2")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.optimize_knowledge, "v2")

    def test_optimize_command_accepts_optimize_knowledge_v3(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["optimize", "-i", "kernel.py", "--optimize-knowledge", "v3"]
        )
        self.assertEqual(args.optimize_knowledge, "v3")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.optimize_knowledge, "v3")

    def test_optimize_batch_defaults_optimize_knowledge_to_v1(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize-batch", "-i", "kernels"])
        self.assertEqual(args.optimize_knowledge, "v1")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.optimize_knowledge, "v1")

    def test_optimize_batch_accepts_optimize_knowledge_v2(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["optimize-batch", "-i", "kernels", "--optimize-knowledge", "v2"]
        )
        self.assertEqual(args.optimize_knowledge, "v2")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.optimize_knowledge, "v2")

    def test_optimize_batch_accepts_optimize_knowledge_v3(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["optimize-batch", "-i", "kernels", "--optimize-knowledge", "v3"]
        )
        self.assertEqual(args.optimize_knowledge, "v3")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.optimize_knowledge, "v3")

    def test_optimize_command_defaults_target_chip_to_a5(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py"])
        self.assertEqual(args.target_chip, "A5")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.target_chip, "A5")

    def test_optimize_command_accepts_target_chip(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--target-chip", "A3"])
        self.assertEqual(args.target_chip, "A3")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.target_chip, "A3")

    def test_optimize_command_accepts_supervise_modes(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--supervise", "on"])
        self.assertEqual(args.supervise, "on")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.supervise, "on")

    def test_optimize_command_accepts_supervisor_alias(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--supervisor", "on"])
        self.assertEqual(args.supervise, "on")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.supervise, "on")

    def test_optimize_command_accepts_user_prompt(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--prompt", "Focus on memory access."])
        self.assertEqual(args.prompt, "Focus on memory access.")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.prompt, "Focus on memory access.")

    def test_optimize_command_defaults_supervise_off(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py"])
        self.assertEqual(args.supervise, "off")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.supervise, "off")

    def test_optimize_run_options_rejects_invalid_supervise(self) -> None:
        args = argparse.Namespace(
            agent="codex",
            interact=False,
            verbose=False,
            show_output=False,
            remote=None,
            remote_workdir=None,
            min_rounds=None,
            resume="auto",
            reset_optimize=False,
            no_agent_session=False,
            supervise="maybe",
            output=None,
            test_mode=None,
            bench_mode=None,
            prompt=None,
        )
        with self.assertRaises(ValueError):
            optimize_run_options_from_args(args)

    def test_optimize_batch_maps_to_command_kind(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize-batch", "-i", "kernels"])
        self.assertEqual(args.command_kind, CommandKind.OPTIMIZE_BATCH)
        self.assertEqual(args.max_concurrency, 1)
        self.assertEqual(args.agent, "codex")
        self.assertFalse(hasattr(args, "interact"))
        self.assertFalse(args.show_output)

    def test_optimize_batch_accepts_supervise_modes(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize-batch", "-i", "kernels", "--supervise", "on"])
        self.assertEqual(args.supervise, "on")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.supervise, "on")

    def test_optimize_batch_accepts_supervisor_alias(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize-batch", "-i", "kernels", "--supervisor", "on"])
        self.assertEqual(args.supervise, "on")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.supervise, "on")

    def test_optimize_batch_accepts_user_prompt(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["optimize-batch", "-i", "kernels", "--prompt", "Avoid numerics changes."]
        )
        self.assertEqual(args.prompt, "Avoid numerics changes.")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.prompt, "Avoid numerics changes.")

    def test_optimize_batch_defaults_supervise_off(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize-batch", "-i", "kernels"])
        self.assertEqual(args.supervise, "off")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.supervise, "off")
        self.assertEqual(args.target_chip, "A5")
        self.assertEqual(options.target_chip, "A5")

    def test_status_maps_to_command_kind(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["status", "-i", "kernels"])
        self.assertEqual(args.command_kind, CommandKind.STATUS)
        self.assertTrue(args.verbose is False)
        self.assertEqual(args.format, "text")
        self.assertFalse(hasattr(args, "agent"))
        self.assertFalse(hasattr(args, "interact"))
        self.assertFalse(hasattr(args, "remote"))
        self.assertFalse(hasattr(args, "output"))

    def test_status_accepts_markdown_format(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["status", "-i", "kernels", "--format", "markdown"])
        self.assertEqual(args.format, "markdown")

    def test_optimize_status_no_longer_parses(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["optimize-status", "-i", "kernels"])

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
                "--target-chip",
                "A3",
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
        self.assertEqual(args.target_chip, "A3")
        self.assertTrue(args.no_agent_session)
        self.assertEqual(args.max_concurrency, 3)
        self.assertTrue(args.show_output)

    def test_optimize_batch_defaults_resume_to_auto(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize-batch", "-i", "kernels"])
        self.assertEqual(args.resume, "auto")

    def test_optimize_batch_rejects_require_analysis(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["optimize-batch", "-i", "kernels", "--require-analysis"])

    def test_optimize_batch_accepts_reset_optimize(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize-batch", "-i", "kernels", "--reset-optimize"])
        self.assertTrue(args.reset_optimize)
        options = optimize_run_options_from_args(args)
        self.assertTrue(options.reset_optimize)


class PathResolutionTests(unittest.TestCase):
    def test_main_status_rejects_missing_root(self) -> None:
        stderr = StringIO()
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as exc:
                main(["status", "-i", "/tmp/definitely-missing-triton-agent-root"])

        self.assertEqual(exc.exception.code, 2)
        self.assertIn("Input path does not exist", stderr.getvalue())

    def test_main_status_reports_empty_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stderr = StringIO()

            with redirect_stderr(stderr):
                exit_code = main(["status", "-i", str(root)])

            self.assertEqual(exit_code, 1)
            self.assertIn("No operator workspaces found under", stderr.getvalue())

    def test_main_status_reports_no_session_workspaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fresh").mkdir()
            stdout = StringIO()

            with redirect_stdout(stdout):
                exit_code = main(["status", "-i", str(root)])

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertIn("[NO-SESSION] fresh", rendered)
            self.assertIn("Summary: 0 ok, 0 warning, 1 no-session", rendered)

    def test_main_status_accepts_single_workspace_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            (workspace / "kernel_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            round_one = workspace / "opt-round-1"
            round_one.mkdir()
            (round_one / "opt_kernel_perf.txt").write_text(
                "latency-a: 8\nlatency-b: 16\n",
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["status", "-i", str(workspace)])

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertIn(f"[OK] {workspace.name}", rendered)
            self.assertIn("Best round: round-1", rendered)
            self.assertNotIn("[NO-SESSION] opt-round-1", rendered)

    def test_main_status_sorts_no_session_first_then_remaining_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "gamma").mkdir()

            warning_workspace = root / "zeta"
            warning_workspace.mkdir()
            (warning_workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            (warning_workspace / "kernel_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            warning_round = warning_workspace / "opt-round-1"
            warning_round.mkdir()
            (warning_round / "opt_kernel_perf.txt").write_text(
                "latency-a: 8\n",
                encoding="utf-8",
            )

            ok_workspace = root / "alpha"
            ok_workspace.mkdir()
            (ok_workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            (ok_workspace / "kernel_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            ok_round = ok_workspace / "opt-round-1"
            ok_round.mkdir()
            (ok_round / "opt_kernel_perf.txt").write_text(
                "latency-a: 8\nlatency-b: 16\n",
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["status", "-i", str(root)])

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertLess(rendered.index("[NO-SESSION] gamma"), rendered.index("[OK] alpha"))
            self.assertLess(rendered.index("[OK] alpha"), rendered.index("[WARN] zeta"))

    def test_main_status_reports_numeric_best_and_logged_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "matmul"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            (workspace / "kernel_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            (workspace / "opt-note.md").write_text(
                "\n".join(
                    [
                        "## Round 1",
                        "Best status: validated branch",
                        "## Round 2",
                        "Best status: validated branch",
                        "## Round 3",
                        "Best status: current best",
                        "",
                        "## Overall Summary",
                        "Final best round: round-1",
                        "Geomean speedup: 1.16x",
                        "Total speedup: 1.18x",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            round_one = workspace / "opt-round-1"
            round_two = workspace / "opt-round-2"
            round_one.mkdir()
            round_two.mkdir()
            (round_one / "opt_kernel_perf.txt").write_text(
                "latency-a: 8\nlatency-b: 18\n",
                encoding="utf-8",
            )
            (round_two / "opt_kernel_perf.txt").write_text(
                "latency-a: 9\nlatency-b: 10\n",
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["status", "-i", str(root)])

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertIn("[OK] matmul", rendered)
            self.assertIn("Baseline mean: 15.000000", rendered)
            self.assertIn("Best mean: 9.500000", rendered)
            self.assertIn("Avg improvement: +30.0%", rendered)
            self.assertIn("Geomean speedup: 1.49x", rendered)
            self.assertIn("Total speedup: 1.58x", rendered)
            self.assertIn("Best round: round-2", rendered)
            self.assertIn("Logged best: round-1", rendered)
            self.assertIn(
                "Warning: numeric best round != logged best. "
                "computed speedup: 1.49x, 1.58x; "
                "logged speedup: 1.16x, 1.18x",
                rendered,
            )

    def test_main_status_prefers_overall_summary_logged_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "matmul"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
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
            (round_one / "opt_kernel_perf.txt").write_text(
                "latency-a: 8\nlatency-b: 18\n",
                encoding="utf-8",
            )
            (round_two / "opt_kernel_perf.txt").write_text(
                "latency-a: 9\nlatency-b: 10\n",
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["status", "-i", str(root)])

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertIn("[OK] matmul", rendered)
            self.assertIn("Best round: round-2", rendered)
            self.assertIn("Logged best: round-2", rendered)
            self.assertIn(
                "Warning: overall summary best round differs from legacy current best marker",
                rendered,
            )

    def test_main_status_warns_when_perf_ids_do_not_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "layernorm"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            (workspace / "kernel_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            round_one = workspace / "opt-round-1"
            round_one.mkdir()
            (round_one / "opt_kernel_perf.txt").write_text(
                "latency-a: 8\nlatency-c: 18\n",
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["status", "-i", str(root)])

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertIn("[WARN] layernorm", rendered)
            self.assertIn("Best mean: unknown", rendered)
            self.assertIn("Avg improvement: unknown", rendered)
            self.assertIn("Geomean speedup: unknown", rendered)
            self.assertIn("Total speedup: unknown", rendered)
            self.assertIn("Warning: ", rendered)
            self.assertIn("missing required latency ids", rendered)
            self.assertIn("Summary: 0 ok, 1 warning, 0 no-session", rendered)

    def test_main_status_prefers_non_opt_top_level_perf_as_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "matmul"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            (workspace / "kernel_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            (workspace / "opt_kernel_perf.txt").write_text(
                "latency-a: 8\nlatency-b: 18\n",
                encoding="utf-8",
            )
            round_one = workspace / "opt-round-1"
            round_one.mkdir()
            (round_one / "opt_kernel_perf.txt").write_text(
                "latency-a: 9\nlatency-b: 15\n",
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["status", "-i", str(root)])

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertIn("[OK] matmul", rendered)
            self.assertIn("Best round: round-1", rendered)
            self.assertNotIn("Warning: found multiple baseline perf files", rendered)

    def test_main_status_renders_markdown_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fresh").mkdir()

            warning_workspace = root / "zeta"
            warning_workspace.mkdir()
            (warning_workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            (warning_workspace / "kernel_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            (warning_workspace / "opt-round-1").mkdir()

            ok_workspace = root / "beta"
            ok_workspace.mkdir()
            (ok_workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            (ok_workspace / "kernel_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            (ok_workspace / "opt-note.md").write_text(
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
            ok_round = ok_workspace / "opt-round-1"
            best_round = ok_workspace / "opt-round-2"
            ok_round.mkdir()
            best_round.mkdir()
            (ok_round / "opt_kernel_perf.txt").write_text(
                "latency-a: 8\nlatency-b: 18\n",
                encoding="utf-8",
            )
            (best_round / "opt_kernel_perf.txt").write_text(
                "latency-a: 9\nlatency-b: 10\n",
                encoding="utf-8",
            )
            verify_dir = ok_workspace / "opt-verify" / "verify-20260421-120000"
            verify_dir.mkdir(parents=True)
            (verify_dir / "verify-state.json").write_text(
                "\n".join(
                    [
                        "{",
                        '  "verify-result": {',
                        '    "test": {"status": "passed"},',
                        '    "rerun_baseline_bench": {"status": "passed"},',
                        '    "rerun_best_bench": {"status": "passed"},',
                        '    "compare_perf": {"status": "passed"},',
                        '    "speedup": {',
                        '      "geomean_speedup": 1.22,',
                        '      "total_speedup": 1.28',
                        "    }",
                        "  }",
                        "}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["status", "-i", str(root), "--format", "markdown"])

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertIn(
                "| 名称 | Geomean speedup | Total speedup | Verified | "
                "Verified Geomean speedup | Verified Total speedup | Notes |",
                rendered,
            )
            self.assertIn("| beta | 1.49x | 1.58x | Verified | 1.22x | 1.28x | best≠log |", rendered)
            self.assertIn("| zeta | - | - | - |  |  | warn |", rendered)
            self.assertLess(rendered.index("| beta |"), rendered.index("| zeta |"))
            self.assertNotIn("fresh", rendered)
            self.assertNotIn("Summary:", rendered)

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

    def test_main_optimize_batch_accepts_root_as_single_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "operator.py").write_text("print('x')\n", encoding="utf-8")

            seen_inputs: list[Path] = []

            def _fake_run(request):
                seen_inputs.append(request.input_path)
                return AgentResult(return_code=0, stdout="", stderr="")

            with patch("triton_agent.optimize.batch.run_optimize_request", side_effect=_fake_run):
                exit_code = main(["optimize-batch", "-i", str(root), "--resume", "fresh"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(seen_inputs, [(root / "operator.py").resolve()])

    def test_main_optimize_batch_accepts_root_with_non_workspace_child_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "operator.py").write_text("print('x')\n", encoding="utf-8")
            (root / "artifacts").mkdir()
            (root / "logs").mkdir()

            seen_inputs: list[Path] = []

            def _fake_run(request):
                seen_inputs.append(request.input_path)
                return AgentResult(return_code=0, stdout="", stderr="")

            with patch("triton_agent.optimize.batch.run_optimize_request", side_effect=_fake_run):
                exit_code = main(["optimize-batch", "-i", str(root), "--resume", "fresh"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(seen_inputs, [(root / "operator.py").resolve()])

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

    def test_main_optimize_batch_ignores_triton_agent_tooling_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good"
            tooling = root / ".triton-agent"
            good.mkdir()
            tooling.mkdir()
            (good / "kernel.py").write_text("print('ok')\n", encoding="utf-8")
            (tooling / "round-brief.md").write_text("Pending\n", encoding="utf-8")

            stdout = StringIO()
            seen_inputs: list[Path] = []

            def _fake_run(request):
                seen_inputs.append(request.input_path)
                return AgentResult(return_code=0, stdout="", stderr="")

            with patch("triton_agent.optimize.batch.run_optimize_request", side_effect=_fake_run):
                with redirect_stdout(stdout):
                    exit_code = main(["optimize-batch", "-i", str(root), "--resume", "fresh"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(seen_inputs, [(good / "kernel.py").resolve()])
            self.assertNotIn(".triton-agent", stdout.getvalue())
            self.assertIn("Summary: 1 succeeded, 0 failed", stdout.getvalue())

    def test_main_optimize_batch_reports_skipped_workspaces_from_status_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            alpha = root / "alpha"
            beta = root / "beta"
            alpha.mkdir()
            beta.mkdir()
            (alpha / "kernel.py").write_text("print('a')\n", encoding="utf-8")
            (beta / "kernel.py").write_text("print('b')\n", encoding="utf-8")
            (root / "optimize-batch-status.json").write_text(
                '{"version": 1, "workspaces": {"alpha": {"status": "completed", "operator_file": "kernel.py"}}}\n',
                encoding="utf-8",
            )

            stdout = StringIO()
            seen_inputs: list[Path] = []

            def _fake_run(request):
                seen_inputs.append(request.input_path)
                return AgentResult(return_code=0, stdout="", stderr="")

            with patch("triton_agent.optimize.batch.run_optimize_request", side_effect=_fake_run):
                with redirect_stdout(stdout):
                    exit_code = main(["optimize-batch", "-i", str(root)])

            self.assertEqual(exit_code, 0)
            self.assertEqual([path.parent.name for path in seen_inputs], ["beta"])
            self.assertIn("[SKIP] alpha: already completed", stdout.getvalue())
            self.assertIn("Summary: 1 succeeded, 0 failed, 1 skipped", stdout.getvalue())

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

    def test_main_optimize_batch_resume_auto_accepts_explicit_bench_mode_for_mixed_workspaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resumable = root / "resume_ws"
            fresh = root / "fresh_ws"
            resumable.mkdir()
            fresh.mkdir()
            resumable_operator = resumable / "kernel.py"
            fresh_operator = fresh / "kernel.py"
            resumable_operator.write_text("print('resume')\n", encoding="utf-8")
            fresh_operator.write_text("print('fresh')\n", encoding="utf-8")

            (resumable / "opt-note.md").write_text("history\n", encoding="utf-8")
            (resumable / "opt-round-1").mkdir()
            baseline_dir = resumable / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text(
                "\n".join(
                    [
                        "{",
                        '  "baseline_kind": "original",',
                        '  "source_operator": "kernel.py",',
                        '  "baseline_operator": "baseline/kernel.py",',
                        '  "test_file": "differential_test_kernel.py",',
                        '  "test_mode": "differential",',
                        '  "bench_file": "bench_kernel.py",',
                        '  "bench_mode": "standalone",',
                        '  "perf_artifact": "baseline/perf.txt",',
                        '  "correctness_status": "passed",',
                        '  "benchmark_status": "passed",',
                        '  "baseline_established": true',
                        "}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text("latency-a: 1.0\n", encoding="utf-8")
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            (resumable / "differential_test_kernel.py").write_text(
                "# test-mode: differential\nprint('test')\n", encoding="utf-8"
            )
            (resumable / "bench_kernel.py").write_text(
                "# bench-mode: standalone\n# kernel: k\nprint('bench')\n", encoding="utf-8"
            )

            captured_modes: dict[str, Optional[str]] = {}

            def _fake_run(request):
                captured_modes[request.workdir.name] = request.bench_mode
                return AgentResult(return_code=0, stdout="", stderr="")

            with patch("triton_agent.optimize.batch.run_optimize_request", side_effect=_fake_run):
                exit_code = main(
                    [
                        "optimize-batch",
                        "-i",
                        str(root),
                        "--resume",
                        "auto",
                        "--bench-mode",
                        "msprof",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured_modes["resume_ws"], "standalone")
            self.assertEqual(captured_modes["fresh_ws"], "msprof")

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

            with patch("triton_agent.generation.batch.run_generation_request", side_effect=_fake_run):
                exit_code = main(["gen-eval-batch", "-i", str(root)])

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                seen_inputs,
                [
                    (first / "kernel.py").resolve(),
                    (second / "matmul_impl.py").resolve(),
                ],
            )

    def test_main_gen_eval_batch_accepts_root_as_single_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "operator.py").write_text("print('x')\n", encoding="utf-8")

            seen_inputs: list[Path] = []

            def _fake_run(request, stdout=None, stderr=None):
                seen_inputs.append(request.input_path)
                return AgentResult(return_code=0, stdout="", stderr="")

            with patch("triton_agent.generation.batch.run_generation_request", side_effect=_fake_run):
                exit_code = main(["gen-eval-batch", "-i", str(root)])

            self.assertEqual(exit_code, 0)
            self.assertEqual(seen_inputs, [(root / "operator.py").resolve()])

    def test_main_gen_eval_batch_accepts_root_with_non_workspace_child_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "operator.py").write_text("print('x')\n", encoding="utf-8")
            (root / "artifacts").mkdir()
            (root / "logs").mkdir()

            seen_inputs: list[Path] = []

            def _fake_run(request, stdout=None, stderr=None):
                seen_inputs.append(request.input_path)
                return AgentResult(return_code=0, stdout="", stderr="")

            with patch("triton_agent.generation.batch.run_generation_request", side_effect=_fake_run):
                exit_code = main(["gen-eval-batch", "-i", str(root)])

            self.assertEqual(exit_code, 0)
            self.assertEqual(seen_inputs, [(root / "operator.py").resolve()])

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

            with patch("triton_agent.generation.batch.run_generation_request", side_effect=_fake_run):
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

            with patch("triton_agent.generation.batch.run_generation_request", side_effect=_fake_run):
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

            with patch("triton_agent.generation.batch.run_generation_request", side_effect=_fake_run):
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

            with patch("triton_agent.execution.load_operator_eval_script_module", return_value=runtime) as mocked_loader:
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
                supervise="off",
                target_chip=None,
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

            with patch("triton_agent.generation.orchestration.build_prompt", side_effect=_fake_build_prompt):
                with patch("triton_agent.generation.orchestration.create_runner", side_effect=_fake_create_runner):
                    with patch("triton_agent.generation.orchestration.SkillLinkManager.prepare_skills", return_value=[]):
                        with patch("triton_agent.generation.orchestration.SkillLinkManager.cleanup", return_value=[]):
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

    def test_main_optimize_reset_optimize_requires_fresh_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")

            stderr = StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exc:
                    main(["optimize", "-i", str(operator), "--reset-optimize"])

            self.assertEqual(exc.exception.code, 2)
            self.assertIn("--reset-optimize requires --resume fresh", stderr.getvalue())

    def test_main_optimize_rejects_cann_ext_api_without_a5(self) -> None:
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
                            "--target-chip",
                            "A3",
                            "--enable-cann-ext-api",
                        ]
                    )

            self.assertEqual(exc.exception.code, 2)
            self.assertIn("--enable-cann-ext-api requires --target-chip A5", stderr.getvalue())

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
                supervise="off",
                target_chip=None,
            ):
                captured["test_mode"] = test_mode
                captured["bench_mode"] = bench_mode
                captured["resume_existing_session"] = resume_existing_session
                return "Prompt body"

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch("triton_agent.optimize.orchestration.build_prompt", side_effect=_fake_build_prompt):
                with patch(
                    "triton_agent.optimize.execution.OptimizeRunLoop.run",
                    return_value=fake_result,
                ):
                    with patch("triton_agent.optimize.orchestration.create_runner", return_value=object()):
                        with patch(
                            "triton_agent.optimize.orchestration.SkillLinkManager.prepare_skills",
                            return_value=[],
                        ):
                            with patch(
                                "triton_agent.optimize.orchestration.SkillLinkManager.cleanup",
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

    def test_main_optimize_resume_auto_treats_prepared_harnesses_as_no_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            (root / "test_kernel.py").write_text(
                "# test-mode: standalone\nprint('test')\n",
                encoding="utf-8",
            )
            (root / "differential_test_kernel.py").write_text(
                "# test-mode: differential\nprint('test')\n",
                encoding="utf-8",
            )
            (root / "bench_kernel.py").write_text(
                "# bench-mode: standalone\n# kernel: k\nprint('bench')\n",
                encoding="utf-8",
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
                supervise="off",
                target_chip=None,
            ):
                captured["test_mode"] = test_mode
                captured["bench_mode"] = bench_mode
                captured["resume_existing_session"] = resume_existing_session
                return "Prompt body"

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch("triton_agent.optimize.orchestration.build_prompt", side_effect=_fake_build_prompt):
                with patch(
                    "triton_agent.optimize.execution.OptimizeRunLoop.run",
                    return_value=fake_result,
                ):
                    with patch("triton_agent.optimize.orchestration.create_runner", return_value=object()):
                        with patch(
                            "triton_agent.optimize.orchestration.SkillLinkManager.prepare_skills",
                            return_value=[],
                        ):
                            with patch(
                                "triton_agent.optimize.orchestration.SkillLinkManager.cleanup",
                                return_value=[],
                            ):
                                exit_code = main(["optimize", "-i", str(operator), "--resume", "auto"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["test_mode"], "differential")
            self.assertEqual(captured["bench_mode"], "standalone")
            self.assertFalse(captured["resume_existing_session"])

    def test_main_optimize_resume_auto_allows_baseline_without_opt_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text(
                "\n".join(
                    [
                        "{",
                        '  "baseline_kind": "prepared",',
                        '  "source_operator": "kernel.py",',
                        '  "baseline_operator": "baseline/kernel.py",',
                        '  "test_file": "differential_test_kernel.py",',
                        '  "test_mode": "differential",',
                        '  "bench_file": "bench_kernel.py",',
                        '  "bench_mode": "standalone",',
                        '  "perf_artifact": "baseline/perf.txt",',
                        '  "correctness_status": "passed",',
                        '  "benchmark_status": "passed",',
                        '  "baseline_established": true',
                        "}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text("latency-a: 1.0\n", encoding="utf-8")
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")

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
                supervise="off",
                target_chip=None,
            ):
                captured["test_mode"] = test_mode
                captured["bench_mode"] = bench_mode
                captured["resume_existing_session"] = resume_existing_session
                return "Prompt body"

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch("triton_agent.optimize.orchestration.build_prompt", side_effect=_fake_build_prompt):
                with patch(
                    "triton_agent.optimize.execution.OptimizeRunLoop.run",
                    return_value=fake_result,
                ):
                    with patch("triton_agent.optimize.orchestration.create_runner", return_value=object()):
                        with patch(
                            "triton_agent.optimize.orchestration.SkillLinkManager.prepare_skills",
                            return_value=[],
                        ):
                            with patch(
                                "triton_agent.optimize.orchestration.SkillLinkManager.cleanup",
                                return_value=[],
                            ):
                                exit_code = main(["optimize", "-i", str(operator), "--resume", "auto"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["test_mode"], "differential")
            self.assertEqual(captured["bench_mode"], "standalone")
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
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text(
                "\n".join(
                    [
                        "{",
                        '  "baseline_kind": "original",',
                        '  "source_operator": "kernel.py",',
                        '  "baseline_operator": "baseline/kernel.py",',
                        '  "test_file": "differential_test_kernel.py",',
                        '  "test_mode": "differential",',
                        '  "bench_file": "bench_kernel.py",',
                        '  "bench_mode": "standalone",',
                        '  "perf_artifact": "baseline/perf.txt",',
                        '  "correctness_status": "passed",',
                        '  "benchmark_status": "passed",',
                        '  "baseline_established": true',
                        "}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text("latency-a: 1.0\n", encoding="utf-8")
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")
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

    def test_main_optimize_resume_continue_requires_established_baseline(self) -> None:
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
                "# bench-mode: standalone\n# kernel: k\nprint('bench')\n", encoding="utf-8"
            )

            stderr = StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exc:
                    main(["optimize", "-i", str(operator), "--resume", "continue"])

            self.assertEqual(exc.exception.code, 2)
            self.assertIn("resume continue requires established baseline/", stderr.getvalue())
            self.assertIn("missing established baseline/", stderr.getvalue())

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

    def test_main_optimize_resume_fresh_with_reset_cleans_session_artifacts_and_keeps_harnesses(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            (root / "opt-note.md").write_text("history\n", encoding="utf-8")
            (root / "learned_lessons.md").write_text("notes\n", encoding="utf-8")
            (root / "opt-round-1").mkdir()
            (root / ".triton-agent").mkdir()
            (root / "triton-agent-logs").mkdir()
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            (root / "opt_kernel.py").write_text("print('opt')\n", encoding="utf-8")
            test_harness = root / "differential_test_kernel.py"
            test_harness.write_text(
                "# test-mode: differential\nprint('test')\n",
                encoding="utf-8",
            )
            bench_harness = root / "bench_kernel.py"
            bench_harness.write_text(
                "# bench-mode: standalone\n# kernel: k\nprint('bench')\n",
                encoding="utf-8",
            )

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch("triton_agent.optimize.execution.OptimizeRunLoop.run", return_value=fake_result):
                with patch("triton_agent.optimize.orchestration.create_runner", return_value=object()):
                    with patch(
                        "triton_agent.optimize.orchestration.SkillLinkManager.prepare_skills",
                        return_value=[],
                    ):
                        with patch(
                            "triton_agent.optimize.orchestration.SkillLinkManager.cleanup",
                            return_value=[],
                        ):
                            exit_code = main(
                                [
                                    "optimize",
                                    "-i",
                                    str(operator),
                                    "--resume",
                                    "fresh",
                                    "--reset-optimize",
                                ]
                            )

            self.assertEqual(exit_code, 0)
            self.assertFalse((root / "opt-note.md").exists())
            self.assertFalse((root / "learned_lessons.md").exists())
            self.assertFalse((root / "opt-round-1").exists())
            self.assertFalse((root / ".triton-agent").exists())
            self.assertFalse((root / "triton-agent-logs").exists())
            self.assertFalse((root / "baseline").exists())
            self.assertFalse((root / "opt_kernel.py").exists())
            self.assertTrue(test_harness.exists())
            self.assertTrue(bench_harness.exists())

    def test_main_optimize_resume_auto_uses_continue_for_resumable_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            (root / "opt-note.md").write_text("history\n", encoding="utf-8")
            (root / "opt-round-1").mkdir()
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text(
                "\n".join(
                    [
                        "{",
                        '  "baseline_kind": "original",',
                        '  "source_operator": "kernel.py",',
                        '  "baseline_operator": "baseline/kernel.py",',
                        '  "test_file": "differential_test_kernel.py",',
                        '  "test_mode": "differential",',
                        '  "bench_file": "bench_kernel.py",',
                        '  "bench_mode": "msprof",',
                        '  "perf_artifact": "baseline/perf.txt",',
                        '  "correctness_status": "passed",',
                        '  "benchmark_status": "passed",',
                        '  "baseline_established": true',
                        "}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text("latency-a: 1.0\n", encoding="utf-8")
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")
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
                supervise="off",
                target_chip=None,
            ):
                captured["test_mode"] = test_mode
                captured["bench_mode"] = bench_mode
                captured["resume_existing_session"] = resume_existing_session
                return "Prompt body"

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch("triton_agent.optimize.orchestration.build_prompt", side_effect=_fake_build_prompt):
                with patch(
                    "triton_agent.optimize.execution.OptimizeRunLoop.run",
                    return_value=fake_result,
                ) as mocked:
                    with patch("triton_agent.optimize.orchestration.create_runner", return_value=object()):
                        with patch(
                            "triton_agent.optimize.orchestration.SkillLinkManager.prepare_skills",
                            return_value=[],
                        ):
                            with patch(
                                "triton_agent.optimize.orchestration.SkillLinkManager.cleanup",
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

    def test_main_optimize_resume_auto_accepts_explicit_bench_mode_for_resumable_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            (root / "opt-note.md").write_text("history\n", encoding="utf-8")
            (root / "opt-round-1").mkdir()
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text(
                "\n".join(
                    [
                        "{",
                        '  "baseline_kind": "original",',
                        '  "source_operator": "kernel.py",',
                        '  "baseline_operator": "baseline/kernel.py",',
                        '  "test_file": "differential_test_kernel.py",',
                        '  "test_mode": "differential",',
                        '  "bench_file": "bench_kernel.py",',
                        '  "bench_mode": "msprof",',
                        '  "perf_artifact": "baseline/perf.txt",',
                        '  "correctness_status": "passed",',
                        '  "benchmark_status": "passed",',
                        '  "baseline_established": true',
                        "}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text("latency-a: 1.0\n", encoding="utf-8")
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")
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
                supervise="off",
                target_chip=None,
            ):
                captured["test_mode"] = test_mode
                captured["bench_mode"] = bench_mode
                captured["resume_existing_session"] = resume_existing_session
                return "Prompt body"

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch("triton_agent.optimize.orchestration.build_prompt", side_effect=_fake_build_prompt):
                with patch(
                    "triton_agent.optimize.execution.OptimizeRunLoop.run",
                    return_value=fake_result,
                ) as mocked:
                    with patch("triton_agent.optimize.orchestration.create_runner", return_value=object()):
                        with patch(
                            "triton_agent.optimize.orchestration.SkillLinkManager.prepare_skills",
                            return_value=[],
                        ):
                            with patch(
                                "triton_agent.optimize.orchestration.SkillLinkManager.cleanup",
                                return_value=[],
                            ):
                                exit_code = main(
                                    [
                                        "optimize",
                                        "-i",
                                        str(operator),
                                        "--resume",
                                        "auto",
                                        "--bench-mode",
                                        "standalone",
                                    ]
                                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["test_mode"], "differential")
            self.assertEqual(captured["bench_mode"], "msprof")
            self.assertTrue(captured["resume_existing_session"])
            request = mocked.call_args.args[1]
            self.assertEqual(request.test_mode, "differential")
            self.assertEqual(request.bench_mode, "msprof")

    def test_main_optimize_resume_auto_uses_continue_for_baseline_only_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            (root / "opt-note.md").write_text("baseline prepared\n", encoding="utf-8")
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text(
                "\n".join(
                    [
                        "{",
                        '  "baseline_kind": "prepared",',
                        '  "source_operator": "kernel.py",',
                        '  "baseline_operator": "baseline/kernel.py",',
                        '  "test_file": "differential_test_kernel.py",',
                        '  "test_mode": "differential",',
                        '  "bench_file": "bench_kernel.py",',
                        '  "bench_mode": "standalone",',
                        '  "perf_artifact": "baseline/perf.txt",',
                        '  "correctness_status": "passed",',
                        '  "benchmark_status": "passed",',
                        '  "baseline_established": true',
                        "}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text("latency-a: 1.0\n", encoding="utf-8")
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            (root / "differential_test_kernel.py").write_text(
                "# test-mode: differential\nprint('test')\n", encoding="utf-8"
            )
            (root / "bench_kernel.py").write_text(
                "# bench-mode: standalone\n# kernel: k\nprint('bench')\n", encoding="utf-8"
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
                supervise="off",
                target_chip=None,
            ):
                captured["resume_existing_session"] = resume_existing_session
                captured["test_mode"] = test_mode
                captured["bench_mode"] = bench_mode
                return "Prompt body"

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch("triton_agent.optimize.orchestration.build_prompt", side_effect=_fake_build_prompt):
                with patch(
                    "triton_agent.optimize.execution.OptimizeRunLoop.run",
                    return_value=fake_result,
                ):
                    with patch("triton_agent.optimize.orchestration.create_runner", return_value=object()):
                        with patch(
                            "triton_agent.optimize.orchestration.SkillLinkManager.prepare_skills",
                            return_value=[],
                        ):
                            with patch(
                                "triton_agent.optimize.orchestration.SkillLinkManager.cleanup",
                                return_value=[],
                            ):
                                exit_code = main(
                                    ["optimize", "-i", str(operator), "--resume", "auto"]
                                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(captured["resume_existing_session"])
            self.assertEqual(captured["test_mode"], "differential")
            self.assertEqual(captured["bench_mode"], "standalone")

    def test_main_optimize_accepts_workspace_directory_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch("triton_agent.optimize.orchestration.build_prompt", return_value="Prompt body"):
                with patch(
                    "triton_agent.optimize.execution.OptimizeRunLoop.run",
                    return_value=fake_result,
                ) as mocked:
                    with patch("triton_agent.optimize.orchestration.create_runner", return_value=object()):
                        with patch(
                            "triton_agent.optimize.orchestration.SkillLinkManager.prepare_skills",
                            return_value=[],
                        ):
                            with patch(
                                "triton_agent.optimize.orchestration.SkillLinkManager.cleanup",
                                return_value=[],
                            ):
                                exit_code = main(["optimize", "-i", str(root)])

            self.assertEqual(exit_code, 0)
            request = mocked.call_args.args[1]
            self.assertEqual(request.input_path, operator.resolve())
            self.assertEqual(request.operator_path, operator.resolve())
            self.assertEqual(request.workdir, root.resolve())

    def test_main_optimize_accepts_dot_workspace_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            original_cwd = Path.cwd()
            try:
                os.chdir(root)
                with patch("triton_agent.optimize.orchestration.build_prompt", return_value="Prompt body"):
                    with patch(
                        "triton_agent.optimize.execution.OptimizeRunLoop.run",
                        return_value=fake_result,
                    ) as mocked:
                        with patch("triton_agent.optimize.orchestration.create_runner", return_value=object()):
                            with patch(
                                "triton_agent.optimize.orchestration.SkillLinkManager.prepare_skills",
                                return_value=[],
                            ):
                                with patch(
                                    "triton_agent.optimize.orchestration.SkillLinkManager.cleanup",
                                    return_value=[],
                                ):
                                    exit_code = main(["optimize", "-i", "."])
            finally:
                os.chdir(original_cwd)

            self.assertEqual(exit_code, 0)
            request = mocked.call_args.args[1]
            self.assertEqual(request.input_path, operator.resolve())
            self.assertEqual(request.operator_path, operator.resolve())
            self.assertEqual(request.workdir, root.resolve())

    def test_main_optimize_rejects_workspace_directory_with_multiple_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("print('a')\n", encoding="utf-8")
            (root / "b.py").write_text("print('b')\n", encoding="utf-8")

            stderr = StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exc:
                    main(["optimize", "-i", str(root)])

            self.assertEqual(exc.exception.code, 2)
            self.assertIn("found multiple candidate operator files", stderr.getvalue())

    def test_main_optimize_rejects_require_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            stderr = StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exc:
                    main(["optimize", "-i", str(operator), "--require-analysis"])

            self.assertEqual(exc.exception.code, 2)
            self.assertIn("--require-analysis", stderr.getvalue())

    def test_main_optimize_passes_supervise_mode_to_prompt_and_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")

            captured = {}
            fake_result = AgentResult(return_code=0, stdout="", stderr="")

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
                supervise="off",
                target_chip=None,
            ):
                captured["supervise"] = supervise
                captured["target_chip"] = target_chip
                return "Prompt body"

            with patch("triton_agent.optimize.orchestration.build_prompt", side_effect=_fake_build_prompt):
                with patch(
                    "triton_agent.optimize.execution.OptimizeRunLoop.run",
                    return_value=fake_result,
                ) as mocked:
                    with patch("triton_agent.optimize.orchestration.create_runner", return_value=object()):
                        with patch(
                            "triton_agent.optimize.orchestration.SkillLinkManager.prepare_skills",
                            return_value=[],
                        ):
                            with patch(
                                "triton_agent.optimize.orchestration.SkillLinkManager.cleanup",
                                return_value=[],
                            ):
                                exit_code = main(
                                    ["optimize", "-i", str(operator), "--supervise", "on"]
                                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["supervise"], "on")
            self.assertEqual(captured["target_chip"], "A5")
            request = mocked.call_args.args[1]
            self.assertEqual(request.supervise, "on")
            self.assertEqual(request.optimize_role, "worker")
            self.assertEqual(request.target_chip, "A5")

    def test_main_optimize_passes_target_chip_to_prompt_and_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")

            captured = {}
            fake_result = AgentResult(return_code=0, stdout="", stderr="")

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
                supervise="off",
                target_chip=None,
            ):
                captured["target_chip"] = target_chip
                return "Prompt body"

            with patch("triton_agent.optimize.orchestration.build_prompt", side_effect=_fake_build_prompt):
                with patch(
                    "triton_agent.optimize.execution.OptimizeRunLoop.run",
                    return_value=fake_result,
                ) as mocked:
                    with patch("triton_agent.optimize.orchestration.create_runner", return_value=object()):
                        with patch(
                            "triton_agent.optimize.orchestration.SkillLinkManager.prepare_skills",
                            return_value=[],
                        ):
                            with patch(
                                "triton_agent.optimize.orchestration.SkillLinkManager.cleanup",
                                return_value=[],
                            ):
                                exit_code = main(
                                    ["optimize", "-i", str(operator), "--target-chip", "A3"]
                                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["target_chip"], "A3")
            request = mocked.call_args.args[1]
            self.assertEqual(request.target_chip, "A3")

    def test_main_optimize_passes_no_agent_session_to_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch("triton_agent.optimize.orchestration.build_prompt", return_value="Prompt body"):
                with patch(
                    "triton_agent.optimize.execution.OptimizeRunLoop.run",
                    return_value=fake_result,
                ) as mocked:
                    with patch("triton_agent.optimize.orchestration.create_runner", return_value=object()):
                        with patch(
                            "triton_agent.optimize.orchestration.SkillLinkManager.prepare_skills",
                            return_value=[],
                        ):
                            with patch(
                                "triton_agent.optimize.orchestration.SkillLinkManager.cleanup",
                                return_value=[],
                            ):
                                exit_code = main(
                                    ["optimize", "-i", str(operator), "--no-agent-session"]
                                )

            self.assertEqual(exit_code, 0)
            request = mocked.call_args.args[1]
            self.assertTrue(request.no_agent_session)
            self.assertIsNone(request.optimize_role)

    def test_main_optimize_appends_user_prompt_to_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch("triton_agent.optimize.orchestration.build_prompt", return_value="Prompt body"):
                with patch(
                    "triton_agent.optimize.execution.OptimizeRunLoop.run",
                    return_value=fake_result,
                ) as mocked:
                    with patch("triton_agent.optimize.orchestration.create_runner", return_value=object()):
                        with patch(
                            "triton_agent.optimize.orchestration.SkillLinkManager.prepare_skills",
                            return_value=[],
                        ):
                            with patch(
                                "triton_agent.optimize.orchestration.SkillLinkManager.cleanup",
                                return_value=[],
                            ):
                                exit_code = main(
                                    [
                                        "optimize",
                                        "-i",
                                        str(operator),
                                        "--prompt",
                                        "Focus on memory coalescing.",
                                    ]
                                )

            self.assertEqual(exit_code, 0)
            request = mocked.call_args.args[1]
            self.assertIn("Prompt body", request.prompt)
            self.assertIn("Additional user instructions:", request.prompt)
            self.assertIn("Focus on memory coalescing.", request.prompt)

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
            self.assertNotIn("Hint: use `compare-result`", stdout.getvalue())
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
            self.assertEqual(
                stdout.getvalue(),
                (
                    "Return code: 0\n"
                    f"Archived result: {archive}\n"
                    "Hint: use `compare-result` to inspect this archived result instead of reading it directly.\n"
                ),
            )

    def test_main_run_test_auto_compares_differential_result_when_oracle_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "differential_test_kernel.py"
            archive = root / "kernel_result.pt"
            oracle = root / "oracle_result.pt"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')", encoding="utf-8")
            oracle.write_text("oracle", encoding="utf-8")

            stdout = StringIO()
            fake_result = AgentResult(return_code=0, stdout="", stderr="")

            with patch("triton_agent.commands.execution.run_local_test", return_value=(fake_result, archive)):
                with patch("triton_agent.commands.execution.compare_result_files", return_value=1) as compare_mock:
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
                                "--oracle-result",
                                str(oracle),
                            ]
                        )

            self.assertEqual(exit_code, 1)
            compare_mock.assert_called_once_with(
                oracle.resolve(),
                archive,
                "balanced",
            )
            self.assertIn(f"Archived result: {archive}\n", stdout.getvalue())
            self.assertNotIn("Hint: use `compare-result`", stdout.getvalue())

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
            self.assertEqual(
                stdout.getvalue(),
                (
                    f"Perf file: {perf_file}\n"
                    "Hint: use `compare-perf` to inspect this perf artifact instead of reading it directly.\n"
                ),
            )
            self.assertNotIn("latency-a", stdout.getvalue())
            self.assertEqual(stderr.getvalue(), "")

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

            with patch("triton_agent.execution.load_operator_eval_script_module", return_value=runtime) as mocked_loader:
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
    def test_build_optimize_worker_prompt_mentions_single_round_boundary(self) -> None:
        prompt = build_optimize_worker_prompt(
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
        )
        self.assertIn("This invocation is the optimize worker role.", prompt)
        self.assertIn("This invocation owns exactly one round.", prompt)
        self.assertIn("Read `.triton-agent/round-brief.md`", prompt)
        self.assertNotIn("optimize-worker.md", prompt)
        self.assertIn("Use the staged `triton-npu-prepare-optimize-baseline` skill", prompt)
        self.assertIn("Use the staged `triton-npu-optimize-check` skill", prompt)
        self.assertIn(
            "baseline preparation is needed, use the staged `triton-npu-prepare-optimize-baseline` skill",
            prompt.lower(),
        )
        self.assertNotIn(
            "use the staged `triton-npu-optimize-check` skill to run `check-baseline`",
            prompt.lower(),
        )
        self.assertIn(
            "use the staged `triton-npu-optimize-check` skill to run `check-round`",
            prompt.lower(),
        )
        self.assertIn(
            "must pass `check-round` through `triton-npu-optimize-check` before the invocation ends",
            prompt,
        )
        self.assertIn("Establish or reuse `baseline/` before creating `opt-round-1`.", prompt)
        self.assertIn("Use `baseline/perf.txt` for canonical performance comparisons.", prompt)
        self.assertIn("Use `compare-perf` as the only authority for claimed speedups or benchmark deltas.", prompt)
        self.assertIn("Use the staged `triton-npu-analyze-round-performance` skill", prompt)
        self.assertIn("write `opt-round-n/perf-analysis.md`", prompt.lower())
        self.assertIn(
            "When pattern triage is used, record candidate patterns, the selected pattern if one is chosen, and why that pattern looks plausible in `opt-round-N/attempts.md`.",
            prompt,
        )
        self.assertIn(
            "When a named pattern guides the round, record the final selected pattern direction in `opt-round-N/summary.md`.",
            prompt,
        )
        self.assertIn(
            "`learned_lessons.md` is only for reusable, evidence-backed optimization or profiling rules",
            prompt,
        )
        self.assertIn("Do not put round narrative, command failures, or operator-specific details", prompt)
        self.assertIn("Write `baseline/state.json` with these required fields:", prompt)
        self.assertIn("`baseline_kind`", prompt)
        self.assertIn("`source_operator`", prompt)
        self.assertIn("`baseline_operator`", prompt)
        self.assertIn("`test_file`", prompt)
        self.assertIn("`test_mode`", prompt)
        self.assertIn("`bench_file`", prompt)
        self.assertIn("`bench_mode`", prompt)
        self.assertIn("`perf_artifact`", prompt)
        self.assertIn("`correctness_status`", prompt)
        self.assertIn("`benchmark_status`", prompt)
        self.assertIn("`baseline_established`", prompt)
        self.assertIn("Set `baseline_established` to `true` only after", prompt)
        self.assertIn("PyTorch-facing public API may remain as a wrapper", prompt)
        self.assertIn("must continue optimizing the Triton Ascend NPU kernel path itself", prompt)
        self.assertIn("Do not replace the core computation with a pure PyTorch implementation", prompt)
        self.assertIn("does not count as a successful optimize round", prompt)
        self.assertIn("Target chip for this optimize session: A5.", prompt)
        self.assertIn("prefer changes that fit A5", prompt)

    def test_build_optimize_worker_prompt_mentions_compiler_source_when_enabled(self) -> None:
        prompt = build_optimize_worker_prompt(
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
            compiler_source_path=Path("/tmp/AscendNPU-IR"),
            compiler_source_commit="abc123",
        )

        self.assertIn("Compiler source analysis is enabled", prompt)
        self.assertIn("Compiler source path: /tmp/AscendNPU-IR", prompt)
        self.assertIn("Compiler source commit: abc123.", prompt)
        self.assertIn("Treat the compiler source checkout as read-only.", prompt)
        self.assertIn("Do not run git clone, git fetch, git pull", prompt)
        self.assertIn("then IR evidence, then compiler source", prompt)
        self.assertNotIn("https://gitcode.com/Ascend/AscendNPU-IR.git", prompt)

    def test_build_optimize_worker_prompt_mentions_cann_ext_api_when_enabled(self) -> None:
        prompt = build_optimize_worker_prompt(
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
            enable_cann_ext_api=True,
        )

        self.assertIn("CANN Triton extension API pattern access is enabled for this optimize run.", prompt)
        self.assertIn("Use the staged `triton-npu-cann-ext-api-patterns` skill", prompt)

    def test_build_optimize_worker_prompt_omits_cann_ext_api_when_disabled(self) -> None:
        prompt = build_optimize_worker_prompt(
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
        )

        self.assertNotIn("CANN Triton extension API pattern access is enabled", prompt)
        self.assertNotIn("triton-npu-cann-ext-api-patterns", prompt)

    def test_build_optimize_unsupervised_prompt_mentions_baseline_state_contract(self) -> None:
        prompt = build_optimize_unsupervised_prompt(
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
        )
        self.assertIn("This invocation is an unsupervised optimize run.", prompt)
        self.assertIn("Use the staged `triton-npu-prepare-optimize-baseline` skill", prompt)
        self.assertIn("Use the staged `triton-npu-optimize-check` skill", prompt)
        self.assertIn(
            "baseline preparation is needed, use the staged `triton-npu-prepare-optimize-baseline` skill",
            prompt.lower(),
        )
        self.assertNotIn(
            "use the staged `triton-npu-optimize-check` skill to run `check-baseline`",
            prompt.lower(),
        )
        self.assertIn(
            "use the staged `triton-npu-optimize-check` skill to run `check-round`",
            prompt.lower(),
        )
        self.assertIn("continue optimizing until the session should stop", prompt)
        self.assertIn(
            "Do not begin the next round until the current round passes `check-round` through `triton-npu-optimize-check`.",
            prompt,
        )
        self.assertIn("Use the staged `triton-npu-analyze-round-performance` skill", prompt)
        self.assertIn("write `opt-round-n/perf-analysis.md`", prompt.lower())
        self.assertIn(
            "When pattern triage is used, record candidate patterns, the selected pattern if one is chosen, and why that pattern looks plausible in `opt-round-N/attempts.md`.",
            prompt,
        )
        self.assertIn(
            "When a named pattern guides the round, record the final selected pattern direction in `opt-round-N/summary.md`.",
            prompt,
        )
        self.assertIn(
            "`learned_lessons.md` is only for reusable, evidence-backed optimization or profiling rules",
            prompt,
        )
        self.assertIn("Do not put round narrative, command failures, or operator-specific details", prompt)
        self.assertIn("Write `baseline/state.json` with these required fields:", prompt)
        self.assertIn("`baseline_established`", prompt)
        self.assertIn("Set `baseline_established` to `true` only after", prompt)
        self.assertIn("PyTorch-facing public API may remain as a wrapper", prompt)
        self.assertIn("must continue optimizing the Triton Ascend NPU kernel path itself", prompt)
        self.assertIn("Do not replace the core computation with a pure PyTorch implementation", prompt)
        self.assertIn("does not count as a successful optimize round", prompt)
        self.assertIn("Target chip for this optimize session: A5.", prompt)
        self.assertIn("prefer changes that fit A5", prompt)

    def test_build_optimize_unsupervised_prompt_mentions_compiler_source_when_enabled(self) -> None:
        prompt = build_optimize_unsupervised_prompt(
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
            compiler_source_path=Path("/tmp/AscendNPU-IR"),
            compiler_source_commit="abc123",
        )

        self.assertIn("Compiler source analysis is enabled", prompt)
        self.assertIn("Compiler source path: /tmp/AscendNPU-IR", prompt)
        self.assertIn("Compiler source commit: abc123.", prompt)
        self.assertIn("Use the staged `triton-npu-analyze-compiler-source` skill", prompt)
        self.assertNotIn("https://gitcode.com/Ascend/AscendNPU-IR.git", prompt)

    def test_build_optimize_unsupervised_prompt_mentions_cann_ext_api_when_enabled(self) -> None:
        prompt = build_optimize_unsupervised_prompt(
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
            enable_cann_ext_api=True,
        )

        self.assertIn("CANN Triton extension API pattern access is enabled for this optimize run.", prompt)
        self.assertIn("Use the staged `triton-npu-cann-ext-api-patterns` skill", prompt)

    def test_build_optimize_resume_prompt_preserves_compiler_source_when_enabled(self) -> None:
        prompt = build_optimize_resume_prompt(
            "Round gate passed.",
            compiler_source_path=Path("/tmp/AscendNPU-IR"),
            compiler_source_commit="abc123",
        )

        self.assertIn("Compiler source analysis is enabled", prompt)
        self.assertIn("Compiler source path: /tmp/AscendNPU-IR", prompt)
        self.assertIn("Compiler source commit: abc123.", prompt)
        self.assertIn("Round gate passed.", prompt)
        self.assertIn(
            "When pattern triage is used, record candidate patterns, the selected pattern if one is chosen, and why that pattern looks plausible in `opt-round-N/attempts.md`.",
            prompt,
        )
        self.assertIn(
            "When a named pattern guides the round, record the final selected pattern direction in `opt-round-N/summary.md`.",
            prompt,
        )
        self.assertNotIn("https://gitcode.com/Ascend/AscendNPU-IR.git", prompt)

    def test_build_optimize_unsupervised_prompt_mentions_min_rounds_when_requested(self) -> None:
        prompt = build_optimize_unsupervised_prompt(
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
            min_rounds=4,
        )
        self.assertIn("Complete at least 4 optimization rounds", prompt)
        self.assertIn("Once 4 optimization rounds are complete", prompt)
        self.assertIn(
            "stop the session after the current round passes `check-round` through `triton-npu-optimize-check`",
            prompt,
        )

    def test_build_optimize_supervisor_prompt_mentions_audit_role(self) -> None:
        prompt = build_optimize_supervisor_prompt(
            Path("/tmp"),
            latest_round_dir=Path("/tmp/opt-round-3"),
        )
        self.assertIn("This invocation is the optimize supervisor role.", prompt)
        self.assertIn("This invocation is an audit and handoff pass", prompt)
        self.assertIn("Read `/tmp/opt-round-3`", prompt)
        self.assertIn("Use only existing `compare-perf` results", prompt)
        self.assertIn("`triton-npu-prepare-optimize-baseline`", prompt)
        self.assertIn("`triton-npu-optimize-check`", prompt)
        self.assertIn("Write `.triton-agent/supervisor-report.md`", prompt)
        self.assertIn("Write `.triton-agent/round-brief.md`", prompt)
        self.assertIn("Do not edit the operator implementation", prompt)
        self.assertIn("replace the Triton kernel path with pure PyTorch computation", prompt)
        self.assertNotIn("optimize-supervisor.md", prompt)

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
        self.assertIn("triton-npu-gen-eval-suite", prompt)
        self.assertIn("Requested test output: /tmp/differential_test_op.py", prompt)
        self.assertIn("Requested benchmark output: /tmp/bench_op.py", prompt)
        self.assertIn("may edit the original operator file directly", prompt)
        self.assertIn("both generated artifacts must be executed", prompt)
        self.assertNotIn("Requested output:", prompt)

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
        self.assertIn("triton-npu-gen-test", prompt)
        self.assertIn("primary workflow contract", prompt)
        self.assertIn("helper scripts or subcommands", prompt)
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

    def test_gen_eval_prompt_mentions_force_overwrite_for_both_outputs(self) -> None:
        prompt = build_prompt(
            CommandKind.GEN_EVAL,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            None,
            test_mode="differential",
            bench_mode="standalone",
            force_overwrite=True,
        )
        self.assertIn("Overwrite any existing generated test, benchmark, or archived execution output files", prompt)

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
        self.assertIn("After generating the artifact, execute the generated test case", prompt)
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
        self.assertIn("After generating the artifact, execute the generated benchmark case", prompt)
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
        self.assertIn("When you execute generated test cases in this task", prompt)
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

    def test_convert_prompt_mentions_differential_validation_without_baseline(self) -> None:
        prompt = build_prompt(
            CommandKind.CONVERT,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/triton_op.py"),
            test_mode="differential",
            bench_mode=None,
            force_overwrite=False,
        )
        self.assertIn("Do not execute the original input operator file.", prompt)
        self.assertIn("Preserve the trailing input-helper block", prompt)
        self.assertIn("Treat the input operator file as source material and the differential correctness oracle.", prompt)
        self.assertIn("Generate a differential test for the converted output and execute it.", prompt)
        self.assertIn("Validate the converted output by comparing it against the original operator behavior.", prompt)
        self.assertIn("Do not introduce unnecessary wrappers, compatibility branches, helper layers, or scaffolding.", prompt)
        self.assertIn("real Triton Ascend NPU kernel path", prompt)
        self.assertIn("PyTorch-facing wrapper or module API may remain", prompt)
        self.assertIn("A pure PyTorch rewrite does not satisfy this convert task", prompt)
        self.assertIn("Target Ascend NPU only for this conversion flow", prompt)
        self.assertIn("Do not benchmark this workflow.", prompt)
        self.assertIn("Do not create `baseline/`.", prompt)
        self.assertNotIn("triton-npu-prepare-optimize-baseline", prompt)
        self.assertIn("Requested output: /tmp/triton_op.py", prompt)

    def test_optimize_prompt_mentions_requested_modes(self) -> None:
        prompt = build_prompt(
            CommandKind.OPTIMIZE,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
            force_overwrite=False,
            supervise="on",
        )
        self.assertIn("This invocation is the optimize worker role.", prompt)
        self.assertIn("This invocation owns exactly one round.", prompt)
        self.assertIn("Requested test mode: differential", prompt)
        self.assertIn("Requested bench mode: standalone", prompt)
        self.assertIn("Reuse existing correctness tests and benchmark cases when they already exist", prompt)
        self.assertIn("State the optimization hypothesis and why it may help", prompt)
        self.assertIn("Explain what evidence supports the change", prompt)
        self.assertIn("If you skip profiling or IR capture", prompt)

    def test_optimize_prompt_keeps_min_rounds_out_of_worker_prompt(self) -> None:
        prompt = build_prompt(
            CommandKind.OPTIMIZE,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
            force_overwrite=False,
            min_rounds=4,
            supervise="on",
        )
        self.assertNotIn("Complete at least 4 optimization rounds", prompt)
        self.assertIn("This invocation owns exactly one round.", prompt)

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
            supervise="on",
        )
        self.assertIn("Continue the existing optimization session", prompt)
        self.assertIn("Read `opt-note.md`", prompt)

    def test_optimize_prompt_defaults_to_layered_analysis(self) -> None:
        prompt = build_prompt(
            CommandKind.OPTIMIZE,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
            force_overwrite=False,
            supervise="on",
        )
        self.assertIn("Choose the analysis level for the round before editing code.", prompt)
        self.assertIn(
            "Escalate analysis in this order: pattern triage, profiling diagnosis, IR attribution, compiler-source escalation.",
            prompt,
        )
        self.assertIn(
            "Use profiling diagnosis as the default deeper entrypoint when pattern triage is not enough.",
            prompt,
        )
        self.assertIn(
            "Record the round's primary analysis level separately from its supporting evidence.",
            prompt,
        )
        self.assertIn(
            "Use the staged `triton-npu-optimize-knowledge` skill for generic pattern and symptom references.",
            prompt,
        )
        self.assertIn(
            "Read the staged `triton-npu-optimize-knowledge` skill's generated `references/pattern_index.md` before detailed pattern references.",
            prompt,
        )
        self.assertIn(
            "Inspect the operator file directly when code structure is still unclear at pattern triage.",
            prompt,
        )
        self.assertIn(
            "Use the staged `triton-npu-optimize-knowledge` skill's symptom cards to narrow pattern candidates after structured profiler or IR evidence exists.",
            prompt,
        )
        self.assertIn("Do not begin with blind tiling or launch-parameter search", prompt)

    def test_optimize_prompt_defaults_to_unsupervised_mode(self) -> None:
        prompt = build_prompt(
            CommandKind.OPTIMIZE,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
            force_overwrite=False,
        )
        self.assertIn("This invocation is an unsupervised optimize run.", prompt)
        self.assertNotIn("This invocation is the optimize worker role.", prompt)

    def test_optimize_prompt_unsupervised_avoids_role_brief_artifacts(self) -> None:
        prompt = build_prompt(
            CommandKind.OPTIMIZE,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
            force_overwrite=False,
            supervise="off",
        )
        self.assertIn("This invocation is an unsupervised optimize run.", prompt)
        self.assertIn("Own the end-to-end optimize session", prompt)
        self.assertNotIn("optimize worker role", prompt)
        self.assertNotIn("owns exactly one round", prompt)
        self.assertNotIn(".triton-agent/roles/optimize-worker.md", prompt)
        self.assertNotIn(".triton-agent/round-brief.md", prompt)

    def test_append_additional_user_instructions_adds_section(self) -> None:
        prompt = append_additional_user_instructions(
            "Optimize the operator implementation.",
            "Prefer shared-memory reductions.",
        )
        self.assertIn("Optimize the operator implementation.", prompt)
        self.assertIn("Additional user instructions:", prompt)
        self.assertIn("Prefer shared-memory reductions.", prompt)

    def test_append_additional_user_instructions_skips_blank_values(self) -> None:
        prompt = append_additional_user_instructions(
            "Optimize the operator implementation.",
            "   ",
        )
        self.assertEqual(prompt, "Optimize the operator implementation.")


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
