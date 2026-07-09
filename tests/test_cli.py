import argparse
import json
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
from triton_agent.terminal.render import render_result
from triton_agent.paths import (
    default_generated_output_path,
    resolve_execution_target,
)
from triton_agent.prompts import (
    append_additional_user_instructions,
    build_optimize_round_prompt,
    build_prompt,
)
from triton_agent.optimize.prompts import (
    build_optimize_baseline_prompt,
    build_optimize_resume_prompt,
    build_optimize_supervisor_prompt,
)
from triton_agent.optimize.env import optimize_min_speedup_env_name
from triton_agent.remote.env import remote_target_env_name, remote_workdir_env_name
from triton_agent.eval.runners import _normalize_agent_result as normalize_agent_result


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
        self.assertEqual(args.concurrency, 1)
        self.assertEqual(args.test_mode, "differential")
        self.assertEqual(args.bench_mode, "torch-npu-profiler")
        self.assertEqual(args.agent, "codex")
        self.assertFalse(hasattr(args, "interact"))
        self.assertFalse(hasattr(args, "output"))

    def test_gen_eval_batch_accepts_max_concurrency_keyword(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["gen-eval-batch", "-i", "kernels", "--concurrency", "max"])
        self.assertEqual(args.concurrency, "max")

    def test_concurrency_short_alias_is_parseable_for_supported_commands(self) -> None:
        parser = build_parser()
        for argv, expected in (
            (["gen-eval", "-i", "kernels", "-c", "max"], "max"),
            (["gen-eval-batch", "-i", "kernels", "-c", "max"], "max"),
            (["convert", "-i", "kernels", "-c", "2"], 2),
            (["convert-batch", "-i", "kernels", "-c", "2"], 2),
            (["log-check", "-i", "kernels", "-c", "4"], 4),
            (["log-check-batch", "-i", "kernels", "-c", "4"], 4),
            (["optimize", "-i", "kernels", "-c", "3"], 3),
            (["optimize-batch", "-i", "kernels", "-c", "3"], 3),
            (["report", "-i", "kernels", "-c", "2"], 2),
            (["verify", "-i", "kernels", "-c", "2"], 2),
            (["distill", "-i", "operators", "-c", "2"], 2),
        ):
            with self.subTest(argv=argv):
                args = parser.parse_args(argv)
                self.assertEqual(args.concurrency, expected)

    def test_single_commands_do_not_default_to_batch_mode(self) -> None:
        parser = build_parser()
        for command, input_value in (
            ("gen-eval", "kernel.py"),
            ("convert", "kernel.py"),
            ("log-check", "workspace"),
            ("optimize", "kernel.py"),
            ("report", "workspace"),
            ("verify", "workspace"),
        ):
            with self.subTest(command=command):
                args = parser.parse_args([command, "-i", input_value])
                self.assertIsNone(args.concurrency)

    def test_gen_eval_batch_accepts_operator_filter(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["gen-eval-batch", "-i", "kernels", "--operator-filter", "kernel*.py"]
        )
        self.assertEqual(args.operator_filter, "kernel*.py")

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
                "--concurrency",
                "4",
                "--verbose",
                "--no-stream-output",
            ]
        )
        self.assertEqual(args.command, "log-check-batch")
        self.assertEqual(args.command_kind, CommandKind.LOG_CHECK_BATCH)
        self.assertEqual(args.check_result_file, "custom_check.txt")
        self.assertEqual(args.summary_file, "custom_summary.txt")
        self.assertEqual(args.concurrency, 4)
        self.assertTrue(args.verbose)
        self.assertFalse(args.stream_output)

    def test_log_check_with_explicit_concurrency_accepts_batch_result_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "log-check",
                "-i",
                "kernels",
                "--check-result-file",
                "custom_check.txt",
                "--summary-file",
                "custom_summary.txt",
                "--concurrency",
                "4",
            ]
        )
        self.assertEqual(args.command, "log-check")
        self.assertEqual(args.command_kind, CommandKind.LOG_CHECK)
        self.assertEqual(args.check_result_file, "custom_check.txt")
        self.assertEqual(args.summary_file, "custom_summary.txt")
        self.assertEqual(args.concurrency, 4)

    def test_log_check_with_explicit_concurrency_uses_batch_handler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captured: dict[str, object] = {}

            def _fake_run_log_check_batch(
                root_path,
                *,
                output_file,
                summary_file,
                agent_name,
                verbose,
                show_output,
                log_tools,
                language,
                max_concurrency,
            ):
                del output_file, agent_name, verbose, show_output, log_tools, language
                captured["root"] = root_path
                captured["summary_file"] = summary_file
                captured["max_concurrency"] = max_concurrency
                return 0

            with patch(
                "triton_agent.commands.log_check.run_log_check_batch",
                side_effect=_fake_run_log_check_batch,
            ):
                exit_code = main(
                    [
                        "log-check",
                        "-i",
                        str(root),
                        "--concurrency",
                        "4",
                        "--summary-file",
                        "custom_summary.md",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["root"], root.resolve())
            self.assertEqual(captured["summary_file"], "custom_summary.md")
            self.assertEqual(captured["max_concurrency"], 4)

    def test_log_check_batch_rejects_max_concurrency_keyword(self) -> None:
        parser = build_parser()
        stderr = StringIO()
        with self.assertRaises(SystemExit) as exc, redirect_stderr(stderr):
            parser.parse_args(["log-check-batch", "-i", "kernels", "--concurrency", "max"])
        self.assertEqual(exc.exception.code, 2)
        self.assertIn("--concurrency", stderr.getvalue())

    def test_gen_eval_maps_to_command_kind(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["gen-eval", "-i", "kernel.py"])
        self.assertEqual(args.command, "gen-eval")
        self.assertEqual(args.command_kind, CommandKind.GEN_EVAL)
        self.assertEqual(args.test_mode, "differential")
        self.assertEqual(args.bench_mode, "torch-npu-profiler")
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

    def test_agent_generation_commands_accept_log_tools_option(self) -> None:
        from triton_agent.commands.generation import generation_options_from_args

        parser = build_parser()
        for command in ("gen-eval", "gen-eval-batch", "gen-test", "gen-bench"):
            with self.subTest(command=command):
                args = parser.parse_args([command, "-i", "kernel.py", "--log-tool"])
                self.assertTrue(args.log_tools)
                options = generation_options_from_args(args)
                self.assertTrue(options.log_tools)

    def test_enable_mcp_is_available_on_agent_backed_run_eval_commands(self) -> None:
        parser = build_parser()
        cases = (
            ("gen-eval", "kernel.py"),
            ("gen-eval-batch", "kernels"),
            ("convert", "kernel.py"),
            ("convert-batch", "kernels"),
            ("optimize", "kernel.py"),
            ("optimize-batch", "kernels"),
        )

        for command, input_value in cases:
            with self.subTest(command=command):
                args = parser.parse_args([command, "-i", input_value, "--enable-mcp"])
                self.assertTrue(args.enable_mcp)

    def test_optimize_commands_accept_enable_report_option(self) -> None:
        parser = build_parser()
        cases = (
            ("optimize", "kernel.py"),
            ("optimize-batch", "kernels"),
        )

        for command, input_value in cases:
            with self.subTest(command=command):
                args = parser.parse_args([command, "-i", input_value, "--enable-report"])
                self.assertTrue(args.enable_report)

    def test_enable_mcp_is_not_available_on_direct_execution_commands(self) -> None:
        parser = build_parser()
        stderr = StringIO()
        with self.assertRaises(SystemExit) as exc, redirect_stderr(stderr):
            parser.parse_args(
                [
                    "run-bench",
                    "--bench-file",
                    "bench.py",
                    "--operator-file",
                    "kernel.py",
                    "--enable-mcp",
                ]
            )
        self.assertEqual(exc.exception.code, 2)
        self.assertIn("--enable-mcp", stderr.getvalue())

    def test_run_eval_mcp_server_maps_to_command_kind(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run-eval-mcp-server"])
        self.assertEqual(args.command, "run-eval-mcp-server")
        self.assertEqual(args.command_kind, CommandKind.RUN_EVAL_MCP_SERVER)
        self.assertEqual(args.port, 0)

    def test_run_eval_mcp_server_accepts_explicit_port(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run-eval-mcp-server", "--port", "8765"])
        self.assertEqual(args.port, 8765)

    def test_run_eval_mcp_server_accepts_batch_affinity_cli_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["run-eval-mcp-server", "--npu-devices", "0,1", "--workers-per-npu", "2"]
        )
        self.assertEqual(args.npu_devices, "0,1")
        self.assertEqual(args.workers_per_npu, "2")

    def test_run_eval_mcp_server_accepts_batch_affinity_cli_aliases(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["run-eval-mcp-server", "--npu-device", "0,1", "--worker-per-npu", "2"]
        )
        self.assertEqual(args.npu_devices, "0,1")
        self.assertEqual(args.workers_per_npu, "2")

    def test_handle_run_eval_mcp_server_passes_explicit_batch_affinity_values(self) -> None:
        from triton_agent.commands.mcp_server import handle_run_eval_mcp_server

        args = argparse.Namespace(port=1234, npu_devices="0,1", workers_per_npu="2")
        with patch("triton_agent.commands.mcp_server.serve_http_server_forever", return_value=0) as mocked:
            exit_code = handle_run_eval_mcp_server(argparse.ArgumentParser(), args)
        self.assertEqual(exit_code, 0)
        mocked.assert_called_once_with(port=1234, npu_devices="0,1", workers_per_npu="2")

    def test_generation_run_eval_commands_accept_enable_mcp_option(self) -> None:
        from triton_agent.commands.generation import generation_options_from_args

        parser = build_parser()
        for command, input_value in (
            ("gen-eval", "kernel.py"),
            ("gen-eval-batch", "kernels"),
        ):
            with self.subTest(command=command):
                args = parser.parse_args([command, "-i", input_value, "--enable-mcp"])
                self.assertTrue(args.enable_mcp)
                options = generation_options_from_args(args)
                self.assertTrue(options.enable_mcp)

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
        self.assertEqual(args.concurrency, 1)
        self.assertEqual(args.test_mode, "differential")
        self.assertEqual(args.agent, "codex")
        self.assertFalse(hasattr(args, "interact"))
        self.assertFalse(hasattr(args, "output"))

    def test_convert_batch_accepts_operator_filter(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["convert-batch", "-i", "kernels", "--operator-filter", "kernel*.py"]
        )
        self.assertEqual(args.operator_filter, "kernel*.py")

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

    def test_convert_commands_accept_log_tools_option(self) -> None:
        from triton_agent.commands.convert import convert_options_from_args

        parser = build_parser()
        for command in ("convert", "convert-batch"):
            with self.subTest(command=command):
                args = parser.parse_args([command, "-i", "kernel.py", "--log-tool"])
                self.assertTrue(args.log_tools)
                options = convert_options_from_args(args)
                self.assertTrue(options.log_tools)

    def test_convert_commands_accept_enable_mcp_option(self) -> None:
        from triton_agent.commands.convert import convert_options_from_args

        parser = build_parser()
        for command, input_value in (("convert", "kernel.py"), ("convert-batch", "kernels")):
            with self.subTest(command=command):
                args = parser.parse_args([command, "-i", input_value, "--enable-mcp"])
                self.assertTrue(args.enable_mcp)
                options = convert_options_from_args(args)
                self.assertTrue(options.enable_mcp)

    def test_log_check_commands_accept_log_tools_option(self) -> None:
        parser = build_parser()
        for command in ("log-check", "log-check-batch"):
            with self.subTest(command=command):
                args = parser.parse_args([command, "-i", "workspace", "--log-tool"])
                self.assertTrue(args.log_tools)

    def test_convert_commands_accept_standalone_test_mode(self) -> None:
        parser = build_parser()
        for command, input_value in (("convert", "kernel.py"), ("convert-batch", "kernels")):
            with self.subTest(command=command):
                args = parser.parse_args([command, "-i", input_value, "--test-mode", "standalone"])
                self.assertEqual(args.test_mode, "standalone")

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

    def test_build_generation_request_carries_batch_affinity_values(self) -> None:
        from triton_agent.generation.models import GenerationOptions
        from triton_agent.generation.orchestration import build_generation_request

        options = GenerationOptions(
            interact=False,
            verbose=False,
            stream_output=False,
            force_overwrite=False,
            agent_name="codex",
            remote=None,
            remote_workdir=None,
            min_rounds=None,
            continue_optimize=False,
            output=None,
            test_mode=None,
            bench_mode=None,
            npu_devices="0,1",
            workers_per_npu="2",
            prompt=None,
            log_tools=False,
            enable_mcp=True,
        )

        request = build_generation_request(
            CommandKind.GEN_EVAL,
            Path("kernel.py"),
            Path("kernel.py"),
            Path.cwd(),
            options,
        )

        self.assertEqual(request.npu_devices, "0,1")
        self.assertEqual(request.workers_per_npu, "2")

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
            ("run_eval_mcp_server", CommandKind.RUN_EVAL_MCP_SERVER),
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
                elif expected_kind == CommandKind.RUN_EVAL_MCP_SERVER:
                    argv = [alias]
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
        self.assertIn("run-eval-mcp-server", help_text)
        self.assertIn("compare-result", help_text)
        self.assertIn("compare-perf", help_text)
        self.assertIn("status", help_text)
        self.assertIn("verify", help_text)
        self.assertIn("verify-batch", help_text)
        self.assertIn("optimize-batch", help_text)
        self.assertNotIn("gen-convert", help_text)
        self.assertNotIn("gen_eval_batch", help_text)


class CliMCPServerCommandTests(unittest.TestCase):
    def test_main_routes_run_eval_mcp_server_command(self) -> None:
        with patch(
            "triton_agent.commands.mcp_server.serve_http_server_forever",
            return_value=7,
        ) as mocked:
            exit_code = main(["run-eval-mcp-server", "--port", "8765"])

        self.assertEqual(exit_code, 7)
        mocked.assert_called_once_with(port=8765, npu_devices=None, workers_per_npu=None)

    def test_main_prints_build_commit_for_long_version_flag(self) -> None:
        stdout = StringIO()
        with patch("triton_agent.cli.get_build_info_display", return_value="deadbeefcafe"):
            with redirect_stdout(stdout):
                with self.assertRaises(SystemExit) as exc:
                    main(["--version"])
        self.assertEqual(exc.exception.code, 0)
        self.assertEqual(stdout.getvalue(), "deadbeefcafe\n")

    def test_main_prints_build_commit_for_short_version_flag(self) -> None:
        stdout = StringIO()
        with patch("triton_agent.cli.get_build_info_display", return_value="deadbeefcafe"):
            with redirect_stdout(stdout):
                with self.assertRaises(SystemExit) as exc:
                    main(["-v"])
        self.assertEqual(exc.exception.code, 0)
        self.assertEqual(stdout.getvalue(), "deadbeefcafe\n")

    def test_top_level_help_groups_commands_and_examples(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()
        self.assertIn("Generate, run, verify, and optimize NPU operator workflows.", help_text)
        self.assertIn("Build info:", help_text)
        self.assertIn("Git commit:", help_text)
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
        self.assertIn("TRITON_AGENT_BENCH_OUTPUT_DIR", help_text)
        self.assertIn(remote_target_env_name(), help_text)
        self.assertIn(remote_workdir_env_name(), help_text)
        self.assertIn("TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES", help_text)
        self.assertIn(optimize_min_speedup_env_name(), help_text)
        self.assertIn("TRITON_AGENT_OPTIMIZE_LOCAL_OPTIMUM_WINDOW", help_text)
        self.assertIn("TRITON_AGENT_OPTIMIZE_LOCAL_OPTIMUM_MAX_GEOMEAN_GAIN", help_text)
        self.assertIn("TRITON_AGENT_COMPILER_SOURCE_CACHE_DIR", help_text)
        self.assertIn("TRITON_AGENT_STALL_TIMEOUT_SECONDS", help_text)
        self.assertIn("TRITON_AGENT_SSH_TIMEOUT_SECONDS", help_text)
        self.assertIn("TRITON_AGENT_SCP_TIMEOUT_SECONDS", help_text)
        self.assertIn("TRITON_AGENT_EVAL_TIMEOUT_SECONDS", help_text)
        self.assertNotIn("TRITON_AGENT_TEST_TIMEOUT_SECONDS", help_text)
        self.assertNotIn("TRITON_AGENT_BENCH_TIMEOUT_SECONDS", help_text)
        self.assertIn("TRITON_AGENT_PROFILE_TIMEOUT_SECONDS", help_text)
        self.assertIn("TRITON_AGENT_DEBUG", help_text)
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
        self.assertFalse(hasattr(args, "stream_output"))

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

    def test_verify_with_explicit_concurrency_accepts_batch_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "verify",
                "-i",
                "workspace-root",
                "--concurrency",
                "4",
                "--force-verify",
            ]
        )
        self.assertEqual(args.command_kind, CommandKind.VERIFY)
        self.assertEqual(args.concurrency, 4)
        self.assertTrue(args.force_verify)

    def test_verify_with_explicit_concurrency_uses_batch_handler_and_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captured: dict[str, object] = {}
            stderr = StringIO()

            def _fake_run_verify_batch(root_path, *, force_verify, options):
                del options
                captured["root"] = root_path
                captured["force_verify"] = force_verify
                return 0

            with patch(
                "triton_agent.commands.verify.run_verify_batch",
                side_effect=_fake_run_verify_batch,
            ):
                with redirect_stderr(stderr):
                    exit_code = main(
                        ["verify", "-i", str(root), "--concurrency", "4", "--force-verify"]
                    )

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["root"], root.resolve())
            self.assertTrue(captured["force_verify"])
            self.assertIn("ignores --concurrency", stderr.getvalue())

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

    def test_run_bench_accepts_npu_devices_option(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run-bench",
                "--bench-file",
                "bench_kernel.py",
                "--operator-file",
                "kernel.py",
                "--npu-devices",
                "0,2-3",
            ]
        )
        self.assertEqual(args.command_kind, CommandKind.RUN_BENCH)
        self.assertEqual(args.npu_devices, "0,2-3")

    def test_run_test_requires_test_and_operator_files(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["run-test", "--test-file", "test_kernel.py", "--operator-file", "kernel.py"]
        )
        self.assertEqual(args.command_kind, CommandKind.RUN_TEST)
        self.assertEqual(args.test_file, "test_kernel.py")
        self.assertEqual(args.operator_file, "kernel.py")
        self.assertEqual(args.accuracy_mode, "npu-contract")
        self.assertFalse(hasattr(args, "agent"))

    def test_run_test_accepts_accuracy_mode(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run-test",
                "--test-file",
                "test_kernel.py",
                "--operator-file",
                "kernel.py",
                "--accuracy-mode",
                "dtype-close",
            ]
        )

        self.assertEqual(args.command_kind, CommandKind.RUN_TEST)
        self.assertEqual(args.accuracy_mode, "dtype-close")

    def test_compare_result_accepts_accuracy_mode(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "compare-result",
                "--ref-result",
                "ref_result.pt",
                "--new-result",
                "new_result.pt",
                "--accuracy-mode",
                "dtype-close",
            ]
        )

        self.assertEqual(args.command_kind, CommandKind.COMPARE_RESULT)
        self.assertEqual(args.accuracy_mode, "dtype-close")

    def test_run_bench_requires_bench_and_operator_files(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["run-bench", "--bench-file", "bench_kernel.py", "--operator-file", "kernel.py"]
        )
        self.assertEqual(args.command_kind, CommandKind.RUN_BENCH)
        self.assertEqual(args.bench_file, "bench_kernel.py")
        self.assertEqual(args.operator_file, "kernel.py")

    def test_run_bench_accepts_baseline_operator_file(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run-bench",
                "--bench-file",
                "bench_kernel.py",
                "--operator-file",
                "opt_kernel.py",
                "--baseline-operator-file",
                "kernel.py",
            ]
        )
        self.assertEqual(args.command_kind, CommandKind.RUN_BENCH)
        self.assertEqual(args.baseline_operator_file, "kernel.py")

    def test_run_bench_accepts_compare_perf_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run-bench",
                "--bench-file",
                "bench_kernel.py",
                "--operator-file",
                "opt_kernel.py",
                "--baseline-operator-file",
                "kernel.py",
                "--skip-latency-errors",
                "--metric-source",
                "all",
            ]
        )
        self.assertTrue(args.skip_latency_errors)
        self.assertEqual(args.metric_source, "all")

    def test_run_bench_accepts_metric_source_short_alias(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run-bench",
                "--bench-file",
                "bench_kernel.py",
                "--operator-file",
                "opt_kernel.py",
                "-m",
                "kernel",
            ]
        )
        self.assertEqual(args.metric_source, "kernel")

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
                "--ref-result",
                "ref_result.pt",
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
                "--ref-result",
                "ref_result.pt",
                "--new-result",
                "new_result.pt",
            ]
        )
        self.assertEqual(args.command_kind, CommandKind.COMPARE_RESULT)
        self.assertEqual(args.ref_result, "ref_result.pt")
        self.assertEqual(args.new_result, "new_result.pt")

    def test_compare_result_accepts_legacy_oracle_result_alias(self) -> None:
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
        self.assertEqual(args.ref_result, "oracle_result.pt")
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
        self.assertFalse(args.skip_latency_errors)

    def test_compare_perf_accepts_skip_latency_errors_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "compare-perf",
                "--baseline",
                "baseline_perf.txt",
                "--compare",
                "candidate_perf.txt",
                "--skip-latency-errors",
            ]
        )
        self.assertEqual(args.command_kind, CommandKind.COMPARE_PERF)
        self.assertTrue(args.skip_latency_errors)

    def test_compare_perf_accepts_metric_source_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "compare-perf",
                "--baseline",
                "baseline_perf.txt",
                "--compare",
                "candidate_perf.txt",
                "--metric-source",
                "total-op",
            ]
        )
        self.assertEqual(args.command_kind, CommandKind.COMPARE_PERF)
        self.assertEqual(args.metric_source, "total-op")

    def test_compare_perf_accepts_metric_source_short_alias(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "compare-perf",
                "--baseline",
                "baseline_perf.txt",
                "--compare",
                "candidate_perf.txt",
                "-m",
                "total-op",
            ]
        )
        self.assertEqual(args.command_kind, CommandKind.COMPARE_PERF)
        self.assertEqual(args.metric_source, "total-op")

    def test_compare_perf_defaults_metric_source_to_auto(self) -> None:
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
        self.assertEqual(args.metric_source, "auto")

    def test_probe_bench_requires_bench_operator_and_baseline_files(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "probe-bench",
                "--bench-file",
                "bench_kernel.py",
                "--operator-file",
                "opt_kernel.py",
                "--baseline-operator-file",
                "baseline_kernel.py",
            ]
        )
        self.assertEqual(args.command_kind, CommandKind.PROBE_BENCH)
        self.assertEqual(args.bench_file, "bench_kernel.py")
        self.assertEqual(args.operator_file, "opt_kernel.py")
        self.assertEqual(args.baseline_operator_file, "baseline_kernel.py")

    def test_probe_bench_requires_baseline_operator_file(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "probe-bench",
                    "--bench-file",
                    "bench_kernel.py",
                    "--operator-file",
                    "opt_kernel.py",
                ]
            )

    def test_probe_bench_accepts_metric_source_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "probe-bench",
                "--bench-file",
                "bench_kernel.py",
                "--operator-file",
                "opt_kernel.py",
                "--baseline-operator-file",
                "baseline_kernel.py",
                "--metric-source",
                "total-op",
            ]
        )
        self.assertEqual(args.command_kind, CommandKind.PROBE_BENCH)
        self.assertEqual(args.metric_source, "total-op")

    def test_probe_bench_accepts_metric_source_short_alias(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "probe-bench",
                "--bench-file",
                "bench_kernel.py",
                "--operator-file",
                "opt_kernel.py",
                "--baseline-operator-file",
                "baseline_kernel.py",
                "-m",
                "total-op",
            ]
        )
        self.assertEqual(args.command_kind, CommandKind.PROBE_BENCH)
        self.assertEqual(args.metric_source, "total-op")

    def test_probe_bench_defaults_metric_source_to_auto(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "probe-bench",
                "--bench-file",
                "bench_kernel.py",
                "--operator-file",
                "opt_kernel.py",
                "--baseline-operator-file",
                "baseline_kernel.py",
            ]
        )
        self.assertEqual(args.metric_source, "auto")

    def test_probe_bench_rejects_metric_source_all(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "probe-bench",
                    "--bench-file",
                    "bench_kernel.py",
                    "--operator-file",
                    "opt_kernel.py",
                    "--baseline-operator-file",
                    "baseline_kernel.py",
                    "--metric-source",
                    "all",
                ]
            )

    def test_probe_bench_rejects_output_flag(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "probe-bench",
                    "--bench-file",
                    "bench_kernel.py",
                    "--operator-file",
                    "opt_kernel.py",
                    "--baseline-operator-file",
                    "baseline_kernel.py",
                    "-o",
                    "out.txt",
                ]
            )

    def test_probe_bench_has_common_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "probe-bench",
                "--bench-file",
                "bench_kernel.py",
                "--operator-file",
                "opt_kernel.py",
                "--baseline-operator-file",
                "baseline_kernel.py",
                "--bench-mode",
                "torch-npu-profiler",
                "--npu-devices",
                "0,2-3",
                "--verbose",
            ]
        )
        self.assertEqual(args.bench_mode, "torch-npu-profiler")
        self.assertEqual(args.npu_devices, "0,2-3")
        self.assertTrue(args.verbose)

    def test_verbose_option_is_available(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["gen-test", "-i", "kernel.py", "--verbose"])
        self.assertTrue(args.verbose)

    def test_no_stream_output_option_is_available(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["gen-test", "-i", "kernel.py", "--no-stream-output"])
        self.assertFalse(args.stream_output)

    def test_agent_commands_stream_output_by_default(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["gen-test", "-i", "kernel.py"])
        self.assertTrue(args.stream_output)

    def test_show_output_option_no_longer_parses(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["gen-test", "-i", "kernel.py", "--show-output"])

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

    def test_run_test_accepts_ref_result(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run-test",
                "--test-file",
                "differential_test_kernel.py",
                "--operator-file",
                "kernel.py",
                "--ref-result",
                "ref_result.pt",
            ]
        )
        self.assertEqual(args.ref_result, "ref_result.pt")
        self.assertIsNone(args.ref_operator_file)

    def test_run_test_accepts_legacy_baseline_result_alias(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run-test",
                "--test-file",
                "differential_test_kernel.py",
                "--operator-file",
                "kernel.py",
                "--baseline-result",
                "baseline_result.pt",
            ]
        )
        self.assertEqual(args.ref_result, "baseline_result.pt")
        self.assertIsNone(args.ref_operator_file)

    def test_run_test_accepts_ref_operator_file(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run-test",
                "--test-file",
                "differential_test_kernel.py",
                "--operator-file",
                "opt_kernel.py",
                "--ref-operator-file",
                "kernel.py",
            ]
        )
        self.assertIsNone(args.ref_result)
        self.assertEqual(args.ref_operator_file, "kernel.py")

    def test_run_test_accepts_legacy_baseline_operator_file_alias(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run-test",
                "--test-file",
                "differential_test_kernel.py",
                "--operator-file",
                "opt_kernel.py",
                "--baseline-operator-file",
                "kernel.py",
            ]
        )
        self.assertIsNone(args.ref_result)
        self.assertEqual(args.ref_operator_file, "kernel.py")

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
        gen_args = parser.parse_args(
            ["gen-bench", "-i", "kernel.py", "--bench-mode", "torch-npu-profiler"]
        )
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
        self.assertEqual(gen_args.bench_mode, "torch-npu-profiler")
        self.assertEqual(run_args.bench_mode, "msprof")

    def test_bench_commands_default_to_torch_npu_profiler_mode(self) -> None:
        parser = build_parser()
        gen_args = parser.parse_args(["gen-bench", "-i", "kernel.py"])
        run_args = parser.parse_args(
            ["run-bench", "--bench-file", "bench_kernel.py", "--operator-file", "kernel.py"]
        )
        self.assertEqual(gen_args.bench_mode, "torch-npu-profiler")
        self.assertIsNone(run_args.bench_mode)

    def test_run_simulator_accepts_case_and_kernel_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run-simulator",
                "--bench-file",
                "bench_kernel.py",
                "--operator-file",
                "kernel.py",
                "--case-id",
                "case-a",
                "--kernel-name",
                "KernelA",
            ]
        )
        self.assertEqual(args.command_kind, CommandKind.RUN_SIMULATOR)
        self.assertEqual(args.bench_file, "bench_kernel.py")
        self.assertEqual(args.operator_file, "kernel.py")
        self.assertEqual(args.case_id, "case-a")
        self.assertEqual(args.kernel_name, "KernelA")

    def test_run_simulator_rejects_bench_mode_and_remote_options(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "run-simulator",
                    "--bench-file",
                    "bench_kernel.py",
                    "--operator-file",
                    "kernel.py",
                    "--bench-mode",
                    "msprof",
                ]
            )
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "run-simulator",
                    "--bench-file",
                    "bench_kernel.py",
                    "--operator-file",
                    "kernel.py",
                    "--remote",
                    "alice@example.com",
                ]
            )

    def test_gen_eval_defaults_to_torch_npu_profiler_bench_mode(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["gen-eval", "-i", "kernel.py"])
        self.assertEqual(args.bench_mode, "torch-npu-profiler")

    def test_gen_eval_batch_defaults_to_torch_npu_profiler_bench_mode(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["gen-eval-batch", "-i", "kernels"])
        self.assertEqual(args.bench_mode, "torch-npu-profiler")

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
                "--concurrency",
                "3",
                "--no-stream-output",
            ]
        )
        self.assertEqual(args.agent, "pi")
        self.assertEqual(args.remote, "alice@example.com")
        self.assertEqual(args.remote_workdir, "/tmp/eval")
        self.assertEqual(args.test_mode, "standalone")
        self.assertEqual(args.bench_mode, "msprof")
        self.assertEqual(args.concurrency, 3)
        self.assertFalse(args.stream_output)

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
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--min-round", "3"])
        self.assertEqual(args.min_rounds, 3)

    def test_optimize_command_defaults_min_rounds(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py"])
        self.assertEqual(args.min_rounds, 5)
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.min_rounds, 5)

    def test_optimize_command_defaults_min_speedup_to_none(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py"])
        self.assertIsNone(args.min_speedup)
        options = optimize_run_options_from_args(args)
        self.assertIsNone(options.min_speedup)

    def test_optimize_command_accepts_min_speedup(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--min-speedup", "1.2"])
        self.assertEqual(args.min_speedup, 1.2)
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.min_speedup, 1.2)

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
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--enable-agent-hook"])

        self.assertTrue(args.enable_agent_hooks)
        options = optimize_run_options_from_args(args)
        self.assertTrue(options.enable_agent_hooks)

    def test_optimize_accepts_enable_subagent_option(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--enable-subagent"])

        self.assertTrue(args.enable_subagent)
        options = optimize_run_options_from_args(args)
        self.assertTrue(options.enable_subagent)

    def test_optimize_accepts_log_tools_option(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--log-tool"])

        self.assertTrue(args.log_tools)
        options = optimize_run_options_from_args(args)
        self.assertTrue(options.log_tools)

    def test_optimize_batch_accepts_enable_subagent_option(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize-batch", "-i", "operators", "--enable-subagent"])

        self.assertTrue(args.enable_subagent)
        options = optimize_run_options_from_args(args)
        self.assertTrue(options.enable_subagent)

    def test_optimize_batch_accepts_agent_hooks_option(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize-batch", "-i", "operators", "--enable-agent-hook"])

        self.assertTrue(args.enable_agent_hooks)
        options = optimize_run_options_from_args(args)
        self.assertTrue(options.enable_agent_hooks)

    def test_optimize_batch_accepts_log_tools_option(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize-batch", "-i", "operators", "--log-tool"])

        self.assertTrue(args.log_tools)
        options = optimize_run_options_from_args(args)
        self.assertTrue(options.log_tools)

    def test_optimize_commands_accept_enable_mcp_option(self) -> None:
        parser = build_parser()
        for command, input_value in (("optimize", "kernel.py"), ("optimize-batch", "operators")):
            with self.subTest(command=command):
                args = parser.parse_args([command, "-i", input_value, "--enable-mcp"])
                self.assertTrue(args.enable_mcp)
                options = optimize_run_options_from_args(args)
                self.assertTrue(options.enable_mcp)

    def test_canonical_plural_flags_remain_parseable(self) -> None:
        """Backward compatibility: published plural flag names must continue to work."""
        parser = build_parser()

        # --log-tools (generation)
        args = parser.parse_args(["gen-eval", "-i", "kernel.py", "--log-tools"])
        self.assertTrue(args.log_tools)

        # --min-rounds (optimize)
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--min-rounds", "3"])
        self.assertEqual(args.min_rounds, 3)

        # --enable-agent-hooks (optimize)
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--enable-agent-hooks"])
        self.assertTrue(args.enable_agent_hooks)

        # --enable-agent-hooks (optimize-batch)
        args = parser.parse_args(["optimize-batch", "-i", "kernels", "--enable-agent-hooks"])
        self.assertTrue(args.enable_agent_hooks)

        # --skip-latency-errors (compare-perf)
        args = parser.parse_args(
            ["compare-perf", "--baseline", "a.txt", "--compare", "b.txt", "--skip-latency-errors"]
        )
        self.assertTrue(args.skip_latency_errors)

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

    def test_optimize_command_defaults_optimize_target_to_kernel(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py"])
        self.assertEqual(args.optimize_target, "kernel")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.optimize_target, "kernel")

    def test_optimize_command_accepts_operator_optimize_target(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["optimize", "-i", "kernel.py", "--optimize-target", "operator"]
        )
        self.assertEqual(args.optimize_target, "operator")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.optimize_target, "operator")

    def test_optimize_batch_defaults_optimize_target_to_kernel(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize-batch", "-i", "kernels"])
        self.assertEqual(args.optimize_target, "kernel")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.optimize_target, "kernel")

    def test_optimize_batch_accepts_operator_optimize_target(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["optimize-batch", "-i", "kernels", "--optimize-target", "operator"]
        )
        self.assertEqual(args.optimize_target, "operator")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.optimize_target, "operator")

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

    def test_optimize_command_accepts_round_modes(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--round-mode", "checked"])
        self.assertEqual(args.round_mode, "checked")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.round_mode, "checked")

    def test_optimize_command_accepts_user_prompt(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--prompt", "Focus on memory access."])
        self.assertEqual(args.prompt, "Focus on memory access.")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.prompt, "Focus on memory access.")

    def test_optimize_command_defaults_round_mode_checked(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py"])
        self.assertEqual(args.round_mode, "checked")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.round_mode, "checked")

    def test_optimize_command_accepts_round_batch_size(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--round-batch-size", "3"])
        self.assertEqual(args.round_batch_size, 3)
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.round_batch_size, 3)

    def test_optimize_run_options_rejects_invalid_round_mode(self) -> None:
        args = argparse.Namespace(
            agent="codex",
            interact=False,
            verbose=False,
            stream_output=True,
            remote=None,
            remote_workdir=None,
            min_rounds=5,
            resume="auto",
            reset_optimize=False,
            no_agent_session=False,
            round_mode="invalid",
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
        self.assertEqual(args.concurrency, 1)
        self.assertEqual(args.agent, "codex")
        self.assertFalse(hasattr(args, "interact"))
        self.assertTrue(args.stream_output)

    def test_optimize_batch_accepts_operator_filter(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["optimize-batch", "-i", "kernels", "--operator-filter", "kernel*.py"]
        )
        self.assertEqual(args.operator_filter, "kernel*.py")

    def test_optimize_batch_accepts_round_modes(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize-batch", "-i", "kernels", "--round-mode", "supervised"])
        self.assertEqual(args.round_mode, "supervised")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.round_mode, "supervised")

    def test_optimize_batch_accepts_user_prompt(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["optimize-batch", "-i", "kernels", "--prompt", "Avoid numerics changes."]
        )
        self.assertEqual(args.prompt, "Avoid numerics changes.")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.prompt, "Avoid numerics changes.")

    def test_optimize_batch_accepts_post_optimize_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["optimize-batch", "-i", "kernels", "--post-optimize-command", "echo done"]
        )
        self.assertEqual(args.post_optimize_command, "echo done")
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.post_optimize_command, "echo done")

    def test_optimize_batch_defaults_round_batch_size_to_five(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize-batch", "-i", "kernels"])
        self.assertEqual(args.round_mode, "checked")
        self.assertEqual(args.round_batch_size, 5)
        options = optimize_run_options_from_args(args)
        self.assertEqual(options.round_mode, "checked")
        self.assertEqual(options.round_batch_size, 5)
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

    def test_status_defaults_input_to_current_directory(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["status"])
        self.assertEqual(args.input, ".")

    def test_status_accepts_markdown_format(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["status", "-i", "kernels", "--format", "markdown"])
        self.assertEqual(args.format, "markdown")

    def test_status_accepts_json_format(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["status", "-i", "kernels", "--format", "json"])
        self.assertEqual(args.format, "json")

    def test_status_accepts_html_format(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["status", "-i", "kernels", "--format", "html"])
        self.assertEqual(args.format, "html")

    def test_status_accepts_view_option(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["status", "-i", "kernels", "--view", "trend"])
        self.assertEqual(args.command_kind, CommandKind.STATUS)
        self.assertEqual(args.view, "trend")

    def test_status_defaults_to_best_view(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["status", "-i", "kernels"])
        self.assertEqual(args.view, "best")

    def test_status_accepts_metric_source_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["status", "-i", "kernels", "--metric-source", "total-op"])
        self.assertEqual(args.metric_source, "total-op")

    def test_status_accepts_metric_source_short_alias(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["status", "-i", "kernels", "-m", "kernel"])
        self.assertEqual(args.metric_source, "kernel")

    def test_status_defaults_metric_source_to_none(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["status", "-i", "kernels"])
        self.assertIsNone(args.metric_source)

    def test_status_rejects_metric_source_all(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["status", "-i", "kernels", "--metric-source", "all"])

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
                "--min-round",
                "4",
                "--resume",
                "continue",
                "--target-chip",
                "A3",
                "--no-agent-session",
                "--concurrency",
                "3",
                "--no-stream-output",
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
        self.assertEqual(args.concurrency, 3)
        self.assertFalse(args.stream_output)

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

    def test_upload_optimize_command_parses_input_and_verbose(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["upload-optimize", "-i", "workspace-root", "--verbose"]
        )
        self.assertEqual(args.command, "upload-optimize")
        self.assertEqual(args.input, "workspace-root")
        self.assertTrue(args.verbose)

    def test_clean_maps_to_command_kind(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["clean", "-i", "workspace"])
        self.assertEqual(args.command, "clean")
        self.assertEqual(args.command_kind, CommandKind.CLEAN)
        self.assertFalse(args.deep)
        self.assertFalse(hasattr(args, "agent"))

    def test_clean_defaults_input_to_current_directory(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["clean"])
        self.assertEqual(args.input, ".")

    def test_clean_accepts_deep_option(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["clean", "-i", "workspace", "--deep", "--verbose"])
        self.assertTrue(args.deep)
        self.assertTrue(args.verbose)

    def test_optimize_command_defaults_upload_enabled(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py"])
        options = optimize_run_options_from_args(args)
        self.assertTrue(options.upload_enabled)

    def test_optimize_command_accepts_no_upload(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--no-upload"])
        options = optimize_run_options_from_args(args)
        self.assertFalse(options.upload_enabled)

    def test_optimize_batch_accepts_no_upload(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["optimize-batch", "-i", "workspace-root", "--no-upload"]
        )
        options = optimize_run_options_from_args(args)
        self.assertFalse(options.upload_enabled)


class PathResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._upload_patches = [
            patch(
                "triton_agent.commands.optimize.upload_optimize_workspace",
                return_value=None,
            ),
            patch(
                "triton_agent.optimize.batch.upload_optimize_workspace",
                return_value=None,
            ),
        ]
        for p in self._upload_patches:
            p.start()

    def tearDown(self) -> None:
        for p in self._upload_patches:
            p.stop()
        super().tearDown()

    def _write_round_state(
        self,
        round_dir: Path,
        *,
        perf_artifact: str,
        correctness_status: str = "passed",
        benchmark_status: str = "passed",
        effective_metric_source: str = "kernel",
    ) -> None:
        (round_dir / "round-state.json").write_text(
            json.dumps(
                {
                    "round": round_dir.name,
                    "parent_round": "baseline",
                    "hypothesis": "test round",
                    "evidence_sources": ["benchmark"],
                    "correctness_status": correctness_status,
                    "benchmark_status": benchmark_status,
                    "perf_artifact": perf_artifact,
                    "comparison_target_path": "../baseline/perf.txt",
                    "effective_metric_source": effective_metric_source,
                    "summary_path": "summary.md",
                    "opt_note_updated": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )

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

    def test_main_status_skips_hidden_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "visible").mkdir()
            (root / ".hidden").mkdir()
            (root / ".hidden" / "kernel_perf.txt").write_text(
                "latency-a: 10\n", encoding="utf-8"
            )
            stdout = StringIO()

            with redirect_stdout(stdout):
                exit_code = main(["status", "-i", str(root)])

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertIn("[NO-SESSION] visible", rendered)
            self.assertNotIn("hidden", rendered)

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
            self._write_round_state(round_one, perf_artifact="opt_kernel_perf.txt")

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["status", "-i", str(workspace)])

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertIn(f"[OK] {workspace.name}", rendered)
            self.assertIn("Best round: round-1", rendered)
            self.assertNotIn("[NO-SESSION] opt-round-1", rendered)

    def test_main_status_honors_metric_source_override_for_batch_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "kernel_case"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            baseline_dir = workspace / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "perf.txt").write_text(
                "\n".join(
                    [
                        "latency-a: 10",
                        '# raw-op-statistic-a: {"ops":[{"op_type":"OpA","avg_time_us":50.0}]}',
                        "latency-b: 10",
                        '# raw-op-statistic-b: {"ops":[{"op_type":"OpB","avg_time_us":50.0}]}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
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
                )
                + "\n",
                encoding="utf-8",
            )
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            round_one = workspace / "opt-round-1"
            round_two = workspace / "opt-round-2"
            round_one.mkdir()
            round_two.mkdir()
            (round_one / "opt_kernel_perf.txt").write_text(
                "\n".join(
                    [
                        "latency-a: 5",
                        '# raw-op-statistic-a: {"ops":[{"op_type":"OpA","avg_time_us":80.0}]}',
                        "latency-b: 5",
                        '# raw-op-statistic-b: {"ops":[{"op_type":"OpB","avg_time_us":80.0}]}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (round_two / "opt_kernel_perf.txt").write_text(
                "\n".join(
                    [
                        "latency-a: 8",
                        '# raw-op-statistic-a: {"ops":[{"op_type":"OpA","avg_time_us":40.0}]}',
                        "latency-b: 8",
                        '# raw-op-statistic-b: {"ops":[{"op_type":"OpB","avg_time_us":40.0}]}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            self._write_round_state(
                round_one,
                perf_artifact="opt_kernel_perf.txt",
                effective_metric_source="kernel",
            )
            self._write_round_state(
                round_two,
                perf_artifact="opt_kernel_perf.txt",
                effective_metric_source="kernel",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["status", "-i", str(root), "--metric-source", "total-op"])

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertIn("[OK] kernel_case", rendered)
            self.assertIn("Best round: round-2", rendered)

    def test_main_status_defaults_input_to_current_directory(self) -> None:
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
            self._write_round_state(round_one, perf_artifact="opt_kernel_perf.txt")

            stdout = StringIO()
            previous_cwd = os.getcwd()
            try:
                os.chdir(workspace)
                with redirect_stdout(stdout):
                    exit_code = main(["status"])
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertIn(f"[OK] {workspace.name}", rendered)
            self.assertIn("Best round: round-1", rendered)

    def test_main_clean_single_workspace_preserves_cases_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            operator.write_text("print('source')\n", encoding="utf-8")
            test_harness = workspace / "test_kernel.py"
            test_harness.write_text("# test-mode: standalone\n", encoding="utf-8")
            bench_harness = workspace / "bench_kernel.py"
            bench_harness.write_text("# bench-mode: torch-npu-profiler\n", encoding="utf-8")
            generated_operator = workspace / "triton_kernel.py"
            generated_operator.write_text("print('gen')\n", encoding="utf-8")
            extra_info = workspace / "extra-info.json"
            extra_info.write_text("{}", encoding="utf-8")
            prof_dir = workspace / "PROF_demo"
            prof_dir.mkdir()

            exit_code = main(["clean", "-i", str(workspace)])

            self.assertEqual(exit_code, 0)
            self.assertTrue(operator.exists())
            self.assertTrue(test_harness.exists())
            self.assertTrue(bench_harness.exists())
            self.assertFalse(generated_operator.exists())
            self.assertFalse(extra_info.exists())
            self.assertFalse(prof_dir.exists())

    def test_main_clean_defaults_input_to_current_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            operator.write_text("print('source')\n", encoding="utf-8")
            generated_operator = workspace / "triton_kernel.py"
            generated_operator.write_text("print('gen')\n", encoding="utf-8")

            previous_cwd = os.getcwd()
            try:
                os.chdir(workspace)
                exit_code = main(["clean"])
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(exit_code, 0)
            self.assertTrue(operator.exists())
            self.assertFalse(generated_operator.exists())

    def test_main_clean_single_workspace_deep_removes_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            operator.write_text("print('source')\n", encoding="utf-8")
            test_harness = workspace / "differential_test_kernel.py"
            test_harness.write_text("# test-mode: differential\n", encoding="utf-8")
            bench_harness = workspace / "bench_kernel.py"
            bench_harness.write_text("# bench-mode: msprof\n", encoding="utf-8")

            exit_code = main(["clean", "-i", str(workspace), "--deep"])

            self.assertEqual(exit_code, 0)
            self.assertTrue(operator.exists())
            self.assertFalse(test_harness.exists())
            self.assertFalse(bench_harness.exists())

    def test_main_clean_batch_root_removes_batch_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "case-a"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            (workspace / "opt_kernel.py").write_text("print('opt')\n", encoding="utf-8")
            batch_status = root / "optimize-batch-status.json"
            batch_status.write_text("{}", encoding="utf-8")
            batch_summary = root / "log_check_summary.md"
            batch_summary.write_text("# summary\n", encoding="utf-8")
            batch_state = root / "report-batch-state.json"
            batch_state.write_text("{}", encoding="utf-8")
            batch_report = root / "report-batch.md"
            batch_report.write_text("# report\n", encoding="utf-8")

            exit_code = main(["clean", "-i", str(root)])

            self.assertEqual(exit_code, 0)
            self.assertFalse((workspace / "opt_kernel.py").exists())
            self.assertFalse(batch_status.exists())
            self.assertFalse(batch_summary.exists())
            self.assertFalse(batch_state.exists())
            self.assertFalse(batch_report.exists())

    def test_main_clean_batch_root_without_children_reports_status_style_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stderr = StringIO()

            with redirect_stderr(stderr):
                exit_code = main(["clean", "-i", str(root)])

            self.assertEqual(exit_code, 1)
            self.assertIn("No operator workspaces found under", stderr.getvalue())

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
            self._write_round_state(ok_round, perf_artifact="opt_kernel_perf.txt")

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
            self._write_round_state(round_one, perf_artifact="opt_kernel_perf.txt")
            self._write_round_state(round_two, perf_artifact="opt_kernel_perf.txt")

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["status", "-i", str(root)])

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertIn("[OK] matmul", rendered)
            self.assertIn("Avg improvement: +30.0%", rendered)
            self.assertIn("Geomean speedup: 1.49x", rendered)
            self.assertIn("Best round: round-2", rendered)
            self.assertIn("Logged best: round-1", rendered)
            self.assertIn(
                "Warning: numeric best round != logged best. "
                "computed speedup: 1.49x; "
                "logged speedup: 1.16x",
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
            self._write_round_state(round_one, perf_artifact="opt_kernel_perf.txt")
            self._write_round_state(round_two, perf_artifact="opt_kernel_perf.txt")

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
            self._write_round_state(round_one, perf_artifact="opt_kernel_perf.txt")

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["status", "-i", str(root)])

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertIn("[WARN] layernorm", rendered)
            self.assertIn("Avg improvement: unknown", rendered)
            self.assertIn("Geomean speedup: unknown", rendered)
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
            self._write_round_state(round_one, perf_artifact="opt_kernel_perf.txt")

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["status", "-i", str(root)])

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertIn("[OK] matmul", rendered)
            self.assertIn("Best round: round-1", rendered)
            self.assertNotIn("Warning: found multiple baseline perf files", rendered)

    def test_main_status_renders_best_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fresh").mkdir()

            workspace = root / "matmul"
            workspace.mkdir()
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
            self._write_round_state(round_one, perf_artifact="opt_kernel_perf.txt")

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["status", "-i", str(root), "--format", "json"])

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual([item["name"] for item in payload["operators"]], ["fresh", "matmul"])
            self.assertEqual(payload["operators"][0]["state"], "no-session")
            self.assertEqual(payload["operators"][1]["state"], "ok")
            self.assertEqual(payload["operators"][1]["best_round"], "round-1")
            self.assertAlmostEqual(payload["operators"][1]["geomean_speedup"], 1.25)

    def test_main_status_renders_trend_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fresh").mkdir()

            alpha = root / "alpha"
            alpha.mkdir()
            (alpha / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            (alpha / "kernel_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            alpha_round = alpha / "opt-round-2"
            alpha_round.mkdir()
            (alpha_round / "opt_kernel_perf.txt").write_text(
                "latency-a: 5\nlatency-b: 10\n",
                encoding="utf-8",
            )
            self._write_round_state(alpha_round, perf_artifact="opt_kernel_perf.txt")

            beta = root / "beta"
            beta.mkdir()
            (beta / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            (beta / "kernel_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            beta_round_one = beta / "opt-round-1"
            beta_round_three = beta / "opt-round-3"
            beta_round_one.mkdir()
            beta_round_three.mkdir()
            (beta_round_one / "opt_kernel_perf.txt").write_text(
                "latency-a: 8\nlatency-b: 16\n",
                encoding="utf-8",
            )
            (beta_round_three / "opt_kernel_perf.txt").write_text(
                "latency-a: 4\nlatency-b: 8\n",
                encoding="utf-8",
            )
            self._write_round_state(beta_round_one, perf_artifact="opt_kernel_perf.txt")
            self._write_round_state(beta_round_three, perf_artifact="opt_kernel_perf.txt")

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["status", "-i", str(root), "--view", "trend", "--format", "markdown"])

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertIn("| Name | round-1 | round-2 | round-3 |", rendered)
            self.assertIn("| alpha | - | 2.00x | - |", rendered)
            self.assertIn("| beta | 1.25x | - | 2.50x |", rendered)
            self.assertNotIn("fresh", rendered)

    def test_main_status_renders_trend_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fresh").mkdir()

            alpha = root / "alpha"
            alpha.mkdir()
            (alpha / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            (alpha / "kernel_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            alpha_round = alpha / "opt-round-2"
            alpha_round.mkdir()
            (alpha_round / "opt_kernel_perf.txt").write_text(
                "latency-a: 5\nlatency-b: 10\n",
                encoding="utf-8",
            )
            self._write_round_state(alpha_round, perf_artifact="opt_kernel_perf.txt")

            beta = root / "beta"
            beta.mkdir()
            (beta / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            (beta / "kernel_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            beta_round_one = beta / "opt-round-1"
            beta_round_three = beta / "opt-round-3"
            beta_round_one.mkdir()
            beta_round_three.mkdir()
            (beta_round_one / "opt_kernel_perf.txt").write_text(
                "latency-a: 8\nlatency-b: 16\n",
                encoding="utf-8",
            )
            (beta_round_three / "opt_kernel_perf.txt").write_text(
                "latency-a: 4\nlatency-b: 8\n",
                encoding="utf-8",
            )
            self._write_round_state(beta_round_one, perf_artifact="opt_kernel_perf.txt")
            self._write_round_state(beta_round_three, perf_artifact="opt_kernel_perf.txt")

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["status", "-i", str(root), "--view", "trend", "--format", "json"])

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(
                payload["operators"],
                [
                    {
                        "name": "alpha",
                        "round_speedups": {
                            "round-1": None,
                            "round-2": 2.0,
                            "round-3": None,
                        },
                    },
                    {
                        "name": "beta",
                        "round_speedups": {
                            "round-1": 1.25,
                            "round-2": None,
                            "round-3": 2.5,
                        },
                    },
                ],
            )

    def test_main_status_renders_trend_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "matmul"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            (workspace / "kernel_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            round_one = workspace / "opt-round-1"
            round_two = workspace / "opt-round-2"
            round_one.mkdir()
            round_two.mkdir()
            (round_one / "opt_kernel_perf.txt").write_text(
                "latency-a: 8\nlatency-b: 16\n",
                encoding="utf-8",
            )
            (round_two / "opt_kernel_perf.txt").write_text(
                "latency-a: 5\nlatency-b: 10\n",
                encoding="utf-8",
            )
            self._write_round_state(round_one, perf_artifact="opt_kernel_perf.txt")
            self._write_round_state(round_two, perf_artifact="opt_kernel_perf.txt")

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["status", "-i", str(root), "--view", "trend", "--format", "html"])

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertIn("<!doctype html>", rendered.lower())
            self.assertIn("Operator Speedup Trends", rendered)
            self.assertIn("matmul", rendered)
            self.assertIn("<svg", rendered)

    def test_main_status_rejects_best_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "matmul"
            workspace.mkdir()
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
            self._write_round_state(round_one, perf_artifact="opt_kernel_perf.txt")

            stderr = StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exc:
                    main(["status", "-i", str(root), "--format", "html"])

            self.assertEqual(exc.exception.code, 2)
            self.assertIn("HTML format only supports --view trend", stderr.getvalue())

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
            self._write_round_state(ok_round, perf_artifact="opt_kernel_perf.txt")
            self._write_round_state(best_round, perf_artifact="opt_kernel_perf.txt")
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
                        '      "geomean_speedup": 1.22',
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
                "| 名称 | Geomean speedup | Verified | "
                "Verified Geomean speedup | Notes |",
                rendered,
            )
            self.assertIn("| beta | 1.49x | Verified | 1.22x | best≠log |", rendered)
            self.assertIn("| zeta | - | - |  | warn |", rendered)
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

            def _fake_run(request, stdout=None, stderr=None):
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

            def _fake_run(request, stdout=None, stderr=None):
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

            def _fake_run(request, stdout=None, stderr=None):
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

            def _fake_run(request, stdout=None, stderr=None):
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

    def test_main_optimize_batch_ignores_hidden_triton_agent_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good"
            hidden_triton_agent_dir = root / ".triton-agent"
            good.mkdir()
            hidden_triton_agent_dir.mkdir()
            (good / "kernel.py").write_text("print('ok')\n", encoding="utf-8")
            (hidden_triton_agent_dir / "round-brief.md").write_text("Pending\n", encoding="utf-8")

            stdout = StringIO()
            seen_inputs: list[Path] = []

            def _fake_run(request, stdout=None, stderr=None):
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

            def _fake_run(request, stdout=None, stderr=None):
                seen_inputs.append(request.input_path)
                return AgentResult(return_code=0, stdout="", stderr="")

            with patch("triton_agent.optimize.batch.run_optimize_request", side_effect=_fake_run):
                with redirect_stdout(stdout):
                    exit_code = main(["optimize-batch", "-i", str(root)])

            self.assertEqual(exit_code, 0)
            self.assertEqual([path.parent.name for path in seen_inputs], ["beta"])
            self.assertIn("[SKIP] alpha: already completed", stdout.getvalue())
            self.assertIn("Summary: 1 succeeded, 0 failed, 1 skipped", stdout.getvalue())

    def test_main_optimize_batch_honors_concurrency(self) -> None:
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

            with patch("triton_agent.optimize.batch.run_optimize_request", side_effect=_fake_run):
                exit_code = main(["optimize-batch", "-i", str(root), "--concurrency", "2"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(max_active, 2)

    def test_main_optimize_batch_resolves_max_concurrency_keyword(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captured: dict[str, object] = {}

            def _fake_run_optimize_batch(root, options, max_concurrency, operator_filter=None):
                del root, options
                self.assertIsNone(operator_filter)
                captured["max_concurrency"] = max_concurrency
                return 0

            with patch.dict(
                os.environ,
                {
                    "TRITON_AGENT_BATCH_NPU_DEVICES": "0,1",
                    "TRITON_AGENT_BATCH_WORKERS_PER_NPU": "2",
                },
                clear=False,
            ):
                with patch(
                    "triton_agent.commands.optimize.run_optimize_batch",
                    side_effect=_fake_run_optimize_batch,
                ):
                    exit_code = main(
                        ["optimize-batch", "-i", str(root), "--concurrency", "max"]
                    )

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["max_concurrency"], 4)

    def test_main_optimize_batch_mcp_max_concurrency_ignores_workers_per_npu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captured: dict[str, object] = {}

            def _fake_run_optimize_batch(root, options, max_concurrency, operator_filter=None):
                del root, options, operator_filter
                captured["max_concurrency"] = max_concurrency
                return 0

            with patch(
                "triton_agent.commands.optimize.run_optimize_batch",
                side_effect=_fake_run_optimize_batch,
            ):
                exit_code = main(
                    [
                        "optimize-batch",
                        "-i",
                        str(root),
                        "--concurrency",
                        "max",
                        "--enable-mcp",
                        "--npu-devices",
                        "0,1",
                        "--workers-per-npu",
                        "2",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["max_concurrency"], 2)

    def test_main_prefers_explicit_batch_affinity_options_over_legacy_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captured: dict[str, object] = {}

            def _fake_handler(parser, args):
                del parser
                captured["npu_devices"] = args.npu_devices
                captured["workers_per_npu"] = args.workers_per_npu
                return 0

            original = cli_module._COMMAND_SPECS[CommandKind.OPTIMIZE_BATCH]
            patched = cli_module._CommandSpec(
                handler=_fake_handler,
                help_group=original.help_group,
                help_summary=original.help_summary,
                description=original.description,
                input_mode=original.input_mode,
                input_default=original.input_default,
                has_output=original.has_output,
                has_verbose=original.has_verbose,
                has_remote=original.has_remote,
                keep_remote_workdir=original.keep_remote_workdir,
                has_agent=original.has_agent,
                agent_default=original.agent_default,
                has_interact=original.has_interact,
                has_show_output=original.has_show_output,
                has_test_mode=original.has_test_mode,
                test_mode_default=original.test_mode_default,
                test_mode_choices=original.test_mode_choices,
                has_bench_mode=original.has_bench_mode,
                bench_mode_default=original.bench_mode_default,
                has_npu_devices=original.has_npu_devices,
                has_batch_affinity=original.has_batch_affinity,
                has_optimize_options=original.has_optimize_options,
                has_prompt=original.has_prompt,
                concurrency_default=original.concurrency_default,
                optional_concurrency=original.optional_concurrency,
                concurrency_accepts_max=original.concurrency_accepts_max,
                has_force_overwrite=original.has_force_overwrite,
                has_format=original.has_format,
                has_language=original.has_language,
                has_verify_phase=original.has_verify_phase,
                has_force_verify=original.has_force_verify,
                has_log_tools=original.has_log_tools,
                has_url=original.has_url,
                has_distill_options=original.has_distill_options,
                has_operator_filter=original.has_operator_filter,
            )

            with patch.dict(
                os.environ,
                {
                    "TRITON_AGENT_BATCH_NPU_DEVICES": "4,5",
                    "TRITON_AGENT_BATCH_WORKERS_PER_NPU": "9",
                },
                clear=False,
            ):
                with patch.dict(
                    cli_module._COMMAND_SPECS,
                    {CommandKind.OPTIMIZE_BATCH: patched},
                    clear=False,
                ):
                    exit_code = main(
                        [
                            "optimize-batch",
                            "-i",
                            str(root),
                            "--npu-devices",
                            "0,1",
                            "--workers-per-npu",
                            "2",
                        ]
                    )

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["npu_devices"], "0,1")
            self.assertEqual(captured["workers_per_npu"], "2")

    def test_main_uses_legacy_env_when_batch_affinity_options_are_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captured: dict[str, object] = {}

            def _fake_handler(parser, args):
                del parser
                captured["npu_devices"] = args.npu_devices
                captured["workers_per_npu"] = args.workers_per_npu
                return 0

            original = cli_module._COMMAND_SPECS[CommandKind.OPTIMIZE_BATCH]
            patched = cli_module._CommandSpec(
                handler=_fake_handler,
                help_group=original.help_group,
                help_summary=original.help_summary,
                description=original.description,
                input_mode=original.input_mode,
                input_default=original.input_default,
                has_output=original.has_output,
                has_verbose=original.has_verbose,
                has_remote=original.has_remote,
                keep_remote_workdir=original.keep_remote_workdir,
                has_agent=original.has_agent,
                agent_default=original.agent_default,
                has_interact=original.has_interact,
                has_show_output=original.has_show_output,
                has_test_mode=original.has_test_mode,
                test_mode_default=original.test_mode_default,
                test_mode_choices=original.test_mode_choices,
                has_bench_mode=original.has_bench_mode,
                bench_mode_default=original.bench_mode_default,
                has_npu_devices=original.has_npu_devices,
                has_batch_affinity=original.has_batch_affinity,
                has_optimize_options=original.has_optimize_options,
                has_prompt=original.has_prompt,
                concurrency_default=original.concurrency_default,
                optional_concurrency=original.optional_concurrency,
                concurrency_accepts_max=original.concurrency_accepts_max,
                has_force_overwrite=original.has_force_overwrite,
                has_format=original.has_format,
                has_language=original.has_language,
                has_verify_phase=original.has_verify_phase,
                has_force_verify=original.has_force_verify,
                has_log_tools=original.has_log_tools,
                has_url=original.has_url,
                has_distill_options=original.has_distill_options,
                has_operator_filter=original.has_operator_filter,
            )

            with patch.dict(
                os.environ,
                {
                    "TRITON_AGENT_BATCH_NPU_DEVICES": "0,1",
                    "TRITON_AGENT_BATCH_WORKERS_PER_NPU": "2",
                },
                clear=False,
            ):
                with patch.dict(
                    cli_module._COMMAND_SPECS,
                    {CommandKind.OPTIMIZE_BATCH: patched},
                    clear=False,
                ):
                    exit_code = main(["optimize-batch", "-i", str(root)])

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["npu_devices"], "0,1")
            self.assertEqual(captured["workers_per_npu"], "2")

    def test_main_optimize_batch_forwards_operator_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captured: dict[str, object] = {}

            def _fake_run_optimize_batch(root, options, max_concurrency, operator_filter=None):
                del options
                captured["root"] = root
                captured["max_concurrency"] = max_concurrency
                captured["operator_filter"] = operator_filter
                return 0

            with patch(
                "triton_agent.commands.optimize.run_optimize_batch",
                side_effect=_fake_run_optimize_batch,
            ):
                exit_code = main(
                    ["optimize-batch", "-i", str(root), "--operator-filter", "kernel*.py"]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["root"], root.resolve())
            self.assertEqual(captured["max_concurrency"], 1)
            self.assertEqual(captured["operator_filter"], "kernel*.py")

    def test_main_optimize_with_explicit_concurrency_uses_batch_handler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captured: dict[str, object] = {}

            def _fake_run_optimize_batch(root, options, max_concurrency, operator_filter=None):
                captured["root"] = root
                captured["max_concurrency"] = max_concurrency
                captured["operator_filter"] = operator_filter
                captured["post_optimize_command"] = options.post_optimize_command
                return 0

            with patch(
                "triton_agent.commands.optimize.run_optimize_batch",
                side_effect=_fake_run_optimize_batch,
            ):
                exit_code = main(
                    [
                        "optimize",
                        "-i",
                        str(root),
                        "--concurrency",
                        "2",
                        "--operator-filter",
                        "kernel*.py",
                        "--post-optimize-command",
                        "echo done",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["root"], root.resolve())
            self.assertEqual(captured["max_concurrency"], 2)
            self.assertEqual(captured["operator_filter"], "kernel*.py")
            self.assertEqual(captured["post_optimize_command"], "echo done")

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
                    exit_code = main(["optimize-batch", "-i", str(root)])

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
            (resumable / "differential_test_kernel.py").write_text(
                "# test-mode: differential\nprint('test')\n", encoding="utf-8"
            )
            (resumable / "bench_kernel.py").write_text(
                "# bench-mode: msprof\n# kernel: k\nprint('bench')\n", encoding="utf-8"
            )

            captured_modes: dict[str, Optional[str]] = {}

            def _fake_run(request, stdout=None, stderr=None):
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
            self.assertEqual(captured_modes["resume_ws"], "msprof")
            self.assertEqual(captured_modes["fresh_ws"], "msprof")

    def test_main_optimize_batch_auto_resumable_mismatch_fails_per_workspace(self) -> None:
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
            (resumable / "differential_test_kernel.py").write_text(
                "# test-mode: differential\nprint('test')\n", encoding="utf-8"
            )
            (resumable / "bench_kernel.py").write_text(
                "# bench-mode: msprof\n# kernel: k\nprint('bench')\n", encoding="utf-8"
            )

            captured_modes: dict[str, Optional[str]] = {}

            def _fake_run(request, stdout=None, stderr=None):
                captured_modes[request.workdir.name] = request.bench_mode
                return AgentResult(return_code=0, stdout="", stderr="")

            stdout = StringIO()
            with patch("triton_agent.optimize.batch.run_optimize_request", side_effect=_fake_run):
                with redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "optimize-batch",
                            "-i",
                            str(root),
                            "--resume",
                            "auto",
                            "--bench-mode",
                            "torch-npu-profiler",
                        ]
                    )

            rendered = stdout.getvalue()
            self.assertEqual(exit_code, 1)
            self.assertIn("[FAIL] resume_ws", rendered)
            self.assertIn("conflicts with existing harness bench-mode msprof", rendered)
            self.assertIn("[OK] fresh_ws", rendered)
            self.assertEqual(captured_modes.get("fresh_ws"), "torch-npu-profiler")
            self.assertNotIn("resume_ws", captured_modes)
            self.assertIn("Summary: 1 succeeded, 1 failed", rendered)

    def test_main_optimize_batch_rejects_invalid_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stderr = StringIO()

            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exc:
                    main(["optimize-batch", "-i", str(root), "--concurrency", "0"])

            self.assertEqual(exc.exception.code, 2)
            self.assertIn("--concurrency must be at least 1", stderr.getvalue())

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

    def test_main_gen_eval_batch_honors_concurrency(self) -> None:
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
                exit_code = main(["gen-eval-batch", "-i", str(root), "--concurrency", "2"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(max_active, 2)

    def test_main_gen_eval_batch_forwards_operator_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captured: dict[str, object] = {}

            def _fake_run(root_path, options, max_concurrency, operator_filter=None):
                del options
                captured["root"] = root_path
                captured["max_concurrency"] = max_concurrency
                captured["operator_filter"] = operator_filter
                return 0

            with patch(
                "triton_agent.commands.generation.run_gen_eval_batch",
                side_effect=_fake_run,
            ):
                exit_code = main(
                    ["gen-eval-batch", "-i", str(root), "--operator-filter", "kernel*.py"]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["root"], root.resolve())
            self.assertEqual(captured["max_concurrency"], 1)
            self.assertEqual(captured["operator_filter"], "kernel*.py")

    def test_main_gen_eval_with_explicit_concurrency_uses_batch_handler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captured: dict[str, object] = {}

            def _fake_run(root_path, options, max_concurrency, operator_filter=None):
                del options
                captured["root"] = root_path
                captured["max_concurrency"] = max_concurrency
                captured["operator_filter"] = operator_filter
                return 0

            with patch(
                "triton_agent.commands.generation.run_gen_eval_batch",
                side_effect=_fake_run,
            ):
                exit_code = main(
                    ["gen-eval", "-i", str(root), "--concurrency", "2", "--operator-filter", "kernel*.py"]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["root"], root.resolve())
            self.assertEqual(captured["max_concurrency"], 2)
            self.assertEqual(captured["operator_filter"], "kernel*.py")

    def test_main_gen_eval_batch_mcp_max_concurrency_ignores_workers_per_npu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captured: dict[str, object] = {}

            def _fake_run(root_path, options, max_concurrency, operator_filter=None):
                del root_path, options, operator_filter
                captured["max_concurrency"] = max_concurrency
                return 0

            with patch(
                "triton_agent.commands.generation.run_gen_eval_batch",
                side_effect=_fake_run,
            ):
                exit_code = main(
                    [
                        "gen-eval-batch",
                        "-i",
                        str(root),
                        "--concurrency",
                        "max",
                        "--enable-mcp",
                        "--npu-devices",
                        "0,1",
                        "--workers-per-npu",
                        "2",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["max_concurrency"], 2)

    def test_main_convert_batch_forwards_operator_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captured: dict[str, object] = {}

            def _fake_run(root_path, options, max_concurrency, operator_filter=None):
                del options
                captured["root"] = root_path
                captured["max_concurrency"] = max_concurrency
                captured["operator_filter"] = operator_filter
                return 0

            with patch(
                "triton_agent.commands.convert.run_convert_batch",
                side_effect=_fake_run,
            ):
                exit_code = main(
                    ["convert-batch", "-i", str(root), "--operator-filter", "kernel*.py"]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["root"], root.resolve())
            self.assertEqual(captured["max_concurrency"], 1)
            self.assertEqual(captured["operator_filter"], "kernel*.py")

    def test_main_convert_with_explicit_concurrency_uses_batch_handler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captured: dict[str, object] = {}

            def _fake_run(root_path, options, max_concurrency, operator_filter=None):
                del options
                captured["root"] = root_path
                captured["max_concurrency"] = max_concurrency
                captured["operator_filter"] = operator_filter
                return 0

            with patch(
                "triton_agent.commands.convert.run_convert_batch",
                side_effect=_fake_run,
            ):
                exit_code = main(
                    ["convert", "-i", str(root), "--concurrency", "2", "--operator-filter", "kernel*.py"]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["root"], root.resolve())
            self.assertEqual(captured["max_concurrency"], 2)
            self.assertEqual(captured["operator_filter"], "kernel*.py")

    def test_main_convert_batch_mcp_max_concurrency_ignores_workers_per_npu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captured: dict[str, object] = {}

            def _fake_run(root_path, options, max_concurrency, operator_filter=None):
                del root_path, options, operator_filter
                captured["max_concurrency"] = max_concurrency
                return 0

            with patch(
                "triton_agent.commands.convert.run_convert_batch",
                side_effect=_fake_run,
            ):
                exit_code = main(
                    [
                        "convert-batch",
                        "-i",
                        str(root),
                        "--concurrency",
                        "max",
                        "--enable-mcp",
                        "--npu-devices",
                        "0,1",
                        "--workers-per-npu",
                        "2",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["max_concurrency"], 2)

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
                    exit_code = main(["gen-eval-batch", "-i", str(root)])

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
                    main(["gen-eval-batch", "-i", str(root), "--concurrency", "0"])

            self.assertEqual(exc.exception.code, 2)
            self.assertIn("--concurrency must be at least 1", stderr.getvalue())

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
                accuracy_mode="npu-contract",
                verbose=False,
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
                accuracy_mode="npu-contract",
                verbose=False,
            )

    def test_main_sets_remote_env_before_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "test_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: standalone\nprint('test')\n", encoding="utf-8")

            seen_env: dict[str, Optional[str]] = {}

            def fake_run_remote_test(*args, **kwargs):
                del args, kwargs
                seen_env["remote"] = os.environ.get(remote_target_env_name())
                seen_env["remote_workdir"] = os.environ.get(remote_workdir_env_name())
                return AgentResult(return_code=0, stdout="", stderr=""), None, "/tmp/triton-agent-123"

            with patch.object(cli_module, "ensure_remote_ssh_ready", return_value=None), patch(
                "triton_agent.commands.execution.run_remote_test",
                side_effect=fake_run_remote_test,
            ):
                exit_code = main(
                    [
                        "run-test",
                        "--test-file",
                        str(test_file),
                        "--operator-file",
                        str(operator),
                        "--remote",
                        "alice@example.com",
                        "--remote-workdir",
                        "/tmp/triton-agent",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(seen_env["remote"], "alice@example.com")
            self.assertEqual(seen_env["remote_workdir"], "/tmp/triton-agent")

    def test_main_clears_stale_remote_workdir_when_remote_has_no_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "test_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: standalone\nprint('test')\n", encoding="utf-8")

            seen_env: dict[str, Optional[str]] = {}

            def fake_run_remote_test(*args, **kwargs):
                del args, kwargs
                seen_env["remote"] = os.environ.get(remote_target_env_name())
                seen_env["remote_workdir"] = os.environ.get(remote_workdir_env_name())
                return AgentResult(return_code=0, stdout="", stderr=""), None, "/tmp/triton-agent-123"

            with patch.dict(
                os.environ,
                {remote_workdir_env_name(): "/tmp/stale-workdir"},
                clear=False,
            ):
                with patch.object(cli_module, "ensure_remote_ssh_ready", return_value=None), patch(
                    "triton_agent.commands.execution.run_remote_test",
                    side_effect=fake_run_remote_test,
                ):
                    exit_code = main(
                        [
                            "run-test",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator),
                            "--remote",
                            "alice@example.com",
                        ]
                    )

            self.assertEqual(exit_code, 0)
            self.assertEqual(seen_env["remote"], "alice@example.com")
            self.assertIsNone(seen_env["remote_workdir"])

    def test_main_clears_stale_remote_env_for_non_remote_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(
                os.environ,
                {
                    remote_target_env_name(): "alice@example.com",
                    remote_workdir_env_name(): "/tmp/stale-workdir",
                },
                clear=False,
            ):
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as stderr:
                    exit_code = main(["status", "-i", str(root)])

            self.assertEqual(exit_code, 1)
            self.assertNotIn(remote_target_env_name(), os.environ)
            self.assertNotIn(remote_workdir_env_name(), os.environ)
            self.assertIn("No operator workspaces found", stderr.getvalue())

    def test_main_explicit_remote_runs_ssh_preflight_before_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "test_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: standalone\nprint('test')\n", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch.object(cli_module, "ensure_remote_ssh_ready") as preflight, patch(
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
                    ]
                )

        self.assertEqual(exit_code, 0)
        preflight.assert_called_once_with("alice@example.com:2200")
        mocked.assert_called_once()

    def test_main_explicit_remote_preflight_failure_returns_1_and_skips_handler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "test_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: standalone\nprint('test')\n", encoding="utf-8")

            stderr = StringIO()
            with patch.object(
                cli_module,
                "ensure_remote_ssh_ready",
                side_effect=RuntimeError(
                    "Run `ssh-copy-id alice@example.com` and enter the remote login password."
                ),
            ), patch("triton_agent.commands.execution.run_remote_test") as mocked:
                with redirect_stderr(stderr):
                    exit_code = main(
                        [
                            "run-test",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator),
                            "--remote",
                            "alice@example.com",
                        ]
                    )

        self.assertEqual(exit_code, 1)
        mocked.assert_not_called()
        self.assertIn("ssh-copy-id alice@example.com", stderr.getvalue())

    def test_main_local_run_test_skips_ssh_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "test_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: standalone\nprint('test')\n", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch.object(cli_module, "ensure_remote_ssh_ready") as preflight, patch(
                "triton_agent.commands.execution.run_local_test",
                return_value=(fake_result, None),
            ) as mocked:
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
        preflight.assert_not_called()
        mocked.assert_called_once()

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

            with patch("triton_agent.eval.runners.load_operator_eval_script_module", return_value=runtime) as mocked_loader:
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
                round_mode="checked",
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

    def test_main_optimize_resume_continue_rejects_test_mode_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            (root / "opt-note.md").write_text("history\n", encoding="utf-8")
            (root / "opt-round-1").mkdir()
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "original",
                        "source_operator": "../kernel.py",
                        "baseline_operator": "baseline/opt_kernel.py",
                        "test_file": "../differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "../bench_kernel.py",
                        "bench_mode": "torch-npu-profiler",
                        "perf_artifact": "baseline/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text("latency-a: 1.0\n", encoding="utf-8")
            (baseline_dir / "opt_kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            (root / "differential_test_kernel.py").write_text(
                "# test-mode: differential\nprint('test')\n", encoding="utf-8"
            )
            (root / "bench_kernel.py").write_text(
                "# bench-mode: torch-npu-profiler\n# kernel: k\nprint('bench')\n",
                encoding="utf-8",
            )

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
                            "standalone",
                        ]
                    )

            self.assertEqual(exc.exception.code, 2)
            self.assertIn(
                "--test-mode standalone conflicts with existing harness test-mode differential",
                stderr.getvalue(),
            )

    def test_main_optimize_resume_continue_rejects_bench_mode_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            (root / "opt-note.md").write_text("history\n", encoding="utf-8")
            (root / "opt-round-1").mkdir()
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "original",
                        "source_operator": "../kernel.py",
                        "baseline_operator": "baseline/opt_kernel.py",
                        "test_file": "../differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "../bench_kernel.py",
                        "bench_mode": "torch-npu-profiler",
                        "perf_artifact": "baseline/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text("latency-a: 1.0\n", encoding="utf-8")
            (baseline_dir / "opt_kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            (root / "differential_test_kernel.py").write_text(
                "# test-mode: differential\nprint('test')\n", encoding="utf-8"
            )
            (root / "bench_kernel.py").write_text(
                "# bench-mode: torch-npu-profiler\n# kernel: k\nprint('bench')\n",
                encoding="utf-8",
            )

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
                            "msprof",
                        ]
                    )

            self.assertEqual(exc.exception.code, 2)
            self.assertIn(
                "--bench-mode msprof conflicts with existing harness bench-mode torch-npu-profiler",
                stderr.getvalue(),
            )

    def test_main_optimize_resume_continue_allows_matching_test_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            (root / "opt-note.md").write_text("history\n", encoding="utf-8")
            (root / "opt-round-1").mkdir()
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "original",
                        "source_operator": "../kernel.py",
                        "baseline_operator": "baseline/opt_kernel.py",
                        "test_file": "../differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "../bench_kernel.py",
                        "bench_mode": "torch-npu-profiler",
                        "perf_artifact": "baseline/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text("latency-a: 1.0\n", encoding="utf-8")
            (baseline_dir / "opt_kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            (root / "differential_test_kernel.py").write_text(
                "# test-mode: differential\nprint('test')\n", encoding="utf-8"
            )
            (root / "bench_kernel.py").write_text(
                "# bench-mode: torch-npu-profiler\n# kernel: k\nprint('bench')\n",
                encoding="utf-8",
            )

            captured_modes: dict[str, Optional[str]] = {}

            def _fake_run(request):
                captured_modes["test_mode"] = request.test_mode
                captured_modes["bench_mode"] = request.bench_mode
                return AgentResult(return_code=0, stdout="", stderr="")

            with patch(
                "triton_agent.commands.optimize.run_optimize_request",
                side_effect=_fake_run,
            ):
                exit_code = main(
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

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured_modes["test_mode"], "differential")
            self.assertEqual(captured_modes["bench_mode"], "torch-npu-profiler")

    def test_main_optimize_resume_continue_allows_matching_bench_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            (root / "opt-note.md").write_text("history\n", encoding="utf-8")
            (root / "opt-round-1").mkdir()
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "original",
                        "source_operator": "../kernel.py",
                        "baseline_operator": "baseline/opt_kernel.py",
                        "test_file": "../differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "../bench_kernel.py",
                        "bench_mode": "torch-npu-profiler",
                        "perf_artifact": "baseline/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text("latency-a: 1.0\n", encoding="utf-8")
            (baseline_dir / "opt_kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            (root / "differential_test_kernel.py").write_text(
                "# test-mode: differential\nprint('test')\n", encoding="utf-8"
            )
            (root / "bench_kernel.py").write_text(
                "# bench-mode: torch-npu-profiler\n# kernel: k\nprint('bench')\n",
                encoding="utf-8",
            )

            captured_modes: dict[str, Optional[str]] = {}

            def _fake_run(request):
                captured_modes["test_mode"] = request.test_mode
                captured_modes["bench_mode"] = request.bench_mode
                return AgentResult(return_code=0, stdout="", stderr="")

            with patch(
                "triton_agent.commands.optimize.run_optimize_request",
                side_effect=_fake_run,
            ):
                exit_code = main(
                    [
                        "optimize",
                        "-i",
                        str(operator),
                        "--resume",
                        "continue",
                        "--bench-mode",
                        "torch-npu-profiler",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured_modes["test_mode"], "differential")
            self.assertEqual(captured_modes["bench_mode"], "torch-npu-profiler")

    def test_main_optimize_resume_auto_resumable_rejects_test_mode_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            (root / "opt-note.md").write_text("history\n", encoding="utf-8")
            (root / "opt-round-1").mkdir()
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "original",
                        "source_operator": "../kernel.py",
                        "baseline_operator": "baseline/opt_kernel.py",
                        "test_file": "../differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "../bench_kernel.py",
                        "bench_mode": "torch-npu-profiler",
                        "perf_artifact": "baseline/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text("latency-a: 1.0\n", encoding="utf-8")
            (baseline_dir / "opt_kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            (root / "differential_test_kernel.py").write_text(
                "# test-mode: differential\nprint('test')\n", encoding="utf-8"
            )
            (root / "bench_kernel.py").write_text(
                "# bench-mode: torch-npu-profiler\n# kernel: k\nprint('bench')\n",
                encoding="utf-8",
            )

            stderr = StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exc:
                    main(
                        [
                            "optimize",
                            "-i",
                            str(operator),
                            "--resume",
                            "auto",
                            "--test-mode",
                            "standalone",
                        ]
                    )

            self.assertEqual(exc.exception.code, 2)
            self.assertIn(
                "--test-mode standalone conflicts with existing harness test-mode differential",
                stderr.getvalue(),
            )

    def test_main_optimize_resume_auto_resumable_allows_matching_test_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            (root / "opt-note.md").write_text("history\n", encoding="utf-8")
            (root / "opt-round-1").mkdir()
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "original",
                        "source_operator": "../kernel.py",
                        "baseline_operator": "baseline/opt_kernel.py",
                        "test_file": "../differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "../bench_kernel.py",
                        "bench_mode": "torch-npu-profiler",
                        "perf_artifact": "baseline/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text("latency-a: 1.0\n", encoding="utf-8")
            (baseline_dir / "opt_kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            (root / "differential_test_kernel.py").write_text(
                "# test-mode: differential\nprint('test')\n", encoding="utf-8"
            )
            (root / "bench_kernel.py").write_text(
                "# bench-mode: torch-npu-profiler\n# kernel: k\nprint('bench')\n",
                encoding="utf-8",
            )

            captured_modes: dict[str, Optional[str]] = {}

            def _fake_run(request):
                captured_modes["test_mode"] = request.test_mode
                captured_modes["bench_mode"] = request.bench_mode
                return AgentResult(return_code=0, stdout="", stderr="")

            with patch(
                "triton_agent.commands.optimize.run_optimize_request",
                side_effect=_fake_run,
            ):
                exit_code = main(
                    [
                        "optimize",
                        "-i",
                        str(operator),
                        "--resume",
                        "auto",
                        "--test-mode",
                        "differential",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured_modes["test_mode"], "differential")
            self.assertEqual(captured_modes["bench_mode"], "torch-npu-profiler")

    def test_main_optimize_resume_auto_resumable_rejects_bench_mode_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            (root / "opt-note.md").write_text("history\n", encoding="utf-8")
            (root / "opt-round-1").mkdir()
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "original",
                        "source_operator": "../kernel.py",
                        "baseline_operator": "baseline/opt_kernel.py",
                        "test_file": "../differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "../bench_kernel.py",
                        "bench_mode": "torch-npu-profiler",
                        "perf_artifact": "baseline/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text("latency-a: 1.0\n", encoding="utf-8")
            (baseline_dir / "opt_kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            (root / "differential_test_kernel.py").write_text(
                "# test-mode: differential\nprint('test')\n", encoding="utf-8"
            )
            (root / "bench_kernel.py").write_text(
                "# bench-mode: torch-npu-profiler\n# kernel: k\nprint('bench')\n",
                encoding="utf-8",
            )

            stderr = StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exc:
                    main(
                        [
                            "optimize",
                            "-i",
                            str(operator),
                            "--resume",
                            "auto",
                            "--bench-mode",
                            "msprof",
                        ]
                    )

            self.assertEqual(exc.exception.code, 2)
            self.assertIn(
                "--bench-mode msprof conflicts with existing harness bench-mode torch-npu-profiler",
                stderr.getvalue(),
            )

    def test_main_optimize_resume_auto_resumable_allows_matching_bench_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            (root / "opt-note.md").write_text("history\n", encoding="utf-8")
            (root / "opt-round-1").mkdir()
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "original",
                        "source_operator": "../kernel.py",
                        "baseline_operator": "baseline/opt_kernel.py",
                        "test_file": "../differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "../bench_kernel.py",
                        "bench_mode": "torch-npu-profiler",
                        "perf_artifact": "baseline/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text("latency-a: 1.0\n", encoding="utf-8")
            (baseline_dir / "opt_kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            (root / "differential_test_kernel.py").write_text(
                "# test-mode: differential\nprint('test')\n", encoding="utf-8"
            )
            (root / "bench_kernel.py").write_text(
                "# bench-mode: torch-npu-profiler\n# kernel: k\nprint('bench')\n",
                encoding="utf-8",
            )

            captured_modes: dict[str, Optional[str]] = {}

            def _fake_run(request):
                captured_modes["test_mode"] = request.test_mode
                captured_modes["bench_mode"] = request.bench_mode
                return AgentResult(return_code=0, stdout="", stderr="")

            with patch(
                "triton_agent.commands.optimize.run_optimize_request",
                side_effect=_fake_run,
            ):
                exit_code = main(
                    [
                        "optimize",
                        "-i",
                        str(operator),
                        "--resume",
                        "auto",
                        "--bench-mode",
                        "torch-npu-profiler",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured_modes["test_mode"], "differential")
            self.assertEqual(captured_modes["bench_mode"], "torch-npu-profiler")

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

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch(
                "triton_agent.optimize.orchestration.optimize_execution.execute_multi_invocation_optimize",
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
                                    "--test-mode",
                                    "standalone",
                                    "--bench-mode",
                                    "msprof",
                                ]
                            )

            self.assertEqual(exit_code, 0)
            request = mocked.call_args.args[2]
            self.assertEqual(request.test_mode, "standalone")
            self.assertEqual(request.bench_mode, "msprof")
            self.assertFalse(request.continue_optimize)
            self.assertEqual(request.prompt, "")

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
                "# bench-mode: torch-npu-profiler\n# kernel: k\nprint('bench')\n",
                encoding="utf-8",
            )

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch(
                "triton_agent.optimize.orchestration.optimize_execution.execute_multi_invocation_optimize",
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
                            exit_code = main(["optimize", "-i", str(operator), "--resume", "auto"])

            self.assertEqual(exit_code, 0)
            request = mocked.call_args.args[2]
            self.assertEqual(request.test_mode, "differential")
            self.assertEqual(request.bench_mode, "torch-npu-profiler")
            self.assertFalse(request.continue_optimize)
            self.assertEqual(request.prompt, "")

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
                        '  "bench_mode": "torch-npu-profiler",',
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

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch(
                "triton_agent.optimize.orchestration.optimize_execution.execute_multi_invocation_optimize",
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
                            exit_code = main(["optimize", "-i", str(operator), "--resume", "auto"])

            self.assertEqual(exit_code, 0)
            request = mocked.call_args.args[2]
            self.assertEqual(request.test_mode, "differential")
            self.assertEqual(request.bench_mode, "torch-npu-profiler")
            self.assertFalse(request.continue_optimize)
            self.assertEqual(request.prompt, "")

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
                        '  "bench_mode": "torch-npu-profiler",',
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
                "# bench-mode: torch-npu-profiler\n# kernel: k\nprint('bench')\n", encoding="utf-8"
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
                "# bench-mode: torch-npu-profiler\n# kernel: k\nprint('bench')\n", encoding="utf-8"
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
                "# bench-mode: torch-npu-profiler\n# kernel: k\nprint('bench')\n",
                encoding="utf-8",
            )

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch("triton_agent.optimize.orchestration.optimize_execution.execute_multi_invocation_optimize", return_value=fake_result):
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

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch(
                "triton_agent.optimize.orchestration.optimize_execution.execute_multi_invocation_optimize",
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
            request = mocked.call_args.args[2]
            self.assertEqual(request.test_mode, "differential")
            self.assertEqual(request.bench_mode, "msprof")
            self.assertTrue(request.continue_optimize)
            self.assertEqual(request.prompt, "")

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

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch(
                "triton_agent.optimize.orchestration.optimize_execution.execute_multi_invocation_optimize",
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
                                    "msprof",
                                ]
                            )

            self.assertEqual(exit_code, 0)
            request = mocked.call_args.args[2]
            self.assertEqual(request.test_mode, "differential")
            self.assertEqual(request.bench_mode, "msprof")
            self.assertTrue(request.continue_optimize)
            self.assertEqual(request.prompt, "")

    def test_handle_optimize_rejects_enable_subagent_for_pi(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["optimize", "-i", "kernel.py", "--agent", "pi", "--enable-subagent"]
        )

        stderr = StringIO()
        with self.assertRaises(SystemExit) as exc, redirect_stderr(stderr):
            optimize_commands.handle_optimize(parser, args)

        self.assertEqual(exc.exception.code, 2)
        self.assertIn("--enable-subagent only supports", stderr.getvalue())

    def test_handle_optimize_batch_rejects_enable_subagent_for_pi(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["optimize-batch", "-i", "kernels", "--agent", "pi", "--enable-subagent"]
        )

        stderr = StringIO()
        with self.assertRaises(SystemExit) as exc, redirect_stderr(stderr):
            optimize_commands.handle_optimize_batch(parser, args)

        self.assertEqual(exc.exception.code, 2)
        self.assertIn("--enable-subagent only supports", stderr.getvalue())

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
                        '  "bench_mode": "torch-npu-profiler",',
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
                "# bench-mode: torch-npu-profiler\n# kernel: k\nprint('bench')\n", encoding="utf-8"
            )

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch(
                "triton_agent.optimize.orchestration.optimize_execution.execute_multi_invocation_optimize",
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
            request = mocked.call_args.args[2]
            self.assertTrue(request.continue_optimize)
            self.assertEqual(request.test_mode, "differential")
            self.assertEqual(request.bench_mode, "torch-npu-profiler")
            self.assertEqual(request.prompt, "")

    def test_main_optimize_accepts_workspace_directory_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch(
                "triton_agent.optimize.orchestration.optimize_execution.execute_multi_invocation_optimize",
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
            request = mocked.call_args.args[2]
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
                with patch(
                    "triton_agent.optimize.orchestration.optimize_execution.execute_multi_invocation_optimize",
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
            request = mocked.call_args.args[2]
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

    def test_main_optimize_passes_round_mode_to_prompt_and_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")

            with patch(
                "triton_agent.optimize.orchestration.optimize_execution.execute_multi_invocation_optimize",
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
                                ["optimize", "-i", str(operator), "--round-mode", "checked"]
                            )

            self.assertEqual(exit_code, 0)
            request = mocked.call_args.args[2]
            self.assertEqual(request.round_mode, "checked")
            self.assertEqual(request.target_chip, "A5")
            self.assertEqual(request.prompt, "")

    def test_main_optimize_passes_target_chip_to_prompt_and_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")

            with patch(
                "triton_agent.optimize.orchestration.optimize_execution.execute_multi_invocation_optimize",
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
            request = mocked.call_args.args[2]
            self.assertEqual(request.target_chip, "A3")
            self.assertEqual(request.prompt, "")

    def test_main_optimize_passes_no_agent_session_to_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch(
                "triton_agent.optimize.orchestration.optimize_execution.execute_multi_invocation_optimize",
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
            request = mocked.call_args.args[2]
            self.assertTrue(request.no_agent_session)

    def test_main_optimize_appends_user_prompt_to_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            operator.write_text("print('x')", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch(
                "triton_agent.optimize.orchestration.optimize_execution.execute_multi_invocation_optimize",
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
            request = mocked.call_args.args[2]
            self.assertEqual(request.prompt, "")
            self.assertEqual(request.user_prompt, "Focus on memory coalescing.")

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
            self.assertIn("test stdout", stdout.getvalue())
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

    def test_main_run_test_auto_compares_differential_result_when_ref_result_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "differential_test_kernel.py"
            archive = root / "kernel_result.pt"
            baseline_result = root / "baseline_result.pt"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')", encoding="utf-8")
            baseline_result.write_text("baseline", encoding="utf-8")

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
                            "--ref-result",
                            str(baseline_result),
                        ]
                    )

            self.assertEqual(exit_code, 1)
            compare_mock.assert_called_once_with(
                baseline_result.resolve(),
                archive,
                accuracy_mode="npu-contract",
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
            with patch.object(cli_module, "ensure_remote_ssh_ready", return_value=None), patch(
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
                accuracy_mode="npu-contract",
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
            with patch.object(cli_module, "ensure_remote_ssh_ready", return_value=None), patch(
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
            bench_file.write_text("# bench-mode: torch-npu-profiler\nprint('bench')", encoding="utf-8")

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
                "torch-npu-profiler",
                None,
                verbose=False,
                output=None,
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

    def test_main_run_bench_threads_output_to_local_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            perf_file = root / "custom_perf.txt"
            operator.write_text("print('x')", encoding="utf-8")
            bench_file.write_text("# bench-mode: torch-npu-profiler\nprint('bench')", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch("triton_agent.commands.execution.run_local_bench", return_value=(fake_result, perf_file)) as mocked:
                exit_code = main(
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
            mocked.assert_called_once_with(
                bench_file.resolve(),
                operator.resolve(),
                "torch-npu-profiler",
                None,
                verbose=False,
                output=str(perf_file),
            )

    def test_main_run_bench_defaults_to_torch_npu_profiler_when_flag_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            bench_file.write_text("# kernel: k\nprint('bench')", encoding="utf-8")

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
                "torch-npu-profiler",
                None,
                verbose=False,
                output=None,
            )

    def test_main_run_bench_threads_npu_devices_to_local_runner(self) -> None:
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
                        "--npu-devices",
                        "0,2-3",
                    ]
                )

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(
                bench_file.resolve(),
                operator.resolve(),
                "torch-npu-profiler",
                "0,2-3",
                verbose=False,
                output=None,
            )

    def test_run_bench_accepts_npu_device_alias(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run-bench",
                "--bench-file",
                "bench.py",
                "--operator-file",
                "kernel.py",
                "--npu-device",
                "0,2-3",
            ]
        )
        self.assertEqual(args.npu_devices, "0,2-3")

    def test_run_bench_wrapper_calls_loaded_skill_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            bench_file.write_text("# bench-mode: torch-npu-profiler\nprint('bench')", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            runtime = SimpleNamespace(
                parse_bench_metadata=lambda _path: {"bench-mode": "torch-npu-profiler"},
                run_local_bench=lambda *_args, **_kwargs: (fake_result, None),
            )

            with patch("triton_agent.eval.runners.load_operator_eval_script_module", return_value=runtime) as mocked_loader:
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
            bench_file.write_text("# bench-mode: torch-npu-profiler\nprint('bench')", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch.object(cli_module, "ensure_remote_ssh_ready", return_value=None), patch(
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
                "torch-npu-profiler",
                "alice@example.com",
                None,
                None,
                keep_remote_workdir=False,
                verbose=False,
                stderr=sys.stderr,
                output=None,
            )

    def test_main_run_bench_threads_npu_devices_to_remote_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            bench_file.write_text("# bench-mode: torch-npu-profiler\nprint('bench')", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch.object(cli_module, "ensure_remote_ssh_ready", return_value=None), patch(
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
                        "--npu-devices",
                        "4-5",
                    ]
                )

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(
                bench_file.resolve(),
                operator.resolve(),
                "torch-npu-profiler",
                "alice@example.com",
                None,
                "4-5",
                keep_remote_workdir=False,
                verbose=False,
                stderr=sys.stderr,
                output=None,
            )

    def test_main_run_bench_prints_remote_workspace_when_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            bench_file.write_text("# bench-mode: torch-npu-profiler\nprint('bench')", encoding="utf-8")

            stdout = StringIO()
            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch.object(cli_module, "ensure_remote_ssh_ready", return_value=None), patch(
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
            bench_file.write_text("# bench-mode: torch-npu-profiler\nprint('bench')", encoding="utf-8")

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
            mocked.assert_called_once_with(
                oracle.resolve(),
                new.resolve(),
                accuracy_mode="npu-contract",
            )

    def test_main_compare_result_uses_remote_comparison_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oracle = root / "abs_result.pt"
            new = root / "opt_abs_result.pt"
            oracle.write_text("oracle", encoding="utf-8")
            new.write_text("new", encoding="utf-8")

            with patch.object(cli_module, "ensure_remote_ssh_ready", return_value=None), patch(
                "triton_agent.commands.comparison.compare_remote_result_files", return_value=0
            ) as mocked:
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
                "alice@example.com:2200",
                None,
                accuracy_mode="npu-contract",
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
            mocked.assert_called_once_with(
                baseline.resolve(),
                compare.resolve(),
                skip_latency_errors=False,
                metric_source="auto",
            )

    def test_main_compare_perf_forwards_skip_latency_errors_flag(self) -> None:
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
                        "--skip-latency-errors",
                    ]
                )

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(
                baseline.resolve(),
                compare.resolve(),
                skip_latency_errors=True,
                metric_source="auto",
            )

    def test_main_compare_perf_forwards_metric_source_flag(self) -> None:
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
                        "--metric-source",
                        "kernel",
                    ]
                )

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(
                baseline.resolve(),
                compare.resolve(),
                skip_latency_errors=False,
                metric_source="kernel",
            )

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
        self.assertIn("Complete optimize rounds strictly one at a time in sequence.", prompt)
        self.assertIn("Do not use subagents to implement or advance multiple optimize rounds in parallel.", prompt)

    def test_build_optimize_resume_prompt_mentions_operator_target_contract(self) -> None:
        prompt = build_optimize_resume_prompt(
            "Round gate passed.",
            optimize_target="operator",
        )

        self.assertIn("Target optimization scope for this optimize session: operator.", prompt)
        self.assertIn("Optimize end-to-end operator latency.", prompt)
        self.assertIn(
            "Use the staged `torch-npu-optimize-knowledge` skill for Torch NPU and operator-level pattern references.",
            prompt,
        )

    def test_build_optimize_resume_prompt_requires_pre_round_reflection(self) -> None:
        prompt = build_optimize_resume_prompt("Round gate passed.")

        self.assertIn("Before editing code for the next round, stop and reflect on the best entrypoint.", prompt)
        self.assertIn(
            "Choose which operator, kernel path, or wrapper bottleneck should anchor the round before making the next code change.",
            prompt,
        )
        self.assertIn(
            "Decide whether existing benchmark and compare-perf evidence is already sufficient or whether profiling is needed first.",
            prompt,
        )
        self.assertIn(
            "Escalate to IR only after profiler evidence narrows the bottleneck but still does not explain it.",
            prompt,
        )
        self.assertIn(
            "Use compiler-source analysis only after profiler and IR evidence have narrowed a concrete compiler-side question.",
            prompt,
        )
        self.assertIn(
            "Do not use agents or subagents to optimize multiple rounds in parallel; keep the optimize session one round at a time.",
            prompt,
        )
        self.assertIn(
            "Do not treat the next round as a parameter-only tuning sweep; make a bottleneck-backed change instead.",
            prompt,
        )

    def test_build_optimize_supervisor_prompt_mentions_audit_pass(self) -> None:
        prompt = build_optimize_supervisor_prompt(
            Path("/tmp"),
            latest_round_dir=Path("/tmp/opt-round-3"),
        )
        self.assertIn("This invocation is the optimize supervisor pass.", prompt)
        self.assertIn("This invocation is an audit and handoff pass", prompt)
        self.assertIn("Read `/tmp/opt-round-3`", prompt)
        self.assertIn("Use only existing `compare-perf` results", prompt)
        self.assertIn("`ascend-npu-prepare-optimize-baseline`", prompt)
        self.assertIn("`ascend-npu-optimize-state`", prompt)
        self.assertIn("`submit-baseline`", prompt)
        self.assertIn("`submit-round`", prompt)
        self.assertIn("`start-round`", prompt)
        self.assertIn("`set-current-round-state`", prompt)
        self.assertIn("Write `supervisor-report.md`", prompt)
        self.assertNotIn(".triton-agent/supervisor-report.md", prompt)
        self.assertIn("The CLI will read that supervisor report", prompt)
        self.assertIn("Do not edit the operator implementation", prompt)
        self.assertIn("replace the Triton kernel path with pure PyTorch computation", prompt)
        self.assertNotIn("optimize-supervisor.md", prompt)

    def test_build_optimize_supervisor_prompt_mentions_operator_target_contract(self) -> None:
        prompt = build_optimize_supervisor_prompt(
            Path("/tmp"),
            latest_round_dir=Path("/tmp/opt-round-3"),
            optimize_target="operator",
        )

        self.assertIn("Target optimization scope for this optimize session: operator.", prompt)
        self.assertIn("whole-operator restructuring", prompt)
        self.assertIn("total-op conclusion", prompt)
        self.assertIn("pure PyTorch computation", prompt)

    def test_build_optimize_supervisor_prompt_includes_cli_followup_summary_when_provided(self) -> None:
        prompt = build_optimize_supervisor_prompt(
            Path("/tmp"),
            latest_round_dir=Path("/tmp/opt-round-3"),
            cli_followup_summary=(
                "CLI round follow-up from the previous round:\n"
                "- Decision: pass\n"
                "- Continue required: yes\n"
                "- Issues: none"
            ),
        )

        self.assertIn("Read this CLI round follow-up summary before auditing the round:", prompt)
        self.assertIn("CLI round follow-up from the previous round:", prompt)
        self.assertIn("- Decision: pass", prompt)
        self.assertIn("- Continue required: yes", prompt)

    def test_build_optimize_baseline_prompt_uses_explicit_context_parameters(self) -> None:
        prompt = build_optimize_baseline_prompt(
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="torch-npu-profiler",
            target_chip="A5",
            optimize_target="operator",
            compiler_source_path=Path("/tmp/AscendNPU-IR"),
            compiler_source_commit="abc123",
            enable_cann_ext_api=True,
            baseline_state="missing",
            base_prompt="Focus only on baseline establishment.",
            remote="alice@example.com:2200",
            remote_workdir="/tmp/remote",
        )
        self.assertIn("Operator input: /tmp/op.py", prompt)
        self.assertNotIn("Requested output:", prompt)
        self.assertIn("Requested test mode: differential", prompt)
        self.assertIn("Requested bench mode: torch-npu-profiler", prompt)
        self.assertIn("Remote execution target: alice@example.com:2200", prompt)
        self.assertIn("Remote execution root: /tmp/remote", prompt)
        self.assertIn("Target optimization scope for this optimize session: operator.", prompt)
        self.assertIn("Target chip for this optimize session: A5.", prompt)
        self.assertIn("Compiler source analysis is enabled for this optimize run.", prompt)
        self.assertIn("Compiler source path: /tmp/AscendNPU-IR", prompt)
        self.assertIn("Compiler source commit: abc123.", prompt)
        self.assertIn("CANN Triton extension API pattern access is enabled for this optimize run.", prompt)
        self.assertIn("Additional user instructions:", prompt)
        self.assertIn("Focus only on baseline establishment.", prompt)

    def test_gen_eval_prompt_mentions_operator_repair_and_dual_outputs(self) -> None:
        prompt = build_prompt(
            CommandKind.GEN_EVAL,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="torch-npu-profiler",
            force_overwrite=False,
        )
        self.assertIn("ascend-npu-gen-eval-suite", prompt)
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
        self.assertIn("ascend-npu-gen-test", prompt)
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
            bench_mode="torch-npu-profiler",
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
            bench_mode="torch-npu-profiler",
            force_overwrite=False,
        )
        self.assertIn("Requested bench mode: torch-npu-profiler", prompt)

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
        self.assertIn("repeated runs of the same harness produce identical inputs", prompt)

    def test_gen_bench_prompt_requires_execute_and_autofix(self) -> None:
        prompt = build_prompt(
            CommandKind.GEN_BENCH,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/bench_op.py"),
            test_mode=None,
            bench_mode="torch-npu-profiler",
            force_overwrite=False,
        )
        self.assertIn("After generating the artifact, execute the generated benchmark case", prompt)
        self.assertIn("repair the generated artifact and retry automatically", prompt)
        self.assertIn("repeated runs of the same harness produce identical inputs", prompt)

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
            bench_mode="torch-npu-profiler",
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
        self.assertIn("Do not modify the original input operator file.", prompt)
        self.assertIn("Preserve the trailing input-helper block", prompt)
        self.assertIn("Treat the input operator file as source material and the differential correctness oracle.", prompt)
        self.assertIn("Generate a differential test for the converted output and execute it.", prompt)
        self.assertIn("Validate the converted output by comparing it against the original operator behavior.", prompt)
        self.assertIn("Do not introduce unnecessary wrappers, compatibility branches, helper layers, or scaffolding.", prompt)
        self.assertIn("real Triton Ascend NPU kernel path", prompt)
        self.assertIn("PyTorch-facing wrapper or module API may remain", prompt)
        self.assertIn("A pure PyTorch rewrite does not satisfy this convert task", prompt)
        self.assertIn("Target Ascend NPU only for this conversion flow", prompt)
        self.assertIn("If a suitable test already exists in the workspace, reuse it", prompt)
        self.assertIn("This includes existing standalone or differential test cases when they already cover the operator workspace", prompt)
        self.assertIn("Only generate a new test when no suitable reusable test exists", prompt)
        self.assertIn("Do not modify the original input operator file.", prompt)
        self.assertIn("Read the original input operator file, but treat it as immutable source material and a correctness oracle only.", prompt)
        self.assertNotIn("ascend-npu-prepare-optimize-baseline", prompt)
        self.assertIn("Requested output: /tmp/triton_op.py", prompt)

    def test_convert_prompt_mentions_standalone_validation_when_requested(self) -> None:
        prompt = build_prompt(
            CommandKind.CONVERT,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/triton_op.py"),
            test_mode="standalone",
            bench_mode=None,
            force_overwrite=False,
        )
        self.assertIn("Requested test mode: standalone", prompt)
        self.assertIn("Generate a standalone test for the converted output and execute it.", prompt)
        self.assertIn(
            "Validate the converted output by executing the standalone test against the converted operator.",
            prompt,
        )
        self.assertNotIn("Generate a differential test for the converted output and execute it.", prompt)

    def test_optimize_prompt_mentions_requested_modes(self) -> None:
        prompt = build_prompt(
            CommandKind.OPTIMIZE,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="torch-npu-profiler",
            force_overwrite=False,
            round_mode="checked",
        )
        self.assertIn("This invocation owns rounds 1 through 5.", prompt)
        self.assertIn("Execute those rounds strictly one at a time.", prompt)
        self.assertIn("Do not pre-plan the full batch before acting.", prompt)
        self.assertNotIn("Requested output:", prompt)
        self.assertIn("Requested test mode: differential", prompt)
        self.assertIn("Requested bench mode: torch-npu-profiler", prompt)
        self.assertIn(
            "For each round, write the optimized operator snapshot as `opt_<original-operator>.py` inside `opt-round-N/`.",
            prompt,
        )
        self.assertIn("Reuse existing correctness tests and benchmark cases when they already exist", prompt)
        self.assertIn("State the optimization hypothesis and why it may help", prompt)
        self.assertIn("Explain what evidence supports the change", prompt)
        self.assertIn("If you skip profiling or IR capture", prompt)

    def test_optimize_prompt_defaults_min_rounds_to_five(self) -> None:
        prompt = build_prompt(
            CommandKind.OPTIMIZE,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="torch-npu-profiler",
            force_overwrite=False,
        )
        self.assertIn("This invocation owns rounds 1 through 5.", prompt)
        self.assertNotIn("This invocation is a continuous optimize run.", prompt)

    def test_optimize_prompt_keeps_min_rounds_out_of_worker_prompt(self) -> None:
        prompt = build_prompt(
            CommandKind.OPTIMIZE,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="torch-npu-profiler",
            force_overwrite=False,
            min_rounds=4,
            round_mode="checked",
        )
        self.assertNotIn("Complete at least 4 optimization rounds", prompt)
        self.assertIn("This invocation owns rounds 1 through 4.", prompt)

    def test_optimize_prompt_mentions_min_speedup_target(self) -> None:
        prompt = build_prompt(
            CommandKind.OPTIMIZE,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="torch-npu-profiler",
            force_overwrite=False,
            min_speedup=1.2,
            round_mode="checked",
        )
        self.assertIn(
            "Optimize session target: reach at least 1.20x geomean speedup over the baseline.",
            prompt,
        )
        self.assertIn(
            "If `submit-round` reports that this target is satisfied, stop the optimize session immediately.",
            prompt,
        )
        self.assertIn(
            "The optimize runner injects this target into `submit-round` automatically; do not guess or override a different speedup target.",
            prompt,
        )
        self.assertIn(
            "submit-round --round-dir opt-round-N --current-round N --final-round M",
            prompt,
        )
        self.assertNotIn("--min-speedup 1.20", prompt)

    def test_optimize_prompt_supervised_worker_does_not_mention_audit_pass(self) -> None:
        prompt = build_prompt(
            CommandKind.OPTIMIZE,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="torch-npu-profiler",
            force_overwrite=False,
            round_mode="supervised",
        )
        self.assertIn("This invocation owns rounds 1 through 5.", prompt)
        self.assertNotIn("supervisor audit pass", prompt)
        self.assertNotIn("will review it", prompt)

    def test_build_optimize_round_prompt_mentions_current_and_final_round(self) -> None:
        prompt = build_optimize_round_prompt(
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="torch-npu-profiler",
            round_mode="checked",
            current_round=2,
            final_round=4,
            round_batch_size=3,
        )
        self.assertIn("This invocation owns rounds 2 through 4.", prompt)
        self.assertIn("Execute those rounds strictly one at a time.", prompt)
        self.assertIn("Do not pre-plan the full batch before acting.", prompt)
        self.assertIn(
            "When a round in this invocation is complete, run `submit-round --round-dir opt-round-N --current-round N --final-round M` with the actual round numbers from this worker batch.",
            prompt,
        )

    def test_build_optimize_round_prompt_interactive_baseline_guidance(self) -> None:
        prompt = build_optimize_round_prompt(
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="torch-npu-profiler",
            round_mode="checked",
            baseline_ready=False,
            current_round=1,
            final_round=30,
            round_batch_size=99,
        )
        self.assertIn("repair or establish `baseline/` before `opt-round-1`", prompt)
        self.assertIn("Do not rely on a separate baseline-preflight invocation", prompt)
        self.assertIn(
            "You must run the staged `ascend-npu-optimize-state` skill's `submit-round` subcommand after each completed round.",
            prompt,
        )
        self.assertIn("The CLI will validate the completed batch after the invocation exits.", prompt)
        self.assertNotIn("Interactive mode will not run CLI round checks", prompt)
        self.assertNotIn("The baseline has already been validated before this worker batch.", prompt)

    def test_optimize_prompt_mentions_continue_mode_for_resolved_resume(self) -> None:
        prompt = build_prompt(
            CommandKind.OPTIMIZE,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="torch-npu-profiler",
            force_overwrite=False,
            resume_existing_session=True,
            round_mode="checked",
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
            bench_mode="torch-npu-profiler",
            force_overwrite=False,
            round_mode="checked",
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
        self.assertNotIn("torch-npu-optimize-knowledge", prompt)
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

    def test_tilelang_optimize_prompt_stops_at_profiling_diagnosis(self) -> None:
        prompt = build_prompt(
            CommandKind.OPTIMIZE,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="torch-npu-profiler",
            force_overwrite=False,
            round_mode="checked",
            language="tilelang",
            compiler_source_path=Path("/tmp/compiler"),
            compiler_source_commit="deadbeef",
        )
        self.assertIn("Choose the analysis level for the round before editing code.", prompt)
        self.assertIn(
            "Escalate analysis in this order: pattern triage, profiling diagnosis.",
            prompt,
        )
        self.assertIn(
            "Use the staged `tilelang-npu-optimize-knowledge` skill for generic pattern and symptom references.",
            prompt,
        )
        self.assertNotIn("IR attribution", prompt)
        self.assertNotIn("compiler-source escalation", prompt)
        self.assertNotIn("tilelang-npu-analyze-ir", prompt)
        self.assertNotIn("Compiler source analysis is enabled for this optimize run.", prompt)

    def test_tilelang_optimize_resume_prompt_stops_at_profiling_diagnosis(self) -> None:
        prompt = build_optimize_resume_prompt(
            "Round gate passed.",
            language="tilelang",
            compiler_source_path=Path("/tmp/compiler"),
            compiler_source_commit="deadbeef",
        )
        self.assertIn(
            "Use the staged `ascend-npu-optimize-state` skill's `start-round` subcommand before opening the next round.",
            prompt,
        )
        self.assertIn(
            "That `start-round` call initializes the next round's workflow-owned `round_strategy`, `analysis_policy`, and `reason`.",
            prompt,
        )
        self.assertIn(
            "Use profiling diagnosis as the default deeper entrypoint when pattern triage is not enough.",
            prompt,
        )
        self.assertNotIn("Escalate to IR only after profiler evidence narrows the bottleneck", prompt)
        self.assertNotIn("Use compiler-source analysis only after profiler and IR evidence", prompt)
        self.assertNotIn("Compiler source analysis is enabled for this optimize run.", prompt)

    def test_tilelang_optimize_supervisor_prompt_stops_at_profiling_diagnosis(self) -> None:
        prompt = build_optimize_supervisor_prompt(
            Path("/tmp/workdir"),
            language="tilelang",
            latest_round_dir=Path("/tmp/workdir/opt-round-1"),
        )
        self.assertIn(
            "Read the staged `tilelang-npu-optimize`, `ascend-npu-prepare-optimize-baseline`, and `ascend-npu-optimize-state` skills as the workflow contract that the worker round was supposed to follow.",
            prompt,
        )
        self.assertIn(
            "Audit the worker against this analysis ladder: pattern triage, profiling diagnosis.",
            prompt,
        )
        self.assertNotIn("IR attribution", prompt)
        self.assertNotIn("compiler-source escalation", prompt)

    def test_optimize_prompt_defaults_to_checked_batch_mode(self) -> None:
        prompt = build_prompt(
            CommandKind.OPTIMIZE,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="torch-npu-profiler",
            force_overwrite=False,
        )
        self.assertIn("This invocation owns rounds 1 through 5.", prompt)
        self.assertNotIn("This invocation is a continuous optimize run.", prompt)
        self.assertNotIn("Own the end-to-end optimize session", prompt)

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
    def test_render_result_skips_stdout_when_skip_enabled(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        result = AgentResult(return_code=0, stdout="streamed\n", stderr="")
        render_result(result, skip_stdout=True, stdout=stdout, stderr=stderr)
        self.assertEqual(stdout.getvalue(), "")

    def test_render_result_prints_stdout_when_skip_disabled(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        result = AgentResult(return_code=0, stdout="final\n", stderr="")
        render_result(result, skip_stdout=False, stdout=stdout, stderr=stderr)
        self.assertEqual(stdout.getvalue(), "final\n")


class ResultNormalizationTests(unittest.TestCase):
    def test_invalid_skill_result_payload_raises_actionable_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required keys"):
            normalize_agent_result({"stdout": "", "stderr": ""})


class _TtyStringIO(StringIO):
    def isatty(self) -> bool:
        return True


class CliHelpColorTests(unittest.TestCase):
    def test_format_help_remains_plain_text(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()
        self.assertNotIn("\033[", help_text)

    def test_top_level_help_prints_color_on_tty(self) -> None:
        parser = build_parser()
        stdout = _TtyStringIO()
        with patch.dict(os.environ, {}, clear=True):
            parser.print_help(file=stdout)
        output = stdout.getvalue()
        self.assertIn("\033[36m-h\033[0m", output)
        self.assertIn("\033[36m--help\033[0m", output)
        self.assertIn("\033[36mgen-eval\033[0m", output)
        self.assertIn("\033[36mconvert-batch\033[0m", output)
        self.assertIn("\033[36mTRITON_AGENT_BATCH_NPU_DEVICES\033[0m", output)
        self.assertNotIn("Generate, run, \033[36mverify\033[0m", output)
        self.assertNotIn("Show optimization \033[36mstatus\033[0m for one workspace.", output)

    def test_top_level_help_prints_plain_text_without_tty(self) -> None:
        parser = build_parser()
        stdout = StringIO()
        parser.print_help(file=stdout)
        output = stdout.getvalue()
        self.assertNotIn("\033[", output)

    def test_top_level_help_respects_no_color_on_tty(self) -> None:
        parser = build_parser()
        stdout = _TtyStringIO()
        with patch.dict(os.environ, {"NO_COLOR": "1"}, clear=False):
            parser.print_help(file=stdout)
        output = stdout.getvalue()
        self.assertNotIn("\033[", output)

    def test_subcommand_help_prints_color_on_tty(self) -> None:
        parser = build_parser()
        stdout = _TtyStringIO()
        with self.assertRaises(SystemExit):
            with patch.dict(os.environ, {}, clear=True):
                with patch.object(sys, "stdout", stdout):
                    parser.parse_args(["gen-test", "--help"])
        output = stdout.getvalue()
        self.assertIn("\033[36m--help\033[0m", output)
        self.assertIn("\033[36m--agent\033[0m", output)

    def test_subcommand_help_respects_no_color_on_tty(self) -> None:
        parser = build_parser()
        stdout = _TtyStringIO()
        with self.assertRaises(SystemExit):
            with patch.dict(os.environ, {"NO_COLOR": "1"}, clear=False):
                with patch.object(sys, "stdout", stdout):
                    parser.parse_args(["gen-test", "--help"])
        output = stdout.getvalue()
        self.assertNotIn("\033[", output)


if __name__ == "__main__":
    unittest.main()
