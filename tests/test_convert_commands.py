import importlib.util
import json
import sys
import tempfile
import unittest
from io import StringIO
from os import environ
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from contextlib import redirect_stdout

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.cli import build_parser
from triton_agent.convert.models import ConvertOptions
from triton_agent.models import AgentRequest, AgentResult, CommandKind
from triton_agent.otel_trace import TRACE_PATH_ENV
from triton_agent.remote_execution_env import remote_target_env_name, remote_workdir_env_name


class ConvertCommandModuleTests(unittest.TestCase):
    def test_convert_command_module_exists(self) -> None:
        self.assertIsNotNone(importlib.util.find_spec("triton_agent.commands.convert"))

    def test_generation_command_module_no_longer_exports_convert_handlers(self) -> None:
        import triton_agent.commands.generation as generation_commands

        self.assertFalse(hasattr(generation_commands, "handle_gen_convert"))

    def test_convert_orchestration_module_exists(self) -> None:
        self.assertIsNotNone(importlib.util.find_spec("triton_agent.convert.orchestration"))

    def test_convert_outputs_module_exists(self) -> None:
        self.assertIsNotNone(importlib.util.find_spec("triton_agent.convert.outputs"))


class ConvertRuntimeTests(unittest.TestCase):
    def test_resolve_convert_output_path_uses_triton_prefix(self) -> None:
        from triton_agent.convert.outputs import resolve_convert_output_path

        output_path = resolve_convert_output_path(
            Path("/tmp/kernel.py"),
            explicit_output=None,
        )

        self.assertEqual(output_path, Path("/tmp/triton_kernel.py"))

    def test_prepare_convert_target_rejects_existing_artifact_without_overwrite(self) -> None:
        from triton_agent.convert.outputs import prepare_convert_target

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "triton_kernel.py"
            output.write_text("existing converted operator", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "Output file already exists"):
                prepare_convert_target(output, force_overwrite=False)

    def test_build_convert_request_uses_convert_only_skills(self) -> None:
        from triton_agent.convert.orchestration import build_convert_request

        request = build_convert_request(
            Path("/tmp/kernel.py"),
            Path("/tmp/kernel.py"),
            Path("/tmp"),
            ConvertOptions(
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                remote=None,
                remote_workdir=None,
                output=None,
                test_mode="differential",
                prompt=None,
            ),
        )

        self.assertEqual(request.command_kind, CommandKind.CONVERT)
        self.assertEqual(
            request.staged_skill_names,
            (
                "triton-npu-convert-pytorch-operator",
                "triton-npu-gen-test",
                "triton-npu-run-eval",
                "triton-npu-repair-guide",
            ),
        )
        self.assertIsNone(request.staged_skill_sources)
        self.assertEqual(request.skill_name, "triton-npu-convert-pytorch-operator")
        self.assertEqual(request.output_path, Path("/tmp/triton_kernel.py"))
        self.assertIsNone(request.mcp_servers)

    def test_build_convert_request_attaches_mcp_servers_when_enabled(self) -> None:
        from triton_agent.convert.orchestration import build_convert_request

        request = build_convert_request(
            Path("/tmp/kernel.py"),
            Path("/tmp/kernel.py"),
            Path("/tmp"),
            ConvertOptions(
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                remote=None,
                remote_workdir=None,
                output=None,
                test_mode="differential",
                prompt=None,
                enable_mcp=True,
            ),
        )

        self.assertEqual(
            request.staged_skill_sources,
            {"triton-npu-run-eval": "triton-npu-run-eval-mcp"},
        )
        self.assertEqual(request.mcp_servers, ("triton-agent-run-eval",))

    def test_build_convert_request_injects_remote_env(self) -> None:
        from triton_agent.convert.orchestration import build_convert_request

        request = build_convert_request(
            Path("/tmp/kernel.py"),
            Path("/tmp/kernel.py"),
            Path("/tmp"),
            ConvertOptions(
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                remote="alice@example.com",
                remote_workdir="/tmp/triton-agent",
                output=None,
                test_mode="differential",
                prompt=None,
            ),
        )

        self.assertEqual(request.remote, "alice@example.com")
        self.assertEqual(request.remote_workdir, "/tmp/triton-agent")
        self.assertIsNotNone(request.extra_env)
        assert request.extra_env is not None
        self.assertEqual(request.extra_env[remote_target_env_name()], "alice@example.com")
        self.assertEqual(request.extra_env[remote_workdir_env_name()], "/tmp/triton-agent")

    def test_build_convert_request_appends_user_prompt(self) -> None:
        from triton_agent.convert.orchestration import build_convert_request

        request = build_convert_request(
            Path("/tmp/kernel.py"),
            Path("/tmp/kernel.py"),
            Path("/tmp"),
            ConvertOptions(
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                remote=None,
                remote_workdir=None,
                output=None,
                test_mode="differential",
                prompt="Keep the exported function name.",
            ),
        )

        self.assertIn("Additional user instructions:", request.prompt)
        self.assertIn("Keep the exported function name.", request.prompt)

    def test_build_convert_request_enables_tool_trace_when_requested(self) -> None:
        from triton_agent.convert.orchestration import build_convert_request

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            request = build_convert_request(
                workdir / "kernel.py",
                workdir / "kernel.py",
                workdir,
                ConvertOptions(
                    interact=False,
                    verbose=False,
                    stream_output=False,
                    force_overwrite=False,
                    agent_name="codex",
                    remote=None,
                    remote_workdir=None,
                    output=None,
                    test_mode="differential",
                    prompt=None,
                    log_tools=True,
                ),
            )

            self.assertTrue(request.log_tools)
            self.assertIsNotNone(request.extra_env)
            assert request.extra_env is not None
            trace_path = Path(request.extra_env[TRACE_PATH_ENV])
            self.assertEqual(trace_path.parent.parent, workdir / "triton-agent-logs")
            self.assertTrue(trace_path.parent.name.startswith("convert-"))
            self.assertEqual(trace_path.name, "tool-traces.jsonl")

    def test_handle_convert_builds_request_with_default_output(self) -> None:
        from triton_agent.commands.convert import handle_convert

        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            operator = Path(tmp) / "kernel.py"
            converted = Path(tmp) / "triton_kernel.py"
            test_file = Path(tmp) / "differential_test_kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\n", encoding="utf-8")
            args = parser.parse_args(
                [
                    "convert",
                    "-i",
                    str(operator),
                ]
            )

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            captured: dict[str, object] = {}

            def _fake_run(request):
                captured["output_path"] = request.output_path
                captured["staged_skill_names"] = request.staged_skill_names
                converted.write_text("converted\n", encoding="utf-8")
                return fake_result

            with patch("triton_agent.commands.convert.run_convert_request", side_effect=_fake_run):
                with patch(
                    "triton_agent.commands.convert._verify_converted_output",
                    return_value=type(
                        "Verify",
                        (),
                        {"return_code": 0, "summary": "ok", "baseline_result": None},
                    )(),
                ):
                    exit_code = handle_convert(parser, args)

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                captured["output_path"],
                (Path(tmp) / "triton_kernel.py").resolve(),
            )
            self.assertEqual(
                captured["staged_skill_names"],
                (
                    "triton-npu-convert-pytorch-operator",
                    "triton-npu-gen-test",
                    "triton-npu-run-eval",
                    "triton-npu-repair-guide",
                ),
            )

    def test_handle_convert_workspace_input_uses_workspace_as_workdir(self) -> None:
        from triton_agent.commands.convert import handle_convert

        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            converted = workspace / "triton_kernel.py"
            test_file = workspace / "differential_test_kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\n", encoding="utf-8")
            args = parser.parse_args(
                [
                    "convert",
                    "-i",
                    str(workspace),
                ]
            )

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            captured: dict[str, object] = {}

            def _fake_run(request):
                captured["input_path"] = request.input_path
                captured["workdir"] = request.workdir
                captured["output_path"] = request.output_path
                converted.write_text("converted\n", encoding="utf-8")
                return fake_result

            with patch("triton_agent.commands.convert.run_convert_request", side_effect=_fake_run):
                with patch(
                    "triton_agent.commands.convert._verify_converted_output",
                    return_value=type(
                        "Verify",
                        (),
                        {"return_code": 0, "summary": "ok", "baseline_result": None},
                    )(),
                ):
                    exit_code = handle_convert(parser, args)

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["input_path"], operator.resolve())
            self.assertEqual(captured["workdir"], workspace.resolve())
            self.assertEqual(
                captured["output_path"],
                (workspace / "triton_kernel.py").resolve(),
            )

    def test_handle_convert_reuses_default_differential_test_for_cli_verification(self) -> None:
        from triton_agent.commands.convert import handle_convert

        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            test_file = workspace / "differential_test_kernel.py"
            operator.write_text("print('src')\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\n", encoding="utf-8")
            args = parser.parse_args(["convert", "-i", str(operator)])

            run_result = AgentResult(return_code=0, stdout="", stderr="")
            archived_oracle = workspace / "kernel_result.pt"
            archived_candidate = workspace / "triton_kernel_result.pt"

            observed_test_calls: list[tuple[Path, Path, str]] = []

            def _fake_run_convert(request):
                assert request.output_path is not None
                request.output_path.write_text("print('dst')\n", encoding="utf-8")
                self.assertEqual(request.output_path, (workspace / "triton_kernel.py").resolve())
                return run_result

            def _fake_run_local_test(test_path, operator_path, test_mode, *, verbose=False):
                del verbose
                observed_test_calls.append((test_path, operator_path, test_mode))
                if operator_path == operator.resolve():
                    return AgentResult(return_code=0, stdout="oracle\n", stderr=""), archived_oracle
                return AgentResult(return_code=0, stdout="candidate\n", stderr=""), archived_candidate

            with patch("triton_agent.commands.convert.run_convert_request", side_effect=_fake_run_convert):
                with patch("triton_agent.commands.convert.run_local_test", side_effect=_fake_run_local_test):
                    with patch("triton_agent.commands.convert.compare_result_files", return_value=0) as compare_mock:
                        exit_code = handle_convert(parser, args)

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                observed_test_calls,
                [
                    (test_file.resolve(), operator.resolve(), "differential"),
                    ((workspace / "differential_test_kernel.py").resolve(), (workspace / "triton_kernel.py").resolve(), "differential"),
                ],
            )
            compare_mock.assert_called_once_with(
                archived_oracle,
                archived_candidate,
            )

    def test_resolve_convert_test_file_falls_back_to_unique_standalone_test(self) -> None:
        from triton_agent.commands.convert import _resolve_convert_test_file

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            standalone_test = workspace / "test_kernel.py"
            operator.write_text("print('src')\n", encoding="utf-8")
            standalone_test.write_text("# test-mode: standalone\n", encoding="utf-8")

            request = AgentRequest(
                command_kind=CommandKind.CONVERT,
                input_path=operator.resolve(),
                operator_path=operator.resolve(),
                output_path=(workspace / "triton_kernel.py").resolve(),
                test_mode="differential",
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-convert-pytorch-operator",
                prompt="convert",
                workdir=workspace.resolve(),
            )

            resolved = _resolve_convert_test_file(request)

        self.assertEqual(resolved, standalone_test.resolve())

    def test_resolve_convert_test_file_prefers_differential_over_standalone(self) -> None:
        from triton_agent.commands.convert import _resolve_convert_test_file

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            differential_test = workspace / "differential_test_kernel.py"
            standalone_test = workspace / "test_kernel.py"
            operator.write_text("print('src')\n", encoding="utf-8")
            differential_test.write_text("# test-mode: differential\n", encoding="utf-8")
            standalone_test.write_text("# test-mode: standalone\n", encoding="utf-8")

            request = AgentRequest(
                command_kind=CommandKind.CONVERT,
                input_path=operator.resolve(),
                operator_path=operator.resolve(),
                output_path=(workspace / "triton_kernel.py").resolve(),
                test_mode="differential",
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-convert-pytorch-operator",
                prompt="convert",
                workdir=workspace.resolve(),
            )

            resolved = _resolve_convert_test_file(request)

        self.assertEqual(resolved, differential_test.resolve())

    def test_resolve_convert_test_file_prefers_requested_standalone_over_differential(self) -> None:
        from triton_agent.commands.convert import _resolve_convert_test_file

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            differential_test = workspace / "differential_test_kernel.py"
            standalone_test = workspace / "test_kernel.py"
            operator.write_text("print('src')\n", encoding="utf-8")
            differential_test.write_text("# test-mode: differential\n", encoding="utf-8")
            standalone_test.write_text("# test-mode: standalone\n", encoding="utf-8")

            request = AgentRequest(
                command_kind=CommandKind.CONVERT,
                input_path=operator.resolve(),
                operator_path=operator.resolve(),
                output_path=(workspace / "triton_kernel.py").resolve(),
                test_mode="standalone",
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-convert-pytorch-operator",
                prompt="convert",
                workdir=workspace.resolve(),
            )

            resolved = _resolve_convert_test_file(request)

        self.assertEqual(resolved, standalone_test.resolve())

    def test_verify_converted_output_uses_standalone_mode_without_result_comparison(self) -> None:
        from triton_agent.commands.convert import _verify_converted_output

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            converted = workspace / "triton_kernel.py"
            standalone_test = workspace / "test_kernel.py"
            operator.write_text("print('src')\n", encoding="utf-8")
            converted.write_text("print('dst')\n", encoding="utf-8")
            standalone_test.write_text("# test-mode: standalone\n", encoding="utf-8")

            request = AgentRequest(
                command_kind=CommandKind.CONVERT,
                input_path=operator.resolve(),
                operator_path=operator.resolve(),
                output_path=converted.resolve(),
                test_mode="differential",
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-convert-pytorch-operator",
                prompt="convert",
                workdir=workspace.resolve(),
            )

            observed_modes: list[str] = []
            observed_targets: list[Path] = []

            def _fake_run_local_test(test_path, operator_path, test_mode, *, verbose=False):
                del test_path, verbose
                observed_modes.append(test_mode)
                observed_targets.append(operator_path)
                return AgentResult(return_code=0, stdout="ok\n", stderr=""), None

            with patch("triton_agent.commands.convert.run_local_test", side_effect=_fake_run_local_test):
                with patch("triton_agent.commands.convert.compare_result_files") as compare_mock:
                    verification = _verify_converted_output(request)

        self.assertEqual(verification.return_code, 0)
        self.assertEqual(observed_modes, ["standalone"])
        self.assertEqual(observed_targets, [converted.resolve()])
        compare_mock.assert_not_called()

    def test_verify_converted_output_prefers_requested_standalone_test_when_both_exist(self) -> None:
        from triton_agent.commands.convert import _verify_converted_output

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            converted = workspace / "triton_kernel.py"
            differential_test = workspace / "differential_test_kernel.py"
            standalone_test = workspace / "test_kernel.py"
            operator.write_text("print('src')\n", encoding="utf-8")
            converted.write_text("print('dst')\n", encoding="utf-8")
            differential_test.write_text("# test-mode: differential\n", encoding="utf-8")
            standalone_test.write_text("# test-mode: standalone\n", encoding="utf-8")

            request = AgentRequest(
                command_kind=CommandKind.CONVERT,
                input_path=operator.resolve(),
                operator_path=operator.resolve(),
                output_path=converted.resolve(),
                test_mode="standalone",
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-convert-pytorch-operator",
                prompt="convert",
                workdir=workspace.resolve(),
            )

            observed_test_calls: list[tuple[Path, Path, str]] = []

            def _fake_run_local_test(test_path, operator_path, test_mode, *, verbose=False):
                del verbose
                observed_test_calls.append((test_path, operator_path, test_mode))
                return AgentResult(return_code=0, stdout="ok\n", stderr=""), None

            with patch("triton_agent.commands.convert.run_local_test", side_effect=_fake_run_local_test):
                with patch("triton_agent.commands.convert.compare_result_files") as compare_mock:
                    verification = _verify_converted_output(request)

        self.assertEqual(verification.return_code, 0)
        self.assertEqual(
            observed_test_calls,
            [(standalone_test.resolve(), converted.resolve(), "standalone")],
        )
        compare_mock.assert_not_called()

    def test_verify_converted_output_reports_standalone_failure_context(self) -> None:
        from triton_agent.commands.convert import _verify_converted_output

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            converted = workspace / "triton_kernel.py"
            standalone_test = workspace / "test_kernel.py"
            operator.write_text("print('src')\n", encoding="utf-8")
            converted.write_text("print('dst')\n", encoding="utf-8")
            standalone_test.write_text("# test-mode: standalone\n", encoding="utf-8")

            request = AgentRequest(
                command_kind=CommandKind.CONVERT,
                input_path=operator.resolve(),
                operator_path=operator.resolve(),
                output_path=converted.resolve(),
                test_mode="differential",
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-convert-pytorch-operator",
                prompt="convert",
                workdir=workspace.resolve(),
            )

            def _fake_run_local_test(test_path, operator_path, test_mode, *, verbose=False):
                del test_path, test_mode, verbose
                self.assertEqual(operator_path, converted.resolve())
                return AgentResult(return_code=7, stdout="", stderr="boom\n"), None

            with patch("triton_agent.commands.convert.run_local_test", side_effect=_fake_run_local_test):
                verification = _verify_converted_output(request)

        self.assertEqual(verification.return_code, 7)
        self.assertIn("Standalone test file:", verification.summary)
        self.assertNotIn("Differential test file:", verification.summary)
        self.assertIn("Converted output:", verification.summary)

    def test_verify_converted_output_reuses_existing_differential_baseline_result(self) -> None:
        from triton_agent.commands.convert import _verify_converted_output

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            converted = workspace / "triton_kernel.py"
            test_file = workspace / "differential_test_kernel.py"
            baseline_result = workspace / "kernel_result.pt"
            candidate_result = workspace / "triton_kernel_result.pt"
            operator.write_text("print('src')\n", encoding="utf-8")
            converted.write_text("print('dst')\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\n", encoding="utf-8")
            baseline_result.write_text("baseline\n", encoding="utf-8")

            request = AgentRequest(
                command_kind=CommandKind.CONVERT,
                input_path=operator.resolve(),
                operator_path=operator.resolve(),
                output_path=converted.resolve(),
                test_mode="differential",
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-convert-pytorch-operator",
                prompt="convert",
                workdir=workspace.resolve(),
            )

            observed_test_calls: list[tuple[Path, Path, str]] = []

            def _fake_run_local_test(test_path, operator_path, test_mode, *, verbose=False):
                del verbose
                observed_test_calls.append((test_path, operator_path, test_mode))
                candidate_result.write_text("candidate\n", encoding="utf-8")
                return AgentResult(return_code=0, stdout="candidate\n", stderr=""), candidate_result

            with patch("triton_agent.commands.convert.run_local_test", side_effect=_fake_run_local_test):
                with patch("triton_agent.commands.convert.compare_result_files", return_value=0) as compare_mock:
                    verification = _verify_converted_output(request)

        self.assertEqual(verification.return_code, 0)
        self.assertEqual(
            observed_test_calls,
            [(test_file.resolve(), converted.resolve(), "differential")],
        )
        compare_mock.assert_called_once_with(
            baseline_result.resolve(),
            candidate_result,
        )

    def test_handle_convert_repairs_once_after_cli_verification_failure(self) -> None:
        from triton_agent.commands.convert import handle_convert

        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            test_file = workspace / "differential_test_kernel.py"
            operator.write_text("print('src')\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\n", encoding="utf-8")
            args = parser.parse_args(["convert", "-i", str(operator)])

            prompts: list[str] = []
            observed_test_calls: list[tuple[Path, Path, str]] = []

            def _fake_run_convert(request):
                prompts.append(request.prompt)
                assert request.output_path is not None
                request.output_path.write_text("print('dst')\n", encoding="utf-8")
                return AgentResult(return_code=0, stdout="agent ok\n", stderr="")

            call_index = {"value": 0}

            def _fake_run_local_test(test_path, operator_path, test_mode, *, verbose=False):
                del verbose
                observed_test_calls.append((test_path, operator_path, test_mode))
                call_index["value"] += 1
                current = call_index["value"]
                archive = workspace / f"archive_{current}.pt"
                if current in {1, 3}:
                    archive.write_text("oracle\n", encoding="utf-8")
                    return AgentResult(return_code=0, stdout=f"oracle-{current}\n", stderr=""), archive
                return AgentResult(return_code=0, stdout=f"candidate-{current}\n", stderr=""), archive

            compare_codes = iter([1, 0])

            with patch("triton_agent.commands.convert.run_convert_request", side_effect=_fake_run_convert):
                with patch("triton_agent.commands.convert.run_local_test", side_effect=_fake_run_local_test):
                    with patch("triton_agent.commands.convert.compare_result_files", side_effect=lambda *_args: next(compare_codes)):
                        exit_code = handle_convert(parser, args)

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(prompts), 2)
            self.assertIn("Follow-up convert verification failed.", prompts[1])
            self.assertIn("Differential test file:", prompts[1])
            self.assertIn("Converted operator:", prompts[1])
            self.assertIn("Comparison result: compare-result failed.", prompts[1])
            self.assertEqual(
                observed_test_calls,
                [
                    (test_file.resolve(), operator.resolve(), "differential"),
                    ((workspace / "differential_test_kernel.py").resolve(), (workspace / "triton_kernel.py").resolve(), "differential"),
                    ((workspace / "differential_test_kernel.py").resolve(), (workspace / "triton_kernel.py").resolve(), "differential"),
                ],
            )

    def test_handle_convert_stops_after_second_cli_verification_failure(self) -> None:
        from triton_agent.commands.convert import handle_convert

        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            test_file = workspace / "differential_test_kernel.py"
            operator.write_text("print('src')\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\n", encoding="utf-8")
            args = parser.parse_args(["convert", "-i", str(operator)])

            def _fake_run_convert(request):
                assert request.output_path is not None
                request.output_path.write_text("print('dst')\n", encoding="utf-8")
                return AgentResult(return_code=0, stdout="agent ok\n", stderr="")

            run_calls = {"count": 0}

            def _fake_run_local_test(test_path, operator_path, test_mode, *, verbose=False):
                del test_path, operator_path, test_mode, verbose
                run_calls["count"] += 1
                archive = workspace / f"archive_{run_calls['count']}.pt"
                if run_calls["count"] == 1:
                    archive.write_text("oracle\n", encoding="utf-8")
                return AgentResult(return_code=0, stdout="", stderr=""), archive

            with patch("triton_agent.commands.convert.run_convert_request", side_effect=_fake_run_convert):
                with patch("triton_agent.commands.convert.run_local_test", side_effect=_fake_run_local_test):
                    with patch("triton_agent.commands.convert.compare_result_files", return_value=1):
                        exit_code = handle_convert(parser, args)

            self.assertEqual(exit_code, 1)
            self.assertEqual(run_calls["count"], 3)

    def test_run_convert_request_writes_tool_trace_summary(self) -> None:
        from triton_agent.convert.orchestration import build_convert_request, run_convert_request

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            trace_path = workdir / "triton-agent-logs" / "run-001" / "tool-traces.jsonl"
            request = build_convert_request(
                workdir / "kernel.py",
                workdir / "kernel.py",
                workdir,
                ConvertOptions(
                    interact=False,
                    verbose=False,
                    stream_output=False,
                    force_overwrite=False,
                    agent_name="codex",
                    remote=None,
                    remote_workdir=None,
                    output=None,
                    test_mode="differential",
                    prompt=None,
                    log_tools=True,
                ),
            )
            assert request.extra_env is not None
            request.extra_env[TRACE_PATH_ENV] = str(trace_path)

            class DummyRunner:
                def run(self, request):
                    del request
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    trace_path.write_text(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "type": "agent_invocation",
                                "phase": "end",
                                "command_kind": "convert",
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    return AgentResult(return_code=0, stdout="", stderr="")

            with patch("triton_agent.convert.orchestration.SkillLinkManager.prepare_skills", return_value=()):
                with patch("triton_agent.convert.orchestration.SkillLinkManager.cleanup", return_value=[]):
                    with patch("triton_agent.convert.orchestration.create_runner", return_value=DummyRunner()):
                        result = run_convert_request(request)

            self.assertEqual(result.return_code, 0)
            summary = json.loads((trace_path.parent / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["command_kind"], "convert")
            self.assertEqual(summary["tool_trace_capability"], "agent_invocation_only")


class ConvertBatchTests(unittest.TestCase):
    def test_resolve_batch_convert_operator_file_excludes_triton_prefixed_candidates(self) -> None:
        from triton_agent.convert.batch import resolve_batch_convert_operator_file

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "triton_kernel.py").write_text("converted\n", encoding="utf-8")
            (workspace / "kernel.py").write_text("source\n", encoding="utf-8")

            resolved = resolve_batch_convert_operator_file(workspace)

            self.assertEqual(resolved, workspace / "kernel.py")

    def test_run_convert_batch_accepts_root_as_single_workspace(self) -> None:
        from triton_agent.convert.batch import run_convert_batch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "operator.py").write_text("print('x')\n", encoding="utf-8")

            seen_inputs: list[Path] = []

            def _fake_run(request, stdout=None, stderr=None):
                del stdout, stderr
                seen_inputs.append(request.input_path)
                return AgentResult(return_code=0, stdout="", stderr="")

            exit_code = run_convert_batch(
                root,
                ConvertOptions(
                    interact=False,
                    verbose=False,
                    stream_output=False,
                    force_overwrite=False,
                    agent_name="codex",
                    remote=None,
                    remote_workdir=None,
                    output=None,
                    test_mode="differential",
                    prompt=None,
                ),
                max_concurrency=1,
                run_request=_fake_run,
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(seen_inputs, [root / "operator.py"])

    def test_run_convert_batch_applies_user_prompt_to_each_workspace_request(self) -> None:
        from triton_agent.convert.batch import run_convert_batch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("kernel_a", "kernel_b"):
                workspace = root / name
                workspace.mkdir()
                operator = workspace / "kernel.py"
                operator.write_text("print('x')\n", encoding="utf-8")

            options = ConvertOptions(
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                remote=None,
                remote_workdir=None,
                output=None,
                test_mode="differential",
                prompt="Avoid changing numerics.",
            )

            captured_prompts: list[str] = []

            def _fake_run(request, stdout=None, stderr=None):
                del stdout, stderr
                captured_prompts.append(request.prompt)
                return AgentResult(return_code=0, stdout="ok", stderr="")

            exit_code = run_convert_batch(
                root,
                options,
                max_concurrency=1,
                run_request=_fake_run,
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(captured_prompts), 2)
            for prompt in captured_prompts:
                self.assertIn("Additional user instructions:", prompt)
                self.assertIn("Avoid changing numerics.", prompt)

    def test_run_convert_batch_assigns_affinity_env_per_workspace(self) -> None:
        from triton_agent.convert.batch import run_convert_batch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("alpha", "beta"):
                workspace = root / name
                workspace.mkdir()
                operator = workspace / "kernel.py"
                operator.write_text("print('x')\n", encoding="utf-8")

            seen_devices: list[Optional[str]] = []

            def _fake_run(request, stdout=None, stderr=None):
                del stdout, stderr
                seen_devices.append((request.extra_env or {}).get("ASCEND_RT_VISIBLE_DEVICES"))
                return AgentResult(return_code=0, stdout="ok", stderr="")

            with patch.dict(environ, {"TRITON_AGENT_BATCH_NPU_DEVICES": "0,1"}, clear=False):
                exit_code = run_convert_batch(
                    root,
                    ConvertOptions(
                        interact=False,
                        verbose=False,
                        stream_output=False,
                        force_overwrite=False,
                        agent_name="codex",
                        remote=None,
                        remote_workdir=None,
                        output=None,
                        test_mode="differential",
                        prompt=None,
                    ),
                    max_concurrency=2,
                    run_request=_fake_run,
                )

            self.assertEqual(exit_code, 0)
            self.assertCountEqual(seen_devices, ["0", "1"])

    def test_run_convert_batch_allows_same_device_when_workers_per_npu_gt_1(self) -> None:
        from triton_agent.convert.batch import run_convert_batch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("alpha", "beta"):
                workspace = root / name
                workspace.mkdir()
                (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

            seen_devices: list[Optional[str]] = []

            def _fake_run(request, stdout=None, stderr=None):
                del stdout, stderr
                seen_devices.append((request.extra_env or {}).get("ASCEND_RT_VISIBLE_DEVICES"))
                return AgentResult(return_code=0, stdout="ok", stderr="")

            env_vars = {
                "TRITON_AGENT_BATCH_NPU_DEVICES": "0",
                "TRITON_AGENT_BATCH_WORKERS_PER_NPU": "2",
            }
            with patch.dict(environ, env_vars, clear=False):
                exit_code = run_convert_batch(
                    root,
                    ConvertOptions(
                        interact=False,
                        verbose=False,
                        stream_output=False,
                        force_overwrite=False,
                        agent_name="codex",
                        remote=None,
                        remote_workdir=None,
                        output=None,
                        test_mode="differential",
                        prompt=None,
                    ),
                    max_concurrency=2,
                    run_request=_fake_run,
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(seen_devices), 2)
            self.assertEqual(seen_devices, ["0", "0"])

    def test_run_convert_batch_does_not_inject_affinity_env_when_mcp_enabled(self) -> None:
        from triton_agent.convert.batch import run_convert_batch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("alpha", "beta"):
                workspace = root / name
                workspace.mkdir()
                (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

            seen_devices: list[Optional[str]] = []

            def _fake_run(request, stdout=None, stderr=None):
                del stdout, stderr
                seen_devices.append((request.extra_env or {}).get("ASCEND_RT_VISIBLE_DEVICES"))
                return AgentResult(return_code=0, stdout="ok", stderr="")

            with patch.dict(environ, {"TRITON_AGENT_BATCH_NPU_DEVICES": "0"}, clear=False):
                exit_code = run_convert_batch(
                    root,
                    ConvertOptions(
                        interact=False,
                        verbose=False,
                        stream_output=False,
                        force_overwrite=False,
                        agent_name="codex",
                        remote=None,
                        remote_workdir=None,
                        output=None,
                        test_mode="differential",
                        prompt=None,
                        enable_mcp=True,
                    ),
                    max_concurrency=2,
                    run_request=_fake_run,
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(seen_devices, [None, None])

    def test_run_convert_request_enters_managed_mcp_scope_when_request_requires_mcp(self) -> None:
        from triton_agent.convert.orchestration import run_convert_request

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            request = AgentRequest(
                command_kind=CommandKind.CONVERT,
                input_path=workspace / "op.py",
                operator_path=workspace / "op.py",
                output_path=workspace / "triton_op.py",
                test_mode="differential",
                bench_mode=None,
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                skill_name="triton-npu-convert-pytorch-operator",
                prompt="Prompt body",
                workdir=workspace,
                mcp_servers=("triton-agent-run-eval",),
            )

            entered: list[str] = []

            class _DummyScope:
                def __enter__(self):
                    entered.append("enter")
                    return None

                def __exit__(self, exc_type, exc, tb):
                    entered.append("exit")
                    return False

            class DummyRunner:
                def run(self, request):
                    del request
                    return AgentResult(return_code=0, stdout="", stderr="")

            with patch("triton_agent.convert.orchestration.SkillLinkManager.prepare_skills", return_value=()):
                with patch("triton_agent.convert.orchestration.SkillLinkManager.cleanup", return_value=[]):
                    with patch("triton_agent.convert.orchestration.managed_mcp_scope", return_value=_DummyScope()):
                        with patch("triton_agent.convert.orchestration.create_runner", return_value=DummyRunner()):
                            result = run_convert_request(request)

            self.assertEqual(result.return_code, 0)
            self.assertEqual(entered, ["enter", "exit"])

    def test_render_batch_convert_results_renders_summary(self) -> None:
        from triton_agent.convert.batch import BatchConvertResult, render_batch_convert_results

        stream = StringIO()
        exit_code = render_batch_convert_results(
            [
                BatchConvertResult(Path("/tmp/a"), True, "converted a.py"),
                BatchConvertResult(Path("/tmp/b"), False, "boom"),
            ],
            stdout=stream,
        )

        self.assertEqual(exit_code, 1)
        output = stream.getvalue()
        self.assertIn("[OK] a: converted a.py", output)
        self.assertIn("[FAIL] b: boom", output)
        self.assertIn("Summary: 1 succeeded, 1 failed", output)

    def test_main_convert_batch_show_output_prefixes_workspace_streams(self) -> None:
        from triton_agent.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("alpha", "beta"):
                workspace = root / name
                workspace.mkdir()
                (workspace / f"{name}.py").write_text("print('x')\n", encoding="utf-8")

            stdout = StringIO()

            def _fake_run(request, stdout=None, stderr=None):
                if stdout is not None:
                    stdout.write("convert start\n")
                if stderr is not None:
                    stderr.write("warn line\n")
                return AgentResult(return_code=0, stdout="convert start\n", stderr="warn line\n")

            with patch("triton_agent.convert.batch.run_convert_request", side_effect=_fake_run):
                with redirect_stdout(stdout):
                    exit_code = main(["convert-batch", "-i", str(root)])

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertIn("[alpha] convert start", rendered)
            self.assertIn("[beta] convert start", rendered)
            self.assertIn("Summary: 2 succeeded, 0 failed", rendered)


if __name__ == "__main__":
    unittest.main()
