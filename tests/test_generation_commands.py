import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.cli import build_parser
from triton_agent.commands.generation import handle_gen_eval, handle_gen_test
from triton_agent.generation.models import GenerationOptions
from triton_agent.generation.outputs import (
    prepare_generation_targets,
    resolve_generation_output_path,
)
from triton_agent.generation.orchestration import build_generation_request
from triton_agent.models import AgentResult, CommandKind


class GenerationHelpersTests(unittest.TestCase):
    def test_generation_orchestration_module_replaces_runtime_module(self) -> None:
        self.assertIsNotNone(importlib.util.find_spec("triton_agent.generation.orchestration"))
        self.assertIsNone(importlib.util.find_spec("triton_agent.generation.runtime"))

    def test_resolve_generation_output_path_uses_differential_name_for_gen_test(self) -> None:
        operator = Path("/tmp/kernel.py")

        output_path = resolve_generation_output_path(
            CommandKind.GEN_TEST,
            operator,
            explicit_output=None,
            test_mode="differential",
        )

        self.assertEqual(output_path, Path("/tmp/differential_test_kernel.py"))

    def test_prepare_generation_target_rejects_existing_file_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "test_kernel.py"
            output.write_text("existing", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "Output file already exists"):
                prepare_generation_targets(
                    CommandKind.GEN_TEST,
                    Path(tmp) / "kernel.py",
                    output,
                    test_mode="standalone",
                    force_overwrite=False,
                )

    def test_prepare_generation_targets_rejects_existing_gen_eval_artifacts_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "kernel.py"
            input_path.write_text("print('x')\n", encoding="utf-8")
            test_output = Path(tmp) / "differential_test_kernel.py"
            test_output.write_text("existing test", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "Output file already exists"):
                prepare_generation_targets(
                    CommandKind.GEN_EVAL,
                    input_path,
                    None,
                    test_mode="differential",
                    force_overwrite=False,
                )

    def test_prepare_generation_targets_removes_existing_gen_eval_artifacts_with_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "kernel.py"
            input_path.write_text("print('x')\n", encoding="utf-8")
            test_output = Path(tmp) / "differential_test_kernel.py"
            bench_output = Path(tmp) / "bench_kernel.py"
            result_output = Path(tmp) / "kernel_result.pt"
            perf_output = Path(tmp) / "kernel_perf.txt"
            test_output.write_text("existing test", encoding="utf-8")
            bench_output.write_text("existing bench", encoding="utf-8")
            result_output.write_text("existing result", encoding="utf-8")
            perf_output.write_text("existing perf", encoding="utf-8")

            messages = prepare_generation_targets(
                CommandKind.GEN_EVAL,
                input_path,
                None,
                test_mode="differential",
                force_overwrite=True,
            )

            self.assertFalse(test_output.exists())
            self.assertFalse(bench_output.exists())
            self.assertFalse(result_output.exists())
            self.assertFalse(perf_output.exists())
            self.assertEqual(
                messages,
                [
                    f"removed existing output file {test_output}",
                    f"removed existing output file {bench_output}",
                    f"removed existing output file {result_output}",
                    f"removed existing output file {perf_output}",
                ],
            )

    def test_build_generation_request_for_gen_eval_uses_restricted_skills(self) -> None:
        request = build_generation_request(
            CommandKind.GEN_EVAL,
            Path("/tmp/kernel.py"),
            Path("/tmp/kernel.py"),
            Path("/tmp"),
            GenerationOptions(
                interact=False,
                verbose=False,
                show_output=False,
                force_overwrite=False,
                agent_name="codex",
                remote="alice@example.com",
                remote_workdir="/tmp/triton-agent",
                min_rounds=None,
                continue_optimize=False,
                output=None,
                test_mode="differential",
                bench_mode="standalone",
            ),
        )

        self.assertEqual(
            request.staged_skill_names,
            (
                "triton-npu-gen-eval-suite",
                "triton-npu-gen-test",
                "triton-npu-gen-bench",
                "triton-npu-run-eval",
            ),
        )
        self.assertEqual(request.skill_name, "triton-npu-gen-eval-suite")

    def test_build_generation_request_for_gen_eval_omits_single_output_path(self) -> None:
        request = build_generation_request(
            CommandKind.GEN_EVAL,
            Path("/tmp/kernel.py"),
            Path("/tmp/kernel.py"),
            Path("/tmp"),
            GenerationOptions(
                interact=False,
                verbose=False,
                show_output=False,
                force_overwrite=False,
                agent_name="codex",
                remote=None,
                remote_workdir=None,
                min_rounds=None,
                continue_optimize=False,
                output=None,
                test_mode="differential",
                bench_mode="standalone",
            ),
        )

        self.assertIsNone(request.output_path)


class GenerationCommandHandlerTests(unittest.TestCase):
    def test_handle_gen_test_rejects_openhands_interactive_mode(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            operator = Path(tmp) / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            args = parser.parse_args(
                [
                    "gen-test",
                    "-i",
                    str(operator),
                    "--agent",
                    "openhands",
                    "--interact",
                ]
            )

            with self.assertRaises(SystemExit) as exc:
                handle_gen_test(parser, args)

            self.assertEqual(exc.exception.code, 2)

    def test_handle_gen_test_builds_request_with_default_output(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            operator = Path(tmp) / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            args = parser.parse_args(
                [
                    "gen-test",
                    "-i",
                    str(operator),
                    "--test-mode",
                    "differential",
                ]
            )

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            captured: dict[str, Optional[Path]] = {}

            def _fake_run(request):
                captured["output_path"] = request.output_path
                return fake_result

            with patch("triton_agent.commands.generation.run_generation_request", side_effect=_fake_run):
                exit_code = handle_gen_test(parser, args)

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                captured["output_path"],
                (Path(tmp) / "differential_test_kernel.py").resolve(),
            )

    def test_handle_gen_eval_builds_request_with_restricted_skills(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            operator = Path(tmp) / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            args = parser.parse_args(
                [
                    "gen-eval",
                    "-i",
                    str(operator),
                    "--remote",
                    "alice@example.com",
                    "--remote-workdir",
                    "/tmp/triton-agent",
                ]
            )

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            captured: dict[str, object] = {}

            def _fake_run(request):
                captured["staged_skill_names"] = request.staged_skill_names
                captured["test_mode"] = request.test_mode
                captured["bench_mode"] = request.bench_mode
                return fake_result

            with patch("triton_agent.commands.generation.run_generation_request", side_effect=_fake_run):
                exit_code = handle_gen_eval(parser, args)

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                captured["staged_skill_names"],
                (
                    "triton-npu-gen-eval-suite",
                    "triton-npu-gen-test",
                    "triton-npu-gen-bench",
                    "triton-npu-run-eval",
                ),
            )
            self.assertEqual(captured["test_mode"], "differential")
            self.assertEqual(captured["bench_mode"], "standalone")

    def test_handle_gen_eval_rejects_existing_generated_artifacts_without_overwrite(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            operator = Path(tmp) / "kernel.py"
            existing_test = Path(tmp) / "differential_test_kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            existing_test.write_text("existing test\n", encoding="utf-8")
            args = parser.parse_args(["gen-eval", "-i", str(operator)])

            with self.assertRaises(SystemExit) as exc:
                handle_gen_eval(parser, args)

            self.assertEqual(exc.exception.code, 2)

    def test_handle_gen_eval_force_overwrite_removes_existing_generated_artifacts(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            operator = Path(tmp) / "kernel.py"
            existing_test = Path(tmp) / "differential_test_kernel.py"
            existing_bench = Path(tmp) / "bench_kernel.py"
            existing_result = Path(tmp) / "kernel_result.pt"
            existing_perf = Path(tmp) / "kernel_perf.txt"
            operator.write_text("print('x')\n", encoding="utf-8")
            existing_test.write_text("existing test\n", encoding="utf-8")
            existing_bench.write_text("existing bench\n", encoding="utf-8")
            existing_result.write_text("existing result\n", encoding="utf-8")
            existing_perf.write_text("existing perf\n", encoding="utf-8")
            args = parser.parse_args(["gen-eval", "-i", str(operator), "--force-overwrite"])

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch("triton_agent.commands.generation.run_generation_request", return_value=fake_result):
                exit_code = handle_gen_eval(parser, args)

            self.assertEqual(exit_code, 0)
            self.assertFalse(existing_test.exists())
            self.assertFalse(existing_bench.exists())
            self.assertFalse(existing_result.exists())
            self.assertFalse(existing_perf.exists())


if __name__ == "__main__":
    unittest.main()
