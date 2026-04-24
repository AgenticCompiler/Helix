import importlib.util
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from contextlib import redirect_stdout

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.cli import build_parser
from triton_agent.generation.models import GenerationOptions
from triton_agent.models import AgentResult, CommandKind


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
                bench_mode=None,
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
        self.assertEqual(request.skill_name, "triton-npu-convert-pytorch-operator")
        self.assertEqual(request.output_path, Path("/tmp/triton_kernel.py"))

    def test_handle_convert_builds_request_with_default_output(self) -> None:
        from triton_agent.commands.convert import handle_convert

        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            operator = Path(tmp) / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
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
                return fake_result

            with patch("triton_agent.commands.convert.run_convert_request", side_effect=_fake_run):
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
                    bench_mode=None,
                ),
                max_concurrency=1,
                run_request=_fake_run,
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(seen_inputs, [root / "operator.py"])

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
                    exit_code = main(["convert-batch", "-i", str(root), "--show-output"])

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue()
            self.assertIn("[alpha] convert start", rendered)
            self.assertIn("[beta] convert start", rendered)
            self.assertIn("Summary: 2 succeeded, 0 failed", rendered)


if __name__ == "__main__":
    unittest.main()
