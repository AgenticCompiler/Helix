import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.cli import build_parser
from triton_agent.commands.generation import handle_gen_test
from triton_agent.generation import prepare_generation_target, resolve_generation_output_path
from triton_agent.models import AgentResult, CommandKind


class GenerationHelpersTests(unittest.TestCase):
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
                prepare_generation_target(CommandKind.GEN_TEST, output, force_overwrite=False)


class GenerationCommandHandlerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
