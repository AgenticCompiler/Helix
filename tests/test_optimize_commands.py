import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.cli import build_parser
from triton_agent.commands.optimize import handle_optimize


class OptimizeCommandHandlerTests(unittest.TestCase):
    def test_handle_optimize_rejects_openhands_interactive_mode(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            operator = Path(tmp) / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            args = parser.parse_args(
                [
                    "optimize",
                    "-i",
                    str(operator),
                    "--agent",
                    "openhands",
                    "--interact",
                ]
            )

            with self.assertRaises(SystemExit) as exc:
                handle_optimize(parser, args)

            self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
