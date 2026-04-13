import sys
import unittest
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.verbose import emit_verbose


class _TtyStringIO(StringIO):
    def isatty(self) -> bool:
        return True


class VerboseTests(unittest.TestCase):
    def test_emit_verbose_colors_agents_prefix_on_tty(self) -> None:
        stream = _TtyStringIO()

        emit_verbose(stream, "agents", "message")

        output = stream.getvalue()
        self.assertTrue(output.startswith("\033[36m[agents]\033[0m "))


if __name__ == "__main__":
    unittest.main()
