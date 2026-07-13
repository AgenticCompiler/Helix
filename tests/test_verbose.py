import sys
import unittest
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helix.terminal.verbose import _supports_color, emit_command_block, emit_verbose


class _TtyStringIO(StringIO):
    def isatty(self) -> bool:
        return True


class _TruthyIsattyStream(StringIO):
    def isatty(self) -> object:
        return object()


class VerboseTests(unittest.TestCase):
    def test_emit_verbose_colors_agents_prefix_on_tty(self) -> None:
        stream = _TtyStringIO()

        emit_verbose(stream, "agents", "message")

        output = stream.getvalue()
        self.assertTrue(output.startswith("\033[36m[agents]\033[0m "))

    def test_emit_verbose_colors_command_prefix_on_tty(self) -> None:
        stream = _TtyStringIO()

        emit_verbose(stream, "command", "message")

        output = stream.getvalue()
        self.assertTrue(output.startswith("\033[34m[command]\033[0m "))

    def test_emit_verbose_colors_hooks_prefix_on_tty(self) -> None:
        stream = _TtyStringIO()

        emit_verbose(stream, "hooks", "message")

        output = stream.getvalue()
        self.assertTrue(output.startswith("\033[32m[hooks]\033[0m "))

    def test_emit_command_block_renders_plain_indented_body_without_prefixes(self) -> None:
        stream = StringIO()

        emit_command_block(stream, ["opencode", "/tmp/workspace", "first line\nsecond line"])

        self.assertEqual(
            stream.getvalue(),
            "[command] opencode /tmp/workspace '<prompt>'\n"
            "[command] prompt:\n"
            "  first line\n"
            "  second line\n",
        )

    def test_emit_command_block_renders_tty_body_with_muted_text(self) -> None:
        stream = _TtyStringIO()

        emit_command_block(stream, ["opencode", "/tmp/workspace", "first line\nsecond line"])

        output = stream.getvalue()
        self.assertIn("\033[34m[command]\033[0m opencode /tmp/workspace '<prompt>'\n", output)
        self.assertIn("\033[34m[command]\033[0m prompt:\n", output)
        self.assertIn("\033[38;5;245m  first line\033[0m\n", output)
        self.assertIn("\033[38;5;245m  second line\033[0m\n", output)
        self.assertNotIn("[command]   first line", output)

    def test_supports_color_coerces_truthy_isatty_results_to_bool(self) -> None:
        self.assertIs(_supports_color(_TruthyIsattyStream()), True)


if __name__ == "__main__":
    unittest.main()
