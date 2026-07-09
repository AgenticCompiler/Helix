import sys
import unittest
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.terminal.help import style_help_text, supports_color


class _TtyStringIO(StringIO):
    def isatty(self) -> bool:
        return True


class _NonTtyStringIO(StringIO):
    def isatty(self) -> bool:
        return False


class _NoIsattyStream:
    pass


_EMPTY_ENV: dict[str, str] = {}


class SupportsColorTests(unittest.TestCase):
    def test_tty_enables_color_by_default(self) -> None:
        self.assertTrue(supports_color(_TtyStringIO(), environ=_EMPTY_ENV))

    def test_non_tty_disables_color_by_default(self) -> None:
        self.assertFalse(supports_color(_NonTtyStringIO(), environ=_EMPTY_ENV))

    def test_stream_without_isatty_disables_color(self) -> None:
        self.assertFalse(supports_color(_NoIsattyStream(), environ=_EMPTY_ENV))  # type: ignore[arg-type]

    def test_no_color_disables_color_even_on_tty(self) -> None:
        self.assertFalse(supports_color(_TtyStringIO(), environ={"NO_COLOR": ""}))
        self.assertFalse(supports_color(_TtyStringIO(), environ={"NO_COLOR": "1"}))

    def test_clicolor_zero_disables_color(self) -> None:
        self.assertFalse(supports_color(_TtyStringIO(), environ={"CLICOLOR": "0"}))

    def test_clicolor_force_overrides_non_tty(self) -> None:
        self.assertTrue(supports_color(_NonTtyStringIO(), environ={"CLICOLOR_FORCE": "1"}))

    def test_no_color_disables_color_even_when_forced(self) -> None:
        self.assertFalse(
            supports_color(
                _TtyStringIO(),
                environ={"NO_COLOR": "1", "CLICOLOR_FORCE": "1"},
            )
        )


class StyleHelpTextTests(unittest.TestCase):
    def test_plain_text_returned_when_color_disabled(self) -> None:
        text = "usage: triton-agent [-h] [-v] COMMAND ..."
        result = style_help_text(
            text,
            _NonTtyStringIO(),
            option_tokens={"-h", "--help"},
            env_var_tokens=set(),
            command_tokens=set(),
        )
        self.assertEqual(result, text)
        self.assertNotIn("\033[", result)

    def test_option_token_colored_on_tty(self) -> None:
        text = (
            "usage: triton-agent [-h] [-v] COMMAND ...\n\n"
            "options:\n  -h, --help  show this help\n  -v, --version  show version"
        )
        result = style_help_text(
            text,
            _TtyStringIO(),
            option_tokens={"-h", "--help", "-v", "--version"},
            env_var_tokens=set(),
            command_tokens=set(),
            environ=_EMPTY_ENV,
        )
        self.assertIn("\033[36m-h\033[0m", result)
        self.assertIn("\033[36m--help\033[0m", result)
        self.assertIn("\033[36m-v\033[0m", result)
        self.assertIn("\033[36m--version\033[0m", result)

    def test_env_var_token_colored_on_tty(self) -> None:
        text = "Environment variables:\n  TRITON_AGENT_BATCH_NPU_DEVICES    device pool"
        result = style_help_text(
            text,
            _TtyStringIO(),
            option_tokens=set(),
            env_var_tokens={"TRITON_AGENT_BATCH_NPU_DEVICES"},
            command_tokens=set(),
            environ=_EMPTY_ENV,
        )
        self.assertIn("\033[36mTRITON_AGENT_BATCH_NPU_DEVICES\033[0m", result)

    def test_command_token_colored_on_tty_for_command_list_entry(self) -> None:
        text = "  gen-eval            Generate test harnesses."
        result = style_help_text(
            text,
            _TtyStringIO(),
            option_tokens=set(),
            env_var_tokens=set(),
            command_tokens={"gen-eval"},
            environ=_EMPTY_ENV,
        )
        self.assertIn("\033[36mgen-eval\033[0m", result)

    def test_command_token_word_boundary_avoids_partial_matches_for_command_list(self) -> None:
        text = "  gen-eval-batch      Generate evaluation harnesses."
        result = style_help_text(
            text,
            _TtyStringIO(),
            option_tokens=set(),
            env_var_tokens=set(),
            command_tokens={"gen-eval", "gen-eval-batch"},
            environ=_EMPTY_ENV,
        )
        self.assertIn("\033[36mgen-eval-batch\033[0m", result)
        self.assertNotIn("\033[36mgen-eval\033[0m-batch", result)

    def test_command_token_not_colored_inside_description_prose(self) -> None:
        text = "Generate, run, verify, and optimize NPU operator workflows."
        result = style_help_text(
            text,
            _TtyStringIO(),
            option_tokens=set(),
            env_var_tokens=set(),
            command_tokens={"verify", "optimize"},
            environ=_EMPTY_ENV,
        )
        self.assertEqual(result, text)

    def test_command_token_not_colored_inside_option_or_env_var_text(self) -> None:
        text = (
            "Enable ordinary optimize PT cleanup; "
            "default keeps PT files and does not affect --reset-optimize."
        )
        result = style_help_text(
            text,
            _TtyStringIO(),
            option_tokens={"--reset-optimize"},
            env_var_tokens=set(),
            command_tokens={"optimize"},
            environ=_EMPTY_ENV,
        )
        self.assertNotIn("\033[36moptimize\033[0m", result)
        self.assertIn("\033[36m--reset-optimize\033[0m", result)

    def test_longer_option_matched_before_shorter_prefix(self) -> None:
        text = "  --helpful, --help  show help"
        result = style_help_text(
            text,
            _TtyStringIO(),
            option_tokens={"--help", "--helpful"},
            env_var_tokens=set(),
            command_tokens=set(),
            environ=_EMPTY_ENV,
        )
        self.assertIn("\033[36m--helpful\033[0m", result)
        self.assertIn("\033[36m--help\033[0m", result)

    def test_longer_option_does_not_bleed_into_shorter_prefix(self) -> None:
        text = "  --helpful  show help"
        result = style_help_text(
            text,
            _TtyStringIO(),
            option_tokens={"--help", "--helpful"},
            env_var_tokens=set(),
            command_tokens=set(),
            environ=_EMPTY_ENV,
        )
        self.assertIn("\033[36m--helpful\033[0m", result)
        self.assertNotIn("\033[36m--help\033[0m", result)

    def test_option_word_boundary_avoids_partial_matches(self) -> None:
        text = "The co-here option is not --help."
        result = style_help_text(
            text,
            _TtyStringIO(),
            option_tokens={"-h", "--help"},
            env_var_tokens=set(),
            command_tokens=set(),
            environ=_EMPTY_ENV,
        )
        self.assertNotIn("co\033[36m-h\033[0mere", result)
        self.assertIn("\033[36m--help\033[0m", result)

    def test_env_var_word_boundary_avoids_partial_matches(self) -> None:
        text = "TRITON_AGENT_BATCH_NPU_DEVICES_EXTRA is not supported."
        result = style_help_text(
            text,
            _TtyStringIO(),
            option_tokens=set(),
            env_var_tokens={"TRITON_AGENT_BATCH_NPU_DEVICES"},
            command_tokens=set(),
            environ=_EMPTY_ENV,
        )
        self.assertNotIn("\033[36mTRITON_AGENT_BATCH_NPU_DEVICES\033[0m_EXTRA", result)

    def test_no_color_environment_disables_styling(self) -> None:
        text = "usage: triton-agent [-h] [-v] COMMAND ..."
        result = style_help_text(
            text,
            _TtyStringIO(),
            option_tokens={"-h"},
            env_var_tokens=set(),
            command_tokens=set(),
            environ={"NO_COLOR": "1"},
        )
        self.assertEqual(result, text)

    def test_clicolor_force_enables_styling_for_non_tty(self) -> None:
        text = "usage: triton-agent [-h] [-v] COMMAND ..."
        result = style_help_text(
            text,
            _NonTtyStringIO(),
            option_tokens={"-h"},
            env_var_tokens=set(),
            command_tokens=set(),
            environ={"CLICOLOR_FORCE": "1"},
        )
        self.assertIn("\033[36m-h\033[0m", result)


if __name__ == "__main__":
    unittest.main()
