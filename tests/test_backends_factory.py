import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.backends.claude import ClaudeRunner
from triton_agent.backends.codex import CodexRunner
from triton_agent.backends.factory import create_runner
from triton_agent.backends.openhands import OpenHandsRunner
from triton_agent.backends.opencode import OpenCodeRunner
from triton_agent.backends.pi import PiRunner
from triton_agent.backends.traecli import TraeCLIRunner


class BackendFactoryTests(unittest.TestCase):
    def test_create_runner_returns_expected_backend_type(self) -> None:
        self.assertIsInstance(create_runner("codex"), CodexRunner)
        self.assertIsInstance(create_runner("opencode"), OpenCodeRunner)
        self.assertIsInstance(create_runner("pi"), PiRunner)
        self.assertIsInstance(create_runner("claude"), ClaudeRunner)
        self.assertIsInstance(create_runner("openhands"), OpenHandsRunner)
        self.assertIsInstance(create_runner("traecli"), TraeCLIRunner)

    def test_create_runner_rejects_unknown_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported agent backend"):
            create_runner("unknown")

    def test_create_runner_defaults_stall_timeout_to_900(self) -> None:
        env = os.environ.copy()
        env.pop("TRITON_AGENT_STALL_TIMEOUT_SECONDS", None)
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(create_runner("codex").stall_timeout_seconds, 900)

    def test_create_runner_reads_stall_timeout_from_env(self) -> None:
        with patch.dict(os.environ, {"TRITON_AGENT_STALL_TIMEOUT_SECONDS": "1800"}, clear=False):
            self.assertEqual(create_runner("opencode").stall_timeout_seconds, 1800)

    def test_create_runner_disables_stall_timeout_when_env_is_zero(self) -> None:
        with patch.dict(os.environ, {"TRITON_AGENT_STALL_TIMEOUT_SECONDS": "0"}, clear=False):
            self.assertEqual(create_runner("codex").stall_timeout_seconds, 0)


if __name__ == "__main__":
    unittest.main()
