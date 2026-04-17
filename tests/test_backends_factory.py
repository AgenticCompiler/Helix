import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
