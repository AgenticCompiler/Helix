import sys
import unittest
from errno import EIO
from io import StringIO
from typing import Optional
from unittest.mock import patch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from triton_agent.process_runner import (
    run_buffered_process,
    run_interactive_process,
    run_process,
    run_streaming_process,
)


class BufferedProcessRunnerTests(unittest.TestCase):
    def test_collects_stdout_and_session_id(self) -> None:
        process = _BufferedFakeProcess(stdout_lines=["hello\n"], stderr_text="", returncode=0)
        with patch("triton_agent.process_runner.subprocess.Popen", return_value=process):
            result = run_buffered_process(
                ["codex", "exec"],
                "/tmp",
                stall_timeout_seconds=10,
                session_id_extractor=lambda line: "session-1" if "hello" in line else None,
            )
        self.assertEqual(result.stdout, "hello\n")
        self.assertEqual(result.session_id, "session-1")

    def test_buffered_filter_can_remove_diff_blocks(self) -> None:
        from triton_agent.codex_runner import _UnifiedDiffFilter

        process = _BufferedFakeProcess(
            stdout_lines=[
                "before\n",
                "diff --git a/x b/x\n",
                "new file mode 100644\n",
                "@@ -0,0 +1 @@\n",
                "+hello\n",
                "after\n",
            ],
            stderr_text="",
            returncode=0,
        )
        with patch("triton_agent.process_runner.subprocess.Popen", return_value=process):
            result = run_buffered_process(
                ["codex", "exec"],
                "/tmp",
                stall_timeout_seconds=10,
                session_id_extractor=lambda _line: None,
                output_filter=_UnifiedDiffFilter(),
            )
        self.assertEqual(result.stdout, "before\nafter\n")


class StreamingProcessRunnerTests(unittest.TestCase):
    def test_streams_chunks_and_collects_stdout(self) -> None:
        stdout = StringIO()
        process = _StreamingFakeProcess(wait_code=0)
        chunks = [b"line one\n", b"line two\n", b""]
        with patch("triton_agent.process_runner.pty.openpty", return_value=(11, 12)):
            with patch("triton_agent.process_runner.subprocess.Popen", return_value=process):
                with patch("triton_agent.process_runner.select.select", side_effect=[([11], [], []), ([11], [], []), ([11], [], [])]):
                    with patch("triton_agent.process_runner.os.read", side_effect=chunks):
                        with patch("triton_agent.process_runner.os.close"):
                            result = run_streaming_process(
                                ["codex", "exec"],
                                "/tmp",
                                stall_timeout_seconds=10,
                                stdout=stdout,
                            )
        self.assertEqual(stdout.getvalue(), "line one\nline two\n")
        self.assertEqual(result.stdout, "line one\nline two\n")

    def test_streaming_filter_can_remove_diff_blocks(self) -> None:
        from triton_agent.codex_runner import _UnifiedDiffFilter

        stdout = StringIO()
        process = _StreamingFakeProcess(wait_code=0)
        chunks = [b"before\n", b"diff --git a/x b/x\n@@ -0,0 +1 @@\n+hello\n", b"after\n", b""]
        with patch("triton_agent.process_runner.pty.openpty", return_value=(11, 12)):
            with patch("triton_agent.process_runner.subprocess.Popen", return_value=process):
                with patch(
                    "triton_agent.process_runner.select.select",
                    side_effect=[([11], [], []), ([11], [], []), ([11], [], []), ([11], [], [])],
                ):
                    with patch("triton_agent.process_runner.os.read", side_effect=chunks):
                        with patch("triton_agent.process_runner.os.close"):
                            result = run_streaming_process(
                                ["codex", "exec"],
                                "/tmp",
                                stall_timeout_seconds=10,
                                stdout=stdout,
                                output_filter=_UnifiedDiffFilter(),
                            )
        self.assertEqual(stdout.getvalue(), "before\nafter\n")
        self.assertEqual(result.stdout, "before\nafter\n")

    def test_treats_eio_after_child_exit_as_clean_eof(self) -> None:
        stdout = StringIO()
        process = _StreamingFakeProcess(wait_code=0)
        with patch("triton_agent.process_runner.pty.openpty", return_value=(11, 12)):
            with patch("triton_agent.process_runner.subprocess.Popen", return_value=process):
                with patch("triton_agent.process_runner.select.select", side_effect=[([11], [], [])]):
                    with patch(
                        "triton_agent.process_runner.os.read",
                        side_effect=OSError(EIO, "Input/output error"),
                    ):
                        with patch("triton_agent.process_runner.os.close"):
                            result = run_streaming_process(
                                ["codex", "exec"],
                                "/tmp",
                                stall_timeout_seconds=10,
                                stdout=stdout,
                            )
        self.assertEqual(result.return_code, 0)
        self.assertEqual(result.stdout, "")
        self.assertEqual(stdout.getvalue(), "")


class InteractiveProcessRunnerTests(unittest.TestCase):
    def test_returns_interactive_result(self) -> None:
        completed = _CompletedProcess(returncode=0)
        with patch("triton_agent.process_runner.subprocess.run", return_value=completed):
            result = run_interactive_process(["codex"], "/tmp")
        self.assertEqual(result.return_code, 0)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")


class UnifiedProcessRunnerTests(unittest.TestCase):
    def test_dispatches_interactive_mode(self) -> None:
        with patch("triton_agent.process_runner.run_interactive_process", return_value=_result()) as mocked:
            run_process(["codex"], "/tmp", mode="interactive")
        mocked.assert_called_once()

    def test_dispatches_streaming_mode(self) -> None:
        with patch("triton_agent.process_runner.run_streaming_process", return_value=_result()) as mocked:
            run_process(["codex"], "/tmp", mode="streaming")
        mocked.assert_called_once()

    def test_dispatches_buffered_mode(self) -> None:
        with patch("triton_agent.process_runner.run_buffered_process", return_value=_result()) as mocked:
            run_process(
                ["codex"],
                "/tmp",
                mode="buffered",
                stall_timeout_seconds=10,
                session_id_extractor=lambda _line: None,
            )
        mocked.assert_called_once()


class _BufferedFakeStdout:
    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)

    def readline(self) -> str:
        if self._lines:
            return self._lines.pop(0)
        return ""


class _BufferedFakeStderr:
    def __init__(self, text: str) -> None:
        self._text = text

    def read(self) -> str:
        return self._text


class _BufferedFakeProcess:
    def __init__(self, stdout_lines: list[str], stderr_text: str, returncode: int) -> None:
        self.stdout = _BufferedFakeStdout(stdout_lines)
        self.stderr = _BufferedFakeStderr(stderr_text)
        self.returncode = returncode

    def poll(self) -> Optional[int]:
        if self.stdout._lines:
            return None
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 1


class _StreamingFakeProcess:
    def __init__(self, wait_code: int) -> None:
        self._wait_code = wait_code

    def poll(self) -> Optional[int]:
        return 0

    def wait(self) -> int:
        return self._wait_code

    def terminate(self) -> None:
        self._wait_code = 1


class _CompletedProcess:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


def _result():
    from triton_agent.models import AgentResult

    return AgentResult(return_code=0, stdout="", stderr="")


if __name__ == "__main__":
    unittest.main()
