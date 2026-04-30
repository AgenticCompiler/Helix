import sys
import unittest
from errno import EIO
from io import StringIO
from os import environ
from typing import Any, Optional, cast
from unittest.mock import patch
import signal

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from triton_agent.process_runner import (
    InterruptPolicy,
    run_buffered_process,
    run_interactive_process,
    run_process,
    run_streaming_process,
)


_USE_RETURNCODE = object()


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

    def test_buffered_process_runner_merges_extra_env(self) -> None:
        process = _BufferedFakeProcess(stdout_lines=[], stderr_text="", returncode=0)
        with (
            patch.dict(environ, {"EXISTING_ENV": "base"}, clear=False),
            patch("triton_agent.process_runner.subprocess.Popen", return_value=process) as mocked,
        ):
            run_buffered_process(
                ["codex", "exec"],
                "/tmp",
                stall_timeout_seconds=10,
                session_id_extractor=lambda _line: None,
                extra_env={"ASCEND_RT_VISIBLE_DEVICES": "7"},
            )
        self.assertEqual(mocked.call_args.kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"], "7")
        self.assertEqual(mocked.call_args.kwargs["env"]["EXISTING_ENV"], "base")

    def test_buffered_filter_can_remove_diff_blocks(self) -> None:
        from triton_agent.backends.codex import _UnifiedDiffFilter

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

    def test_buffered_none_returncode_defaults_to_failure(self) -> None:
        process = _BufferedFakeProcess(stdout_lines=[], stderr_text="", returncode=None)
        with patch("triton_agent.process_runner.subprocess.Popen", return_value=process):
            result = run_buffered_process(
                ["codex", "exec"],
                "/tmp",
                stall_timeout_seconds=10,
                session_id_extractor=lambda _line: None,
            )
        self.assertEqual(result.return_code, 1)

    @unittest.skipIf(not hasattr(__import__("os"), "killpg"), "requires POSIX process groups")
    def test_keyboard_interrupt_sends_two_sigints_then_force_kills(self) -> None:
        process = _BufferedFakeProcess(
            stdout_lines=[],
            stderr_text="",
            returncode=None,
            poll_values=[None, None, None, None],
            pid=4321,
            default_poll_return=None,
        )
        with patch("triton_agent.process_runner.subprocess.Popen", return_value=process):
            with patch("triton_agent.process_runner.os.killpg") as mocked_killpg:
                with patch("triton_agent.process_runner.time.sleep"):
                    with patch.object(
                        process.stdout,
                        "readline",
                        side_effect=KeyboardInterrupt,
                    ):
                        result = run_buffered_process(
                            ["codex", "exec"],
                            "/tmp",
                            stall_timeout_seconds=10,
                            session_id_extractor=lambda _line: None,
                            interrupt_policy=InterruptPolicy(
                                first_sigint_grace_seconds=0.01,
                                second_sigint_grace_seconds=0.01,
                            ),
                        )
        self.assertEqual(result.return_code, 130)
        self.assertFalse(result.stalled)
        self.assertEqual(
            mocked_killpg.call_args_list,
            [
                ((4321, signal.SIGINT),),
                ((4321, signal.SIGINT),),
                ((4321, signal.SIGKILL),),
            ],
        )


@unittest.skipIf(sys.platform == "win32", "PTY streaming tests require a POSIX pty")
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

    def test_streaming_collects_session_id(self) -> None:
        stdout = StringIO()
        process = _StreamingFakeProcess(wait_code=0)
        chunks = [b"session id: session-1\n", b""]
        with patch("triton_agent.process_runner.pty.openpty", return_value=(11, 12)):
            with patch("triton_agent.process_runner.subprocess.Popen", return_value=process):
                with patch("triton_agent.process_runner.select.select", side_effect=[([11], [], []), ([11], [], [])]):
                    with patch("triton_agent.process_runner.os.read", side_effect=chunks):
                        with patch("triton_agent.process_runner.os.close"):
                            result = run_streaming_process(
                                ["codex", "exec"],
                                "/tmp",
                                stall_timeout_seconds=10,
                                stdout=stdout,
                                session_id_extractor=lambda text: "session-1" if "session id:" in text else None,
                            )
        self.assertEqual(result.session_id, "session-1")

    def test_streaming_filter_can_remove_diff_blocks(self) -> None:
        from triton_agent.backends.codex import _UnifiedDiffFilter

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

    def test_streaming_filter_preserves_indented_output_after_diff(self) -> None:
        from triton_agent.backends.codex import _UnifiedDiffFilter

        stdout = StringIO()
        process = _StreamingFakeProcess(wait_code=0, poll_values=[None, 0])
        chunks = [
            b"before\n",
            b"diff --git a/x b/x\n@@ -0,0 +1 @@\n+hello\n",
            b"  indented note\nafter\n",
        ]
        with patch("triton_agent.process_runner.pty.openpty", return_value=(11, 12)):
            with patch("triton_agent.process_runner.subprocess.Popen", return_value=process):
                with patch(
                    "triton_agent.process_runner.select.select",
                    side_effect=[
                        ([11], [], []),
                        ([11], [], []),
                        ([11], [], []),
                        ([], [], []),
                        ([], [], []),
                    ],
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
        self.assertEqual(stdout.getvalue(), "before\n  indented note\nafter\n")
        self.assertEqual(result.stdout, "before\n  indented note\nafter\n")

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

    def test_zero_timeout_disables_streaming_stall_detection(self) -> None:
        stdout = StringIO()
        process = _StreamingFakeProcess(wait_code=0, poll_values=[None, 0, 0])
        with patch("triton_agent.process_runner.pty.openpty", return_value=(11, 12)):
            with patch("triton_agent.process_runner.subprocess.Popen", return_value=process):
                with patch(
                    "triton_agent.process_runner.select.select",
                    side_effect=[([], [], []), ([], [], [])],
                ):
                    with patch("triton_agent.process_runner.time.monotonic", side_effect=[0.0, 1.0]):
                        with patch("triton_agent.process_runner.os.close"):
                            result = run_streaming_process(
                                ["codex", "exec"],
                                "/tmp",
                                stall_timeout_seconds=0,
                                stdout=stdout,
                            )
        self.assertFalse(result.stalled)
        self.assertEqual(result.return_code, 0)

    def test_streaming_keyboard_interrupt_uses_interrupt_escalation(self) -> None:
        stdout = StringIO()
        process = _StreamingFakeProcess(
            wait_code=0,
            poll_values=[None, None, None, None],
            pid=1234,
            default_poll_return=None,
        )
        with patch("triton_agent.process_runner.pty.openpty", return_value=(11, 12)):
            with patch("triton_agent.process_runner.subprocess.Popen", return_value=process):
                with patch(
                    "triton_agent.process_runner.select.select",
                    side_effect=KeyboardInterrupt,
                ):
                    with patch("triton_agent.process_runner.os.killpg") as mocked_killpg:
                        with patch("triton_agent.process_runner.time.sleep"):
                            with patch("triton_agent.process_runner.os.close"):
                                result = run_streaming_process(
                                    ["codex", "exec"],
                                    "/tmp",
                                    stall_timeout_seconds=10,
                                    stdout=stdout,
                                    interrupt_policy=InterruptPolicy(
                                        first_sigint_grace_seconds=0.01,
                                        second_sigint_grace_seconds=0.01,
                                    ),
                                )
        self.assertEqual(result.return_code, 130)
        self.assertFalse(result.stalled)
        self.assertEqual(
            mocked_killpg.call_args_list,
            [
                ((1234, signal.SIGINT),),
                ((1234, signal.SIGINT),),
                ((1234, signal.SIGKILL),),
            ],
        )


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
    def __init__(
        self,
        stdout_lines: list[str],
        stderr_text: str,
        returncode: Optional[int],
        poll_values: Optional[list[Optional[int]]] = None,
        pid: int = 1,
        default_poll_return: Any = _USE_RETURNCODE,
    ) -> None:
        self.stdout = _BufferedFakeStdout(stdout_lines)
        self.stderr = _BufferedFakeStderr(stderr_text)
        self.returncode = returncode
        self._poll_values = list(poll_values or [])
        self.pid = pid
        self._default_poll_return = default_poll_return

    def poll(self) -> Optional[int]:
        if self._poll_values:
            return self._poll_values.pop(0)
        if self._default_poll_return is not _USE_RETURNCODE:
            return cast(Optional[int], self._default_poll_return)
        if self.stdout._lines:
            return None
        return 0 if self.returncode is None else self.returncode

    def terminate(self) -> None:
        self.returncode = 1


class _StreamingFakeProcess:
    def __init__(
        self,
        wait_code: int,
        poll_values: Optional[list[Optional[int]]] = None,
        pid: int = 1,
        default_poll_return: Any = 0,
    ) -> None:
        self._wait_code = wait_code
        self._poll_values = list(poll_values or [])
        self.pid = pid
        self._default_poll_return = default_poll_return

    def poll(self) -> Optional[int]:
        if self._poll_values:
            return self._poll_values.pop(0)
        return cast(Optional[int], self._default_poll_return)

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
