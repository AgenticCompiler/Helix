import sys
import unittest
from errno import EIO
from io import StringIO
import multiprocessing
from os import environ
from pathlib import Path
from queue import Empty
import subprocess
import tempfile
import textwrap
from typing import Any, Optional, cast
from unittest.mock import Mock, patch
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
_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)


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

    def test_buffered_filter_can_remove_bare_hunk_fragments(self) -> None:
        from triton_agent.backends.codex import _UnifiedDiffFilter

        process = _BufferedFakeProcess(
            stdout_lines=[
                "baseline/perf.txt 368.0119\n",
                "opt-round-1/opt_triton_10_SwigluQuant_perf.txt 299.39092500000004\n",
                "\n",
                "     tl.store(scale_ptr + row, inv_scale)\n",
                "\n",
                "@@ -10,0 +11,11 @@\n",
                "+@triton.jit\n",
                "+def _round_half_to_even_tl(values):\n",
                "+    abs_values = tl.abs(values)\n",
                "+    base = tl.floor(abs_values)\n",
                "+    frac = abs_values - base\n",
                "+    base_i = base.to(tl.int32)\n",
                "+    is_odd = (base_i & 1) != 0\n",
                "+    rounded_abs = base + ((frac > 0.5) | ((frac == 0.5) & is_odd)).to(tl.float32)\n",
                "+    return tl.where(values < 0, -rounded_abs, rounded_abs)\n",
                "done\n",
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
        self.assertEqual(
            result.stdout,
            "baseline/perf.txt 368.0119\n"
            "opt-round-1/opt_triton_10_SwigluQuant_perf.txt 299.39092500000004\n"
            "\n"
            "     tl.store(scale_ptr + row, inv_scale)\n"
            "\n"
            "done\n",
        )

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

    def test_buffered_process_drains_large_stderr_without_blocking_child_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            helper = workspace / "emit_large_stderr.py"
            created = workspace / "created.txt"
            helper.write_text(
                textwrap.dedent(
                    """\
                    import sys
                    from pathlib import Path

                    target = Path(sys.argv[1])
                    sys.stderr.write("x" * (256 * 1024) + "\\n")
                    sys.stderr.flush()
                    target.write_text("ok", encoding="utf-8")
                    sys.stdout.write("done\\n")
                    sys.stdout.flush()
                    """
                ),
                encoding="utf-8",
            )

            ctx = multiprocessing.get_context("spawn")
            queue = ctx.Queue()
            worker = ctx.Process(
                target=_run_buffered_large_stderr_repro,
                args=(str(helper), str(created), queue),
            )
            worker.start()
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join()
                self.fail("buffered process runner hung while child emitted large stderr")

            self.assertEqual(worker.exitcode, 0)
            try:
                payload = queue.get(timeout=1)
            except Empty as exc:
                raise AssertionError("buffered process worker did not return a result") from exc

            self.assertNotIn("error", payload)
            self.assertEqual(payload["return_code"], 0)
            self.assertFalse(payload["stalled"])
            self.assertTrue(payload["output_exists"])
            self.assertEqual(payload["output_text"], "ok")
            self.assertEqual(payload["stdout"], "done\n")

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
                sleep_calls = {"count": 0}

                def _sleep(_seconds: float) -> None:
                    if sleep_calls["count"] == 0:
                        sleep_calls["count"] += 1
                        raise KeyboardInterrupt
                    sleep_calls["count"] += 1

                with patch("triton_agent.process_runner.time.sleep", side_effect=_sleep):
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
                ((4321, _SIGKILL),),
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

    def test_streaming_can_write_rendered_chunks_to_incremental_sink(self) -> None:
        stdout = StringIO()
        sink = StringIO()

        def write_chunk(text: str) -> None:
            sink.write(text)

        process = _StreamingFakeProcess(wait_code=0)
        chunks = [b"line one\n", b"line two\n", b""]
        with patch("triton_agent.process_runner.pty.openpty", return_value=(11, 12)):
            with patch("triton_agent.process_runner.subprocess.Popen", return_value=process):
                with patch(
                    "triton_agent.process_runner.select.select",
                    side_effect=[([11], [], []), ([11], [], []), ([11], [], [])],
                ):
                    with patch("triton_agent.process_runner.os.read", side_effect=chunks):
                        with patch("triton_agent.process_runner.os.close"):
                            result = run_streaming_process(
                                ["codex", "exec"],
                                "/tmp",
                                stall_timeout_seconds=10,
                                stdout=stdout,
                                rendered_chunk_sink=write_chunk,
                                collect_stdout=False,
                            )
        self.assertEqual(stdout.getvalue(), "line one\nline two\n")
        self.assertEqual(sink.getvalue(), "line one\nline two\n")
        self.assertEqual(result.stdout, "")

    def test_streaming_marks_retryable_failure_from_raw_chunks(self) -> None:
        process = _StreamingFakeProcess(wait_code=1)
        chunks = [b"ERROR: exceeded retry limit, ", b"last status: 429 Too Many Requests\n", b""]
        with patch("triton_agent.process_runner.pty.openpty", return_value=(11, 12)):
            with patch("triton_agent.process_runner.subprocess.Popen", return_value=process):
                with patch(
                    "triton_agent.process_runner.select.select",
                    side_effect=[([11], [], []), ([11], [], []), ([11], [], [])],
                ):
                    with patch("triton_agent.process_runner.os.read", side_effect=chunks):
                        with patch("triton_agent.process_runner.os.close"):
                            result = run_streaming_process(
                                ["codex", "exec"],
                                "/tmp",
                                stall_timeout_seconds=10,
                                collect_stdout=False,
                            )
        self.assertTrue(result.retryable_failure)

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

    def test_streaming_filter_can_remove_bare_hunk_fragments(self) -> None:
        from triton_agent.backends.codex import _UnifiedDiffFilter

        stdout = StringIO()
        process = _StreamingFakeProcess(wait_code=0, poll_values=[None, 0])
        chunks = [
            b"baseline/perf.txt 368.0119\n",
            b"opt-round-1/opt_triton_10_SwigluQuant_perf.txt 299.39092500000004\n\n",
            b"     tl.store(scale_ptr + row, inv_scale)\n\n@@ -10,0 +11,3 @@\n",
            b"+@triton.jit\n+def _round_half_to_even_tl(values):\n+    abs_values = tl.abs(values)\n",
            b"done\n",
        ]
        with patch("triton_agent.process_runner.pty.openpty", return_value=(11, 12)):
            with patch("triton_agent.process_runner.subprocess.Popen", return_value=process):
                with patch(
                    "triton_agent.process_runner.select.select",
                    side_effect=[
                        ([11], [], []),
                        ([11], [], []),
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
        expected = (
            "baseline/perf.txt 368.0119\n"
            "opt-round-1/opt_triton_10_SwigluQuant_perf.txt 299.39092500000004\n\n"
            "     tl.store(scale_ptr + row, inv_scale)\n\n"
            "done\n"
        )
        self.assertEqual(stdout.getvalue(), expected)
        self.assertEqual(result.stdout, expected)

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

    def test_treats_eio_before_poll_reports_exit_as_clean_eof(self) -> None:
        stdout = StringIO()
        process = _StreamingFakeProcess(wait_code=0, poll_values=[None])
        process.wait = Mock(side_effect=[0, 0])
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
        self.assertEqual(process.wait.call_count, 2)

    def test_raises_eio_when_child_does_not_exit_within_grace_window(self) -> None:
        stdout = StringIO()
        process = _StreamingFakeProcess(wait_code=0, poll_values=[None], default_poll_return=None)
        process.wait = Mock(side_effect=subprocess.TimeoutExpired(cmd=["codex", "exec"], timeout=0.1))
        with patch("triton_agent.process_runner.pty.openpty", return_value=(11, 12)):
            with patch("triton_agent.process_runner.subprocess.Popen", return_value=process):
                with patch("triton_agent.process_runner.select.select", side_effect=[([11], [], [])]):
                    with patch(
                        "triton_agent.process_runner.os.read",
                        side_effect=OSError(EIO, "Input/output error"),
                    ):
                        with patch("triton_agent.process_runner.os.close"):
                            with self.assertRaises(OSError):
                                run_streaming_process(
                                    ["codex", "exec"],
                                    "/tmp",
                                    stall_timeout_seconds=10,
                                    stdout=stdout,
                                )

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
                ((1234, _SIGKILL),),
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
        self._chunks = list(lines)

    def read(self, _size: int = -1) -> str:
        if self._chunks:
            return self._chunks.pop(0)
        return ""

    def readline(self) -> str:
        return self.read()


class _BufferedFakeStderr:
    def __init__(self, text: str) -> None:
        self._text = text

    def read(self, _size: int = -1) -> str:
        text = self._text
        self._text = ""
        return text


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
        if self.stdout._chunks:
            return None
        return 0 if self.returncode is None else self.returncode

    def terminate(self) -> None:
        self.returncode = 1

    def kill(self) -> None:
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


def _run_buffered_large_stderr_repro(
    helper_path: str,
    output_path: str,
    queue: Any,
) -> None:
    try:
        result = run_buffered_process(
            [sys.executable, helper_path, output_path],
            str(Path(helper_path).parent),
            stall_timeout_seconds=2,
            session_id_extractor=lambda _line: None,
        )
        target = Path(output_path)
        queue.put(
            {
                "return_code": result.return_code,
                "stalled": result.stalled,
                "stdout": result.stdout,
                "output_exists": target.exists(),
                "output_text": target.read_text(encoding="utf-8") if target.exists() else None,
            }
        )
    except BaseException as exc:  # pragma: no cover - exercised only on failure
        queue.put({"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    unittest.main()
