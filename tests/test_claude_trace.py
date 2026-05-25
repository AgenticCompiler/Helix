from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from triton_agent.backends.claude_trace import (
    ClaudeJsonLineParser,
    ClaudeJsonOutputFilter,
    _parse_timestamp,
    build_claude_trace_env,
)


class TestParseTimestamp(unittest.TestCase):
    def test_iso_with_z_suffix(self) -> None:
        ts = _parse_timestamp("2026-05-17T10:29:18Z")
        self.assertIsNotNone(ts)
        assert ts is not None
        self.assertEqual(ts.year, 2026)
        self.assertEqual(ts.month, 5)
        self.assertEqual(ts.day, 17)

    def test_iso_with_microseconds(self) -> None:
        ts = _parse_timestamp("2026-05-17T10:29:18.123456Z")
        self.assertIsNotNone(ts)

    def test_invalid_returns_none(self) -> None:
        self.assertIsNone(_parse_timestamp("not-a-timestamp"))
        self.assertIsNone(_parse_timestamp(""))


class TestClaudeJsonLineParser(unittest.TestCase):
    def _make_trace_path(self) -> tuple[Path, Path]:
        tmpdir = Path(tempfile.mkdtemp())
        trace_path = tmpdir / "trace.jsonl"
        return tmpdir, trace_path

    def _make_parser(self, trace_path: Path, run_id: str = "test-run-id", role: str = "worker") -> ClaudeJsonLineParser:
        extra_env = {
            "TRITON_AGENT_OTEL_RUN_ID": run_id,
            "TRITON_AGENT_OTEL_ROLE": role,
            "TRITON_AGENT_WORKSPACE_ROOT": str(trace_path.parent.parent),
        }
        return ClaudeJsonLineParser(trace_path, extra_env)

    def test_non_json_line_passed_through(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        result = parser.parse_line("Hello world\n")
        self.assertEqual(result, "Hello world\n")

    def test_parse_assistant_tool_use(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        line = json.dumps({
            "type": "assistant",
            "message": {
                "id": "msg-123",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_00_abcd",
                        "name": "Read",
                        "input": {"file_path": "/abs/path/to/file", "limit": 30},
                    },
                ],
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
            "session_id": "session-uuid",
        })
        result = parser.parse_line(line + "\n")
        assert result is not None
        self.assertIn("[tool:start] Read call_00_abcd", result)
        self.assertIn("file_path: /abs/path/to/file", result)
        self.assertIn("/abs/path/to/file", result)
        events = trace_path.read_text().splitlines()
        tool_events = [json.loads(e) for e in events if json.loads(e).get("type") == "tool_call"]
        self.assertEqual(len(tool_events), 1)
        event = tool_events[0]
        self.assertEqual(event["type"], "tool_call")
        self.assertEqual(event["phase"], "start")
        self.assertEqual(event["tool"], "Read")
        self.assertEqual(event["tool_use_id"], "call_00_abcd")
        self.assertEqual(event["tool_input"]["file_path"], "/abs/path/to/file")
        self.assertEqual(event["source"], "claude_native_json")

    def test_parse_user_tool_result(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        # First, send assistant with tool_use
        start_line = json.dumps({
            "type": "assistant",
            "message": {
                "id": "msg-123",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_00_abcd",
                        "name": "Bash",
                        "input": {"command": "echo hello"},
                    },
                ],
            },
            "session_id": "session-uuid",
        })
        parser.parse_line(start_line + "\n")
        # Then user with tool_result
        end_line = json.dumps({
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "tool_use_id": "call_00_abcd",
                        "type": "tool_result",
                        "content": "hello",
                        "is_error": False,
                    },
                ],
            },
            "timestamp": "2026-05-18T12:51:29.018Z",
            "tool_use_result": {
                "type": "text",
                "stdout": "hello\n",
                "stderr": "",
            },
        })
        result = parser.parse_line(end_line + "\n")
        assert result is not None
        self.assertIn("[tool:end] Bash call_00_abcd ok in", result)
        self.assertIn("rc=0", result)
        self.assertIn("stdout: hello", result)
        events = trace_path.read_text().splitlines()
        # Should have: diagnostic(claude_native_json_active), tool_call start, tool_call end, command
        self.assertGreaterEqual(len(events), 3)
        end_events = [json.loads(e) for e in events if json.loads(e).get("phase") == "end" and json.loads(e).get("type") == "tool_call"]
        self.assertEqual(len(end_events), 1)
        end_event = end_events[0]
        self.assertEqual(end_event["tool_use_id"], "call_00_abcd")
        self.assertEqual(end_event["status"], "ok")
        self.assertGreaterEqual(end_event["duration_ms"], 0)

    def test_parse_multiple_tool_use_in_one_assistant(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        line = json.dumps({
            "type": "assistant",
            "message": {
                "id": "msg-456",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call_01", "name": "Read", "input": {"file_path": "/a"}},
                    {"type": "tool_use", "id": "call_02", "name": "Read", "input": {"file_path": "/b"}},
                ],
            },
        })
        parser.parse_line(line + "\n")
        events = trace_path.read_text().splitlines()
        # Skip diagnostic event
        tool_starts = [json.loads(e) for e in events if json.loads(e).get("phase") == "start"]
        self.assertEqual(len(tool_starts), 2)
        self.assertEqual(tool_starts[0]["tool_use_id"], "call_01")
        self.assertEqual(tool_starts[1]["tool_use_id"], "call_02")

    def test_parse_thinking_block_rendered(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        line = json.dumps({
            "type": "assistant",
            "message": {
                "id": "msg-789",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me think about this..."},
                    {"type": "text", "text": "I will read the file."},
                ],
            },
        })
        result = parser.parse_line(line + "\n")
        self.assertEqual(
            result,
            "[think:full]\nLet me think about this...\n\nI will read the file.\n\n",
        )
        events = trace_path.read_text().splitlines()
        # Only diagnostic, no tool_call event
        tool_events = [json.loads(e) for e in events if json.loads(e).get("type") == "tool_call"]
        self.assertEqual(len(tool_events), 0)

    def test_parse_text_block_rendered(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        line = json.dumps({
            "type": "assistant",
            "message": {
                "id": "msg-999",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me help you with that."},
                ],
            },
        })
        result = parser.parse_line(line + "\n")
        self.assertEqual(result, "Let me help you with that.\n\n")

    def test_unknown_json_event_does_not_render_raw_json(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        line = json.dumps({"type": "mystery", "payload": {"a": 1}})
        result = parser.parse_line(line + "\n")
        assert result is not None
        self.assertIn("Skipped unsupported Claude stream-json event type: mystery", result)
        self.assertNotIn('"payload"', result)

    def test_tool_end_error_renders_exit_code_and_error_line(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        parser.parse_line(json.dumps({
            "type": "assistant",
            "message": {
                "id": "msg-cmd-error",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call_error_01", "name": "Bash", "input": {"command": "python missing.py"}},
                ],
            },
        }) + "\n")
        result = parser.parse_line(json.dumps({
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "tool_use_id": "call_error_01",
                        "type": "tool_result",
                        "content": "Exit code 49\nModuleNotFoundError: No module named 'torch'",
                        "is_error": True,
                    },
                ],
            },
            "timestamp": "2026-05-18T12:52:00Z",
            "tool_use_result": {
                "stdout": "",
                "stderr": "Traceback (most recent call last):\nModuleNotFoundError: No module named 'torch'",
            },
        }) + "\n")
        assert result is not None
        self.assertIn("[tool:end] Bash call_error_01 error", result)
        self.assertIn("rc=49", result)
        self.assertIn("error: ModuleNotFoundError: No module named 'torch'", result)
        self.assertIn("stderr excerpt:", result)

    def test_derive_command_event(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        # Start
        parser.parse_line(json.dumps({
            "type": "assistant",
            "message": {
                "id": "msg-cmd",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call_cmd_01", "name": "Bash", "input": {"command": "python run-bench msprof"}},
                ],
            },
        }) + "\n")
        # End
        parser.parse_line(json.dumps({
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"tool_use_id": "call_cmd_01", "type": "tool_result", "content": "ok", "is_error": False},
                ],
            },
            "timestamp": "2026-05-18T12:52:00Z",
            "tool_use_result": {"stdout": "benchmark output", "stderr": ""},
        }) + "\n")
        events = trace_path.read_text().splitlines()
        command_events = [json.loads(e) for e in events if json.loads(e).get("type") == "command"]
        self.assertEqual(len(command_events), 1)
        self.assertEqual(command_events[0]["command_kind"], "benchmark")
        self.assertEqual(command_events[0]["stdout_excerpt"], "benchmark output")

    def test_derive_file_access_event(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        parser.parse_line(json.dumps({
            "type": "assistant",
            "message": {
                "id": "msg-fa",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call_fa_01", "name": "Grep", "input": {"pattern": "TODO", "path": "/src"}},
                ],
            },
        }) + "\n")
        parser.parse_line(json.dumps({
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"tool_use_id": "call_fa_01", "type": "tool_result", "content": "found", "is_error": False},
                ],
            },
            "timestamp": "2026-05-18T12:53:00Z",
        }) + "\n")
        events = trace_path.read_text().splitlines()
        file_events = [json.loads(e) for e in events if json.loads(e).get("type") == "file_access"]
        self.assertEqual(len(file_events), 1)
        self.assertEqual(file_events[0]["tool_use_id"], "call_fa_01")
        self.assertEqual(file_events[0]["tool"], "Grep")

    def test_powershell_unwrap(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        cmd = '"C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "python ./.codex/skills/script.py run-bench"'
        unwrapped = parser._unwrap_powershell(cmd)
        self.assertEqual(unwrapped, "python ./.codex/skills/script.py run-bench")

    def test_command_classification(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        self.assertEqual(parser._classify_command("python run-bench --bench-mode msprof"), "benchmark")
        self.assertEqual(parser._classify_command("ssh user@host python run-bench"), "remote_bench")
        self.assertEqual(parser._classify_command("pytest test.py"), "correctness_test")
        self.assertEqual(parser._classify_command("compare-perf"), "compare_perf")
        self.assertEqual(parser._classify_command("check-round"), "check_round")
        self.assertEqual(parser._classify_command("check-baseline"), "check_baseline")
        self.assertEqual(parser._classify_command("compare-result"), "compare_result")

    def test_flush_writes_pending_events(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        parser.parse_line(json.dumps({
            "type": "assistant",
            "message": {
                "id": "msg-flush",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call_flush_01", "name": "Read", "input": {"file_path": "/x"}},
                ],
            },
        }) + "\n")
        parser.flush()
        events = trace_path.read_text().splitlines()
        end_events = [json.loads(e) for e in events if json.loads(e).get("phase") == "end" and json.loads(e).get("type") == "tool_call"]
        self.assertTrue(any(e.get("tool_use_id") == "call_flush_01" and e.get("status") == "unknown" for e in end_events))

    def test_deduplication_skips_duplicate(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        line = json.dumps({
            "type": "assistant",
            "message": {
                "id": "msg-dup",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call_dup_01", "name": "Read", "input": {"file_path": "/y"}},
                ],
            },
        }) + "\n"
        parser.parse_line(line)
        parser.parse_line(line)  # duplicate
        events = trace_path.read_text().splitlines()
        tool_call_events = [e for e in events if json.loads(e).get("tool_use_id") == "call_dup_01"]
        self.assertEqual(len(tool_call_events), 1)

    def test_invalid_json_graceful(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        result = parser.parse_line("{invalid json}\n")
        self.assertEqual(result, "{invalid json}\n")

    def test_system_init_session_id(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        line = json.dumps({
            "type": "system",
            "subtype": "init",
            "session_id": "test-session-uuid",
            "tools": ["Read", "Write"],
            "model": "claude-sonnet-4-6",
        })
        result = parser.parse_line(line + "\n")
        assert result is not None
        self.assertIn("test-session-uuid", result)
        self.assertEqual(parser._session_id, "test-session-uuid")

    def test_result_event(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        line = json.dumps({
            "type": "result",
            "subtype": "success",
            "duration_ms": 8956,
            "num_turns": 2,
            "stop_reason": "end_turn",
            "session_id": "uuid",
        })
        result = parser.parse_line(line + "\n")
        assert result is not None
        self.assertIn("success", result)
        self.assertIn("8956ms", result)
        events = trace_path.read_text().splitlines()
        diagnostic_events = [json.loads(e) for e in events if json.loads(e).get("type") == "diagnostic"]
        self.assertEqual(len(diagnostic_events), 2)  # claude_native_json_active + claude_result


class TestClaudeJsonOutputFilter(unittest.TestCase):
    def _make_trace_path(self) -> tuple[Path, Path]:
        tmpdir = Path(tempfile.mkdtemp())
        trace_path = tmpdir / "trace.jsonl"
        return tmpdir, trace_path

    def test_feed_writes_trace_and_returns_human(self) -> None:
        _, trace_path = self._make_trace_path()
        extra_env = {
            "TRITON_AGENT_OTEL_RUN_ID": "test-run",
            "TRITON_AGENT_OTEL_ROLE": "worker",
            "TRITON_AGENT_WORKSPACE_ROOT": str(trace_path.parent.parent),
        }
        filter_obj = ClaudeJsonOutputFilter(trace_path, extra_env)
        result = filter_obj.feed(json.dumps({
            "type": "assistant",
            "message": {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call-1", "name": "Read", "input": {"file_path": "/f"}},
                ],
            },
        }) + "\n", flush=True)
        self.assertIn("[tool:start] Read call-1", result)
        self.assertIn("No Claude native thinking block was present in stdout.", result)
        self.assertTrue(trace_path.exists())
        self.assertGreater(len(trace_path.read_text()), 0)

    def test_output_filter_can_render_without_trace_path(self) -> None:
        filter_obj = ClaudeJsonOutputFilter(None, None)
        result = filter_obj.feed(json.dumps({
            "type": "assistant",
            "message": {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will inspect the workspace."},
                ],
            },
        }) + "\n", flush=True)
        self.assertIn("I will inspect the workspace.", result)

    def test_non_json_lines_pass_through(self) -> None:
        _, trace_path = self._make_trace_path()
        extra_env = {
            "TRITON_AGENT_OTEL_RUN_ID": "test-run",
            "TRITON_AGENT_OTEL_ROLE": "worker",
            "TRITON_AGENT_WORKSPACE_ROOT": str(trace_path.parent.parent),
        }
        filter_obj = ClaudeJsonOutputFilter(trace_path, extra_env)
        result = filter_obj.feed("Hello world\n", flush=True)
        self.assertEqual(result, "Hello world\n")


class TestBuildClaudeTraceEnv(unittest.TestCase):
    def test_build_env_sets_trace_vars(self) -> None:
        trace_path = Path(tempfile.gettempdir()) / "trace.jsonl"
        env = build_claude_trace_env(
            None,
            trace_path=trace_path,
            run_id="run-123",
            role="worker",
            workspace_root=Path(tempfile.gettempdir()),
        )
        self.assertEqual(env["TRITON_AGENT_OTEL_TRACE_PATH"], str(trace_path))
        self.assertEqual(env["TRITON_AGENT_OTEL_RUN_ID"], "run-123")
        self.assertEqual(env["TRITON_AGENT_OTEL_ROLE"], "worker")

    def test_existing_env_preserved(self) -> None:
        trace_path = Path(tempfile.gettempdir()) / "trace.jsonl"
        existing = {"MY_VAR": "my_value"}
        env = build_claude_trace_env(
            existing,
            trace_path=trace_path,
            run_id="run-123",
            role="worker",
            workspace_root=Path(tempfile.gettempdir()),
        )
        self.assertEqual(env["MY_VAR"], "my_value")
        self.assertEqual(env["TRITON_AGENT_OTEL_TRACE_PATH"], str(trace_path))


if __name__ == "__main__":
    unittest.main()
