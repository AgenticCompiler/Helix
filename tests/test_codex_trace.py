from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from triton_agent.backends.codex_trace import (
    CodexJsonLineParser,
    CodexJsonOutputFilter,
    _parse_timestamp,
    build_codex_trace_env,
    remote_from_command,
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
        self.assertIsNone(_parse_timestamp(None))  # type: ignore[arg-type]


class TestRemoteFromCommand(unittest.TestCase):
    def test_ssh_remote(self) -> None:
        self.assertTrue(remote_from_command("ssh user@host echo hello"))
        self.assertTrue(remote_from_command("ssh 192.168.1.1 run-bench"))

    def test_scp_remote(self) -> None:
        self.assertTrue(remote_from_command("scp file.txt user@host:/tmp/"))

    def test_at_sign_remote(self) -> None:
        self.assertTrue(remote_from_command("user@host echo hello"))

    def test_local_command(self) -> None:
        self.assertFalse(remote_from_command("python run.py"))
        self.assertFalse(remote_from_command("echo hello"))


class TestCodexJsonLineParser(unittest.TestCase):
    def _make_trace_path(self) -> tuple[Path, Path]:
        tmpdir = Path(tempfile.mkdtemp())
        trace_path = tmpdir / "trace.jsonl"
        return tmpdir, trace_path

    def _make_parser(self, trace_path: Path | None, run_id: str = "test-run-id") -> CodexJsonLineParser:
        workspace_root = trace_path.parent if trace_path is not None else Path(tempfile.mkdtemp())
        return CodexJsonLineParser(trace_path, run_id=run_id, workspace_root=str(workspace_root))

    def test_non_json_line_passed_through(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        result = parser.parse_line("Hello world\n")
        self.assertEqual(result, "Hello world\n")

    def test_tool_start_writes_trace_event(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        line = json.dumps({"type": "tool_start", "tool": "Read", "tool_use_id": "call-123", "timestamp": "2026-05-17T10:29:18Z"})
        result = parser.parse_line(line + "\n")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("[system] Codex native JSON event stream is active", result)
        self.assertIn("[tool:start] Read call-123", result)
        events = trace_path.read_text().splitlines()
        # First event is diagnostic (codex_native_json_active), second is tool_call
        self.assertGreaterEqual(len(events), 2)
        tool_events = [json.loads(e) for e in events if json.loads(e).get("type") == "tool_call"]
        self.assertEqual(len(tool_events), 1)
        event = tool_events[0]
        self.assertEqual(event["type"], "tool_call")
        self.assertEqual(event["phase"], "start")
        self.assertEqual(event["tool"], "Read")
        self.assertEqual(event["tool_use_id"], "call-123")
        self.assertEqual(event["source"], "codex_native_json")

    def test_tool_end_writes_trace_event_with_duration(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        # First, send tool_start
        start_line = json.dumps({"type": "tool_start", "tool": "Bash", "tool_use_id": "call-123", "timestamp": "2026-05-17T10:29:18Z"})
        parser.parse_line(start_line + "\n")
        # Then tool_end
        end_line = json.dumps({"type": "tool_end", "tool": "Bash", "tool_use_id": "call-123", "timestamp": "2026-05-17T10:35:00Z", "status": "ok", "return_code": 0})
        result = parser.parse_line(end_line + "\n")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("ok", result)
        self.assertIn("[tool:end] Bash call-123", result)
        self.assertIn("rc=0", result)
        events = trace_path.read_text().splitlines()
        # tool_start + tool_end = 2 events, plus diagnostic
        self.assertGreaterEqual(len(events), 2)

    def test_powershell_unwrap(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        cmd = '"C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "python ./.codex/skills/script.py run-bench"'
        unwrapped = parser._unwrap_powershell(cmd)  # pylint: disable=protected-access
        self.assertEqual(unwrapped, "python ./.codex/skills/script.py run-bench")

    def test_command_classification(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        self.assertEqual(parser._classify_command("python run-bench --bench-mode msprof"), "benchmark")
        self.assertEqual(parser._classify_command("ssh user@host python run-bench"), "remote_bench")
        self.assertEqual(parser._classify_command("pytest test.py"), "correctness_test")
        self.assertEqual(parser._classify_command("python run-command.py run-test-baseline --test-file test.py"), "correctness_test")
        self.assertEqual(parser._classify_command("python run-command.py run-test-optimize --test-file differential_test.py"), "correctness_test")
        self.assertEqual(parser._classify_command("compare-perf"), "compare_perf")
        self.assertEqual(parser._classify_command("submit-round"), "check_round")

    def test_flush_writes_pending_events(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        start_line = json.dumps({"type": "tool_start", "tool": "Read", "tool_use_id": "call-456", "timestamp": "2026-05-17T10:00:00Z"})
        parser.parse_line(start_line + "\n")
        parser.flush()
        events = trace_path.read_text().splitlines()
        # At least the pending tool_start flushed as incomplete end event
        end_events = [json.loads(e) for e in events if json.loads(e).get("phase") == "end"]
        self.assertTrue(any(e.get("tool_use_id") == "call-456" and e.get("status") == "unknown" for e in end_events))

    def test_deduplication_skips_duplicate(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        line = json.dumps({"type": "tool_start", "tool": "Read", "tool_use_id": "call-789", "timestamp": "2026-05-17T10:29:18Z"})
        parser.parse_line(line + "\n")
        parser.parse_line(line + "\n")  # duplicate
        events = trace_path.read_text().splitlines()
        tool_call_events = [e for e in events if json.loads(e).get("tool_use_id") == "call-789"]
        # Should be only 1, not 2
        self.assertEqual(len(tool_call_events), 1)

    def test_codex_item_command_execution_writes_command_and_skill_read(self) -> None:
        workspace, trace_path = self._make_trace_path()
        skill_path = workspace / ".codex" / "skills" / "triton-npu-optimize" / "SKILL.md"
        skill_path.parent.mkdir(parents=True)
        skill_path.write_text("# Optimize\n", encoding="utf-8")
        parser = self._make_parser(trace_path)
        command = (
            '"C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" '
            f"-Command 'Get-Content -Path {skill_path.as_posix()}'"
        )

        parser.parse_line(json.dumps({
            "type": "item.started",
            "item": {
                "id": "item_1",
                "type": "command_execution",
                "command": command,
                "aggregated_output": "",
                "exit_code": None,
                "status": "in_progress",
            },
        }) + "\n")
        result = parser.parse_line(json.dumps({
            "type": "item.completed",
            "item": {
                "id": "item_1",
                "type": "command_execution",
                "command": command,
                "aggregated_output": "# Optimize\n",
                "exit_code": 0,
                "status": "completed",
            },
        }) + "\n")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("[tool:end] exec item_1 ok", result)
        self.assertIn("rc=0", result)
        events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        tool_events = [event for event in events if event.get("type") == "tool_call"]
        command_events = [event for event in events if event.get("type") == "command"]
        file_events = [event for event in events if event.get("type") == "file_access"]

        self.assertEqual([event["phase"] for event in tool_events], ["start", "end"])
        self.assertEqual(len(command_events), 1)
        self.assertEqual(command_events[0]["tool_use_id"], "item_1")
        self.assertEqual(command_events[0]["return_code"], 0)
        self.assertEqual(command_events[0]["duration_source"], "runner_clock")
        self.assertEqual(len(file_events), 1)
        self.assertEqual(file_events[0]["path"], ".codex/skills/triton-npu-optimize/SKILL.md")
        self.assertEqual(file_events[0]["path_class"], "skill_md")
        self.assertEqual(file_events[0]["skill_name"], "triton-npu-optimize")

    def test_codex_item_file_change_writes_edit_event(self) -> None:
        workspace, trace_path = self._make_trace_path()
        operator_path = workspace / "opt-round-1" / "opt_triton_5_Cumsum.py"
        operator_path.parent.mkdir(parents=True)
        parser = self._make_parser(trace_path)
        changes = [{"path": str(operator_path), "kind": "update"}]

        parser.parse_line(json.dumps({
            "type": "item.started",
            "item": {
                "id": "item_2",
                "type": "file_change",
                "changes": changes,
                "status": "in_progress",
            },
        }) + "\n")
        result = parser.parse_line(json.dumps({
            "type": "item.completed",
            "item": {
                "id": "item_2",
                "type": "file_change",
                "changes": changes,
                "status": "completed",
            },
        }) + "\n")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("[tool:end] file_change item_2 ok", result)
        events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        edit_events = [event for event in events if event.get("type") == "edit"]
        self.assertEqual(len(edit_events), 1)
        self.assertEqual(edit_events[0]["path"], "opt-round-1/opt_triton_5_Cumsum.py")
        self.assertEqual(edit_events[0]["edit_kind"], "round_artifact")
        self.assertEqual(edit_events[0]["change_kind"], "update")

    def test_codex_item_agent_message_renders_text(self) -> None:
        _, trace_path = self._make_trace_path()
        parser = self._make_parser(trace_path)
        result = parser.parse_line(json.dumps({
            "type": "item.completed",
            "item": {
                "id": "item_3",
                "type": "agent_message",
                "text": "done",
            },
        }) + "\n")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("done", result)
        self.assertIn("[system] Codex native JSON event stream is active", result)


class TestCodexJsonOutputFilter(unittest.TestCase):
    def _make_trace_path(self) -> tuple[Path, Path]:
        tmpdir = Path(tempfile.mkdtemp())
        trace_path = tmpdir / "trace.jsonl"
        return tmpdir, trace_path

    def test_feed_writes_trace_and_returns_human(self) -> None:
        _, trace_path = self._make_trace_path()
        extra_env = {
            "TRITON_AGENT_OTEL_RUN_ID": "test-run",
            "TRITON_AGENT_WORKSPACE_ROOT": str(trace_path.parent.parent),
        }
        filter_obj = CodexJsonOutputFilter(trace_path, extra_env, run_id="test-run", workspace_root=str(trace_path.parent.parent))
        result = filter_obj.feed('{"type":"tool_start","tool":"Read","tool_use_id":"call-1","timestamp":"2026-05-17T10:29:18Z"}\n', flush=True)
        # parse_line returns human text without trailing newline (newline consumed by line split)
        self.assertIn("[tool:start] Read call-1", result)
        self.assertTrue(trace_path.exists())
        self.assertGreater(len(trace_path.read_text()), 0)

    def test_feed_with_none_trace_path_returns_human_without_trace(self) -> None:
        workspace = Path(tempfile.mkdtemp())
        trace_path = workspace / "trace.jsonl"
        extra_env = {
            "TRITON_AGENT_OTEL_RUN_ID": "test-run",
            "TRITON_AGENT_WORKSPACE_ROOT": str(workspace),
        }
        filter_obj = CodexJsonOutputFilter(None, extra_env)

        result = filter_obj.feed(
            '{"type":"tool_start","tool":"Read","tool_use_id":"call-1","timestamp":"2026-05-17T10:29:18Z"}\n',
            flush=True,
        )

        self.assertIn("[tool:start] Read call-1", result)
        self.assertFalse(trace_path.exists())

    def test_item_start_and_completed_render_tool_timeline(self) -> None:
        _, trace_path = self._make_trace_path()
        filter_obj = CodexJsonOutputFilter(trace_path)

        start = filter_obj.feed(json.dumps({
            "type": "item.started",
            "item": {
                "id": "item_1",
                "type": "command_execution",
                "command": "uv run python -m unittest tests/test_codex_runner.py -v",
                "status": "in_progress",
            },
        }) + "\n")
        end = filter_obj.feed(json.dumps({
            "type": "item.completed",
            "item": {
                "id": "item_1",
                "type": "command_execution",
                "command": "uv run python -m unittest tests/test_codex_runner.py -v",
                "aggregated_output": "Ran 12 tests in 0.31s\n",
                "exit_code": 0,
                "status": "completed",
            },
        }) + "\n", flush=True)

        self.assertIn("[tool:start] exec item_1", start)
        self.assertIn("command: uv run python -m unittest tests/test_codex_runner.py -v", start)
        self.assertIn("[tool:end] exec item_1 ok", end)
        self.assertIn("stdout: Ran 12 tests in 0.31s", end)

    def test_non_json_lines_pass_through(self) -> None:
        _, trace_path = self._make_trace_path()
        extra_env = {
            "TRITON_AGENT_OTEL_RUN_ID": "test-run",
            "TRITON_AGENT_WORKSPACE_ROOT": str(trace_path.parent.parent),
        }
        filter_obj = CodexJsonOutputFilter(trace_path, extra_env, run_id="test-run", workspace_root=str(trace_path.parent.parent))
        result = filter_obj.feed("Hello world\n", flush=True)
        self.assertEqual(result, "Hello world\n")


class TestBuildCodexTraceEnv(unittest.TestCase):
    def test_build_env_sets_trace_vars(self) -> None:
        trace_path = Path(tempfile.gettempdir()) / "trace.jsonl"
        env = build_codex_trace_env(None, trace_path=trace_path, run_id="run-123", workspace_root=Path(tempfile.gettempdir()))
        self.assertEqual(env["TRITON_AGENT_OTEL_TRACE_PATH"], str(trace_path))
        self.assertEqual(env["TRITON_AGENT_OTEL_RUN_ID"], "run-123")
        self.assertEqual(env["TRITON_AGENT_WORKSPACE_ROOT"], str(Path(tempfile.gettempdir())))

    def test_existing_env_preserved(self) -> None:
        trace_path = Path(tempfile.gettempdir()) / "trace.jsonl"
        existing = {"MY_VAR": "my_value"}
        env = build_codex_trace_env(existing, trace_path=trace_path, run_id="run-123", workspace_root=Path(tempfile.gettempdir()))
        self.assertEqual(env["MY_VAR"], "my_value")
        self.assertEqual(env["TRITON_AGENT_OTEL_TRACE_PATH"], str(trace_path))


if __name__ == "__main__":
    unittest.main()
