from __future__ import annotations

import json
import unittest

from triton_agent.backends.traecli_trace import (
    TraeCliJsonLineParser,
    TraeCliJsonOutputFilter,
)


class TestTraeCliJsonLineParser(unittest.TestCase):
    def _parser(self) -> TraeCliJsonLineParser:
        return TraeCliJsonLineParser()

    def test_system_init_renders_banner(self) -> None:
        parser = self._parser()
        line = json.dumps(
            {
                "type": "system",
                "subtype": "init",
                "model": "GLM-5.1",
                "session_id": "session-123",
            }
        )
        result = parser.parse_line(line + "\n")
        self.assertEqual(result, "> build · GLM-5.1\n\n")

    def test_stream_event_reasoning_delta_renders_thinking(self) -> None:
        parser = self._parser()
        first = json.dumps(
            {
                "type": "stream_event",
                "delta": {"reasoning_content": "The user wants"},
            }
        )
        second = json.dumps(
            {
                "type": "stream_event",
                "delta": {"reasoning_content": " a test file."},
            }
        )
        self.assertEqual(parser.parse_line(first + "\n"), "Thinking: The user wants")
        self.assertEqual(parser.parse_line(second + "\n"), " a test file.")

    def test_summary_mode_does_not_emit_later_reasoning_chunks(self) -> None:
        parser = TraeCliJsonLineParser({"TRITON_AGENT_SHOW_OUTPUT_THINKING": "summary"})
        first = json.dumps(
            {
                "type": "stream_event",
                "delta": {"reasoning_content": "first line\nsecond line"},
            }
        )
        second = json.dumps(
            {
                "type": "stream_event",
                "delta": {"reasoning_content": " hidden details"},
            }
        )
        result = parser.parse_line(first + "\n")
        assert result is not None
        self.assertIn("Thinking: first line", result)
        self.assertIsNone(parser.parse_line(second + "\n"))

    def test_off_mode_suppresses_streamed_reasoning(self) -> None:
        parser = TraeCliJsonLineParser({"TRITON_AGENT_SHOW_OUTPUT_THINKING": "off"})
        reasoning = json.dumps(
            {
                "type": "stream_event",
                "delta": {"reasoning_content": "hidden reasoning"},
            }
        )
        assistant = json.dumps(
            {
                "type": "assistant",
                "message": {"reasoning_content": "hidden reasoning"},
            }
        )
        self.assertIsNone(parser.parse_line(reasoning + "\n"))
        self.assertIsNone(parser.parse_line(assistant + "\n"))

    def test_assistant_tool_call_renders_arrow_line(self) -> None:
        parser = self._parser()
        line = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "Read",
                                "arguments": json.dumps({"file_path": "D:/work/kernel.py"}),
                            }
                        }
                    ]
                },
            }
        )
        result = parser.parse_line(line + "\n")
        assert result is not None
        self.assertIn("→ Read kernel.py", result)

    def test_assistant_skill_tool_renders_skill_name(self) -> None:
        parser = self._parser()
        line = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "Skill",
                                "arguments": json.dumps({"skill": "triton-npu-gen-test"}),
                            }
                        }
                    ]
                },
            }
        )
        result = parser.parse_line(line + "\n")
        assert result is not None
        self.assertIn('→ Skill "triton-npu-gen-test"', result)

    def test_non_json_line_passes_through(self) -> None:
        parser = self._parser()
        result = parser.parse_line("plain status line\n")
        self.assertEqual(result, "plain status line\n")

    def test_duplicate_reasoning_block_skipped_after_stream(self) -> None:
        parser = self._parser()
        stream = json.dumps(
            {
                "type": "stream_event",
                "delta": {"reasoning_content": "already streamed"},
            }
        )
        assistant = json.dumps(
            {
                "type": "assistant",
                "message": {"reasoning_content": "already streamed"},
            }
        )
        parser.parse_line(stream + "\n")
        result = parser.parse_line(assistant + "\n")
        self.assertIsNone(result)

    def test_unknown_event_renders_text_line(self) -> None:
        parser = self._parser()
        line = json.dumps(
            {
                "type": "tool_result",
                "tool_use_id": "tool-1",
                "error": "boom",
            }
        )
        result = parser.parse_line(line + "\n")
        assert result is not None
        self.assertIn("[event:tool_result]", result)
        self.assertIn("tool-1", result)
        self.assertNotIn("{", result)
        self.assertNotIn("}", result)

    def test_unknown_tool_arguments_render_text_preview(self) -> None:
        parser = self._parser()
        line = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "CustomTool",
                                "arguments": json.dumps({"path": "kernel.py", "count": 3}),
                            }
                        }
                    ]
                },
            }
        )
        result = parser.parse_line(line + "\n")
        assert result is not None
        self.assertIn("→ CustomTool path=kernel.py, count=3", result)
        self.assertNotIn('{"path"', result)


class TestTraeCliJsonOutputFilter(unittest.TestCase):
    def test_feed_across_chunk_boundaries(self) -> None:
        payload = json.dumps(
            {
                "type": "system",
                "subtype": "init",
                "model": "GLM-5.1",
            }
        )
        half = len(payload) // 2
        output_filter = TraeCliJsonOutputFilter()
        first = output_filter.feed(payload[:half])
        second = output_filter.feed(payload[half:] + "\n", flush=True)
        self.assertEqual(first + second, "> build · GLM-5.1\n\n")

    def test_flush_closes_open_thinking(self) -> None:
        output_filter = TraeCliJsonOutputFilter()
        line = json.dumps(
            {
                "type": "stream_event",
                "delta": {"reasoning_content": "still thinking"},
            }
        )
        output_filter.feed(line + "\n")
        trailing = output_filter.feed("", flush=True)
        self.assertEqual(trailing, "\n\n")


if __name__ == "__main__":
    unittest.main()
