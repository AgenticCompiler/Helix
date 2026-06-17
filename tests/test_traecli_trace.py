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
