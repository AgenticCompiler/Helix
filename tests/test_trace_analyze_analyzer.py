import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.trace.analyze import build_summary


class TraceAnalysisAnalyzerTests(unittest.TestCase):
    def test_build_summary_detects_redundant_reads_and_msprof_commands(self) -> None:
        trace_path = Path("/tmp/triton-agent-logs/run-001/trace-batch-1-5.jsonl")

        events = [
            {
                "type": "file_access",
                "action": "read",
                "path": ".codex/skills/ascend-npu-run-eval/scripts/cli.py",
            },
            {
                "type": "file_access",
                "action": "read",
                "path": ".codex/skills/ascend-npu-run-eval/scripts/cli.py",
            },
            {
                "type": "command",
                "command_kind": "remote_bench",
                "command": "python3 -m run-bench --bench-mode msprof",
                "duration_ms": 120000,
                "return_code": 0,
            },
            {
                "type": "command",
                "command_kind": "remote_bench",
                "command": "python3 -m run-bench --bench-mode msprof",
                "duration_ms": 130000,
                "return_code": 0,
            },
        ]

        summary = build_summary(events, trace_path=trace_path)

        self.assertEqual(summary["file_access"]["skill_script_reads"], 2)
        self.assertTrue(summary["tool_trace_enabled"])
        self.assertEqual(summary["tool_trace_capability"], "partial")
        self.assertEqual(
            summary["paths"]["summary_json"],
            "/tmp/triton-agent-logs/run-001/trace-batch-1-5.summary.json",
        )
        self.assertFalse(summary["capabilities"]["agent_invocation"])
        self.assertFalse(summary["capabilities"]["pre_tool_events"])
        self.assertFalse(summary["capabilities"]["tool_completion_events"])
        self.assertTrue(summary["capabilities"]["command_events"])
        self.assertEqual(
            summary["file_access"]["repeated_file_reads"],
            {".codex/skills/ascend-npu-run-eval/scripts/cli.py": 2},
        )
        self.assertEqual(
            summary["commands"]["full_msprof_benchmark_commands"],
            {"python3 -m run-bench --bench-mode msprof": 2},
        )

    def test_build_summary_empty_events(self) -> None:
        trace_path = Path("/tmp/triton-agent-logs/run-001/trace-batch-1-5.jsonl")
        summary = build_summary([], trace_path=trace_path)

        self.assertFalse(summary["tool_trace_enabled"])
        self.assertEqual(summary["tool_trace_capability"], "disabled")
        self.assertEqual(
            summary["paths"]["summary_json"],
            "/tmp/triton-agent-logs/run-001/trace-batch-1-5.summary.json",
        )
        self.assertEqual(summary["event_counts"]["total"], 0)
        self.assertEqual(summary["evidence_gaps"], [
            "No agent invocation events detected — agent lifecycle not traced.",
            "No pre-tool events detected — prompt contents are not captured.",
            "No tool completion events detected — tool outcomes are not captured.",
        ])

    def test_build_summary_handles_non_skill_script_paths(self) -> None:
        trace_path = Path("/tmp/triton-agent-logs/run-001/trace-batch-1-5.jsonl")

        events = [
            {
                "type": "file_access",
                "action": "read",
                "path": ".codex/skills/ascend-npu-run-eval/SKILL.md",
            },
            {
                "type": "file_access",
                "action": "read",
                "path": ".codex/skills/ascend-npu-run-eval/SKILL.md",
            },
            {
                "type": "file_access",
                "action": "read",
                "path": ".codex/skills/ascend-npu-run-eval/references/pattern.md",
            },
        ]

        summary = build_summary(events, trace_path=trace_path)

        self.assertEqual(summary["file_access"]["skill_script_reads"], 0)
        self.assertEqual(summary["file_access"]["skill_md_reads"], 2)
        self.assertEqual(summary["file_access"]["reference_reads"], 1)
        self.assertEqual(
            summary["file_access"]["repeated_file_reads"],
            {".codex/skills/ascend-npu-run-eval/SKILL.md": 2},
        )

    def test_build_summary_keeps_summary_json_for_non_optimize_trace_name(self) -> None:
        trace_path = Path("/tmp/triton-agent-logs/run-001/trace.jsonl")

        summary = build_summary([], trace_path=trace_path)

        self.assertEqual(
            summary["paths"]["summary_json"],
            "/tmp/triton-agent-logs/run-001/summary.json",
        )


if __name__ == "__main__":
    unittest.main()
