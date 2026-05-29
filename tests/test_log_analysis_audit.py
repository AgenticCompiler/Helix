import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.log_analysis.audit import build_summary, render_audit_markdown


class LogAnalysisAuditTests(unittest.TestCase):
    def test_build_summary_detects_redundant_reads_and_msprof_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            trace_path = workdir / "triton-agent-logs" / "run-001" / "otel" / "trace.jsonl"
            show_output_dir = workdir / "triton-agent-logs" / "run-001"
            show_output_path = show_output_dir / "show-output.log"
            agent_sessions_path = (
                workdir / "triton-agent-logs" / "run-001" / "agent-sessions.jsonl"
            )
            show_output_path.parent.mkdir(parents=True)
            show_output_path.write_text("output\n", encoding="utf-8")
            agent_sessions_path.parent.mkdir(parents=True, exist_ok=True)
            agent_sessions_path.write_text(
                json.dumps({"role": "worker", "session_id": "unknown", "agent": "codex"}) + "\n",
                encoding="utf-8",
            )
            (workdir / "baseline").mkdir()
            round_dir = workdir / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "parent_round": "baseline",
                        "hypothesis": "reduce redundant memory traffic",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "round_disposition": "continue",
                        "evidence_sources": ["benchmark", "summary"],
                    }
                ),
                encoding="utf-8",
            )
            (round_dir / "summary.md").write_text("Round improved memory traffic.\n", encoding="utf-8")
            (round_dir / "attempts.md").write_text("Ran benchmark once.\n", encoding="utf-8")

            events = [
                {
                    "type": "file_access",
                    "action": "read",
                    "path": ".codex/skills/triton-npu-run-eval/scripts/run-command.py",
                },
                {
                    "type": "file_access",
                    "action": "read",
                    "path": ".codex/skills/triton-npu-run-eval/scripts/run-command.py",
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

            summary = build_summary(
                events,
                workdir=workdir,
                run_id="run-001",
                trace_path=trace_path,
                show_output_dir=show_output_dir,
                agent_sessions_path=agent_sessions_path,
            )
            audit = render_audit_markdown(summary)

            self.assertEqual(summary["file_access"]["skill_script_reads"], 2)
            self.assertTrue(summary["tool_trace_enabled"])
            self.assertEqual(summary["tool_trace_capability"], "tool_completion_events")
            self.assertTrue(summary["capabilities"]["pre_tool_events"])
            self.assertTrue(summary["capabilities"]["tool_completion_events"])
            self.assertEqual(
                summary["file_access"]["repeated_file_reads"],
                {".codex/skills/triton-npu-run-eval/scripts/run-command.py": 2},
            )
            self.assertEqual(
                summary["commands"]["full_msprof_benchmark_commands"],
                {"python3 -m run-bench --bench-mode msprof": 2},
            )
            self.assertEqual(len(summary["worker_timeline"]), 1)
            self.assertEqual(summary["worker_timeline"][0]["round"], "opt-round-1")
            self.assertEqual(summary["worker_timeline"][0]["hypothesis"], "reduce redundant memory traffic")
            self.assertEqual(summary["worker_timeline"][0]["evidence_sources"], ["benchmark", "summary"])
            self.assertIn("Agent Execution Audit Report", audit)
            self.assertIn("Repeated full msprof benchmark", audit)
            self.assertIn("Worker Timeline", audit)
            self.assertIn("reduce redundant memory traffic", audit)

    def test_build_summary_reports_round_artifact_evidence_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            trace_path = workdir / "triton-agent-logs" / "run-001" / "otel" / "trace.jsonl"
            show_output_dir = workdir / "triton-agent-logs" / "run-001"
            agent_sessions_path = (
                workdir / "triton-agent-logs" / "run-001" / "agent-sessions.jsonl"
            )
            (workdir / "opt-round-1").mkdir()

            summary = build_summary(
                [],
                workdir=workdir,
                run_id="run-001",
                trace_path=trace_path,
                show_output_dir=show_output_dir,
                agent_sessions_path=agent_sessions_path,
            )

            gaps = "\n".join(summary["evidence_gaps"])
            self.assertFalse(summary["tool_trace_enabled"])
            self.assertEqual(summary["tool_trace_capability"], "disabled")
            self.assertIn("opt-round-1 is missing round-state.json", gaps)
            self.assertIn("opt-round-1 is missing summary.md", gaps)
            self.assertIn("opt-round-1 is missing attempts.md", gaps)


if __name__ == "__main__":
    unittest.main()
