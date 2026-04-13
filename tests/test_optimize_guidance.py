import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.optimize_guidance import OptimizeGuidanceManager


class OptimizeGuidanceManagerTests(unittest.TestCase):
    def test_prepare_creates_shared_guidance_and_role_briefs_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            manager = OptimizeGuidanceManager()
            state = manager.prepare(
                workdir,
                operator,
                test_mode="differential",
                bench_mode="standalone",
                agent_name="codex",
            )

            agents_path = workdir / "AGENTS.md"
            shared_content = agents_path.read_text(encoding="utf-8")
            worker_content = state.worker_brief_path.read_text(encoding="utf-8")
            supervisor_content = state.supervisor_brief_path.read_text(encoding="utf-8")

            self.assertTrue(agents_path.exists())
            self.assertEqual(state.guidance_path, agents_path)
            self.assertIsNone(state.backup_path)
            self.assertTrue(state.worker_brief_path.exists())
            self.assertTrue(state.supervisor_brief_path.exists())
            self.assertTrue(state.round_brief_path.exists())
            self.assertTrue(state.supervisor_report_path.exists())

            self.assertIn("## Triton Agent Optimize Orchestration", shared_content)
            self.assertIn("This workspace is under optimize orchestration.", shared_content)
            self.assertIn("Use the staged workspace skills as the workflow source of truth.", shared_content)
            self.assertIn("Read the role brief for this invocation before acting.", shared_content)
            self.assertIn("Do not put worker-only or supervisor-only role assignment", shared_content)
            self.assertNotIn("Improve the Triton operator", shared_content)
            self.assertNotIn("This invocation is an audit and handoff pass", shared_content)

            self.assertIn("## Mission", worker_content)
            self.assertIn("Improve the Triton operator for Ascend NPU", worker_content)
            self.assertIn("Never edit the original operator in place.", worker_content)
            self.assertIn("Record a baseline correctness and benchmark result", worker_content)
            self.assertIn("Update `attempts.md` throughout each round", worker_content)
            self.assertIn("Write a short diagnosis summary before the first code-changing round.", worker_content)
            self.assertIn("Start by consulting the staged `optimize` skill", worker_content)

            self.assertIn("## Supervisor Mission", supervisor_content)
            self.assertIn("This invocation is an audit and handoff pass", supervisor_content)
            self.assertIn("Do not perform open-ended optimization work.", supervisor_content)
            self.assertIn("Repair metadata only when the underlying evidence already exists.", supervisor_content)
            self.assertIn("Emit a gate result for the completed round.", supervisor_content)

            warnings = manager.cleanup(state)
            self.assertEqual(warnings, [])
            self.assertFalse(agents_path.exists())
            self.assertFalse(state.worker_brief_path.exists())
            self.assertFalse(state.supervisor_brief_path.exists())
            self.assertFalse(state.round_brief_path.exists())
            self.assertFalse(state.supervisor_report_path.exists())

    def test_prepare_uses_claude_file_and_restores_existing_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            guidance_path = workdir / "CLAUDE.md"
            guidance_path.write_text("original content\n", encoding="utf-8")

            manager = OptimizeGuidanceManager()
            state = manager.prepare(
                workdir,
                operator,
                test_mode="standalone",
                bench_mode="msprof",
                agent_name="claude",
            )

            shared_content = guidance_path.read_text(encoding="utf-8")
            worker_content = state.worker_brief_path.read_text(encoding="utf-8")

            self.assertIsNotNone(state.backup_path)
            self.assertTrue(state.backup_path is not None and state.backup_path.exists())
            self.assertEqual(state.guidance_path, guidance_path)
            self.assertIn("# CLAUDE.md", shared_content)
            self.assertIn("## Triton Agent Optimize Orchestration", shared_content)
            self.assertIn("Read the role brief for this invocation before acting.", shared_content)
            self.assertNotIn("Improve the Triton operator", shared_content)
            self.assertIn("Use `standalone` correctness validation", worker_content)
            self.assertIn("Use `msprof` benchmark validation", worker_content)

            warnings = manager.cleanup(state)
            self.assertEqual(warnings, [])
            self.assertEqual(guidance_path.read_text(encoding="utf-8"), "original content\n")
            self.assertFalse(state.backup_path is not None and state.backup_path.exists())

    def test_prepare_mentions_strict_analysis_in_worker_and_supervisor_briefs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            manager = OptimizeGuidanceManager()
            state = manager.prepare(
                workdir,
                operator,
                test_mode="differential",
                bench_mode="standalone",
                agent_name="codex",
                require_analysis=True,
            )

            worker_content = state.worker_brief_path.read_text(encoding="utf-8")
            supervisor_content = state.supervisor_brief_path.read_text(encoding="utf-8")
            self.assertIn("Before the first code-changing round, gather profiling or IR-backed evidence.", worker_content)
            self.assertIn("Do not begin with blind tiling or launch-parameter search.", worker_content)
            self.assertIn("Require existing profiling or IR-backed evidence", supervisor_content)

            warnings = manager.cleanup(state)
            self.assertEqual(warnings, [])

    def test_describe_cleanup_lists_individual_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            manager = OptimizeGuidanceManager()
            state = manager.prepare(
                workdir,
                operator,
                test_mode="differential",
                bench_mode="standalone",
                agent_name="codex",
            )

            messages = manager.describe_cleanup(state)

            self.assertTrue(any("AGENTS.md" in message for message in messages))
            self.assertTrue(any("optimize-worker.md" in message for message in messages))
            self.assertTrue(any("optimize-supervisor.md" in message for message in messages))
            self.assertTrue(any("round-brief.md" in message for message in messages))
            self.assertTrue(any("supervisor-report.md" in message for message in messages))

            warnings = manager.cleanup(state)
            self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
