import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.optimize.guidance import OptimizeGuidanceManager


class OptimizeGuidanceManagerTests(unittest.TestCase):
    def test_prepare_unsupervised_session_creates_self_contained_guidance_file_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            manager = OptimizeGuidanceManager()
            state = manager.prepare_unsupervised_session(
                workdir,
                operator_path=operator,
                agent_name="codex",
                test_mode="differential",
                bench_mode="standalone",
            )

            agents_path = workdir / "AGENTS.md"
            guidance_content = agents_path.read_text(encoding="utf-8")

            self.assertTrue(agents_path.exists())
            self.assertEqual(state.guidance_path, agents_path)
            self.assertIsNone(state.backup_path)
            self.assertTrue(state.created_guidance)
            self.assertIn("## Triton Agent Optimize Session", guidance_content)
            self.assertIn("This workspace is under an unsupervised optimize run.", guidance_content)
            self.assertIn("Own the end-to-end optimize session.", guidance_content)
            self.assertIn("Use `differential` correctness validation", guidance_content)
            self.assertIn("Use `standalone` benchmark validation", guidance_content)
            self.assertIn("Use the staged `triton-npu-optimize` skill", guidance_content)
            self.assertNotIn("Read the role brief", guidance_content)
            self.assertNotIn("Worker and supervisor roles", guidance_content)
            self.assertNotIn(".triton-agent/roles/", guidance_content)

            warnings = manager.cleanup_unsupervised_session(state)
            self.assertEqual(warnings, [])
            self.assertFalse(agents_path.exists())
            self.assertFalse((workdir / ".triton-agent").exists())

    def test_prepare_creates_shared_guidance_and_handoff_files_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            manager = OptimizeGuidanceManager()
            state = manager.prepare_supervised_session(
                workdir,
                agent_name="codex",
            )

            agents_path = workdir / "AGENTS.md"
            shared_content = agents_path.read_text(encoding="utf-8")
            self.assertTrue(agents_path.exists())
            self.assertEqual(state.guidance_path, agents_path)
            self.assertIsNone(state.backup_path)
            self.assertTrue(state.round_brief_path.exists())
            self.assertTrue(state.supervisor_report_path.exists())
            self.assertTrue(state.history_dir.exists())
            self.assertEqual(state.archive_root, workdir / "optimize-logs" / "triton-agent")
            self.assertTrue(state.run_archive_dir.parent == state.archive_root)

            self.assertIn("## Triton Agent Optimize Orchestration", shared_content)
            self.assertIn("This workspace is under optimize orchestration.", shared_content)
            self.assertIn("Use the staged workspace skills as the workflow source of truth.", shared_content)
            self.assertIn("Role-specific behavior comes from the launch prompt.", shared_content)
            self.assertIn(
                "Use `.triton-agent/round-brief.md` and `.triton-agent/supervisor-report.md` as live handoff files.",
                shared_content,
            )
            self.assertIn("Treat `baseline/` as the canonical optimize baseline", shared_content)
            self.assertIn("Use `compare-perf` as the authoritative source", shared_content)
            self.assertNotIn("Improve the Triton operator", shared_content)
            self.assertNotIn("This invocation is an audit and handoff pass", shared_content)

            warnings = manager.cleanup_supervised_session(state)
            self.assertEqual(warnings, [])
            self.assertFalse(agents_path.exists())
            self.assertFalse((workdir / ".triton-agent").exists())
            self.assertTrue(state.run_archive_dir.exists())
            self.assertTrue((state.run_archive_dir / "shared-guidance.md").exists())
            self.assertTrue((state.run_archive_dir / "final" / "round-brief.md").exists())
            self.assertTrue((state.run_archive_dir / "final" / "supervisor-report.md").exists())
            self.assertTrue((state.run_archive_dir / "history").exists())

    def test_prepare_uses_claude_file_and_restores_existing_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            guidance_path = workdir / "CLAUDE.md"
            guidance_path.write_text("original content\n", encoding="utf-8")

            manager = OptimizeGuidanceManager()
            state = manager.prepare_supervised_session(
                workdir,
                agent_name="claude",
            )

            shared_content = guidance_path.read_text(encoding="utf-8")
            self.assertIsNotNone(state.backup_path)
            self.assertTrue(state.backup_path is not None and state.backup_path.exists())
            self.assertEqual(state.guidance_path, guidance_path)
            self.assertIn("# CLAUDE.md", shared_content)
            self.assertIn("## Triton Agent Optimize Orchestration", shared_content)
            self.assertIn("Role-specific behavior comes from the launch prompt.", shared_content)
            self.assertNotIn("Improve the Triton operator", shared_content)

            warnings = manager.cleanup_supervised_session(state)
            self.assertEqual(warnings, [])
            self.assertEqual(guidance_path.read_text(encoding="utf-8"), "original content\n")
            self.assertFalse(state.backup_path is not None and state.backup_path.exists())

    def test_prepare_mentions_strict_analysis_in_shared_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            manager = OptimizeGuidanceManager()
            state = manager.prepare_supervised_session(
                workdir,
                agent_name="codex",
                require_analysis=True,
            )

            shared_content = state.guidance_path.read_text(encoding="utf-8")
            self.assertIn(
                "Require profiling or IR-backed evidence before the first code-changing round when possible.",
                shared_content,
            )
            self.assertIn("Do not begin with blind tiling or launch-parameter search.", shared_content)

            warnings = manager.cleanup_supervised_session(state)
            self.assertEqual(warnings, [])

    def test_prepare_rejects_preexisting_nonempty_runtime_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            runtime_root = workdir / ".triton-agent"
            runtime_root.mkdir()
            (runtime_root / "leftover.txt").write_text("busy\n", encoding="utf-8")

            manager = OptimizeGuidanceManager()

            with self.assertRaisesRegex(RuntimeError, "Existing \\.triton-agent/ directory contains data"):
                manager.prepare_supervised_session(
                    workdir,
                    agent_name="codex",
                )

    def test_describe_cleanup_lists_archive_and_runtime_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            manager = OptimizeGuidanceManager()
            state = manager.prepare_supervised_session(
                workdir,
                agent_name="codex",
            )

            messages = manager.describe_cleanup_supervised_session(state)

            self.assertTrue(any("archiving supervised optimize logs" in message for message in messages))
            self.assertTrue(any("round-brief.md" in message for message in messages))
            self.assertTrue(any("supervisor-report.md" in message for message in messages))
            self.assertTrue(
                any("removing temporary optimize runtime directory tree" in message for message in messages)
            )

            warnings = manager.cleanup_supervised_session(state)
            self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
