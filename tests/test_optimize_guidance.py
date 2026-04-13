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
            self.assertIn("Use `compare-perf` output as the only source", worker_content)
            self.assertIn("Do not hand-calculate speedups or percentage improvements", worker_content)

            self.assertIn("## Supervisor Mission", supervisor_content)
            self.assertIn("This invocation is an audit and handoff pass", supervisor_content)
            self.assertIn("Do not perform open-ended optimization work.", supervisor_content)
            self.assertIn("Repair metadata only when the underlying evidence already exists.", supervisor_content)
            self.assertIn("Emit a gate result for the completed round.", supervisor_content)
            self.assertIn("Use only existing `compare-perf` results", supervisor_content)

            warnings = manager.cleanup(state)
            self.assertEqual(warnings, [])
            self.assertFalse(agents_path.exists())
            self.assertFalse(state.worker_brief_path.exists())
            self.assertFalse(state.supervisor_brief_path.exists())
            self.assertFalse(state.round_brief_path.exists())
            self.assertFalse(state.supervisor_report_path.exists())

    def test_prepare_exposes_history_and_archive_paths(self) -> None:
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

            history_dir = workdir / ".triton-agent" / "history"
            archive_root = workdir / "optimize-logs" / "triton-agent"

            self.assertEqual(state.history_dir, history_dir)
            self.assertTrue(history_dir.exists())
            self.assertEqual(state.archive_root, archive_root)
            self.assertEqual(state.run_archive_dir.parent, archive_root)
            self.assertEqual(
                state.shared_guidance_snapshot_path,
                state.run_archive_dir / "shared-guidance.md",
            )
            self.assertFalse(archive_root.exists())
            self.assertTrue(state.created_guidance)

            warnings = manager.cleanup(state)
            self.assertEqual(warnings, [])

    def test_cleanup_archives_supervised_logs_and_removes_runtime_dir(self) -> None:
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
            state.round_brief_path.write_text("final brief\n", encoding="utf-8")
            state.supervisor_report_path.write_text("final report\n", encoding="utf-8")
            (state.history_dir / "round-001-brief.md").write_text("round 1 brief\n", encoding="utf-8")
            (state.history_dir / "round-001-supervisor-report.md").write_text(
                "round 1 report\n", encoding="utf-8"
            )

            warnings = manager.cleanup(state)
            self.assertEqual(warnings, [])

            self.assertTrue(state.run_archive_dir.exists())
            self.assertTrue(state.shared_guidance_snapshot_path.exists())
            self.assertTrue((state.run_archive_dir / "roles" / "optimize-worker.md").exists())
            self.assertTrue((state.run_archive_dir / "roles" / "optimize-supervisor.md").exists())
            self.assertEqual(
                (state.run_archive_dir / "final" / "round-brief.md").read_text(encoding="utf-8"),
                "final brief\n",
            )
            self.assertEqual(
                (state.run_archive_dir / "final" / "supervisor-report.md").read_text(encoding="utf-8"),
                "final report\n",
            )
            self.assertTrue((state.run_archive_dir / "history" / "round-001-brief.md").exists())
            self.assertTrue(
                (state.run_archive_dir / "history" / "round-001-supervisor-report.md").exists()
            )
            self.assertFalse((workdir / ".triton-agent").exists())

    def test_cleanup_warns_when_archive_write_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            # Force an archive failure by preventing directory creation.
            (workdir / "optimize-logs").write_text("not a directory\n", encoding="utf-8")

            manager = OptimizeGuidanceManager()
            state = manager.prepare(
                workdir,
                operator,
                test_mode="differential",
                bench_mode="standalone",
                agent_name="codex",
            )
            (state.history_dir / "round-001-brief.md").write_text("round 1 brief\n", encoding="utf-8")

            warnings = manager.cleanup(state)

            self.assertTrue(any("archive" in warning.lower() for warning in warnings))
            self.assertFalse((workdir / ".triton-agent").exists())

    def test_prepare_requires_cleanup_of_old_triton_agent_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            triton_dir = workdir / ".triton-agent"
            triton_dir.mkdir(parents=True, exist_ok=True)
            (triton_dir / "round-brief.md").write_text("leftover\n", encoding="utf-8")

            manager = OptimizeGuidanceManager()
            with self.assertRaisesRegex(RuntimeError, r"Existing \.triton-agent/ directory contains data"):
                manager.prepare(
                    workdir,
                    operator,
                    test_mode="differential",
                    bench_mode="standalone",
                    agent_name="codex",
                )

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
