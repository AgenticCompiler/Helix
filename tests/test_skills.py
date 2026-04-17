import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.skills import SkillLinkManager


class SkillLinkManagerTests(unittest.TestCase):
    def test_repo_skills_stage_optimize_and_optimize_check_for_codex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            source = Path(__file__).resolve().parents[1] / "skills"

            manager = SkillLinkManager(source)
            links = manager.prepare_codex_skills(
                workspace,
                skill_names=(
                    "triton-npu-optimize",
                    "triton-npu-optimize-check",
                    "triton-npu-analyze-round-performance",
                ),
            )

            target = workspace / ".codex" / "skills"
            self.assertTrue((target / "triton-npu-optimize" / "SKILL.md").exists())
            self.assertTrue((target / "triton-npu-optimize-check" / "SKILL.md").exists())
            self.assertTrue((target / "triton-npu-analyze-round-performance" / "SKILL.md").exists())
            manager.cleanup(links)

    def test_copy_only_requested_skill_dirs_when_codex_skills_dir_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            workspace.mkdir()
            source.mkdir()
            for name in (
                "triton-npu-gen-eval-suite",
                "triton-npu-gen-test",
                "triton-npu-gen-bench",
                "triton-npu-run-eval",
                "triton-npu-optimize",
                "triton-npu-analyze-round-performance",
            ):
                (source / name).mkdir()
                (source / name / "SKILL.md").write_text(f"{name}\n", encoding="utf-8")

            manager = SkillLinkManager(source)
            links = manager.prepare_codex_skills(
                workspace,
                skill_names=(
                    "triton-npu-gen-eval-suite",
                    "triton-npu-gen-test",
                    "triton-npu-gen-bench",
                    "triton-npu-run-eval",
                ),
            )

            target = workspace / ".codex" / "skills"
            self.assertTrue((target / "triton-npu-gen-eval-suite").exists())
            self.assertTrue((target / "triton-npu-gen-test").exists())
            self.assertTrue((target / "triton-npu-gen-bench").exists())
            self.assertTrue((target / "triton-npu-run-eval").exists())
            self.assertFalse((target / "triton-npu-optimize").exists())
            self.assertFalse((target / "triton-npu-analyze-round-performance").exists())
            self.assertEqual(links.created_paths, [target])
            manager.cleanup(links)
            self.assertFalse(target.exists())

    def test_reject_missing_requested_skill_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            workspace.mkdir()
            source.mkdir()
            (source / "triton-npu-gen-test").mkdir()
            (source / "triton-npu-gen-test" / "SKILL.md").write_text("test\n", encoding="utf-8")

            manager = SkillLinkManager(source)

            with self.assertRaisesRegex(RuntimeError, "Requested skill does not exist"):
                manager.prepare_codex_skills(workspace, skill_names=("missing-skill",))

    def test_reject_existing_root_skills_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            target = workspace / ".codex" / "skills"
            workspace.mkdir()
            source.mkdir()
            (source / "triton-npu-gen-test").mkdir()
            target.parent.mkdir(parents=True)
            target.symlink_to(source, target_is_directory=True)

            manager = SkillLinkManager(source)

            with self.assertRaises(RuntimeError):
                manager.prepare_codex_skills(workspace)

    def test_copy_root_skills_directory_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            workspace.mkdir()
            source.mkdir()
            (source / "triton-npu-gen-test").mkdir()
            (source / "triton-npu-optimize-check").mkdir()
            (source / "triton-npu-gen-test" / "SKILL.md").write_text("test skill\n", encoding="utf-8")
            (source / "triton-npu-optimize-check" / "SKILL.md").write_text(
                "optimize-check skill\n",
                encoding="utf-8",
            )

            manager = SkillLinkManager(source)
            links = manager.prepare_codex_skills(workspace)

            target = workspace / ".codex" / "skills"
            copied_skill = target / "triton-npu-gen-test" / "SKILL.md"
            copied_optimize_check_skill = target / "triton-npu-optimize-check" / "SKILL.md"
            self.assertTrue(target.is_dir())
            self.assertFalse(target.is_symlink())
            self.assertEqual(copied_skill.read_text(encoding="utf-8"), "test skill\n")
            self.assertEqual(
                copied_optimize_check_skill.read_text(encoding="utf-8"),
                "optimize-check skill\n",
            )
            self.assertEqual(links.created_paths, [target])
            self.assertTrue(
                any("created skill copy" in message for message in manager.describe_prepare(links))
            )
            self.assertTrue(
                any("removed skill copy" in message for message in manager.describe_cleanup(links))
            )
            manager.cleanup(links)
            self.assertFalse(target.exists())

    def test_copy_missing_per_skill_dirs_when_codex_skills_dir_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            existing_skills = workspace / ".codex" / "skills"
            workspace.mkdir()
            existing_skills.mkdir(parents=True)
            source.mkdir()
            (source / "triton-npu-gen-test").mkdir()
            (source / "triton-npu-gen-bench").mkdir()
            (source / "triton-npu-gen-test" / "SKILL.md").write_text("test\n", encoding="utf-8")
            (source / "triton-npu-gen-bench" / "SKILL.md").write_text("bench\n", encoding="utf-8")
            (existing_skills / "user-skill").mkdir()

            manager = SkillLinkManager(source)
            links = manager.prepare_codex_skills(workspace)

            test_gen_dir = existing_skills / "triton-npu-gen-test"
            bench_gen_dir = existing_skills / "triton-npu-gen-bench"
            self.assertTrue(test_gen_dir.is_dir())
            self.assertTrue(bench_gen_dir.is_dir())
            self.assertFalse(test_gen_dir.is_symlink())
            self.assertFalse(bench_gen_dir.is_symlink())
            self.assertTrue((existing_skills / "user-skill").exists())
            self.assertEqual({path.name for path in links.created_paths}, {"triton-npu-gen-test", "triton-npu-gen-bench"})
            manager.cleanup(links)
            self.assertFalse(test_gen_dir.exists())
            self.assertFalse(bench_gen_dir.exists())
            self.assertTrue((existing_skills / "user-skill").exists())

    def test_reject_existing_opencode_skill_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            link_path = workspace / ".opencode" / "skills" / "triton-npu-gen-test"
            workspace.mkdir()
            source.mkdir()
            skill_dir = source / "triton-npu-gen-test"
            skill_dir.mkdir()
            link_path.parent.mkdir(parents=True)
            link_path.symlink_to(skill_dir, target_is_directory=True)

            manager = SkillLinkManager(source)

            with self.assertRaises(RuntimeError):
                manager.prepare_opencode_skills(workspace)

    def test_copy_opencode_skill_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            workspace.mkdir()
            source.mkdir()
            (source / "triton-npu-gen-test").mkdir()
            (source / "triton-npu-gen-bench").mkdir()
            (source / "triton-npu-gen-test" / "SKILL.md").write_text("test\n", encoding="utf-8")
            (source / "triton-npu-gen-bench" / "SKILL.md").write_text("bench\n", encoding="utf-8")

            manager = SkillLinkManager(source)
            links = manager.prepare_opencode_skills(workspace)

            test_gen_dir = workspace / ".opencode" / "skills" / "triton-npu-gen-test"
            bench_gen_dir = workspace / ".opencode" / "skills" / "triton-npu-gen-bench"
            self.assertTrue(test_gen_dir.is_dir())
            self.assertTrue(bench_gen_dir.is_dir())
            self.assertFalse(test_gen_dir.is_symlink())
            self.assertFalse(bench_gen_dir.is_symlink())
            self.assertEqual((test_gen_dir / "SKILL.md").read_text(encoding="utf-8"), "test\n")
            self.assertEqual((bench_gen_dir / "SKILL.md").read_text(encoding="utf-8"), "bench\n")
            manager.cleanup(links)
            self.assertFalse(test_gen_dir.exists())
            self.assertFalse(bench_gen_dir.exists())

    def test_reject_existing_pi_skills_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            target = workspace / ".pi" / "skills"
            workspace.mkdir()
            source.mkdir()
            (source / "triton-npu-gen-test").mkdir()
            target.parent.mkdir(parents=True)
            target.symlink_to(source, target_is_directory=True)

            manager = SkillLinkManager(source)

            with self.assertRaises(RuntimeError):
                manager.prepare_pi_skills(workspace)

    def test_copy_pi_skills_directory_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            workspace.mkdir()
            source.mkdir()
            (source / "triton-npu-gen-test").mkdir()
            (source / "triton-npu-gen-test" / "SKILL.md").write_text("test skill\n", encoding="utf-8")

            manager = SkillLinkManager(source)
            links = manager.prepare_pi_skills(workspace)

            target = workspace / ".pi" / "skills"
            copied_skill = target / "triton-npu-gen-test" / "SKILL.md"
            self.assertTrue(target.is_dir())
            self.assertFalse(target.is_symlink())
            self.assertEqual(copied_skill.read_text(encoding="utf-8"), "test skill\n")
            self.assertEqual(links.created_paths, [target])
            manager.cleanup(links)
            self.assertFalse(target.exists())

    def test_reject_existing_claude_skills_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            target = workspace / ".claude" / "skills"
            workspace.mkdir()
            source.mkdir()
            (source / "triton-npu-gen-test").mkdir()
            target.parent.mkdir(parents=True)
            target.symlink_to(source, target_is_directory=True)

            manager = SkillLinkManager(source)

            with self.assertRaises(RuntimeError):
                manager.prepare_claude_skills(workspace)

    def test_copy_claude_skills_directory_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            workspace.mkdir()
            source.mkdir()
            (source / "triton-npu-gen-test").mkdir()
            (source / "triton-npu-gen-test" / "SKILL.md").write_text("test skill\n", encoding="utf-8")

            manager = SkillLinkManager(source)
            links = manager.prepare_claude_skills(workspace)

            target = workspace / ".claude" / "skills"
            copied_skill = target / "triton-npu-gen-test" / "SKILL.md"
            self.assertTrue(target.is_dir())
            self.assertFalse(target.is_symlink())
            self.assertEqual(copied_skill.read_text(encoding="utf-8"), "test skill\n")
            self.assertEqual(links.created_paths, [target])
            manager.cleanup(links)
            self.assertFalse(target.exists())

    def test_reject_existing_openhands_skills_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            target = workspace / ".openhands" / "skills"
            workspace.mkdir()
            source.mkdir()
            (source / "triton-npu-gen-test").mkdir()
            target.parent.mkdir(parents=True)
            target.symlink_to(source, target_is_directory=True)

            manager = SkillLinkManager(source)

            with self.assertRaises(RuntimeError):
                manager.prepare_openhands_skills(workspace)

    def test_copy_openhands_skills_directory_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            workspace.mkdir()
            source.mkdir()
            (source / "triton-npu-gen-test").mkdir()
            (source / "triton-npu-gen-test" / "SKILL.md").write_text("test skill\n", encoding="utf-8")

            manager = SkillLinkManager(source)
            links = manager.prepare_openhands_skills(workspace)

            target = workspace / ".openhands" / "skills"
            copied_skill = target / "triton-npu-gen-test" / "SKILL.md"
            self.assertTrue(target.is_dir())
            self.assertFalse(target.is_symlink())
            self.assertEqual(copied_skill.read_text(encoding="utf-8"), "test skill\n")
            self.assertEqual(links.created_paths, [target])
            manager.cleanup(links)
            self.assertFalse(target.exists())

    def test_repo_skills_include_optimize_check_for_codex_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            source = Path(__file__).resolve().parents[1] / "skills"

            manager = SkillLinkManager(source)
            links = manager.prepare_codex_skills(
                workspace,
                skill_names=("triton-npu-optimize", "triton-npu-optimize-check"),
            )

            target = workspace / ".codex" / "skills"
            self.assertTrue((target / "triton-npu-optimize" / "SKILL.md").exists())
            self.assertTrue((target / "triton-npu-optimize-check" / "SKILL.md").exists())
            manager.cleanup(links)


if __name__ == "__main__":
    unittest.main()
