import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.skills import SkillLinkManager


class SkillLinkManagerTests(unittest.TestCase):
    def test_skip_existing_root_skills_symlink_that_already_matches_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            target = workspace / ".codex" / "skills"
            workspace.mkdir()
            source.mkdir()
            (source / "test-gen").mkdir()
            target.parent.mkdir(parents=True)
            target.symlink_to(source, target_is_directory=True)

            manager = SkillLinkManager(source)
            links = manager.prepare_codex_skills(workspace)

            self.assertEqual(links.created_paths, [])
            self.assertTrue(target.is_symlink())
            self.assertEqual(target.resolve(), source.resolve())

    def test_create_root_skills_symlink_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            workspace.mkdir()
            source.mkdir()
            (source / "test-gen").mkdir()

            manager = SkillLinkManager(source)
            links = manager.prepare_codex_skills(workspace)

            target = workspace / ".codex" / "skills"
            self.assertTrue(target.is_symlink())
            self.assertEqual(target.resolve(), source.resolve())
            self.assertTrue(
                any(
                    "created skill link" in message and "->" in message
                    for message in manager.describe_prepare(links)
                )
            )
            self.assertTrue(
                any(
                    "removed skill link" in message and "->" in message
                    for message in manager.describe_cleanup(links)
                )
            )
            manager.cleanup(links)
            self.assertFalse(target.exists())

    def test_create_per_skill_links_when_skills_dir_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            existing_skills = workspace / ".codex" / "skills"
            workspace.mkdir()
            existing_skills.mkdir(parents=True)
            source.mkdir()
            (source / "test-gen").mkdir()
            (source / "bench-gen").mkdir()

            manager = SkillLinkManager(source)
            links = manager.prepare_codex_skills(workspace)

            test_gen_link = existing_skills / "test-gen"
            bench_gen_link = existing_skills / "bench-gen"
            self.assertTrue(test_gen_link.is_symlink())
            self.assertTrue(bench_gen_link.is_symlink())
            prepare_messages = manager.describe_prepare(links)
            self.assertEqual(len(prepare_messages), 2)
            self.assertTrue(
                all("created skill link" in message and "->" in message for message in prepare_messages)
            )
            manager.cleanup(links)
            self.assertFalse(test_gen_link.exists())
            self.assertFalse(bench_gen_link.exists())

    def test_skip_existing_opencode_skill_symlink_that_already_matches_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            link_path = workspace / ".opencode" / "skills" / "test-gen"
            workspace.mkdir()
            source.mkdir()
            skill_dir = source / "test-gen"
            skill_dir.mkdir()
            link_path.parent.mkdir(parents=True)
            link_path.symlink_to(skill_dir, target_is_directory=True)

            manager = SkillLinkManager(source)
            links = manager.prepare_opencode_skills(workspace)

            self.assertEqual(links.created_paths, [])
            self.assertTrue(link_path.is_symlink())
            self.assertEqual(link_path.resolve(), skill_dir.resolve())

    def test_create_opencode_skill_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            workspace.mkdir()
            source.mkdir()
            (source / "test-gen").mkdir()
            (source / "bench-gen").mkdir()

            manager = SkillLinkManager(source)
            links = manager.prepare_opencode_skills(workspace)

            test_gen_link = workspace / ".opencode" / "skills" / "test-gen"
            bench_gen_link = workspace / ".opencode" / "skills" / "bench-gen"
            self.assertTrue(test_gen_link.is_symlink())
            self.assertTrue(bench_gen_link.is_symlink())
            self.assertTrue(
                all("created skill link" in message and "->" in message for message in manager.describe_prepare(links))
            )
            manager.cleanup(links)
            self.assertFalse(test_gen_link.exists())
            self.assertFalse(bench_gen_link.exists())


if __name__ == "__main__":
    unittest.main()
