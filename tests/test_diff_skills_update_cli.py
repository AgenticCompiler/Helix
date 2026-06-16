import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.cli import build_parser
from triton_agent.commands.diff_skills_update import _config_from_args
from triton_agent.models import CommandKind


class DiffSkillsUpdateCliTests(unittest.TestCase):
    def test_diff_skills_update_maps_to_command_kind(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "diff-skills-update",
                "-i",
                "operators",
                "--agent",
                "opencode",
                "--skills-dir",
                "custom-skills",
                "--max-iterations",
                "4",
                "--concurrency",
                "2",
                "--force",
                "--skip-existing",
                "--promote-converged-skills",
                "--no-stream-output",
            ]
        )

        self.assertEqual(args.command, "diff-skills-update")
        self.assertEqual(args.command_kind, CommandKind.DIFF_SKILLS_UPDATE)
        self.assertEqual(args.input, "operators")
        self.assertEqual(args.agent, "opencode")
        self.assertEqual(args.skills_dir, "custom-skills")
        self.assertEqual(args.max_iterations, 4)
        self.assertEqual(args.concurrency, 2)
        self.assertTrue(args.force)
        self.assertTrue(args.skip_existing)
        self.assertTrue(args.promote_converged_skills)
        self.assertFalse(args.stream_output)

    def test_diff_skills_update_defaults_to_skills_and_update_skills_dirs(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["diff-skills-update", "-i", "operators"])

        config = _config_from_args(args)

        self.assertEqual(config.skills_dir, (Path("operators").resolve() / "skills"))
        self.assertEqual(config.update_skills_dir, (Path("operators").resolve() / "update_skills"))


if __name__ == "__main__":
    unittest.main()
