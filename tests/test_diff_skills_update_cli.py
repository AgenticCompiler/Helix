import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.cli import build_parser
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
                "--show-output",
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
        self.assertTrue(args.show_output)


if __name__ == "__main__":
    unittest.main()
