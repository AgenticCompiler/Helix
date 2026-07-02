import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.cli import build_parser
from triton_agent.commands.distill import _config_from_args
from triton_agent.models import CommandKind


class DistillCliTests(unittest.TestCase):
    def test_distill_maps_to_command_kind(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "distill",
                "-i",
                "operators",
                "--agent",
                "opencode",
                "--skills-dir",
                "custom-skills",
                "--source",
                "optimize-process",
                "--export-dir",
                "distilled-skills",
                "--max-refine-rounds",
                "4",
                "--concurrency",
                "2",
                "--force",
                "--skip-existing",
                "--promote-aligned",
                "--no-stream-output",
            ]
        )

        self.assertEqual(args.command, "distill")
        self.assertEqual(args.command_kind, CommandKind.DISTILL)
        self.assertEqual(args.input, "operators")
        self.assertEqual(args.agent, "opencode")
        self.assertEqual(args.skills_dir, "custom-skills")
        self.assertEqual(args.export_dir, "distilled-skills")
        self.assertEqual(args.source, "optimize-process")
        self.assertEqual(args.max_refine_rounds, 4)
        self.assertEqual(args.concurrency, 2)
        self.assertTrue(args.force)
        self.assertTrue(args.skip_existing)
        self.assertTrue(args.promote_aligned)
        self.assertFalse(args.stream_output)

    def test_distill_defaults_to_skills_and_update_skills_dirs(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["distill", "-i", "operators"])

        config = _config_from_args(args)

        self.assertEqual(config.skills_dir, (Path("operators").resolve() / "skills"))
        self.assertEqual(config.update_skills_dir, (Path("operators").resolve() / "update_skills"))
        self.assertEqual(config.source, "code-diff")
        self.assertEqual(config.agent_name, "opencode")

    def test_old_distill_option_names_are_removed(self) -> None:
        parser = build_parser()

        for old_args in (
            ("--update-skills-dir", "value"),
            ("--max-iterations", "2"),
            ("--promote-converged-skills",),
        ):
            with self.subTest(old_args=old_args):
                stderr = StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                    parser.parse_args(["distill", "-i", "operators", *old_args])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(f"unrecognized arguments: {old_args[0]}", stderr.getvalue())

    def test_old_diff_skills_update_command_is_removed(self) -> None:
        parser = build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["diff-skills-update", "-i", "operators"])


if __name__ == "__main__":
    unittest.main()
