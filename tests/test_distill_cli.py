import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.cli import build_parser
from triton_agent.commands.distill import _config_from_args
from triton_agent.distill.models import OperatorDistillResult
from triton_agent.models import CommandKind


class DistillCliTests(unittest.TestCase):
    def test_distill_result_type_names_operator_distillation(self) -> None:
        self.assertEqual(OperatorDistillResult.__name__, "OperatorDistillResult")

    def test_distill_maps_to_command_kind(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "distill",
                "-i",
                "operators",
                "--agent",
                "opencode",
                "--lang",
                "tilelang",
                "--source",
                "optimize",
                "--output-dir",
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
        self.assertEqual(args.lang, "tilelang")
        self.assertEqual(args.output_dir, "distilled-skills")
        self.assertEqual(args.source, "optimize")
        self.assertEqual(args.max_refine_rounds, 4)
        self.assertEqual(args.concurrency, 2)
        self.assertTrue(args.force)
        self.assertTrue(args.skip_existing)
        self.assertTrue(args.promote_aligned)
        self.assertFalse(args.stream_output)

    def test_distill_defaults_to_transient_skills_and_output_dir(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["distill", "-i", "operators"])

        config = _config_from_args(args)

        self.assertEqual(
            config.skills_dir,
            (Path("operators").resolve() / ".triton-agent" / "distill-skills"),
        )
        self.assertEqual(config.output_dir, (Path("operators").resolve() / "distill-output"))
        self.assertTrue(config.cleanup_skills_dir)
        self.assertEqual(config.source, "diff")
        self.assertEqual(config.agent_name, "opencode")
        self.assertEqual(config.language, "triton")

    def test_distill_config_accepts_tilelang_language(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["distill", "-i", "operators", "--lang", "tilelang"])

        config = _config_from_args(args)

        self.assertEqual(config.language, "tilelang")

    def test_distill_post_update_review_defaults_on(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["distill", "-i", "operators"])
        config = _config_from_args(args)
        self.assertTrue(config.post_update_review)

    def test_distill_skip_review_disables_post_update_review(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["distill", "-i", "operators", "--skip-review"])
        config = _config_from_args(args)
        self.assertFalse(config.post_update_review)

    def test_old_distill_option_names_are_removed(self) -> None:
        parser = build_parser()

        for old_args in (
            ("--skills-dir", "value"),
            ("--export-dir", "value"),
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

    def test_old_distill_source_values_are_removed(self) -> None:
        parser = build_parser()

        for old_source in ("code-diff", "optimize-process", "git-repo"):
            with self.subTest(old_source=old_source):
                stderr = StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                    parser.parse_args(["distill", "-i", "operators", "--source", old_source])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("invalid choice", stderr.getvalue())

    def test_old_diff_skills_update_command_is_removed(self) -> None:
        parser = build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["diff-skills-update", "-i", "operators"])


if __name__ == "__main__":
    unittest.main()
