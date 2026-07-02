import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.distill.discovery import discover_operator_pairs


class DistillDiscoveryTests(unittest.TestCase):
    def test_skips_operator_without_opt_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            op_dir = root / "op"
            op_dir.mkdir()
            (op_dir / "foo.py").write_text("x = 1\n", encoding="utf-8")
            stream = StringIO()

            result = discover_operator_pairs(root, stream=stream)

            self.assertEqual(result.pairs, ())
            self.assertEqual(len(result.skips), 1)
            self.assertIn("no opt_*.py", result.skips[0].reason)
            self.assertIn("skip", stream.getvalue())

    def test_skips_opt_without_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            op_dir = root / "op"
            op_dir.mkdir()
            (op_dir / "opt_foo.py").write_text("x = 2\n", encoding="utf-8")

            result = discover_operator_pairs(root)

            self.assertEqual(result.pairs, ())
            self.assertEqual(len(result.skips), 1)
            self.assertIn("missing baseline file foo.py", result.skips[0].reason)

    def test_discovers_pairs_and_excludes_skills_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            op_dir = root / "op"
            op_dir.mkdir()
            (op_dir / "foo.py").write_text("x = 1\n", encoding="utf-8")
            (op_dir / "opt_foo.py").write_text("x = 2\n", encoding="utf-8")
            skills_dir = root / "skills"
            skills_dir.mkdir()

            result = discover_operator_pairs(root, exclude_dirs={skills_dir})

            self.assertEqual(len(result.pairs), 1)
            self.assertEqual(result.pairs[0].baseline_path.name, "foo.py")
            self.assertEqual(result.pairs[0].expected_path.name, "opt_foo.py")
            self.assertEqual(result.skips, ())

    def test_discovers_direct_optimize_process_from_learned_lessons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "learned_lessons.md").write_text("- use tiling\n", encoding="utf-8")
            (root / "opt-note.md").write_text("## Overall Summary\nFinal best round: round-2\n", encoding="utf-8")
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "kernel.py").write_text("x = 1\n", encoding="utf-8")
            (baseline_dir / "state.json").write_text(
                json.dumps({"baseline_operator": "kernel.py"}),
                encoding="utf-8",
            )
            round_one = root / "opt-round-1"
            round_one.mkdir()
            (round_one / "summary.md").write_text("round one\n", encoding="utf-8")
            round_two = root / "opt-round-2"
            round_two.mkdir()
            (round_two / "opt_kernel.py").write_text("x = 2\n", encoding="utf-8")
            (round_two / "attempts.md").write_text("round two attempts\n", encoding="utf-8")

            result = discover_operator_pairs(root, source="optimize-process")

            self.assertEqual(len(result.pairs), 1)
            pair = result.pairs[0]
            self.assertEqual(pair.source_kind, "optimize-process")
            self.assertEqual(pair.baseline_path, (baseline_dir / "kernel.py").resolve())
            self.assertEqual(pair.expected_path, round_two / "opt_kernel.py")
            self.assertEqual(pair.learned_lessons_path, root / "learned_lessons.md")
            self.assertIn(root / "opt-note.md", pair.context_paths)
            self.assertIn(round_one / "summary.md", pair.context_paths)
            self.assertIn(round_two / "attempts.md", pair.context_paths)

    def test_skips_optimize_process_without_final_operator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "learned_lessons.md").write_text("- use tiling\n", encoding="utf-8")
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "kernel.py").write_text("x = 1\n", encoding="utf-8")

            result = discover_operator_pairs(root, source="optimize-process")

            self.assertEqual(result.pairs, ())
            self.assertEqual(len(result.skips), 1)
            self.assertIn("final optimized operator", result.skips[0].reason)

    def test_optimize_process_source_skips_operator_without_learned_lessons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            op_dir = root / "op"
            op_dir.mkdir()
            (op_dir / "foo.py").write_text("x = 1\n", encoding="utf-8")
            (op_dir / "opt_foo.py").write_text("x = 2\n", encoding="utf-8")

            result = discover_operator_pairs(root, source="optimize-process")

            self.assertEqual(result.pairs, ())
            self.assertEqual(len(result.skips), 1)
            self.assertIn("directory does not look like an optimize workspace", result.skips[0].reason)

    def test_optimize_process_source_skips_direct_workspace_without_learned_lessons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "baseline").mkdir()
            (root / "opt-round-1").mkdir()

            result = discover_operator_pairs(root, source="optimize-process")

            self.assertEqual(result.pairs, ())
            self.assertEqual(len(result.skips), 1)
            self.assertEqual(result.skips[0].operator_dir, root)
            self.assertIn("baseline operator not found in optimize workspace", result.skips[0].reason)


if __name__ == "__main__":
    unittest.main()
