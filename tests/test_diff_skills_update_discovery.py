import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.diff_skills_update.discovery import discover_operator_pairs


class DiffSkillsUpdateDiscoveryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
