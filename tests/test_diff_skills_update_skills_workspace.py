import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.diff_skills_update.skills_workspace import (
    export_changed_patterns,
    snapshot_pattern_cards,
)


class DiffSkillsUpdateSkillsWorkspaceTests(unittest.TestCase):
    def test_export_changed_patterns_copies_only_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "skills" / "triton" / "triton-npu-optimize-knowledge"
            patterns = source / "references" / "patterns"
            scripts = source / "scripts"
            patterns.mkdir(parents=True)
            scripts.mkdir(parents=True)
            (patterns / "tiling.md").write_text("# Tiling\n\n## Summary\nold\n", encoding="utf-8")
            (patterns / "autotune.md").write_text("# Autotune\n\n## Summary\nsame\n", encoding="utf-8")
            (scripts / "build_pattern_index.py").write_text(
                "import argparse\nfrom pathlib import Path\n"
                "parser = argparse.ArgumentParser()\n"
                "parser.add_argument('--patterns-dir')\n"
                "parser.add_argument('--output')\n"
                "args = parser.parse_args()\n"
                "Path(args.output).write_text('# Index\\n', encoding='utf-8')\n",
                encoding="utf-8",
            )
            snapshot = snapshot_pattern_cards(source)

            (patterns / "tiling.md").write_text("# Tiling\n\n## Summary\nnew\n", encoding="utf-8")
            (patterns / "new-pattern.md").write_text("# New\n\n## Summary\nadded\n", encoding="utf-8")

            exported = export_changed_patterns(
                source,
                root / "update_skills",
                pattern_snapshot=snapshot,
                updated_pattern_names=["tiling"],
            )

            self.assertEqual(exported, ["new-pattern.md", "tiling.md"])
            update_patterns = root / "update_skills" / "triton-npu-optimize-knowledge" / "references" / "patterns"
            self.assertFalse((update_patterns / "autotune.md").exists())
            self.assertEqual(
                (update_patterns / "tiling.md").read_text(encoding="utf-8"),
                "# Tiling\n\n## Summary\nnew\n",
            )
            manifest = (root / "update_skills" / "updated_patterns.json").read_text(encoding="utf-8")
            self.assertIn("tiling.md", manifest)


if __name__ == "__main__":
    unittest.main()
