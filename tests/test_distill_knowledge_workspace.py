import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.distill.knowledge_workspace import (
    ensure_editable_knowledge_skill,
    export_changed_pattern_cards,
    optimize_knowledge_skill_name,
    rebuild_pattern_index,
    snapshot_pattern_card_texts,
)


class DistillKnowledgeWorkspaceTests(unittest.TestCase):
    def test_optimize_knowledge_skill_name_uses_active_language(self) -> None:
        self.assertEqual(optimize_knowledge_skill_name("triton"), "triton-npu-optimize-knowledge")
        self.assertEqual(optimize_knowledge_skill_name("tilelang"), "tilelang-npu-optimize-knowledge")

    def test_bundled_tilelang_knowledge_skill_rebuilds_pattern_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            knowledge_dir = ensure_editable_knowledge_skill(
                Path(tmp) / "skills",
                language="tilelang",
            )

            rebuild_pattern_index(knowledge_dir)

            index_text = (knowledge_dir / "references" / "pattern_index.md").read_text(
                encoding="utf-8",
            )
            self.assertIn("`autotune`", index_text)

    def test_rebuild_pattern_index_does_not_require_skill_local_builder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            knowledge_dir = ensure_editable_knowledge_skill(
                Path(tmp) / "skills",
                language="tilelang",
            )
            builder = knowledge_dir / "scripts" / "build_pattern_index.py"
            if builder.exists():
                builder.unlink()

            rebuild_pattern_index(knowledge_dir)

            index_text = (knowledge_dir / "references" / "pattern_index.md").read_text(
                encoding="utf-8",
            )
            self.assertIn("## All Patterns", index_text)

    def test_export_changed_pattern_cards_copies_only_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "skills" / "tilelang-npu-optimize-knowledge"
            patterns = source / "references" / "patterns"
            patterns.mkdir(parents=True)
            (patterns / "tiling.md").write_text(
                "# Tiling\n\n## Summary\nold\n\n## Use When\n\n- Old trigger.\n",
                encoding="utf-8",
            )
            (patterns / "autotune.md").write_text(
                "# Autotune\n\n## Summary\nsame\n\n## Use When\n\n- Same trigger.\n",
                encoding="utf-8",
            )
            snapshot = snapshot_pattern_card_texts(source)

            (patterns / "tiling.md").write_text(
                "# Tiling\n\n## Summary\nnew\n\n## Use When\n\n- New trigger.\n",
                encoding="utf-8",
            )
            (patterns / "new-pattern.md").write_text(
                "# New\n\n## Summary\nadded\n\n## Use When\n\n- Added trigger.\n",
                encoding="utf-8",
            )

            exported = export_changed_pattern_cards(
                source,
                root / "distill-output",
                language="tilelang",
                pattern_snapshot=snapshot,
                updated_pattern_names=["tiling"],
            )

            self.assertEqual(exported, ["new-pattern.md", "tiling.md"])
            update_patterns = root / "distill-output" / "tilelang-npu-optimize-knowledge" / "references" / "patterns"
            self.assertFalse((update_patterns / "autotune.md").exists())
            self.assertEqual(
                (update_patterns / "tiling.md").read_text(encoding="utf-8"),
                "# Tiling\n\n## Summary\nnew\n\n## Use When\n\n- New trigger.\n",
            )
            self.assertFalse((root / "distill-output" / "tilelang-npu-optimize-knowledge" / "scripts").exists())
            manifest = (root / "distill-output" / "updated_patterns.json").read_text(encoding="utf-8")
            self.assertIn("tiling.md", manifest)


if __name__ == "__main__":
    unittest.main()
