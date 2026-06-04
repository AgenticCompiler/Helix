import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "triton-npu-pattern-validation-loop"
    / "scripts"
)
sys.path.insert(0, str(SCRIPT_DIR))

from batch_evaluation import load_batch_evaluation, upsert_workspace_entry
from batch_layout import (
    archive_passed_workspace,
    list_active_validation_workspaces,
    list_completed_validation_workspaces,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class PatternValidationBatchLayoutTests(unittest.TestCase):
    def test_archive_passed_workspace_moves_to_completed_and_updates_meta(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            batch_root = Path(tmp) / "batch"
            workspace = batch_root / "chunk_o"
            workspace.mkdir(parents=True)
            (workspace / "chunk_o.py").write_text("kernel\n", encoding="utf-8")
            upsert_workspace_entry(
                batch_root,
                "chunk_o",
                {
                    "operator_filename": "chunk_o.py",
                    "expected_patterns": ["grid-flatten-and-ub-buffering"],
                },
            )
            (workspace / "opt-round-1").mkdir()
            (workspace / "opt-round-1" / "summary.md").write_text(
                "grid-flatten-and-ub-buffering\n",
                encoding="utf-8",
            )

            destination = archive_passed_workspace(workspace, batch_root=batch_root)

            self.assertFalse(workspace.exists())
            self.assertEqual(destination.parent.name, "_completed")
            self.assertTrue(destination.is_dir())
            self.assertFalse((destination / "validation-meta.json").exists())
            registry = load_batch_evaluation(batch_root)
            entry = registry["workspaces"]["chunk_o"]
            self.assertEqual(entry["validation_status"], "completed")
            self.assertIn("archived_at", entry)
            self.assertEqual(list_active_validation_workspaces(batch_root), [])
            self.assertEqual(len(list_completed_validation_workspaces(batch_root)), 1)

    def test_list_active_validation_workspaces_skips_completed_dir(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            batch_root = Path(tmp) / "batch"
            active = batch_root / "chunk_o"
            active.mkdir(parents=True)
            (active / "chunk_o.py").write_text("def chunk_o():\n    pass\n", encoding="utf-8")
            upsert_workspace_entry(batch_root, "chunk_o", {"operator_filename": "chunk_o.py"})
            completed_root = batch_root / "_completed" / "wy_fast"
            completed_root.mkdir(parents=True)

            names = [path.name for path in list_active_validation_workspaces(batch_root)]
            self.assertEqual(names, ["chunk_o"])
            self.assertEqual([path.name for path in list_completed_validation_workspaces(batch_root)], ["wy_fast"])

if __name__ == "__main__":
    unittest.main()
