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

from audit_batch import audit_workspace
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
            (workspace / "validation-meta.json").write_text(
                json.dumps({"expected_patterns": ["grid-flatten-and-ub-buffering"]}),
                encoding="utf-8",
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
            meta = json.loads((destination / "validation-meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["validation_status"], "completed")
            self.assertIn("archived_at", meta)
            self.assertEqual(list_active_validation_workspaces(batch_root), [])
            self.assertEqual(len(list_completed_validation_workspaces(batch_root)), 1)

    def test_list_active_validation_workspaces_skips_completed_dir(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            batch_root = Path(tmp) / "batch"
            active = batch_root / "chunk_o"
            active.mkdir(parents=True)
            (active / "validation-meta.json").write_text("{}", encoding="utf-8")
            completed_root = batch_root / "_completed" / "wy_fast"
            completed_root.mkdir(parents=True)
            (completed_root / "validation-meta.json").write_text("{}", encoding="utf-8")

            names = [path.name for path in list_active_validation_workspaces(batch_root)]
            self.assertEqual(names, ["chunk_o"])
            self.assertEqual([path.name for path in list_completed_validation_workspaces(batch_root)], ["wy_fast"])

    def test_audit_workspace_marks_completed_location(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            workspace = Path(tmp) / "_completed" / "chunk_o"
            workspace.mkdir(parents=True)
            (workspace / "validation-meta.json").write_text(
                json.dumps({"expected_patterns": ["shape-specialization"]}),
                encoding="utf-8",
            )
            round_dir = workspace / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "summary.md").write_text("shape-specialization\n", encoding="utf-8")

            report = audit_workspace(workspace)
            self.assertTrue(report["passed"])
            self.assertEqual(report["location"], "completed")


if __name__ == "__main__":
    unittest.main()
