import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "triton-npu-pattern-validation-loop"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS))

from batch_evaluation import (  # noqa: E402
    BATCH_EVALUATION_FILENAME,
    migrate_legacy_workspace_meta,
    resolve_workspace_meta,
    upsert_workspace_entry,
)
from batch_layout import list_active_validation_workspaces  # noqa: E402

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class BatchEvaluationTests(unittest.TestCase):
    def test_upsert_and_resolve_without_workspace_meta_file(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT / "tests") as tmp:
            batch_root = Path(tmp)
            workspace = batch_root / "chunk_o"
            workspace.mkdir()
            (workspace / "chunk_o.py").write_text("def forward_chunk_o():\n    pass\n", encoding="utf-8")
            upsert_workspace_entry(
                batch_root,
                "chunk_o",
                {
                    "operator_filename": "chunk_o.py",
                    "expected_patterns": ["grid-flatten-and-ub-buffering"],
                    "source_path": "src/kernels/chunk_o.py",
                },
            )
            meta = resolve_workspace_meta(workspace, batch_root=batch_root)
            self.assertFalse((workspace / "validation-meta.json").exists())
            self.assertEqual(meta["expected_patterns"], ["grid-flatten-and-ub-buffering"])
            active = list_active_validation_workspaces(batch_root)
            self.assertEqual([path.name for path in active], ["chunk_o"])

    def test_migrate_legacy_validation_meta(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT / "tests") as tmp:
            batch_root = Path(tmp)
            workspace = batch_root / "demo"
            workspace.mkdir()
            (workspace / "demo.py").write_text("def demo():\n    pass\n", encoding="utf-8")
            (workspace / "validation-meta.json").write_text(
                json.dumps(
                    {
                        "workspace": "demo",
                        "operator_filename": "demo.py",
                        "expected_patterns": ["tiling"],
                    },
                )
                + "\n",
                encoding="utf-8",
            )
            migrated = migrate_legacy_workspace_meta(batch_root)
            self.assertEqual(migrated, ["demo"])
            registry = batch_root / BATCH_EVALUATION_FILENAME
            self.assertTrue(registry.is_file())
            meta = resolve_workspace_meta(workspace, batch_root=batch_root)
            self.assertEqual(meta["expected_patterns"], ["tiling"])


if __name__ == "__main__":
    unittest.main()
