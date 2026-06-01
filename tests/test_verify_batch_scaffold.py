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

from verify_batch_scaffold import verify_workspace

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class VerifyBatchScaffoldTests(unittest.TestCase):
    def test_shared_source_requires_split_metadata(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            batch_root = Path(tmp)
            workspace = batch_root / "chunk_o"
            workspace.mkdir()
            (workspace / "chunk_o.py").write_text(
                "def forward_chunk_o():\n    pass\n\ndef forward_wy_fast():\n    pass\n",
                encoding="utf-8",
            )
            meta = {
                "workspace": "chunk_o",
                "source_path": "src/kernels/fused_ops.py",
                "operator_filename": "chunk_o.py",
                "expected_patterns": ["grid-flatten-and-ub-buffering"],
            }
            shared = {"src/kernels/fused_ops.py": ["chunk_o", "wy_fast"]}
            report = verify_workspace(workspace, meta, shared_sources=shared)

        self.assertFalse(report["passed"])
        self.assertTrue(any("validation_target" in issue for issue in report["issues"]))
        self.assertTrue(any("split_from" in issue for issue in report["issues"]))

    def test_split_workspace_passes_when_extract_is_minimal(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            workspace = Path(tmp) / "chunk_o"
            workspace.mkdir()
            (workspace / "chunk_o.py").write_text(
                "def forward_chunk_o():\n    pass\n",
                encoding="utf-8",
            )
            meta = {
                "workspace": "chunk_o",
                "source_path": "src/kernels/fused_ops.py",
                "operator_filename": "chunk_o.py",
                "validation_target": "forward_chunk_o",
                "split_from": "src/kernels/fused_ops.py",
                "included_symbols": ["forward_chunk_o"],
                "excluded_targets": ["forward_wy_fast"],
            }
            shared = {"src/kernels/fused_ops.py": ["chunk_o", "wy_fast"]}
            report = verify_workspace(workspace, meta, shared_sources=shared)

        self.assertTrue(report["passed"])

    def test_excluded_target_still_in_operator_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            workspace = Path(tmp) / "chunk_o"
            workspace.mkdir()
            (workspace / "chunk_o.py").write_text(
                "def forward_chunk_o():\n    pass\n\ndef forward_wy_fast():\n    pass\n",
                encoding="utf-8",
            )
            meta = {
                "workspace": "chunk_o",
                "source_path": "src/kernels/fused_ops.py",
                "operator_filename": "chunk_o.py",
                "validation_target": "forward_chunk_o",
                "split_from": "src/kernels/fused_ops.py",
                "included_symbols": ["forward_chunk_o"],
                "excluded_targets": ["forward_wy_fast"],
            }
            shared = {"src/kernels/fused_ops.py": ["chunk_o", "wy_fast"]}
            report = verify_workspace(workspace, meta, shared_sources=shared)

        self.assertFalse(report["passed"])
        self.assertTrue(any("excluded_targets" in issue for issue in report["issues"]))


if __name__ == "__main__":
    unittest.main()
