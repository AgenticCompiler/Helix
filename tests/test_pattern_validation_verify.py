import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path

from triton_agent.pattern_validation_loop.scaffold_verify import run_pattern_validation_verify

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class PatternValidationVerifyTests(unittest.TestCase):
    def test_verify_empty_batch_reports_no_active_workspaces(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            batch_root = Path(tmp)
            (batch_root / "workspace-plan.json").write_text("{}", encoding="utf-8")
            captured = StringIO()
            code = run_pattern_validation_verify(batch_root, stream=captured)
            output = captured.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("no active validation workspaces", output)
        self.assertIn("0 active workspaces", output)
        self.assertNotIn("-1 ok", output)

    def test_verify_passes_minimal_workspace(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            batch_root = Path(tmp)
            workspace = batch_root / "chunk_o"
            workspace.mkdir()
            (workspace / "chunk_o.py").write_text("def forward_chunk_o():\n    pass\n", encoding="utf-8")
            (workspace / "validation-meta.json").write_text(
                json.dumps(
                    {
                        "workspace": "chunk_o",
                        "source_path": "src/kernels/chunk_o.py",
                        "operator_filename": "chunk_o.py",
                        "expected_patterns": ["grid-flatten-and-ub-buffering"],
                    }
                ),
                encoding="utf-8",
            )
            code = run_pattern_validation_verify(batch_root, stream=sys.stdout)
        self.assertEqual(code, 0)

    def test_verify_fails_when_shared_source_missing_split_metadata(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            batch_root = Path(tmp)
            for name in ("chunk_o", "wy_fast"):
                workspace = batch_root / name
                workspace.mkdir()
                (workspace / f"{name}.py").write_text(f"def forward_{name}():\n    pass\n", encoding="utf-8")
                (workspace / "validation-meta.json").write_text(
                    json.dumps(
                        {
                            "workspace": name,
                            "source_path": "src/kernels/fused_ops.py",
                            "operator_filename": f"{name}.py",
                            "expected_patterns": ["grid-flatten-and-ub-buffering"],
                        }
                    ),
                    encoding="utf-8",
                )
            code = run_pattern_validation_verify(batch_root, stream=sys.stdout)
        self.assertEqual(code, 1)

    def test_verify_json_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            batch_root = Path(tmp)
            workspace = batch_root / "op"
            workspace.mkdir()
            (workspace / "op.py").write_text("def run():\n    pass\n", encoding="utf-8")
            (workspace / "validation-meta.json").write_text(
                json.dumps(
                    {
                        "workspace": "op",
                        "source_path": "src/op.py",
                        "operator_filename": "op.py",
                        "expected_patterns": [],
                    }
                ),
                encoding="utf-8",
            )
            code = run_pattern_validation_verify(batch_root, json_output=True)
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
