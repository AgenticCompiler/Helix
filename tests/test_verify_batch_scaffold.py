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
from workspace_deps import REPO_PATH_BOOTSTRAP_START, build_repo_path_bootstrap_block

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def _operator_with_repo_bootstrap(body: str) -> str:
    return build_repo_path_bootstrap_block(repo_relative_from_workspace="..") + body


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
                _operator_with_repo_bootstrap("def forward_chunk_o():\n    pass\n"),
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
                "dependency_strategy": "repo_path",
                "import_smoke_passed": True,
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

    def test_extra_root_py_fails_verify(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            workspace = Path(tmp) / "chunk_o"
            workspace.mkdir()
            (workspace / "chunk_o.py").write_text("def forward_chunk_o():\n    pass\n", encoding="utf-8")
            (workspace / "tiling_utils.py").write_text("X = 1\n", encoding="utf-8")
            meta = {
                "workspace": "chunk_o",
                "source_path": "src/kernels/chunk_o.py",
                "operator_filename": "chunk_o.py",
            }
            report = verify_workspace(workspace, meta, shared_sources={})

        self.assertFalse(report["passed"])
        self.assertEqual(report["extra_root_py"], ["tiling_utils.py"])
        self.assertTrue(any("deps/" in issue for issue in report["issues"]))

    def test_extra_triton_kernels_fail_when_kernels_in_operator_set(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            workspace = Path(tmp) / "chunk_fwd"
            workspace.mkdir()
            (workspace / "chunk_fwd.py").write_text(
                "@triton.jit\ndef kernel_a():\n    pass\n\n"
                "@triton.jit\ndef kernel_b():\n    pass\n\n"
                "def chunk_fwd():\n    kernel_a[(1,)](None)\n",
                encoding="utf-8",
            )
            meta = {
                "workspace": "chunk_fwd",
                "kernel_name": "chunk_fwd",
                "operator_filename": "chunk_fwd.py",
                "launch_functions": ["chunk_fwd"],
                "kernels_in_operator": ["kernel_a"],
            }
            report = verify_workspace(workspace, meta, shared_sources={})

        self.assertFalse(report["passed"])
        self.assertTrue(any("kernels_in_operator" in issue for issue in report["issues"]))

    def test_extra_launch_function_fails_when_launch_functions_set(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            workspace = Path(tmp) / "chunk_fwd"
            workspace.mkdir()
            (workspace / "chunk_fwd.py").write_text(
                "@triton.jit\ndef kernel_a():\n    pass\n\n"
                "@triton.jit\ndef kernel_b():\n    pass\n\n"
                "def chunk_fwd():\n    kernel_a[(1,)](None)\n\n"
                "def chunk_bwd():\n    kernel_b[(1,)](None)\n",
                encoding="utf-8",
            )
            meta = {
                "workspace": "chunk_fwd",
                "kernel_name": "chunk_fwd",
                "operator_filename": "chunk_fwd.py",
                "launch_functions": ["chunk_fwd"],
                "kernels_in_operator": ["kernel_a"],
            }
            report = verify_workspace(workspace, meta, shared_sources={})

        self.assertFalse(report["passed"])
        self.assertTrue(any("launch_functions" in issue for issue in report["issues"]))

    def test_copied_dependencies_must_use_deps_prefix(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            workspace = Path(tmp) / "chunk_o"
            workspace.mkdir()
            (workspace / "chunk_o.py").write_text("def forward_chunk_o():\n    pass\n", encoding="utf-8")
            meta = {
                "workspace": "chunk_o",
                "source_path": "src/kernels/chunk_o.py",
                "operator_filename": "chunk_o.py",
                "copied_dependencies": ["tiling_utils.py"],
            }
            report = verify_workspace(workspace, meta, shared_sources={})

        self.assertFalse(report["passed"])
        self.assertTrue(any("copied_dependencies" in issue for issue in report["issues"]))


if __name__ == "__main__":
    unittest.main()
