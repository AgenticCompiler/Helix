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

from verify_batch_scaffold import verify_workspace  # noqa: E402
from workspace_deps import (  # noqa: E402
    DEPS_PATH_BOOTSTRAP_START,
    REPO_PATH_BOOTSTRAP_START,
    has_deps_path_bootstrap,
    has_repo_path_bootstrap,
    normalize_dependency_dir,
    repo_scoped_import_modules,
    sync_workspace_dependencies,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def _write_minimal_fla_repo(repo: Path) -> None:
    for relative in (
        "src/kernels/fla/__init__.py",
        "src/kernels/fla/ops/__init__.py",
        "src/kernels/fla/ops/utils/__init__.py",
        "src/kernels/fla/utils.py",
    ):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
    (repo / "src/kernels/fla/ops/utils/__init__.py").write_text(
        "def prepare_chunk_indices():\n    return None\n",
        encoding="utf-8",
    )
    (repo / "src/kernels/fla/utils.py").write_text("device = 'cpu'\n", encoding="utf-8")
    for init_path in (
        repo / "src/kernels/fla/__init__.py",
        repo / "src/kernels/fla/ops/__init__.py",
    ):
        init_path.write_text("", encoding="utf-8")


class WorkspaceDepsTests(unittest.TestCase):
    def test_normalize_dependency_dir_rejects_placeholder(self) -> None:
        self.assertEqual(normalize_dependency_dir("{deps}"), "deps")
        self.assertEqual(normalize_dependency_dir("deps"), "deps")

    def test_repo_scoped_import_modules_detects_fla(self) -> None:
        source = "from fla.ops.utils import prepare_chunk_indices\nimport torch\n"
        self.assertEqual(
            repo_scoped_import_modules(source),
            ["fla.ops.utils"],
        )

    def test_verify_fails_on_literal_brace_deps_directory(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            workspace = Path(tmp) / "demo"
            workspace.mkdir()
            (workspace / "demo.py").write_text(
                "from fla.utils import device\n\ndef demo():\n    pass\n",
                encoding="utf-8",
            )
            (workspace / "{deps}").mkdir()
            meta = {
                "workspace": "demo",
                "kernel_name": "demo",
                "operator_filename": "demo.py",
                "dependency_dir": "{deps}",
            }
            report = verify_workspace(workspace, meta, shared_sources={})

        self.assertFalse(report["passed"])
        self.assertTrue(any("{deps}" in issue for issue in report["issues"]))

    def test_sync_default_injects_repo_path_and_passes_smoke(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _write_minimal_fla_repo(repo)

            workspace = root / "demo_ws"
            workspace.mkdir()
            (workspace / "demo.py").write_text(
                "from fla.ops.utils import prepare_chunk_indices\n"
                "from fla.utils import device\n",
                encoding="utf-8",
            )
            from batch_evaluation import resolve_workspace_meta, upsert_workspace_entry

            upsert_workspace_entry(
                root,
                "demo_ws",
                {
                    "workspace": "demo_ws",
                    "operator_filename": "demo.py",
                    "dependency_dir": "{deps}",
                },
            )

            report = sync_workspace_dependencies(workspace, repo)
            operator_text = (workspace / "demo.py").read_text(encoding="utf-8")
            meta = resolve_workspace_meta(workspace, batch_root=root)

            self.assertIn(REPO_PATH_BOOTSTRAP_START, operator_text)
            self.assertTrue(has_repo_path_bootstrap(operator_text))
            self.assertEqual(report["dependency_strategy"], "repo_path")
            self.assertTrue(report["import_smoke_passed"])
            self.assertEqual(report["copied_dependencies"], [])
            self.assertTrue(meta["import_smoke_passed"])
            self.assertFalse((workspace / "{deps}").exists())

    def test_sync_force_copy_deps_materializes_fla_tree(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _write_minimal_fla_repo(repo)

            workspace = root / "demo_ws"
            workspace.mkdir()
            (workspace / "demo.py").write_text(
                "from fla.ops.utils import prepare_chunk_indices\n"
                "from fla.utils import device\n",
                encoding="utf-8",
            )
            from batch_evaluation import upsert_workspace_entry

            upsert_workspace_entry(
                root,
                "demo_ws",
                {
                    "workspace": "demo_ws",
                    "operator_filename": "demo.py",
                    "dependency_dir": "deps",
                },
            )

            report = sync_workspace_dependencies(
                workspace,
                repo,
                force_deps_copy=True,
            )

            operator_text = (workspace / "demo.py").read_text(encoding="utf-8")
            self.assertEqual(report["dependency_strategy"], "deps_copy")
            self.assertTrue(report["import_smoke_passed"])
            self.assertIn(DEPS_PATH_BOOTSTRAP_START, operator_text)
            self.assertTrue(has_deps_path_bootstrap(operator_text))
            self.assertNotIn(REPO_PATH_BOOTSTRAP_START, operator_text)
            self.assertTrue((workspace / "deps/fla/ops/utils/__init__.py").is_file())
            self.assertTrue((workspace / "deps/fla/utils.py").is_file())
            self.assertIn("deps/fla/utils.py", report["copied_dependencies"])


if __name__ == "__main__":
    unittest.main()
