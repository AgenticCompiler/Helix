import tempfile
import unittest
from pathlib import Path

from tests.run_skill_test_utils import load_remote_python_bundle_module


class RemotePythonBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_remote_python_bundle_module()

    def test_resolves_fixed_worker_static_import_closure_in_stable_order(self) -> None:
        assert self.module.__file__ is not None
        scripts_root = Path(self.module.__file__).resolve().parent
        bundle = self.module.resolve_remote_python_bundle(
            [scripts_root / "run_bench_remote_worker.py"]
        )

        self.assertEqual(bundle, sorted(bundle, key=lambda path: path.relative_to(scripts_root).as_posix()))
        names = {path.name for path in bundle}
        self.assertTrue(
            {
                "run_bench_remote_worker.py",
                "run_bench_execution.py",
                "bench_contract.py",
                "perf_artifacts.py",
                "profile_csv_parser.py",
                "result_payload.py",
            }.issubset(names)
        )

    def test_ignores_external_imports_and_resolves_relative_package_imports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "entry.py").write_text(
                "import json\nfrom package import helper\n", encoding="utf-8"
            )
            package = root / "package"
            package.mkdir()
            (package / "__init__.py").write_text("from .helper import VALUE\n", encoding="utf-8")
            (package / "helper.py").write_text("import torch\nVALUE = 1\n", encoding="utf-8")

            original_file = self.module.__file__
            try:
                self.module.__file__ = str(root / "remote_python_bundle.py")
                (root / "remote_python_bundle.py").write_text("", encoding="utf-8")
                bundle = self.module.resolve_remote_python_bundle([root / "entry.py"])
            finally:
                self.module.__file__ = original_file

        resolved_root = root.resolve()
        self.assertEqual([path.relative_to(resolved_root).as_posix() for path in bundle], [
            "entry.py", "package/__init__.py", "package/helper.py",
        ])

    def test_stages_entry_and_dependencies_through_callback(self) -> None:
        assert self.module.__file__ is not None
        scripts_root = Path(self.module.__file__).resolve().parent
        staged: list[tuple[Path, str]] = []
        bundle = self.module.stage_remote_python_bundle(
            [scripts_root / "run_profile_remote_worker.py"],
            "/tmp/workspace",
            lambda source, target: staged.append((source, target)),
        )

        self.assertEqual([source for source, _target in staged], bundle)
        self.assertIn((scripts_root / "run_profile_remote_worker.py", "/tmp/workspace/run_profile_remote_worker.py"), staged)
        self.assertIn((scripts_root / "run_bench_execution.py", "/tmp/workspace/run_bench_execution.py"), staged)


if __name__ == "__main__":
    unittest.main()
