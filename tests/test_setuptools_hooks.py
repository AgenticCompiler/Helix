import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from setuptools import Distribution

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helix._setuptools_hooks import BuildPyWithMeta  # noqa: E402


class BuildPyWithMetaTests(unittest.TestCase):
    def test_run_cleans_stale_generated_package_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_pkg = root / "src" / "helix"
            src_pkg.mkdir(parents=True)
            (src_pkg / "__init__.py").write_text("", encoding="utf-8")
            (src_pkg / "module.py").write_text("VALUE = 1\n", encoding="utf-8")

            build_pkg = root / "build-lib" / "helix"
            build_pkg.mkdir(parents=True)
            (build_pkg / "_build_cmd.py").write_text("STALE = True\n", encoding="utf-8")

            dist = Distribution(
                {
                    "packages": ["helix"],
                    "package_dir": {"": str(root / "src")},
                }
            )
            dist.script_name = str(root / "setup.py")

            command = BuildPyWithMeta(dist)
            command.ensure_finalized()
            command.build_lib = str(root / "build-lib")

            fake_commit = "a" * 40
            with patch(
                "helix._setuptools_hooks._resolve_build_commit",
                return_value=fake_commit,
            ):
                command.run()

            built_files = {path.name for path in build_pkg.iterdir()}
            self.assertNotIn("_build_cmd.py", built_files)
            self.assertIn("__init__.py", built_files)
            self.assertIn("module.py", built_files)
            self.assertIn("_build_meta.json", built_files)

            meta_payload = json.loads((build_pkg / "_build_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta_payload["git_commit"], fake_commit)
