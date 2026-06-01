import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import sys

SCRIPTS = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "triton-npu-pattern-validation-loop"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS))

from scaffold_batch import extract_pre_optimization_snapshot, resolve_test_paths

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class ScaffoldPatternValidationBatchTests(unittest.TestCase):
    def test_extract_pre_optimization_snapshot_uses_parent_of_first_in_range_commit(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            repo = _make_git_repo(Path(tmp))
            base = _git(repo, "rev-parse", "HEAD")

            kernel = repo / "src" / "kernel.py"
            kernel.parent.mkdir(parents=True)
            kernel.write_text("def kernel():\n    return 1\n", encoding="utf-8")
            _git(repo, "add", "src/kernel.py")
            _git(repo, "commit", "-m", "add kernel")

            kernel.write_text("def kernel():\n    return 2\n", encoding="utf-8")
            _git(repo, "add", "src/kernel.py")
            _git(repo, "commit", "-m", "optimize kernel")

            snapshot = extract_pre_optimization_snapshot(
                repo=repo,
                source_path="src/kernel.py",
                base_revision=base,
                head_revision="HEAD",
            )
            self.assertIn("return 1", snapshot)
            self.assertNotIn("return 2", snapshot)

    def test_resolve_test_paths_finds_default_test_name(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            (repo / "tests").mkdir()
            (repo / "tests" / "test_kernel.py").write_text("def test_kernel(): pass\n", encoding="utf-8")
            paths = resolve_test_paths(
                repo=repo,
                entry={"test_paths": []},
                source_path="src/kernel.py",
            )
            self.assertEqual(paths[0].name, "test_kernel.py")


def _make_git_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "--template=")
    _git(repo, "config", "user.email", "tester@example.com")
    _git(repo, "config", "user.name", "Tester")
    (repo / "README.md").write_text("init\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial")
    return repo.resolve()


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return result.stdout.strip()


if __name__ == "__main__":
    unittest.main()
