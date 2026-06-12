import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

SCRIPT_DIR = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "triton-npu-analyze-commit-perf"
    / "scripts"
)
sys.path.insert(0, str(SCRIPT_DIR))

from group_commit_context_by_file import group_context_by_file

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class CommitPerfGroupByFileTests(unittest.TestCase):
    def test_groups_commits_by_file_in_chronological_order(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            repo = _make_git_repo(Path(tmp))
            base = _git(repo, "rev-parse", "HEAD")

            kernel = repo / "src" / "kernels" / "wy_fast.py"
            kernel.parent.mkdir(parents=True)
            kernel.write_text("def kernel():\n    return 1\n", encoding="utf-8")
            _git(repo, "add", "src/kernels/wy_fast.py")
            _git(repo, "commit", "-m", "opti(kernel): first change")
            sha1 = _git(repo, "rev-parse", "HEAD")

            kernel.write_text("def kernel():\n    return 2\n", encoding="utf-8")
            _git(repo, "add", "src/kernels/wy_fast.py")
            _git(repo, "commit", "-m", "opti(kernel): second change")
            sha2 = _git(repo, "rev-parse", "HEAD")

            context_path = repo / ".triton-agent" / "commit-perf-context.json"
            context_path.parent.mkdir(parents=True)
            _run_collect(repo, base, context_path)

            grouped = group_context_by_file(input_path=context_path, repo=repo)
            groups = cast(list[dict[str, Any]], grouped["file_groups"])
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0]["path"], "src/kernels/wy_fast.py")
            commits = cast(list[dict[str, Any]], groups[0]["commits"])
            self.assertEqual([c["sha"] for c in commits], [sha1, sha2])
            self.assertEqual(commits[1]["body"], "")
            self.assertIn("second change", commits[1]["message"])

    def test_skips_hard_filtered_commits(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            repo = _make_git_repo(Path(tmp))
            base = _git(repo, "rev-parse", "HEAD")

            (repo / "README.md").write_text("docs\n", encoding="utf-8")
            _git(repo, "add", "README.md")
            _git(repo, "commit", "-m", "docs: update readme")

            context_path = repo / ".triton-agent" / "commit-perf-context.json"
            context_path.parent.mkdir(parents=True)
            _run_collect(repo, base, context_path)

            grouped = group_context_by_file(input_path=context_path, repo=repo)
            self.assertEqual(grouped["file_group_count"], 0)


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


def _run_collect(repo: Path, base: str, output: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "collect_commit_context.py"),
            "--repo",
            repo.as_posix(),
            "--base",
            base,
            "--output",
            output.as_posix(),
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr)


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
