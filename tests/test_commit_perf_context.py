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

from collect_commit_context import collect_context, resolve_git_worktree

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class CommitPerfContextTests(unittest.TestCase):
    def test_collect_context_lists_commits_in_chronological_order_and_marks_hard_skips(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            repo = _make_git_repo(Path(tmp))
            base = _git(repo, "rev-parse", "HEAD")

            (repo / "kernel.py").write_text(
                "def kernel(x):\n    # coalesced vector path\n    return x + 2\n",
                encoding="utf-8",
            )
            _git(repo, "add", "kernel.py")
            _git(
                repo,
                "commit",
                "-m",
                "optimize: use vector-friendly path",
                "-m",
                "Measured 1.3x on A5 after coalescing the inner loop.",
            )
            perf_sha = _git(repo, "rev-parse", "HEAD")

            (repo / "README.md").write_text("usage\n", encoding="utf-8")
            _git(repo, "add", "README.md")
            _git(repo, "commit", "-m", "docs: explain benchmark")
            docs_sha = _git(repo, "rev-parse", "HEAD")

            context = collect_context(repo=repo, base_revision=base, max_context_chars=2000)

        commits = cast(list[dict[str, Any]], context["commits"])
        self.assertEqual([commit["sha"] for commit in commits], [perf_sha, docs_sha])
        self.assertFalse(commits[0]["hard_skip"])
        self.assertTrue(commits[1]["hard_skip"])
        self.assertEqual(commits[1]["hard_skip_reason"], "message prefix marks this as non-performance work")
        self.assertEqual(commits[0]["body"], "Measured 1.3x on A5 after coalescing the inner loop.")
        self.assertIn("Measured 1.3x on A5", commits[0]["message"])
        self.assertIn("coalesced vector path", commits[0]["file_context"][0]["content"])
        self.assertEqual(context["hard_skipped_count"], 1)

    def test_collect_context_marks_non_code_only_commit_as_hard_skip(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            repo = _make_git_repo(Path(tmp))
            base = _git(repo, "rev-parse", "HEAD")

            (repo / "README.md").write_text("notes\n", encoding="utf-8")
            _git(repo, "add", "README.md")
            _git(repo, "commit", "-m", "update notes")

            context = collect_context(repo=repo, base_revision=base, max_context_chars=2000)

        commits = cast(list[dict[str, Any]], context["commits"])
        self.assertTrue(commits[0]["hard_skip"])
        self.assertEqual(commits[0]["hard_skip_reason"], "only non-code, docs, CI, or test paths changed")

    def test_collect_context_parses_multiline_gitcode_merge_body(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            repo = _make_git_repo(Path(tmp))
            base = _git(repo, "rev-parse", "HEAD")

            (repo / "kernel.py").write_text("def kernel():\n    return 1\n", encoding="utf-8")
            _git(repo, "add", "kernel.py")
            _git(
                repo,
                "commit",
                "-m",
                "!2 merge main into main",
                "-m",
                "feat(op): add mojo opset\n\nCreated-by: fengrui886\nSee merge request: TritonAscendTest/Q2TritonKernel!2",
            )
            merge_sha = _git(repo, "rev-parse", "HEAD")

            context = collect_context(repo=repo, base_revision=base, max_context_chars=2000)

        commits = cast(list[dict[str, Any]], context["commits"])
        self.assertEqual(len(commits), 1)
        self.assertEqual(commits[0]["sha"], merge_sha)
        self.assertEqual(commits[0]["subject"], "!2 merge main into main")
        self.assertIn("feat(op): add mojo opset", commits[0]["body"])
        self.assertIn("See merge request: TritonAscendTest/Q2TritonKernel!2", commits[0]["body"])
        self.assertIn("See merge request", commits[0]["message"])

    def test_resolve_git_worktree_accepts_file_path_inside_repo(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            repo = _make_git_repo(Path(tmp))

            self.assertEqual(resolve_git_worktree(repo / "kernel.py"), repo)


def _make_git_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "--template=")
    _git(repo, "config", "user.email", "tester@example.com")
    _git(repo, "config", "user.name", "Tester")
    (repo / "kernel.py").write_text("def kernel(x):\n    return x + 1\n", encoding="utf-8")
    _git(repo, "add", "kernel.py")
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
