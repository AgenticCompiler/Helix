from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TextIO

from triton_agent.skills.catalog import resolve_skill_source_dir

DEFAULT_OPERATOR_WORKSPACES_DIR = "operators"
DEFAULT_WORKSPACE_PLAN_NAME = "workspace-plan.json"
GIT_REPO_PLAN_SKILL_NAME = "ascend-npu-plan-git-operator-workspaces"

_OPERATOR_SOURCE_EXTENSIONS = (".py", ".triton", ".ttir", ".mlir")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_workspace_plan_prompt(
    *,
    repo_root: Path,
    base_revision: str,
    fork_revision: str,
    plan_path: Path,
    language: str = "triton",
) -> str:
    _ext_filter = " ".join(f'"*{ext}"' for ext in _OPERATOR_SOURCE_EXTENSIONS)
    return f"""Use the staged {GIT_REPO_PLAN_SKILL_NAME} skill to analyze Git commits and produce a workspace plan JSON.

Repository root:
  {repo_root.as_posix()}

Operator language:
  {language}

Base branch:
  {base_revision}

Fork point (pre-computed merge-base of {base_revision}..HEAD):
  {fork_revision}

Plan output:
  {plan_path.as_posix()}

The fork point has been pre-computed by the harness.  Use `{fork_revision}`
directly as the baseline revision in all diffs below — do NOT run
`git merge-base` yourself.

Changed source filter:
```bash
git diff --name-only {fork_revision}..HEAD -- {_ext_filter}
```

Follow the skill workflow and `references/output-contract.md`. Write only the
workspace plan JSON to `{plan_path.as_posix()}`; do not extract or write operator
source files because the CLI scaffold script handles that after this step.
"""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def operator_workspaces_created(output_dir: Path) -> bool:
    """Check whether the agent created at least one valid operator workspace."""
    if not output_dir.is_dir():
        return False
    for child in output_dir.iterdir():
        if child.is_dir() and any(
            f.name.startswith("opt_") and f.suffix == ".py"
            for f in child.glob("*.py")
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def detect_git_worktree(path: Path) -> tuple[Path, str] | None:
    """Try to detect a Git repository from *path*.

    Returns ``(repo_root, head_sha)`` on success, or ``None`` when *path* is
    not inside a Git work tree.
    """
    try:
        repo_root = _resolve_git_worktree(path)
    except ValueError:
        return None
    result = _run_git(["rev-parse", "HEAD"], cwd=repo_root)
    if result.returncode != 0:
        return None
    return repo_root, result.stdout.strip()


def compute_fork_point(
    *, repo_root: Path, base_branch: str
) -> str | None:
    """Compute the fork point where the current branch diverged from *base_branch*.

    Returns the merge-base commit SHA, or ``None`` on failure.
    """
    result = _run_git(["merge-base", base_branch, "HEAD"], cwd=repo_root)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def detect_default_base_branch(*, repo_root: Path) -> str:
    """Auto-detect the default base branch from the remote's HEAD ref.

    Returns a qualified ref like ``"origin/main"`` or ``"origin/master"``.
    Falls back to ``"origin/main"`` when detection fails.
    """
    result = _run_git(
        ["symbolic-ref", "refs/remotes/origin/HEAD"], cwd=repo_root
    )
    if result.returncode == 0:
        ref = result.stdout.strip()
        # refs/remotes/origin/main → origin/main
        prefix = "refs/remotes/"
        if ref.startswith(prefix):
            return ref[len(prefix):]
    # Fallback: try common names
    for candidate in ("origin/main", "origin/master"):
        check = _run_git(
            ["rev-parse", "--verify", f"{candidate}^{{commit}}"],
            cwd=repo_root,
        )
        if check.returncode == 0:
            return candidate
    return "origin/main"


def _resolve_git_worktree(path: Path) -> Path:
    """Resolve the root of the Git worktree containing *path*."""
    candidate = path.expanduser().resolve()
    cwd = candidate if candidate.is_dir() else candidate.parent
    if not cwd.exists():
        raise ValueError(f"Input path does not exist: {candidate}")
    result = _run_git(["rev-parse", "--show-toplevel"], cwd=cwd)
    if result.returncode != 0:
        detail = result.stderr.strip() or "not a Git work tree"
        raise ValueError(
            f"Input path is not inside a Git work tree: {candidate} ({detail})"
        )
    return Path(result.stdout.strip()).resolve()


# ---------------------------------------------------------------------------
# Scaffold runner
# ---------------------------------------------------------------------------


def scaffold_operator_workspaces(
    *,
    plan_path: Path,
    output_root: Path,
    base_revision: str = "origin/main",
    fork_revision: str | None = None,
    stream: TextIO | None = None,
) -> int:
    """Run the scaffold_operators.py script to create workspace files from a plan JSON.

    Returns 0 on success (at least one operator created), non-zero on failure.
    """
    scaffold_script = (
        resolve_skill_source_dir(GIT_REPO_PLAN_SKILL_NAME)
        / "scripts"
        / "scaffold_operators.py"
    )
    if not scaffold_script.is_file():
        print(
            f"[scaffold] scaffold script not found: {scaffold_script}",
            file=stream or sys.stderr,
        )
        return 1

    cmd = [
        sys.executable,
        scaffold_script.as_posix(),
        "--plan",
        plan_path.as_posix(),
        "--output",
        output_root.as_posix(),
        "--base",
        base_revision,
    ]
    if fork_revision is not None:
        cmd.extend(["--fork", fork_revision])

    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=300,
    )
    out = stream or sys.stderr
    if result.stdout.strip():
        print(result.stdout.strip(), file=out)
    if result.stderr.strip():
        print(result.stderr.strip(), file=out)
    return result.returncode


def _run_git(
    args: list[str], *, cwd: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
