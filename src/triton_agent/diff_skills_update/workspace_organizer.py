from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TextIO

from triton_agent.skill_catalog import resolve_skill_source_dir

DEFAULT_OPERATORS_DIR = "operators"
DEFAULT_PLAN_NAME = "workspace-plan.json"

_TRITON_EXTENSIONS = (".py", ".triton", ".ttir", ".mlir")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_organize_workspaces_prompt(
    *,
    repo_root: Path,
    base_revision: str,
    fork_revision: str,
    plan_path: Path,
) -> str:
    _ext_filter = " ".join(f'"*{ext}"' for ext in _TRITON_EXTENSIONS)
    return f"""Analyze Git commits and produce a **workspace plan JSON** that maps every
modified Triton kernel to its launch function and source file.

Repository root:
  {repo_root.as_posix()}

Base branch:
  {base_revision}

Fork point (pre-computed merge-base of {base_revision}..HEAD):
  {fork_revision}

Plan output:
  {plan_path.as_posix()}

The fork point has been pre-computed by the harness.  Use `{fork_revision}`
directly as the baseline revision in all diffs below — do NOT run
`git merge-base` yourself.

## What you must produce

Write a single JSON file at `{plan_path.as_posix()}` with this exact schema:

```json
{{
  "schema_version": 1,
  "repo": "{repo_root.as_posix()}",
  "base_revision": "{base_revision}",
  "operators": [
    {{
      "launch_function": "<host-side wrapper name>",
      "source_path": "<repo-relative path, e.g. python/triton/kernels/matmul.py>",
      "kernels": ["<kernel_fn_1>", "<kernel_fn_2>"],
      "notes": "<one-line rationale>"
    }}
  ]
}}
```

The **"launch_function"** is the host-side Python function that sets up grid /
arguments and calls the kernel (e.g. `matmul_kernel[grid](...)`). This is the
function users invoke — it is the workspace name.

The **"source_path"** is the path to the source `.py` file relative to the
repo root. It is the same path that `git show <rev>:<source_path>` accepts.

The **"kernels"** list names every `@triton.jit`-decorated kernel that this
launch function calls.

## Workflow

### Step 1: List changed Triton source files
```bash
git diff --name-only {fork_revision}..HEAD -- {_ext_filter}
```
This already excludes C/C++/CUDA backend files, docs, tests, CI configs.

### Step 2: For EACH changed file, inspect the diff AND the full source
For every file from step 1:

a) Read the diff to see **which functions changed**:
   ```bash
   git diff {fork_revision}..HEAD -- <source_path>
   ```
   The diff hunk headers (lines starting with `@@`) show the function context.
   Only consider a function **modified** when its BODY lines changed — ignore
   changes that only touch imports, comments, docstrings, or whitespace.

b) Read the full file at HEAD to understand the call structure:
   ```bash
   git show HEAD:<source_path>
   ```
   Do NOT print file content to stdout — read it silently.

c) Identify the kernel functions (decorated with `@triton.jit`) in the file.

d) For each kernel whose **body was modified** in the diff, find its
   **launch function**. A launch function is the host-side `def` that calls
   the kernel via `kernel_name[grid](...)`. Trace upward if the caller is a
   private helper — the launch function is the first public entry point.

e) **Critical — cross-check with the diff.** Before finalizing an operator
   entry, verify that the diff actually touches the identified launch function
   or its kernels. If the diff only touched unrelated functions in the same
   file, do NOT include that launch function as an operator.

f) Produce one operators[] entry per launch function. If two modified kernels
   share the same launch function, list both kernels in one entry.

### Step 3: Deduplicate and validate
- If the same launch function appears from different source paths, keep one
  entry and note the conflict in "notes".
- Verify each source_path resolves to a real file at HEAD:
  ```bash
  git cat-file -e HEAD:<source_path> && echo ok || echo missing
  ```
- Skip any entry where the source_path does not exist at HEAD.
- **Skip any entry where no kernel body changed** — the scaffold script will
  also discard identical baseline/opt pairs as a safety net, but you should
  filter them out here first.

### Step 4: Write the JSON
Write `{plan_path.as_posix()}`. It must be valid JSON. The "operators" list
must be non-empty. If no operator had a meaningful change, write an empty
operators list (the workflow will report this clearly).

## What NOT to do

- Do NOT extract or write any `.py` files. A follow-up script handles that.
- Do NOT edit any source files in the repository.
- Do NOT include files with no Triton kernels (utility modules, __init__.py).
- Do NOT include entries for functions whose diff is import-only or
  formatting-only.
- Do NOT include every kernel in a changed file — only those whose body the
  diff actually touched.
- Do NOT guess. If the diff does not clearly show a function body change, skip.
- Do NOT print full file contents to stdout.
"""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def workspace_organizer_succeeded(output_dir: Path) -> bool:
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


def try_detect_git_repo(path: Path) -> tuple[Path, str] | None:
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


def compute_merge_base(
    *, repo_root: Path, base_branch: str
) -> str | None:
    """Compute the fork point where the current branch diverged from *base_branch*.

    Returns the merge-base commit SHA, or ``None`` on failure.
    """
    result = _run_git(["merge-base", base_branch, "HEAD"], cwd=repo_root)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def detect_default_base(*, repo_root: Path) -> str:
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


def run_scaffold_operators(
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
        resolve_skill_source_dir("ascend-npu-analyze-commit-perf")
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
