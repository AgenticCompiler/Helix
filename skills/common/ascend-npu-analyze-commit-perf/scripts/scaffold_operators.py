#!/usr/bin/env python3
"""Create operator workspace directories from workspace-plan.json using git show.

Each operator workspace receives only the launch function, its kernels, and
their transitive local function dependencies — not the entire source file.
"""

from __future__ import annotations

import ast
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scaffold operator workspace directories from a workspace plan JSON.",
    )
    parser.add_argument("--plan", required=True, help="Path to workspace-plan.json")
    parser.add_argument("--output", required=True, help="Output root for operator directories")
    parser.add_argument("--base", default="origin/main", help="Base branch (used for merge-base computation when --fork is not given). Default: origin/main.")
    parser.add_argument("--fork", default=None, help="Pre-computed fork point (merge-base commit). When given, --base is ignored for extraction.")
    args = parser.parse_args(argv)

    plan_path = Path(args.plan).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    base_branch = str(args.base)

    if not plan_path.is_file():
        print(f"scaffold_operators: plan not found: {plan_path}", file=sys.stderr)
        return 2

    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"scaffold_operators: invalid JSON in plan: {exc}", file=sys.stderr)
        return 2

    repo_root = Path(str(plan.get("repo", ""))).expanduser().resolve()
    plan_base = str(plan.get("base_revision", "")).strip()
    if plan_base:
        base_branch = plan_base

    if not repo_root.is_dir():
        print(f"scaffold_operators: repo not found: {repo_root}", file=sys.stderr)
        return 2

    # Determine fork point: use --fork if given, otherwise compute merge-base
    fork_revision: Optional[str] = None
    if args.fork is not None:
        fork_revision = str(args.fork)
        print(
            f"scaffold_operators: using pre-computed fork = {fork_revision[:12]}..."
        )
    else:
        merge_base_result = _run_git(
            ["merge-base", base_branch, "HEAD"], cwd=repo_root
        )
        if merge_base_result.returncode != 0:
            print(
                f"scaffold_operators: merge-base failed: {merge_base_result.stderr.strip() or 'unknown error'}",
                file=sys.stderr,
            )
            return 2
        fork_revision = merge_base_result.stdout.strip()
        print(
            f"scaffold_operators: merge-base of {base_branch}..HEAD = {fork_revision[:12]}..."
        )

    operators: List[Dict[str, Any]] = plan.get("operators", [])
    if not operators:
        print("scaffold_operators: no operators in plan", file=sys.stderr)
        return 2

    created = 0
    skipped = 0

    for entry in operators:
        launch_fn = str(entry.get("launch_function", ""))
        source_path = str(entry.get("source_path", ""))
        raw_kernels: List[str] = [str(k) for k in entry.get("kernels", [])]
        if not launch_fn or not source_path:
            print(
                f"scaffold_operators: skip entry missing launch_function or source_path: {entry}",
                file=sys.stderr,
            )
            skipped += 1
            continue

        # Extract full source for baseline (fork point) and optimized (HEAD)
        baseline_source = _extract_source(
            repo_root=repo_root,
            revision=fork_revision,
            source_path=source_path,
        )
        opt_source = _extract_source(
            repo_root=repo_root,
            revision="HEAD",
            source_path=source_path,
        )

        if baseline_source is None or opt_source is None:
            missing: List[str] = []
            if baseline_source is None:
                missing.append("baseline")
            if opt_source is None:
                missing.append("opt")
            print(
                f"scaffold_operators: skip {launch_fn}/ — failed to extract source: {', '.join(missing)}",
                file=sys.stderr,
            )
            skipped += 1
            continue

        # Resolve kernel names that actually exist in the source file
        keep_functions = [launch_fn] + [k for k in raw_kernels if k in baseline_source]

        try:
            baseline_filtered = _filter_source(baseline_source, keep_functions)
        except SyntaxError as exc:
            print(
                f"scaffold_operators: skip {launch_fn}/ — baseline source parse error ({source_path}): {exc}",
                file=sys.stderr,
            )
            skipped += 1
            continue
        try:
            opt_filtered = _filter_source(opt_source, keep_functions)
        except SyntaxError as exc:
            print(
                f"scaffold_operators: skip {launch_fn}/ — opt source parse error ({source_path}): {exc}",
                file=sys.stderr,
            )
            skipped += 1
            continue

        # Deterministic guard: skip if there is no effective change
        if baseline_filtered == opt_filtered:
            print(
                f"scaffold_operators: skip {launch_fn}/ — baseline and opt are identical",
                file=sys.stderr,
            )
            skipped += 1
            continue

        workspace_dir = output_root / launch_fn
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / f"{launch_fn}.py").write_text(baseline_filtered, encoding="utf-8")
        (workspace_dir / f"opt_{launch_fn}.py").write_text(opt_filtered, encoding="utf-8")
        print(f"scaffold_operators: created {launch_fn}/ ({source_path})")
        created += 1

    print(
        f"scaffold_operators: done — {created} created, {skipped} skipped "
        f"(output: {output_root.as_posix()})"
    )
    return 0 if created > 0 else 1


# ---------------------------------------------------------------------------
# Source extraction and filtering
# ---------------------------------------------------------------------------


def _extract_source(
    *,
    repo_root: Path,
    revision: str,
    source_path: str,
) -> Optional[str]:
    """Return the full source text at *revision*:*source_path*, or None on failure."""
    result = _run_git(
        ["show", f"{revision}:{source_path}"],
        cwd=repo_root,
    )
    if result.returncode != 0:
        print(
            f"scaffold_operators: git show {revision}:{source_path} failed: "
            f"{result.stderr.strip() or 'unknown error'}",
            file=sys.stderr,
        )
        return None
    return result.stdout


def _filter_source(source_code: str, keep_functions: List[str]) -> str:
    """Extract imports, *keep_functions*, and their transitive local dependencies.

    Starting from each name in *keep_functions*, the function walks the AST
    call graph inside *source_code* to discover any other top-level functions
    that are called locally.  The result includes every such dependency,
    together with all module-level ``import`` / ``from ... import`` statements.

    Returns the filtered source text.
    """
    parsed = ast.parse(source_code)

    # Index top-level function definitions by name
    func_nodes: Dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in ast.iter_child_nodes(parsed):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_nodes[node.name] = node

    # Transitive closure over local calls
    needed: Set[str] = set()
    stack = list(keep_functions)
    while stack:
        name = stack.pop()
        if name in needed or name not in func_nodes:
            continue
        needed.add(name)
        for child in ast.walk(func_nodes[name]):
            if isinstance(child, ast.Call):
                called = _direct_call_name(child)
                if called and called in func_nodes and called not in needed:
                    stack.append(called)

    # Collect line ranges: imports first, then needed function defs
    ranges: List[Tuple[int, int]] = []
    for node in ast.iter_child_nodes(parsed):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            ranges.append((node.lineno, _node_end_line(node)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in needed:
                ranges.append((node.lineno, _node_end_line(node)))

    ranges.sort()
    source_lines = source_code.splitlines()
    result: List[str] = []
    last_end = 0
    for start, end in ranges:
        if result and start > last_end + 1:
            result.append("")
        result.extend(source_lines[start - 1 : end])
        last_end = end

    return "\n".join(result).rstrip("\n") + "\n"


def _direct_call_name(call_node: ast.Call) -> Optional[str]:
    """Return the function name for a simple ``name(...)`` call, or ``None``.

    Attribute calls (``obj.method(...)`` or ``ns.func(...)``) are never local
    dependencies, so they return ``None``.
    """
    if isinstance(call_node.func, ast.Name):
        return call_node.func.id
    return None


def _node_end_line(node: ast.stmt) -> int:
    """Best-effort end line for *node*."""
    end = getattr(node, "end_lineno", None)
    if end is not None:
        return end
    return node.lineno


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _run_git(
    args: List[str], *, cwd: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
