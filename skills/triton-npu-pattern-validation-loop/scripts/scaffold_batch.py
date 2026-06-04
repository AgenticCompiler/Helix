#!/usr/bin/env python3
"""Optional batch materializer from agent-authored manifest JSON.

Primary workflow: the orchestrating agent creates workspaces, copies snapshots/tests,
and writes validation-meta.json. See references/workspace-scaffold-contract.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from reference_test import reference_test_destination_name


class ScaffoldError(RuntimeError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create optimize-batch workspaces with pre-optimization operator snapshots.",
    )
    parser.add_argument("--manifest", required=True, help="Path to validation manifest JSON.")
    parser.add_argument("--output", required=True, help="Batch root directory to create.")
    parser.add_argument(
        "--head",
        default="HEAD",
        help="Git HEAD revision for commit range (default: HEAD).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing files.",
    )
    args = parser.parse_args(argv)

    try:
        manifest_path = Path(args.manifest).expanduser().resolve()
        output_root = Path(args.output).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        repo = Path(str(manifest["repo"])).expanduser().resolve()
        base_revision = str(manifest.get("base_revision", "origin/main"))
        head_revision = str(args.head)
        if not repo.is_dir():
            raise ScaffoldError(f"repo not found: {repo}")
        run_git(["rev-parse", "--verify", f"{base_revision}^{{commit}}"], cwd=repo)
        run_git(["rev-parse", "--verify", head_revision], cwd=repo)

        if not args.dry_run:
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        operators = manifest.get("operators", [])
        if not isinstance(operators, list) or not operators:
            raise ScaffoldError("manifest must contain a non-empty operators list")

        for entry in operators:
            if not isinstance(entry, dict):
                raise ScaffoldError("each operators entry must be an object")
            scaffold_operator(
                repo=repo,
                output_root=output_root,
                base_revision=base_revision,
                head_revision=head_revision,
                entry=entry,
                dry_run=bool(args.dry_run),
            )

        print(output_root.as_posix())
        return 0
    except ScaffoldError as exc:
        print(f"scaffold_batch: {exc}", file=sys.stderr)
        return 2


def scaffold_operator(
    *,
    repo: Path,
    output_root: Path,
    base_revision: str,
    head_revision: str,
    entry: dict[str, Any],
    dry_run: bool,
) -> None:
    name = str(entry["name"])
    source_path = str(entry["source_path"])
    operator_filename = str(entry.get("operator_filename", Path(source_path).name))
    workspace = output_root / name

    snapshot = extract_pre_optimization_snapshot(
        repo=repo,
        source_path=source_path,
        base_revision=base_revision,
        head_revision=head_revision,
    )
    test_paths = resolve_test_paths(repo=repo, entry=entry, source_path=source_path)
    bench_paths = discover_bench_paths(repo=repo, operator_stem=Path(operator_filename).stem)

    if dry_run:
        print(f"[dry-run] workspace={workspace.name} operator={operator_filename} tests={len(test_paths)}")
        return

    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / operator_filename).write_text(snapshot, encoding="utf-8")

    copied_tests: list[str] = []
    for test_path in test_paths:
        destination_name = reference_test_destination_name(test_path.name)
        destination = workspace / destination_name
        destination.write_text(test_path.read_text(encoding="utf-8"), encoding="utf-8")
        copied_tests.append(destination_name)

    copied_benches: list[str] = []
    for bench_path in bench_paths:
        destination = workspace / bench_path.name
        destination.write_text(bench_path.read_text(encoding="utf-8"), encoding="utf-8")
        copied_benches.append(destination.name)

    meta = {
        "workspace": name,
        "source_path": source_path,
        "operator_filename": operator_filename,
        "base_revision": base_revision,
        "head_revision": head_revision,
        "expected_patterns": entry.get("expected_patterns", []),
        "notes": entry.get("notes", ""),
        "copied_tests": copied_tests,
        "copied_benches": copied_benches,
    }
    from batch_evaluation import upsert_workspace_entry

    upsert_workspace_entry(output_root, name, meta)


def extract_pre_optimization_snapshot(
    *,
    repo: Path,
    source_path: str,
    base_revision: str,
    head_revision: str,
) -> str:
    range_spec = f"{base_revision}..{head_revision}"
    log = run_git(
        ["log", range_spec, "--reverse", "--format=%H", "--", source_path],
        cwd=repo,
    )
    commits = [line.strip() for line in log.splitlines() if line.strip()]
    if not commits:
        base_blob = run_git_allow_failure(["show", f"{base_revision}:{source_path}"], cwd=repo)
        if base_blob.returncode == 0:
            return base_blob.stdout
        raise ScaffoldError(
            f"could not resolve pre-optimization snapshot for {source_path} in {range_spec}",
        )

    first_commit = commits[0]
    parent_blob = run_git_allow_failure(["show", f"{first_commit}^:{source_path}"], cwd=repo)
    if parent_blob.returncode == 0:
        return parent_blob.stdout

    first_blob = run_git_allow_failure(["show", f"{first_commit}:{source_path}"], cwd=repo)
    if first_blob.returncode == 0:
        return first_blob.stdout

    raise ScaffoldError(
        f"could not resolve pre-optimization snapshot for {source_path} in {range_spec}",
    )


def resolve_test_paths(*, repo: Path, entry: dict[str, Any], source_path: str) -> list[Path]:
    explicit = entry.get("test_paths", [])
    resolved: list[Path] = []
    if isinstance(explicit, list) and explicit:
        for item in explicit:
            path = repo / str(item)
            if not path.is_file():
                raise ScaffoldError(f"test file not found: {path}")
            resolved.append(path)
        return resolved

    stem = Path(source_path).stem
    patterns = [
        f"test_{stem}.py",
        f"test_*{stem}*.py",
        f"differential_test_{stem}.py",
        f"differential_test_*{stem}*.py",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(repo.glob(f"**/{pattern}"))
    candidates = sorted({path.resolve() for path in candidates if path.is_file()})
    if not candidates:
        raise ScaffoldError(
            f"no test file found for {source_path}; add test_paths to the manifest entry",
        )
    return candidates[:3]


def discover_bench_paths(*, repo: Path, operator_stem: str) -> list[Path]:
    candidates = sorted(
        {
            path.resolve()
            for path in repo.glob(f"**/bench_*{operator_stem}*.py")
            if path.is_file()
        }
    )
    return candidates[:2]


def run_git(args: list[str], *, cwd: Path) -> str:
    result = run_git_allow_failure(args, cwd=cwd)
    if result.returncode != 0:
        command = "git " + " ".join(args)
        detail = result.stderr.strip() or result.stdout.strip()
        raise ScaffoldError(f"{command} failed in {cwd}: {detail}")
    return result.stdout


def run_git_allow_failure(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
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
