from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

JsonValue = Union[None, bool, int, float, str, List["JsonValue"], Dict[str, "JsonValue"]]

DEFAULT_INPUT = ".helix/git-operator-context.json"
DEFAULT_OUTPUT = ".helix/git-operator-file-groups.json"

CODE_EXTENSIONS = {
    ".py",
    ".pyi",
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".h",
    ".hpp",
    ".hh",
    ".inc",
    ".triton",
    ".ttir",
    ".mlir",
}


class GroupError(RuntimeError):
    """Raised when file grouping cannot be completed."""


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = Path(args.repo).expanduser().resolve() / input_path
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = Path(args.repo).expanduser().resolve() / output_path
        grouped = group_context_by_file(
            input_path=input_path,
            repo=Path(args.repo).expanduser().resolve(),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(grouped, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(output_path.as_posix())
        return 0
    except GroupError as exc:
        print(f"group_commit_context_by_file: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Group commit context JSON by changed operator source file.",
    )
    parser.add_argument("--repo", default=".", help="Git repository path (default: current directory).")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Commit context JSON path (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Grouped output JSON path (default: {DEFAULT_OUTPUT}).",
    )
    return parser


def group_context_by_file(*, input_path: Path, repo: Path) -> Dict[str, JsonValue]:
    if not input_path.is_file():
        raise GroupError(f"commit context file not found: {input_path}")
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    commits = cast(List[Dict[str, Any]], raw.get("commits", []))
    groups: Dict[str, List[Dict[str, Any]]] = {}

    for commit in commits:
        if commit.get("hard_skip"):
            continue
        sha = str(commit["sha"])
        subject = str(commit["subject"])
        body = str(commit.get("body", ""))
        message = str(commit.get("message", subject if not body else f"{subject}\n\n{body}"))
        changed_files = cast(List[Dict[str, Any]], commit.get("changed_files", []))
        file_contexts = {
            str(item.get("path")): item
            for item in cast(List[Dict[str, Any]], commit.get("file_context", []))
            if isinstance(item, dict) and isinstance(item.get("path"), str)  # pyright: ignore[reportUnnecessaryIsInstance]
        }
        for item in changed_files:
            path = _entry_path(item)
            if path is None or not _is_code_path(path):
                continue
            groups.setdefault(path, []).append(
                {
                    "sha": sha,
                    "short_sha": sha[:12],
                    "subject": subject,
                    "body": body,
                    "message": message,
                    "status": str(item.get("status", "")),
                    "is_revert_or_rollback": bool(commit.get("is_revert_or_rollback")),
                    "file_diff": _file_diff(repo=repo, sha=sha, path=path),
                    "file_context": file_contexts.get(path),
                }
            )

    file_groups: List[Dict[str, Any]] = []
    for path in sorted(groups):
        entries = groups[path]
        file_groups.append(
            {
                "path": path,
                "commit_count": len(entries),
                "commits": entries,
            }
        )

    return {
        "schema_version": 1,
        "source_context_path": input_path.as_posix(),
        "repo": str(raw.get("repo", repo.as_posix())),
        "base_revision": raw.get("base_revision"),
        "head": raw.get("head"),
        "commit_count": raw.get("commit_count", len(commits)),
        "hard_skipped_count": raw.get("hard_skipped_count", 0),
        "file_group_count": len(file_groups),
        "file_groups": file_groups,  # pyright: ignore[reportReturnType]
    }


def _entry_path(item: Dict[str, Any]) -> Optional[str]:
    path = item.get("path")
    if isinstance(path, str) and path:
        return path
    return None


def _is_code_path(path: str) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix in CODE_EXTENSIONS


def _file_diff(*, repo: Path, sha: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", sha, "--", path],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "git show failed"
        return f"<failed to load file diff: {detail}>"
    return result.stdout


if __name__ == "__main__":
    raise SystemExit(main())
