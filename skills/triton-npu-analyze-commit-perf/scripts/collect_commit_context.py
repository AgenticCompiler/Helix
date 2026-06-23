from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

JsonValue = Union[None, bool, int, float, str, List["JsonValue"], Dict[str, "JsonValue"]]

DEFAULT_BASE_REVISION = "origin/main"
DEFAULT_OUTPUT = ".triton-agent/commit-perf-context.json"
DEFAULT_MAX_CONTEXT_CHARS = 80_000
COMMIT_RECORD_END = "==COMMIT_END=="

MESSAGE_SKIP_PREFIXES = (
    "test:",
    "tests:",
    "ci:",
    "docs:",
    "doc:",
    "chore:",
    "style:",
    "format:",
    "typo",
)

NON_CODE_DIRS = {
    ".github",
    ".gitlab",
    "ci",
    "doc",
    "docs",
    "test",
    "tests",
}

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


class GitError(RuntimeError):
    """Raised when a Git command needed for context collection fails."""


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        repo = resolve_git_worktree(Path(args.repo))
        validate_base_revision(repo, str(args.base))
        pull_request_filter = None
        if args.pull_request:
            from knowledge_pull_requests import parse_pull_request_ids

            pull_request_filter = parse_pull_request_ids(list(args.pull_request))
        context = collect_context(
            repo=repo,
            base_revision=str(args.base),
            max_context_chars=int(args.max_context_chars),
            pull_request_filter=pull_request_filter,
        )
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = repo / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(context, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(output_path.as_posix())
        return 0
    except GitError as exc:
        print(f"collect_commit_context: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Git commit context for performance analysis.")
    parser.add_argument("--repo", default=".", help="Git repository path (default: current directory).")
    parser.add_argument("--base", default=DEFAULT_BASE_REVISION, help="Base revision for <base>..HEAD.")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Context JSON output path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help="Maximum post-commit file context characters per changed file.",
    )
    parser.add_argument(
        "--pull-request",
        "--pr",
        action="append",
        default=[],
        metavar="N",
        help=(
            "Limit collection to commits mapped to merge-request IIDs (!N). "
            "Repeat or use comma-separated ids."
        ),
    )
    return parser


def collect_context(
    *,
    repo: Path,
    base_revision: str,
    max_context_chars: int,
    pull_request_filter: Optional[Set[int]] = None,
) -> Dict[str, JsonValue]:
    head = run_git(["rev-parse", "HEAD"], cwd=repo).strip()
    commits = list_commits(repo=repo, base_revision=base_revision)
    if pull_request_filter is not None:
        from knowledge_pull_requests import (
            build_commit_pull_request_map,
            filter_commit_shas_by_pull_requests,
        )

        commit_to_pr = build_commit_pull_request_map(repo, base_revision)
        kept_shas = set(
            filter_commit_shas_by_pull_requests(
                [str(commit["sha"]) for commit in commits],
                pull_request_filter=pull_request_filter,
                commit_to_pr=commit_to_pr,
            )
        )
        commits = [commit for commit in commits if str(commit["sha"]) in kept_shas]
    commit_contexts: List[JsonValue] = []
    skipped_count = 0
    for commit in commits:
        sha = str(commit["sha"])
        subject = str(commit["subject"])
        body = str(commit.get("body", ""))
        message = format_commit_message(subject, body)
        changed_files = changed_files_for_commit(repo=repo, sha=sha)
        hard_skip_reason = hard_skip_reason_for_commit(subject, body, changed_files)
        if hard_skip_reason is not None:
            skipped_count += 1
        commit_contexts.append(
            {
                "sha": sha,
                "short_sha": sha[:12],
                "subject": subject,
                "body": body,
                "message": message,
                "changed_files": changed_files,
                "hard_skip": hard_skip_reason is not None,
                "hard_skip_reason": hard_skip_reason,
                "is_revert_or_rollback": is_revert_or_rollback(subject, body),
                "diff": show_commit_diff(repo=repo, sha=sha),
                "file_context": file_context_for_commit(
                    repo=repo,
                    sha=sha,
                    changed_files=changed_files,
                    max_context_chars=max_context_chars,
                ),
            }
        )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": repo.as_posix(),
        "base_revision": base_revision,
        "head": head,
        "analyzed_pull_requests": sorted(pull_request_filter) if pull_request_filter else None,
        "commit_count": len(commit_contexts),
        "hard_skipped_count": skipped_count,
        "commits": commit_contexts,
    }


def list_commits(*, repo: Path, base_revision: str) -> List[Dict[str, JsonValue]]:
    record_delimiter = COMMIT_RECORD_END
    output = run_git(
        [
            "log",
            f"{base_revision}..HEAD",
            "--reverse",
            f"--format=%H%x00%s%x00%b%x00{record_delimiter}",
        ],
        cwd=repo,
    )
    sentinel = f"\x00{COMMIT_RECORD_END}"
    commits: List[Dict[str, JsonValue]] = []
    for block in output.split(sentinel):
        block = block.strip()
        if not block:
            continue
        if "\x00" not in block:
            raise GitError(f"Unexpected git log record: {block[:80]!r}")
        parts = block.split("\x00", 2)
        sha = parts[0].strip()
        subject = parts[1].strip() if len(parts) > 1 else ""
        raw_body = parts[2] if len(parts) > 2 else ""
        subject, body = normalize_commit_message(subject, raw_body)
        commits.append({"sha": sha, "subject": subject, "body": body})
    return commits


def normalize_commit_message(subject: str, body: str) -> tuple[str, str]:
    return subject.strip(), body.strip()


def format_commit_message(subject: str, body: str) -> str:
    if body:
        return f"{subject}\n\n{body}"
    return subject


def changed_files_for_commit(*, repo: Path, sha: str) -> List[JsonValue]:
    output = run_git(["diff-tree", "--no-commit-id", "--name-status", "-r", "-M", sha], cwd=repo)
    files: List[JsonValue] = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0]
        if status.startswith("R") and len(parts) >= 3:
            files.append({"status": status, "old_path": parts[1], "path": parts[2]})
            continue
        files.append({"status": status, "path": parts[1]})
    return files


def show_commit_diff(*, repo: Path, sha: str) -> str:
    return run_git(
        [
            "show",
            "--format=fuller",
            "--stat",
            "--patch",
            "--find-renames",
            "--find-copies",
            sha,
        ],
        cwd=repo,
    )


def file_context_for_commit(
    *,
    repo: Path,
    sha: str,
    changed_files: List[JsonValue],
    max_context_chars: int,
) -> List[JsonValue]:
    contexts: List[JsonValue] = []
    for item in changed_files:
        if not isinstance(item, dict):
            continue
        path_value = item.get("path")
        status_value = item.get("status")
        if not isinstance(path_value, str) or not isinstance(status_value, str):
            continue
        if status_value.startswith("D"):
            contexts.append({"path": path_value, "status": status_value, "content": None, "truncated": False})
            continue
        content = git_show_file(repo=repo, sha=sha, path=path_value)
        if content is None:
            contexts.append({"path": path_value, "status": status_value, "content": None, "truncated": False})
            continue
        truncated = len(content) > max_context_chars
        if truncated:
            content = content[:max_context_chars]
        contexts.append(
            {
                "path": path_value,
                "status": status_value,
                "content": content,
                "truncated": truncated,
            }
        )
    return contexts


def git_show_file(*, repo: Path, sha: str, path: str) -> Optional[str]:
    result = run_git_allow_failure(["show", f"{sha}:{path}"], cwd=repo)
    if result.returncode != 0:
        return None
    return result.stdout


def hard_skip_reason_for_commit(
    subject: str,
    body: str,
    changed_files: List[JsonValue],
) -> Optional[str]:
    normalized_subject = subject.strip().lower()
    if normalized_subject.startswith(MESSAGE_SKIP_PREFIXES):
        return "message prefix marks this as non-performance work"
    normalized_body = body.strip().lower()
    if normalized_body.startswith(MESSAGE_SKIP_PREFIXES):
        return "message body prefix marks this as non-performance work"
    paths = paths_from_changed_files(changed_files)
    if not paths:
        return "commit has no changed files"
    if all(path_is_non_code(path) for path in paths):
        return "only non-code, docs, CI, or test paths changed"
    return None


def paths_from_changed_files(changed_files: List[JsonValue]) -> List[str]:
    paths: List[str] = []
    for item in changed_files:
        if not isinstance(item, dict):
            continue
        path_value = item.get("path")
        if isinstance(path_value, str):
            paths.append(path_value)
    return paths


def path_is_non_code(path: str) -> bool:
    parts = Path(path).parts
    if any(part in NON_CODE_DIRS for part in parts):
        return True
    suffix = Path(path).suffix.lower()
    return suffix not in CODE_EXTENSIONS


def is_revert_or_rollback(subject: str, body: str) -> bool:
    normalized_message = format_commit_message(subject, body).lower()
    return "revert" in normalized_message or "rollback" in normalized_message


def resolve_git_worktree(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    cwd = candidate if candidate.is_dir() else candidate.parent
    if not cwd.exists():
        raise GitError(f"Input path does not exist: {candidate}")
    output = run_git(["rev-parse", "--show-toplevel"], cwd=cwd).strip()
    return Path(output).resolve()


def validate_base_revision(repo: Path, base_revision: str) -> None:
    if not base_revision.strip():
        raise GitError("--base must not be empty")
    run_git(["rev-parse", "--verify", f"{base_revision}^{{commit}}"], cwd=repo)


def run_git(args: List[str], *, cwd: Path) -> str:
    result = run_git_allow_failure(args, cwd=cwd)
    if result.returncode != 0:
        command = "git " + " ".join(args)
        detail = result.stderr.strip() or result.stdout.strip()
        raise GitError(f"{command} failed in {cwd}: {detail}")
    return result.stdout


def run_git_allow_failure(args: List[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
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
