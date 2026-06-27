#!/usr/bin/env python3
"""Pull-request helpers for workspace planning."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List, Literal, Optional

_SUBJECT_PR_PATTERNS = (
    re.compile(r"(?i)(?:merge pull request|merged pull request)\s+#(\d+)\b"),
    re.compile(
        r"(?i)(?:merge pull request|merged pull request|merge request)\s+!(\d+)\b"
    ),
    re.compile(r"!(\d+)\b"),
)
_RUN_SUMMARY_PR_ROW_RE = re.compile(
    r"^\|\s*Analyzed pull requests\s*\|\s*([^|]+)\|",
    re.IGNORECASE | re.MULTILINE,
)
_ANALYZED_PR_SECTION_RE = re.compile(
    r"^##\s+Analyzed Pull Requests\s*\n(.*?)(?=^##\s+|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
_INT_LIST_RE = re.compile(r"\d+")


def extract_pull_request_id_from_subject(subject: str) -> Optional[int]:
    """Extract a merge-request IID from GitHub (#N) or GitCode/GitLab (!N) subjects."""
    for pattern in _SUBJECT_PR_PATTERNS:
        match = pattern.search(subject)
        if match:
            return int(match.group(1))
    return None


def parse_pull_request_ids(values: Optional[List[str]]) -> set[int]:
    ids: set[int] = set()
    for raw in values or []:
        for part in raw.split(","):
            text = part.strip()
            if not text:
                continue
            if not text.isdigit():
                raise ValueError(f"invalid pull request id {text!r}; expected a positive integer")
            ids.add(int(text))
    return ids


def extract_pull_request_ids_from_knowledge(text: str) -> set[int]:
    """Read PR IIDs recorded in the knowledge base (Run Summary or dedicated section)."""
    ids: set[int] = set()
    summary_match = _RUN_SUMMARY_PR_ROW_RE.search(text)
    if summary_match:
        ids.update(int(item) for item in _INT_LIST_RE.findall(summary_match.group(1)))
    section_match = _ANALYZED_PR_SECTION_RE.search(text)
    if section_match:
        ids.update(int(item) for item in _INT_LIST_RE.findall(section_match.group(1)))
    return ids


def resolve_pull_request_filter(
    explicit_ids: set[int],
    knowledge_text: str,
) -> tuple[Optional[set[int]], Literal["cli", "knowledge-base", "none"]]:
    if explicit_ids:
        return explicit_ids, "cli"
    extracted = extract_pull_request_ids_from_knowledge(knowledge_text)
    if extracted:
        return extracted, "knowledge-base"
    return None, "none"


def normalize_commit_sha(sha: str) -> str:
    return sha.strip().lower()


def sha_matches(commit_sha: str, candidate: str) -> bool:
    left = normalize_commit_sha(commit_sha)
    right = normalize_commit_sha(candidate)
    return left == right or left.startswith(right) or right.startswith(left)


def lookup_commit_pull_request(
    commit_sha: str,
    commit_to_pr: dict[str, int],
) -> Optional[int]:
    normalized = normalize_commit_sha(commit_sha)
    if normalized in commit_to_pr:
        return commit_to_pr[normalized]
    for key, pr_id in commit_to_pr.items():
        if sha_matches(normalized, key):
            return pr_id
    return None


def build_commit_pull_request_map(
    repo_root: Path,
    base_revision: str,
    *,
    head_revision: str = "HEAD",
) -> dict[str, int]:
    """Map commit SHAs (full and short) to GitHub/GitCode merge-request IIDs (#N or !N)."""

    repo_root = Path(repo_root).expanduser().resolve()
    base_revision = str(base_revision).strip()
    if not base_revision:
        return {}

    commit_to_pr: dict[str, int] = {}
    rev_range = f"{base_revision}..{head_revision}"

    merge_lines = _git_lines(
        repo_root,
        "log",
        rev_range,
        "--merges",
        "--format=%H%x09%s",
    )
    for line in merge_lines:
        if "\t" not in line:
            continue
        merge_sha, subject = line.split("\t", 1)
        pr_id = extract_pull_request_id_from_subject(subject)
        if pr_id is None:
            continue
        _register_commit_pr(commit_to_pr, merge_sha, pr_id)
        pr_commits = _git_lines(
            repo_root,
            "log",
            f"{merge_sha}^1..{merge_sha}^2",
            "--format=%H",
        )
        for commit_sha in pr_commits:
            _register_commit_pr(commit_to_pr, commit_sha, pr_id)

    all_lines = _git_lines(
        repo_root,
        "log",
        rev_range,
        "--format=%H%x09%s",
    )
    for line in all_lines:
        if "\t" not in line:
            continue
        commit_sha, subject = line.split("\t", 1)
        pr_id = extract_pull_request_id_from_subject(subject)
        if pr_id is not None:
            _register_commit_pr(commit_to_pr, commit_sha, pr_id)

    return commit_to_pr


def filter_commit_shas_by_pull_requests(
    commit_shas: list[str],
    *,
    pull_request_filter: Optional[set[int]],
    commit_to_pr: dict[str, int],
) -> list[str]:
    if pull_request_filter is None:
        return list(commit_shas)
    kept: list[str] = []
    for sha in commit_shas:
        pr_id = lookup_commit_pull_request(sha, commit_to_pr)
        if pr_id is not None and pr_id in pull_request_filter:
            kept.append(sha)
    return kept


def _register_commit_pr(mapping: dict[str, int], commit_sha: str, pr_id: int) -> None:
    full = normalize_commit_sha(commit_sha)
    mapping[full] = pr_id
    if len(full) >= 12:
        mapping[full[:12]] = pr_id


def _git_lines(repo_root: Path, *args: str) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]
