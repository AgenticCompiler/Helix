#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast


DEFAULT_REPO = "midwinter1993/triton-agent"
API_ROOT = "https://gitcode.com/api/v5/repos"
LOCAL_PROXY_MARKERS = ("127.0.0.1", "localhost")
PROXY_ENV_VARS = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
)


class GitCodeApiError(RuntimeError):
    """Raised when the GitCode PR API cannot satisfy the request."""


JsonDict = dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=Path(__file__).name,
        description="Create, list, and inspect GitCode pull requests through the official API.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    _add_repo_argument(create)
    create.add_argument("--title")
    create.add_argument("--body")
    create.add_argument("--head")
    create.add_argument("--base", default="main")
    create.add_argument("--fill", action="store_true")
    create.add_argument("--draft", action="store_true")
    create.add_argument("--prune-source-branch", action="store_true")
    create.add_argument("--json", action="store_true")

    list_parser = subparsers.add_parser("list")
    _add_repo_argument(list_parser)
    list_parser.add_argument("--state", default="open")
    list_parser.add_argument("--head")
    list_parser.add_argument("--base")
    list_parser.add_argument("--limit", type=int, default=20)
    list_parser.add_argument("--page", type=int, default=1)
    list_parser.add_argument("--sort", default="updated")
    list_parser.add_argument("--direction", default="desc")
    list_parser.add_argument("--json", action="store_true")

    view = subparsers.add_parser("view")
    _add_repo_argument(view)
    view.add_argument("number", type=int)
    view.add_argument("--comments", action="store_true")
    view.add_argument("--json", action="store_true")
    view.add_argument("--time-format", choices=("absolute", "relative"), default="absolute")

    return parser


def _add_repo_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-R",
        "--repo",
        default=DEFAULT_REPO,
        help=f"Target repository in owner/repo format. Defaults to {DEFAULT_REPO}.",
    )


def require_token() -> str:
    token = os.environ.get("GC_TOKEN", "").strip()
    if not token:
        raise GitCodeApiError("GC_TOKEN is required for GitCode PR API requests.")
    return token


def parse_repo(repo: str) -> tuple[str, str]:
    owner, sep, name = repo.partition("/")
    if not sep or not owner or not name:
        raise GitCodeApiError(f"Repository must use owner/repo format, got: {repo}")
    return owner, name


def detect_current_branch() -> str:
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True,
        text=True,
        check=False,
    )
    branch = result.stdout.strip()
    if result.returncode != 0 or not branch:
        raise GitCodeApiError(
            "Could not determine the current Git branch. Rerun with explicit --head."
        )
    return branch


def read_last_commit_message() -> tuple[str, str]:
    result = subprocess.run(
        ["git", "log", "-1", "--pretty=format:%s%x1f%b"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise GitCodeApiError("Could not read the latest Git commit. Do not use --fill here.")
    subject, _, body = result.stdout.partition("\x1f")
    return subject.strip(), body.strip()


def build_url(owner: str, repo: str, *parts: str, query: dict[str, object | None] | None = None) -> str:
    base = f"{API_ROOT}/{owner}/{repo}/pulls"
    if parts:
        base = "/".join([base, *parts])
    if not query:
        return base
    filtered = {key: value for key, value in query.items() if value is not None}
    if not filtered:
        return base
    return base + "?" + urllib.parse.urlencode(filtered)


def _uses_local_proxy() -> bool:
    for key in PROXY_ENV_VARS:
        value = os.environ.get(key, "")
        if any(marker in value for marker in LOCAL_PROXY_MARKERS):
            return True
    return False


def _is_connection_refused(exc: urllib.error.URLError) -> bool:
    reason = exc.reason
    return isinstance(reason, OSError) and getattr(reason, "errno", None) == 61


def request_json(
    token: str,
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, object] | None = None,
    disable_proxy: bool = False,
) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    opener = (
        urllib.request.build_opener(urllib.request.ProxyHandler({}))
        if disable_proxy
        else urllib.request.build_opener()
    )
    try:
        with opener.open(request) as response:
            body = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise GitCodeApiError(f"GitCode API {method} {url} failed: {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        if not disable_proxy and _uses_local_proxy() and _is_connection_refused(exc):
            return request_json(token, url, method=method, payload=payload, disable_proxy=True)
        raise GitCodeApiError(f"GitCode API {method} {url} failed: {exc}") from exc

    if not body:
        return None
    return json.loads(body)


def format_timestamp(raw: object, time_format: str) -> str:
    if not isinstance(raw, str) or not raw:
        return "-"
    if time_format == "absolute":
        return raw
    try:
        normalized = raw.replace("Z", "+00:00")
        instant = datetime.fromisoformat(normalized)
        now = datetime.now(timezone.utc)
        delta = now - instant.astimezone(timezone.utc)
    except ValueError:
        return raw
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


def author_name(record: dict[str, Any]) -> str:
    for key in ("author", "user"):
        candidate = record.get(key)
        if isinstance(candidate, dict):
            candidate_dict = cast(JsonDict, candidate)
            for name_key in ("name", "login", "username"):
                value = candidate_dict.get(name_key)
                if isinstance(value, str) and value:
                    return value
    return "-"


def render_pull(pull: dict[str, Any], *, time_format: str = "absolute") -> str:
    lines = [
        f"PR #{pull.get('number', '-')}: {pull.get('title', '')}",
        f"State: {pull.get('state', '-')}",
        f"Source: {pull.get('source_branch', '-')}",
        f"Target: {pull.get('target_branch', '-')}",
        f"Author: {author_name(pull)}",
        f"Created: {format_timestamp(pull.get('created_at'), time_format)}",
        f"Updated: {format_timestamp(pull.get('updated_at'), time_format)}",
    ]
    url = pull.get("html_url") or pull.get("web_url") or pull.get("url")
    if isinstance(url, str) and url:
        lines.append(f"URL: {url}")
    body = pull.get("body")
    if isinstance(body, str) and body.strip():
        lines.extend(["", "Body:", body.strip()])
    return "\n".join(lines)


def render_pull_list(pulls: list[dict[str, Any]]) -> str:
    if not pulls:
        return "No pull requests matched."
    lines: list[str] = []
    for pull in pulls:
        lines.append(
            f"#{pull.get('number', '-')}: [{pull.get('state', '-')}] "
            f"{pull.get('title', '')} "
            f"({pull.get('source_branch', '-')} -> {pull.get('target_branch', '-')})"
        )
    return "\n".join(lines)


def render_comments(comments: list[dict[str, Any]], *, time_format: str) -> str:
    if not comments:
        return "Comments: none"
    lines = ["Comments:"]
    for comment in comments:
        lines.append(
            f"- {author_name(comment)} @ "
            f"{format_timestamp(comment.get('created_at'), time_format)}"
        )
        body = comment.get("body")
        if isinstance(body, str) and body.strip():
            lines.append(body.strip())
    return "\n".join(lines)


def command_create(args: argparse.Namespace) -> int:
    token = require_token()
    owner, repo = parse_repo(args.repo)
    title = args.title
    body = args.body
    if args.fill:
        commit_title, commit_body = read_last_commit_message()
        if not title:
            title = commit_title
        if not body:
            body = commit_body
    if not title:
        raise GitCodeApiError("PR title is required. Provide --title or use --fill.")
    head = args.head or detect_current_branch()
    payload: dict[str, object] = {
        "title": title,
        "head": head,
        "base": args.base,
    }
    if body:
        payload["body"] = body
    raw_pull = request_json(token, build_url(owner, repo), method="POST", payload=payload)
    if not isinstance(raw_pull, dict):
        raise GitCodeApiError("GitCode API returned an unexpected create response.")
    pull = cast(JsonDict, raw_pull)
    if args.draft:
        number = pull.get("number")
        if not isinstance(number, int):
            raise GitCodeApiError("Created PR response did not include a numeric PR number.")
        patched = request_json(
            token,
            build_url(owner, repo, str(number)),
            method="PATCH",
            payload={"draft": True},
        )
        if isinstance(patched, dict):
            pull = cast(JsonDict, patched)
    if args.prune_source_branch:
        number = pull.get("number")
        if not isinstance(number, int):
            raise GitCodeApiError("Created PR response did not include a numeric PR number.")
        patched = request_json(
            token,
            build_url(owner, repo, str(number)),
            method="PATCH",
            payload={"force_remove_source_branch": True},
        )
        if isinstance(patched, dict):
            pull = cast(JsonDict, patched)
    if args.json:
        print(json.dumps(pull, ensure_ascii=False, indent=2))
    else:
        print(render_pull(pull))
    return 0


def command_list(args: argparse.Namespace) -> int:
    token = require_token()
    owner, repo = parse_repo(args.repo)
    query = {
        "state": args.state,
        "base": args.base,
        "sort": args.sort,
        "direction": args.direction,
        "per_page": args.limit,
        "page": args.page,
    }
    raw_result = request_json(token, build_url(owner, repo, query=query))
    if not isinstance(raw_result, list):
        raise GitCodeApiError("GitCode API returned an unexpected list response.")
    raw_items = cast(list[object], raw_result)
    pulls: list[JsonDict] = [cast(JsonDict, item) for item in raw_items if isinstance(item, dict)]
    if args.head:
        pulls = [pull for pull in pulls if pull.get("source_branch") == args.head]
    if args.json:
        print(json.dumps(pulls, ensure_ascii=False, indent=2))
    else:
        print(render_pull_list(pulls))
    return 0


def command_view(args: argparse.Namespace) -> int:
    token = require_token()
    owner, repo = parse_repo(args.repo)
    raw_pull = request_json(token, build_url(owner, repo, str(args.number)))
    if not isinstance(raw_pull, dict):
        raise GitCodeApiError("GitCode API returned an unexpected view response.")
    pull = cast(JsonDict, raw_pull)
    comments: list[JsonDict] = []
    if args.comments:
        raw_result = request_json(token, build_url(owner, repo, str(args.number), "comments"))
        if not isinstance(raw_result, list):
            raise GitCodeApiError("GitCode API returned an unexpected comments response.")
        raw_comments = cast(list[object], raw_result)
        comments = [cast(JsonDict, item) for item in raw_comments if isinstance(item, dict)]
    if args.json:
        payload: dict[str, Any] = {"pull": pull}
        if args.comments:
            payload["comments"] = comments
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render_pull(pull, time_format=args.time_format))
        if args.comments:
            print()
            print(render_comments(comments, time_format=args.time_format))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            return command_create(args)
        if args.command == "list":
            return command_list(args)
        if args.command == "view":
            return command_view(args)
        parser.error(f"Unknown command: {args.command}")
    except GitCodeApiError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
