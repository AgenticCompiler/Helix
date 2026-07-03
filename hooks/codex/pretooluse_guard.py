#!/usr/bin/env python3
"""
Codex PreToolUse hook wrapper for triton-agent optimize runs.

This wrapper adapts Codex hook stdin/stdout handling to the shared
backend-agnostic guard policy module.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_support_import() -> None:
    current_dir = Path(__file__).resolve().parent
    candidates = (
        current_dir.parent.parent / "src",
        current_dir,
    )
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate.is_dir() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


_bootstrap_support_import()

from hook_runtime.pretooluse_adapter import run_policy_file_wrapper  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return run_policy_file_wrapper(
        argv=argv,
        failure_prefix="triton-agent codex hook",
    )


if __name__ == "__main__":
    raise SystemExit(main())
