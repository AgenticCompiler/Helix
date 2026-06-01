#!/usr/bin/env python3
"""Initialize pattern validation loop state JSON."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create pattern-validation loop state file.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--synthesis", default="PERF_PATTERN_SYNTHESIS.md")
    parser.add_argument("--batch-dir", default="pattern-validation-batch")
    parser.add_argument(
        "--skills-dir",
        default="pattern-validation-skills",
        help="Persistent loop skills workdir under repo (default: pattern-validation-skills).",
    )
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--min-rounds", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument(
        "--state",
        default=".triton-agent/pattern-validation-loop-state.json",
        help="State file path relative to repo unless absolute.",
    )
    args = parser.parse_args(argv)

    repo = Path(args.repo).expanduser().resolve()
    state_path = Path(args.state).expanduser()
    if not state_path.is_absolute():
        state_path = repo / state_path
    state_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 1,
        "status": "running",
        "iteration": 1,
        "max_iterations": int(args.max_iterations),
        "repo": repo.as_posix(),
        "base_revision": str(args.base),
        "batch_dir": str(args.batch_dir),
        "skills_dir": str(args.skills_dir),
        "synthesis_path": str(args.synthesis),
        "min_rounds": int(args.min_rounds),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "history": [],
    }
    state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(state_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
