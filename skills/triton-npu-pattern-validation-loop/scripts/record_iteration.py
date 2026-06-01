#!/usr/bin/env python3
"""Append events to pattern validation loop state."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Update pattern validation loop state.")
    parser.add_argument("--state", required=True, help="Loop state JSON path.")
    parser.add_argument(
        "--phase",
        required=True,
        choices=("skill-update", "scaffold", "optimize", "audit", "complete", "failed"),
    )
    parser.add_argument("--note", default="")
    parser.add_argument("--audit-report", default="")
    parser.add_argument("--increment-iteration", action="store_true")
    args = parser.parse_args(argv)

    state_path = Path(args.state).expanduser().resolve()
    raw: dict[str, Any] = json.loads(state_path.read_text(encoding="utf-8"))
    event: dict[str, Any] = {
        "at": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "iteration": raw.get("iteration", 1),
        "note": args.note,
    }
    if args.audit_report:
        audit_path = Path(args.audit_report).expanduser().resolve()
        if audit_path.is_file():
            event["audit"] = json.loads(audit_path.read_text(encoding="utf-8"))
    raw.setdefault("history", []).append(event)

    if args.phase == "complete":
        raw["status"] = "complete"
    elif args.phase == "failed":
        raw["status"] = "failed"
    elif args.increment_iteration:
        raw["iteration"] = int(raw.get("iteration", 1)) + 1

    state_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
