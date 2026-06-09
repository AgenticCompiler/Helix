from __future__ import annotations

import argparse
import json
from pathlib import Path

from optimize_submit_baseline_contract import (
    BaselineArtifactsInspection,
    BaselineState,
    OptimizeCheckResult,
    baseline_gate_issues,
    check_baseline,
    inspect_baseline_artifacts,
    load_baseline_state,
)

__all__ = [
    "BaselineArtifactsInspection",
    "BaselineState",
    "OptimizeCheckResult",
    "baseline_gate_issues",
    "build_parser",
    "check_baseline",
    "inspect_baseline_artifacts",
    "load_baseline_state",
    "main",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=Path(__file__).name)
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("check-baseline")
    baseline.add_argument("--baseline-dir", required=True)
    return parser


def _build_cli_payload(result: OptimizeCheckResult) -> dict[str, object]:
    return {
        "kind": result.kind,
        "status": result.status,
        "issues": list(result.issues),
        "guideline": result.summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = check_baseline(Path(args.baseline_dir).expanduser().resolve())

    print(json.dumps(_build_cli_payload(result), ensure_ascii=True))
    if result.status == "pass":
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
