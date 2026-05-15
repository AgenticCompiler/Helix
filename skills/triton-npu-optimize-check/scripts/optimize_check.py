from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from kernel_continuity_check import KernelContinuityResult, analyze_triton_kernel_continuity
from optimize_check_contract import (
    BaselineArtifactsInspection,
    BaselineState,
    OptimizeCheckResult,
    RoundArtifactsInspection,
    RoundState,
    baseline_gate_issues,
    check_baseline,
    check_round,
    inspect_baseline_artifacts,
    inspect_round_artifacts,
    load_baseline_state,
    load_round_state,
)

__all__ = [
    "BaselineArtifactsInspection",
    "BaselineState",
    "KernelContinuityResult",
    "OptimizeCheckResult",
    "RoundArtifactsInspection",
    "RoundState",
    "analyze_triton_kernel_continuity",
    "baseline_gate_issues",
    "build_parser",
    "check_baseline",
    "check_round",
    "inspect_baseline_artifacts",
    "inspect_round_artifacts",
    "load_baseline_state",
    "load_round_state",
    "main",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=Path(__file__).name)
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("check-baseline")
    baseline.add_argument("--baseline-dir", required=True)

    round_parser = subparsers.add_parser("check-round")
    round_parser.add_argument("--round-dir", required=True)
    round_parser.add_argument("--min-rounds", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "check-baseline":
        result = check_baseline(Path(args.baseline_dir).expanduser().resolve())
    else:
        result = check_round(Path(args.round_dir).expanduser().resolve(), min_rounds=args.min_rounds)

    print(json.dumps(asdict(result), ensure_ascii=True))
    print(result.summary, file=sys.stderr)
    if result.decision == "pass":
        return 0
    if result.decision == "hard-fail":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
