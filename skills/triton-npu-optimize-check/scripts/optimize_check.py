from __future__ import annotations

import argparse
import json
from pathlib import Path
from kernel_continuity_check import KernelContinuityResult, analyze_triton_kernel_continuity
from optimize_check_contract import (
    BaselineArtifactsInspection,
    BaselineState,
    OptimizeCheckResult,
    RoundArtifactsInspection,
    RoundState,
    baseline_gate_issues,
    cleanup_dir_pt_files,
    check_baseline,
    check_round,
    expected_round_operator_name,
    expected_round_perf_name,
    inspect_baseline_artifacts,
    inspect_round_artifacts,
    load_baseline_state,
    load_round_state,
    ordinary_optimize_pt_cleanup_enabled,
    resolve_round_operator_file,
    resolve_round_perf_file,
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
    "cleanup_dir_pt_files",
    "build_parser",
    "check_baseline",
    "check_round",
    "expected_round_operator_name",
    "expected_round_perf_name",
    "inspect_baseline_artifacts",
    "inspect_round_artifacts",
    "load_baseline_state",
    "load_round_state",
    "main",
    "ordinary_optimize_pt_cleanup_enabled",
    "resolve_round_operator_file",
    "resolve_round_perf_file",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=Path(__file__).name)
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("check-baseline")
    baseline.add_argument("--baseline-dir", required=True)

    round_parser = subparsers.add_parser("check-round")
    round_parser.add_argument("--round-dir", required=True)
    round_parser.add_argument("--min-rounds", type=int, default=None)
    round_parser.add_argument(
        "--optimize-target",
        choices=("kernel", "operator"),
        default=None,
    )
    return parser


def _build_cli_payload(result: OptimizeCheckResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "ok": result.ok,
        "kind": result.kind,
        "decision": result.decision,
        "issues": list(result.issues),
    }
    if result.next_option is not None:
        payload["next_option"] = result.next_option
    payload["guideline"] = result.summary
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "check-baseline":
        result = check_baseline(Path(args.baseline_dir).expanduser().resolve())
    else:
        result = check_round(
            Path(args.round_dir).expanduser().resolve(),
            min_rounds=args.min_rounds,
            optimize_target=args.optimize_target,
        )

    print(json.dumps(_build_cli_payload(result), ensure_ascii=True))
    if result.decision == "pass":
        return 0
    if result.decision == "hard-fail":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
