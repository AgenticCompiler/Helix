from __future__ import annotations

import argparse
import json
from pathlib import Path

from kernel_continuity_check import KernelContinuityResult, analyze_kernel_continuity

from optimize_submit_round_contract import (
    BaselineArtifactsInspection,
    BaselineState,
    OptimizeCheckResult,
    RoundArtifactsInspection,
    RoundState,
    baseline_gate_issues,
    cleanup_dir_pt_files,
    check_round,
    expected_round_operator_name,
    expected_round_perf_name,
    is_completed_round_directory,
    inspect_baseline_artifacts,
    inspect_round_artifacts,
    iter_completed_round_directories,
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
    "analyze_kernel_continuity",
    "baseline_gate_issues",
    "cleanup_dir_pt_files",
    "build_parser",
    "check_round",
    "expected_round_operator_name",
    "expected_round_perf_name",
    "is_completed_round_directory",
    "inspect_baseline_artifacts",
    "inspect_round_artifacts",
    "iter_completed_round_directories",
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

    round_parser = subparsers.add_parser("check-round")
    round_parser.add_argument("--round-dir", required=True)
    round_parser.add_argument("--current-round", type=int, default=None)
    round_parser.add_argument("--final-round", type=int, default=None)
    round_parser.add_argument(
        "--optimize-target",
        choices=("kernel", "operator"),
        default=None,
    )
    return parser


def _build_cli_payload(result: OptimizeCheckResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": result.kind,
        "status": result.status,
        "issues": list(result.issues),
    }
    if result.next_option is not None:
        payload["next_option"] = result.next_option
    payload["guideline"] = result.summary
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = check_round(
        Path(args.round_dir).expanduser().resolve(),
        current_round=args.current_round,
        final_round=args.final_round,
        optimize_target=args.optimize_target,
    )

    print(json.dumps(_build_cli_payload(result), ensure_ascii=True))
    if result.status == "pass":
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
