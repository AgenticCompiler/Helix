from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

from kernel_continuity_check import KernelContinuityResult, analyze_triton_kernel_continuity
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
    "analyze_triton_kernel_continuity",
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


def _resolve_workflow_state_scripts_dir():
    script_dir = Path(__file__).resolve().parent
    skill_dir = script_dir.parent
    skills_dir = skill_dir.parent

    for candidate in (
        skill_dir / ".." / "triton-npu-optimize" / "scripts",
        skills_dir.parent / "triton" / "triton-npu-optimize" / "scripts",
    ):
        candidate = candidate.resolve()
        if (candidate / "optimize_workflow_state.py").exists():
            return candidate
    raise FileNotFoundError(
        "optimize_workflow_state module not found. "
        "Ensure triton-npu-optimize is staged alongside this skill."
    )


def _load_workflow_state_module():
    shared_scripts_dir = _resolve_workflow_state_scripts_dir()
    shared_path = str(shared_scripts_dir)
    inserted = False
    if shared_path not in sys.path:
        sys.path.insert(0, shared_path)
        inserted = True
    try:
        return importlib.import_module("optimize_workflow_state")
    finally:
        if inserted:
            sys.path.remove(shared_path)


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


def _build_workflow_failure_payload(issue: str, guideline: str) -> dict[str, object]:
    return {
        "kind": "round",
        "status": "fail",
        "issues": [issue],
        "guideline": guideline,
    }


def _workflow_failure_guideline(message: str) -> str:
    if (
        "workflow phase is awaiting_round_start" in message
        or "current_round=None" in message
        or "missing workflow state entry" in message
    ):
        return (
            "This round has not been formally started yet. Use the staged "
            "`ascend-npu-optimize-start-round` skill for this `opt-round-N/` before running "
            "round submission."
        )
    if "cannot complete non-active round" in message or "workflow state current_round=" in message:
        return (
            "The requested round is not the active workflow round. Finish the active round, or "
            "use `ascend-npu-optimize-start-round` to open the intended round before "
            "submitting it."
        )
    if "workflow state" in message:
        return (
            "The temporary optimize workflow state is invalid. Stop this attempt and restart "
            "the optimize session so the runner-managed workflow state is rebuilt cleanly."
        )
    return (
        "Round validation passed, but workflow-state completion failed. Repair the optimize "
        "session before continuing."
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    round_dir = Path(args.round_dir).expanduser().resolve()

    result = check_round(
        round_dir,
        current_round=args.current_round,
        final_round=args.final_round,
        optimize_target=args.optimize_target,
    )
    state_path = round_dir.parent / ".triton-agent" / "state.json"
    if result.status == "pass" and state_path.exists():
        try:
            _load_workflow_state_module().complete_round(
                state_path,
                round_dir.name,
                current_round_arg=args.current_round,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(
                json.dumps(
                    _build_workflow_failure_payload(
                        str(exc),
                        _workflow_failure_guideline(str(exc)),
                    ),
                    ensure_ascii=True,
                )
            )
            return 1

    print(json.dumps(_build_cli_payload(result), ensure_ascii=True))
    if result.status == "pass":
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
