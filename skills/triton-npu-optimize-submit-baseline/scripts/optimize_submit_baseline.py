from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

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


def _load_workflow_state_module():
    skills_root = Path(__file__).resolve().parents[2]
    shared_scripts_dir = skills_root / "triton-npu-optimize" / "scripts"
    module_path = shared_scripts_dir / "optimize_workflow_state.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing optimize workflow helper: {module_path}")
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


def _build_workflow_failure_payload(issue: str, guideline: str) -> dict[str, object]:
    return {
        "kind": "baseline",
        "status": "fail",
        "issues": [issue],
        "guideline": guideline,
    }


def _workflow_failure_guideline(message: str) -> str:
    if "workflow state" in message:
        return (
            "The temporary optimize workflow state is invalid. Do not continue to round work. "
            "Ask the runner to restart the optimize session so `.triton-agent/state.json` can "
            "be rebuilt cleanly."
        )
    return (
        "Baseline validation passed, but workflow-state advancement failed. Restart the "
        "optimize session before continuing."
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    result = check_baseline(baseline_dir)
    state_path = baseline_dir.parent / ".triton-agent" / "state.json"
    if result.status == "pass" and state_path.exists():
        try:
            _load_workflow_state_module().mark_baseline_passed(state_path)
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
