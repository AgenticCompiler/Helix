from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Literal


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from triton_agent.optimize.baseline import baseline_gate_issues
from triton_agent.optimize.gate import evaluate_round_gate
from triton_agent.optimize.models import GateDecision, OptimizeCheckResult


def check_baseline(baseline_dir: Path) -> OptimizeCheckResult:
    workspace = baseline_dir.parent
    issues = baseline_gate_issues(workspace)
    if issues:
        return _build_result(
            kind="baseline",
            decision="revise-required",
            issues=issues,
        )

    return _build_result(kind="baseline", decision="pass", issues=())


def check_round(round_dir: Path) -> OptimizeCheckResult:
    gate_result = evaluate_round_gate(round_dir)
    if gate_result.decision in {GateDecision.PASS_CONTINUE, GateDecision.PASS_STOP}:
        return _build_result(kind="round", decision="pass", issues=())
    if gate_result.decision == GateDecision.HARD_FAIL:
        return _build_result(
            kind="round",
            decision="hard-fail",
            issues=gate_result.blocking_issues,
        )
    return _build_result(
        kind="round",
        decision="revise-required",
        issues=gate_result.blocking_issues,
    )


def _build_result(
    *,
    kind: Literal["baseline", "round"],
    decision: Literal["pass", "revise-required", "hard-fail"],
    issues: tuple[str, ...],
) -> OptimizeCheckResult:
    ok = decision == "pass"
    if ok:
        summary = f"{kind} check passed"
    else:
        summary = f"{kind} check requires fixes: {'; '.join(issues)}"
    return OptimizeCheckResult(
        ok=ok,
        kind=kind,
        decision=decision,
        issues=issues,
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=Path(__file__).name)
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("check-baseline")
    baseline.add_argument("--baseline-dir", required=True)

    round_parser = subparsers.add_parser("check-round")
    round_parser.add_argument("--round-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "check-baseline":
        result = check_baseline(Path(args.baseline_dir).expanduser().resolve())
    else:
        result = check_round(Path(args.round_dir).expanduser().resolve())

    print(json.dumps(asdict(result), ensure_ascii=True))
    print(result.summary, file=sys.stderr)
    if result.decision == "pass":
        return 0
    if result.decision == "hard-fail":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
