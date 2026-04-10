from __future__ import annotations

from pathlib import Path

from triton_agent.optimize.models import GateDecision, GateResult
from triton_agent.optimize.round_contract import inspect_round_artifacts, load_round_state


def evaluate_round_gate(round_dir: Path, *, stop_after_round: bool = False) -> GateResult:
    inspection = inspect_round_artifacts(round_dir)
    if inspection.issues:
        return GateResult(
            decision=GateDecision.REVISE_METADATA,
            blocking_issues=inspection.issues,
        )

    round_state = load_round_state(round_dir)

    if round_state.correctness_status != "passed":
        issue = f"correctness_status={round_state.correctness_status}"
        return GateResult(decision=GateDecision.HARD_FAIL, blocking_issues=(issue,))

    if round_state.benchmark_status != "passed":
        issue = f"benchmark_status={round_state.benchmark_status}"
        return GateResult(decision=GateDecision.REVISE_REQUIRED, blocking_issues=(issue,))

    if not round_state.evidence_sources:
        return GateResult(
            decision=GateDecision.REVISE_REQUIRED,
            blocking_issues=("missing supporting evidence sources",),
        )

    next_recommendation = round_state.next_recommendation.strip().lower()
    if stop_after_round or next_recommendation in {"stop", "done", "final", "finalize"}:
        return GateResult(decision=GateDecision.PASS_STOP, blocking_issues=())

    return GateResult(decision=GateDecision.PASS_CONTINUE, blocking_issues=())
