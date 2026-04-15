from __future__ import annotations

from pathlib import Path

from triton_agent.optimize.baseline import inspect_baseline_artifacts, load_baseline_state
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

    baseline_inspection = inspect_baseline_artifacts(round_dir.parent)
    if baseline_inspection.issues:
        return GateResult(
            decision=GateDecision.REVISE_REQUIRED,
            blocking_issues=baseline_inspection.issues,
        )

    try:
        baseline_state = load_baseline_state(round_dir.parent)
    except ValueError as exc:
        return GateResult(
            decision=GateDecision.REVISE_REQUIRED,
            blocking_issues=(str(exc),),
        )

    if not baseline_state.baseline_established:
        return GateResult(
            decision=GateDecision.REVISE_REQUIRED,
            blocking_issues=("baseline/state.json marks baseline as not established",),
        )
    if baseline_state.correctness_status != "passed":
        issue = f"baseline correctness_status={baseline_state.correctness_status}"
        return GateResult(decision=GateDecision.REVISE_REQUIRED, blocking_issues=(issue,))
    if baseline_state.benchmark_status != "passed":
        issue = f"baseline benchmark_status={baseline_state.benchmark_status}"
        return GateResult(decision=GateDecision.REVISE_REQUIRED, blocking_issues=(issue,))

    if round_state.canonical_baseline != "baseline":
        issue = f"canonical_baseline={round_state.canonical_baseline}"
        return GateResult(decision=GateDecision.REVISE_REQUIRED, blocking_issues=(issue,))
    if round_state.comparison_target != "baseline/perf.txt":
        issue = f"comparison_target={round_state.comparison_target}"
        return GateResult(decision=GateDecision.REVISE_REQUIRED, blocking_issues=(issue,))
    if round_state.perf_summary_source != "compare-perf":
        issue = f"perf_summary_source={round_state.perf_summary_source}"
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
