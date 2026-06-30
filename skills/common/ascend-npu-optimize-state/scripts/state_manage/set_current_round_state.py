from __future__ import annotations

import argparse
import json
from pathlib import Path

from state_manage.state_machine import (
    ANALYSIS_POLICIES,
    ROUND_STRATEGIES,
    set_current_round_state as set_current_round_state_in_workflow,
)


def build_parser(*, prog_name: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog_name or Path(__file__).name)
    subparsers = parser.add_subparsers(dest="command", required=True)

    update = subparsers.add_parser("set-current-round-state")
    update.add_argument("--round-strategy", choices=ROUND_STRATEGIES)
    update.add_argument("--analysis-policy", choices=ANALYSIS_POLICIES)
    update.add_argument("--reason", required=True)
    return parser


def _build_failure_payload(issue: str, guideline: str) -> dict[str, object]:
    return {
        "status": "fail",
        "issues": [issue],
        "guideline": guideline,
    }


def _workflow_failure_guideline(message: str) -> str:
    if (
        "workflow state is not available" in message
        or ".triton-agent/state.json" in message
    ):
        return (
            "Optimize workflow state is unavailable. Use the staged "
            "`ascend-npu-optimize-state` skill's `submit-baseline` subcommand to repair "
            "session state, then reopen the intended `opt-round-N/` with `start-round` "
            "before retrying `set-current-round-state`."
        )
    if "no optimize round is currently active" in message:
        return (
            "No optimize round is currently active. Start the next round first before "
            "changing round strategy state."
        )
    if "state update would be a no-op" in message:
        return (
            "This state update would be a no-op. Keep the current round state or change "
            "the strategy or analysis policy before retrying."
        )
    if "analysis_policy cannot become shallower" in message:
        return (
            "The requested analysis_policy would become shallower. Keep the current or a "
            "deeper analysis policy within the same round."
        )
    if "set-current-round-state requires" in message:
        return "Provide --round-strategy and/or --analysis-policy together with --reason."
    if "workflow state" in message:
        return (
            "The temporary optimize workflow state is invalid. Stop this attempt and restart "
            "the optimize session so the runner can rebuild the temporary workflow state."
        )
    return (
        "This set-current-round-state request could not be applied. Repair the optimize "
        "session and retry."
    )


def _find_state_path_from_cwd(cwd: Path) -> Path | None:
    for candidate_dir in (cwd, *cwd.parents):
        state_path = candidate_dir / ".triton-agent" / "state.json"
        if state_path.exists():
            return state_path
    return None


def main(argv: list[str] | None = None, *, prog_name: str | None = None) -> int:
    args = build_parser(prog_name=prog_name).parse_args(argv)
    try:
        state_path = _find_state_path_from_cwd(Path.cwd())
        if state_path is None:
            raise RuntimeError("optimize workflow state is not available")
        workflow_result = set_current_round_state_in_workflow(
            state_path,
            round_strategy=args.round_strategy,
            analysis_policy=args.analysis_policy,
            reason=args.reason,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(
            json.dumps(
                _build_failure_payload(str(exc), _workflow_failure_guideline(str(exc))),
                ensure_ascii=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "status": "pass",
                "round": workflow_result["round"],
                "guideline": (
                    f"Round strategy state for {workflow_result['round']} is now updated. "
                    "Use the new state as the active same-round contract."
                ),
                "round_strategy": workflow_result["round_strategy"],
                "analysis_policy": workflow_result["analysis_policy"],
                "reason": workflow_result["reason"],
                **(
                    {"previous_round_strategy": workflow_result["previous_round_strategy"]}
                    if "previous_round_strategy" in workflow_result
                    else {}
                ),
                **(
                    {"previous_analysis_policy": workflow_result["previous_analysis_policy"]}
                    if "previous_analysis_policy" in workflow_result
                    else {}
                ),
                **(
                    {"warnings": workflow_result["warnings"]}
                    if "warnings" in workflow_result
                    else {}
                ),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
