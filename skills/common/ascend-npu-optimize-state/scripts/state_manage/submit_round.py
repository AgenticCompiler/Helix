from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from round.check import check_round
from shared.cli import build_check_payload, build_workflow_failure_payload
from shared.results import build_check_result
from state_manage.state_machine import complete_round

_MIN_SPEEDUP_ENV = "TRITON_AGENT_OPTIMIZE_MIN_SPEEDUP"


def build_parser(*, prog_name: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog_name or Path(__file__).name)
    subparsers = parser.add_subparsers(dest="command", required=True)

    round_parser = subparsers.add_parser("submit-round")
    round_parser.add_argument("--round-dir", required=True)
    round_parser.add_argument("--current-round", type=int, default=None)
    round_parser.add_argument("--final-round", type=int, default=None)
    round_parser.add_argument(
        "--optimize-target",
        choices=("kernel", "operator"),
        default=None,
    )
    return parser


def _resolve_min_speedup() -> float | None:
    raw_env = os.environ.get(_MIN_SPEEDUP_ENV)
    if raw_env is not None:
        text = raw_env.strip()
        if not text:
            raise ValueError(f"{_MIN_SPEEDUP_ENV} is set but empty")
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"{_MIN_SPEEDUP_ENV} must be a float, got {raw_env!r}") from exc
        if value <= 0:
            raise ValueError(f"{_MIN_SPEEDUP_ENV} must be greater than 0")
        return value
    return None

def _workflow_failure_guideline(message: str) -> str:
    if (
        "workflow state is not available" in message
        or ".triton-agent/state.json" in message
    ):
        return (
            "Optimize workflow state is unavailable. Use the staged "
            "`ascend-npu-optimize-state` skill's `submit-baseline` subcommand to repair "
            "session state, then reopen this `opt-round-N/` with `start-round` before "
            "retrying `submit-round`."
        )
    if (
        "workflow phase is awaiting_round_start" in message
        or "current_round=None" in message
        or "missing workflow state entry" in message
    ):
        return (
            "This round has not been formally started yet. Use the staged "
            "`ascend-npu-optimize-state` skill's `start-round` subcommand for this "
            "`opt-round-N/` before running `submit-round`."
        )
    if "cannot complete non-active round" in message or "workflow state current_round=" in message:
        return (
            "The requested round is not the active workflow round. Finish the active round, or "
            "use `ascend-npu-optimize-state` `start-round` to open the intended round "
            "before submitting it."
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


def _missing_round_directory_payload(round_dir: Path) -> dict[str, object]:
    result = build_check_result(
        kind="round",
        status="fail",
        issues=(f"missing round directory: {round_dir.name}",),
        summary=(
            f"round check requires fixes: missing round directory: {round_dir.name}. "
            "Create or reopen the expected `opt-round-N/` directory before submitting the round."
        ),
    )
    return build_check_payload(result)


def main(argv: list[str] | None = None, *, prog_name: str | None = None) -> int:
    parser = build_parser(prog_name=prog_name)
    args = parser.parse_args(argv)
    round_dir = Path(args.round_dir).expanduser().resolve()
    if not round_dir.is_dir():
        print(json.dumps(_missing_round_directory_payload(round_dir), ensure_ascii=True))
        return 1
    try:
        min_speedup = _resolve_min_speedup()
    except ValueError as exc:
        print(
            json.dumps(
                build_workflow_failure_payload(
                    kind="round",
                    issue=str(exc),
                    guideline=(
                        "The optimize session speedup target is invalid. Stop this attempt and "
                        "restart the optimize session so the runner can inject a valid target, "
                        "or rerun `submit-round` only after the target configuration is repaired."
                    ),
                ),
                ensure_ascii=True,
            )
        )
        return 1

    result = check_round(
        round_dir,
        current_round=args.current_round,
        final_round=args.final_round,
        optimize_target=args.optimize_target,
        min_speedup=min_speedup,
    )
    state_path = round_dir.parent / ".triton-agent" / "state.json"
    if result.status == "pass":
        if not state_path.exists():
            print(
                json.dumps(
                    build_workflow_failure_payload(
                        kind="round",
                        issue="optimize workflow state is not available",
                        guideline=_workflow_failure_guideline("optimize workflow state is not available"),
                    ),
                    ensure_ascii=True,
                )
            )
            return 1
        try:
            complete_round(
                state_path,
                round_dir.name,
                current_round_arg=args.current_round,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(
                json.dumps(
                    build_workflow_failure_payload(
                        kind="round",
                        issue=str(exc),
                        guideline=_workflow_failure_guideline(str(exc)),
                    ),
                    ensure_ascii=True,
                )
            )
            return 1

    print(json.dumps(build_check_payload(result), ensure_ascii=True))
    if result.status == "pass":
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
