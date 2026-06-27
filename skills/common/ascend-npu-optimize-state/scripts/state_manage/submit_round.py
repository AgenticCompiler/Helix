from __future__ import annotations

import argparse
import json
from pathlib import Path

from round.check import check_round
from shared.cli import build_check_payload, build_workflow_failure_payload
from state_manage.state_machine import complete_round


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

def _workflow_failure_guideline(message: str) -> str:
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


def main(argv: list[str] | None = None, *, prog_name: str | None = None) -> int:
    parser = build_parser(prog_name=prog_name)
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
