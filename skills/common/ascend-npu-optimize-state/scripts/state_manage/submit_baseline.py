from __future__ import annotations

import argparse
import json
from pathlib import Path

from baseline.check import check_baseline
from shared.cli import build_check_payload, build_workflow_failure_payload
from state_manage.workflow import mark_baseline_passed


def build_parser(*, prog_name: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog_name or Path(__file__).name)
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("submit-baseline")
    baseline.add_argument("--baseline-dir", required=True)
    return parser

def _workflow_failure_guideline(message: str) -> str:
    if "workflow state" in message:
        return (
            "The temporary optimize workflow state is invalid. Do not continue to round work. "
            "Ask the runner to restart the optimize session so the runner-managed workflow state "
            "can be rebuilt cleanly."
        )
    return (
        "Baseline validation passed, but workflow-state advancement failed. Restart the "
        "optimize session before continuing."
    )


def main(argv: list[str] | None = None, *, prog_name: str | None = None) -> int:
    parser = build_parser(prog_name=prog_name)
    args = parser.parse_args(argv)
    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    result = check_baseline(baseline_dir)
    state_path = baseline_dir.parent / ".triton-agent" / "state.json"
    if result.status == "pass" and state_path.exists():
        try:
            mark_baseline_passed(state_path)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(
                json.dumps(
                    build_workflow_failure_payload(
                        kind="baseline",
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
