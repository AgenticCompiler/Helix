from __future__ import annotations

import argparse
import json
from pathlib import Path

from state_manage.workflow import start_round as start_round_in_workflow_state

_FALLBACK_HARD_RULES = (
    "Only one optimize round may be active at a time.",
    "Do not use a script to create multiple optimize rounds where each round only adjusts parameters in order to speed up the optimization process. This is cheating behavior and is strictly prohibited.",
    "Do not use agents or subagents to advance multiple rounds in parallel while the current round is still in flight.",
    "Do not treat the next round as a blind parameter sweep. If you need to tune parameters, prefer the `autotune` optimization pattern.",
    "Do not burn rounds on hand-tuned launch or tile sweeps unless existing evidence clearly justifies that direction.",
    "Before editing code, decide which operator, kernel path, or wrapper bottleneck should anchor the next round.",
    "Before editing code, decide whether existing evidence is already sufficient or whether profiling, IR, or compiler-source analysis is needed first.",
    "Keep the round goal narrow: one coherent hypothesis, one active round, one evidence-backed change direction.",
)


def _load_hard_rules() -> list[str]:
    skill_path = Path(__file__).resolve().parents[2] / "SKILL.md"
    try:
        lines = skill_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return list(_FALLBACK_HARD_RULES)

    rules: list[str] = []
    in_hard_rules = False
    for line in lines:
        stripped = line.strip()
        if not in_hard_rules:
            if stripped == "## Hard Rules":
                in_hard_rules = True
            continue
        if stripped.startswith("## "):
            break
        if stripped.startswith("- "):
            rules.append(stripped[2:])
    return rules or list(_FALLBACK_HARD_RULES)


def build_parser(*, prog_name: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog_name or Path(__file__).name)
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start-round")
    start.add_argument("--round-dir", required=True)
    return parser


def _build_failure_payload(issue: str, guideline: str) -> dict[str, object]:
    return {
        "status": "fail",
        "issues": [issue],
        "guideline": guideline,
        "hard_rules": _load_hard_rules(),
    }


def _workflow_failure_guideline(message: str) -> str:
    if (
        "workflow state is not available" in message
        or "--enable-agent-hook" in message
    ):
        return (
            "Optimize workflow state is unavailable. This script only works in optimize "
            "sessions started with --enable-agent-hook."
        )
    if "baseline.status=passed" in message:
        return (
            "Baseline has not been accepted yet. Use the staged "
            "`ascend-npu-optimize-state` skill's `submit-baseline` subcommand to repair and "
            "submit `baseline/` until it passes, then run `start-round` again."
        )
    if "cannot reopen completed round" in message:
        return (
            "This round is already completed. Move to the next incomplete `opt-round-N/` "
            "instead of reopening it."
        )
    if "workflow phase is round_active" in message:
        return (
            "Another round is already active. Finish that round before starting a different "
            "round."
        )
    if "workflow state" in message:
        return (
            "The temporary optimize workflow state is invalid. Stop this attempt and restart "
            "the optimize session so the runner can rebuild the temporary workflow state."
        )
    return "This start-round request could not be applied. Repair the optimize session and retry."


def main(argv: list[str] | None = None, *, prog_name: str | None = None) -> int:
    args = build_parser(prog_name=prog_name).parse_args(argv)
    round_dir = Path(args.round_dir).expanduser().resolve()
    try:
        state_path = round_dir.parent / ".triton-agent" / "state.json"
        if not state_path.exists():
            raise RuntimeError(
                "optimize workflow state is not available; start-round requires --enable-agent-hook"
            )
        start_round_in_workflow_state(state_path, round_dir.name)
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
                "round": round_dir.name,
                "guideline": (
                    f"Round {round_dir.name} is now active. Follow the hard rules below while "
                    "working on this round."
                ),
                "hard_rules": _load_hard_rules(),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
