"""Stage-verdict self-check (gate) for the optimize round.

Runs INSIDE the worker agent's round, AFTER the agent writes
``opt-round-N/stage-verdict.json`` and BEFORE it triages/optimizes. The agent
invokes this script (``python scripts/check_stage_verdict.py opt-round-N``) to
verify its ``determined_stage`` does not skip an unresolved prerequisite. If it
does, the script prints FAIL + the dep-order-first actionable stage, and the
agent re-analyzes (per the SKILL stage-analysis procedure). If it passes, the
agent proceeds to triage for the determined stage.

Self-contained (no ``triton_agent`` import) so it can run as a skill script.
Loads the stage contract from ``references/stages.json`` and reconstructs
``addressed`` / ``exhausted`` from the per-round ``stage-addressed.json``
markers (same semantics as the CLI side).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

_CONTRACT_PATH = Path(__file__).resolve().parent.parent / "references" / "stages.json"
_EXHAUSTION_THRESHOLD = 5


def _load_contract() -> dict:
    return json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))


def _prereqs_map(deps: list) -> dict[str, list[str]]:
    prereqs: dict[str, list[str]] = {}
    for edge in deps:
        before, after = str(edge["before"]), str(edge["after"])
        prereqs.setdefault(after, []).append(before)
    return prereqs


def _read_verdict(round_dir: Path) -> dict | None:
    path = round_dir / "stage-verdict.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_markers(workdir: Path) -> tuple[set[str], set[str], dict[str, list[str]]]:
    """Return (addressed, exhausted, patterns_tried) from opt-round-*/stage-addressed.json."""
    entries: list[tuple[int, str, bool]] = []
    addressed: set[str] = set()
    patterns_tried: dict[str, list[str]] = {}
    for round_dir in workdir.glob("opt-round-*"):
        marker = round_dir / "stage-addressed.json"
        if not marker.is_file():
            continue
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stage = data.get("stage")
        if not isinstance(stage, str) or not stage:
            continue
        addressed.add(stage)
        # collect patterns_tried
        tried = data.get("patterns_tried")
        if isinstance(tried, list):
            patterns_tried.setdefault(stage, [])
            patterns_tried[stage].extend(str(p) for p in tried if isinstance(p, str))
        # progress default True (conservative; old markers without progress field)
        progress = bool(data.get("progress", True))
        # extract round number for ordering
        name = round_dir.name
        try:
            num = int(name.split("-")[-1])
        except ValueError:
            continue
        entries.append((num, stage, progress))
    entries.sort(key=lambda x: x[0])
    # exhausted: consecutive same-stage no-progress >= threshold (sticky)
    exhausted: set[str] = set()
    current: Optional[str] = None
    streak = 0
    for _, stage, progress in entries:
        if stage != current:
            current = stage
            streak = 0
        if progress:
            streak = 0
        else:
            streak += 1
            if streak >= _EXHAUSTION_THRESHOLD:
                exhausted.add(stage)
    return addressed, exhausted, patterns_tried


def _verdict_stages(verdict: dict) -> tuple[set[str], set[str]]:
    """Return (clean_stages, issues_stages) from the verdict."""
    clean: set[str] = set()
    issues: set[str] = set()
    for entry in verdict.get("verdicts", []):
        stage = str(entry.get("stage", ""))
        if not stage:
            continue
        v = str(entry.get("verdict", "")).strip().lower()
        if v == "clean":
            clean.add(stage)
        elif v == "issues":
            issues.add(stage)
    return clean, issues


def _first_actionable(
    stage_order: list[str],
    issues_stages: set[str],
    prereqs: dict[str, list[str]],
    resolved: set[str],
) -> str | None:
    """Dep-order first stage with issues whose prereqs are all resolved."""
    for stage in stage_order:
        if stage not in issues_stages:
            continue
        if all(p in resolved for p in prereqs.get(stage, [])):
            return stage
    return None


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        print("usage: check_stage_verdict.py <opt-round-N>", file=sys.stderr)
        return 2
    round_dir = Path(argv[0])
    if not round_dir.is_absolute():
        round_dir = Path.cwd() / round_dir
    workdir = Path.cwd()

    contract = _load_contract()
    stage_order = [str(s["id"]) for s in contract.get("stages", [])]
    prereqs = _prereqs_map(contract.get("dependencies", []))

    verdict = _read_verdict(round_dir)
    if verdict is None:
        print(
            "FAIL: stage-verdict.json not found or malformed. Write it first per "
            "the SKILL stage-analysis procedure, then re-run this check.",
            file=sys.stderr,
        )
        return 1

    determined = verdict.get("determined_stage")
    if not isinstance(determined, str) or not determined:
        print(
            "FAIL: stage-verdict.json has no determined_stage. Set it to the "
            "stage you intend to triage (dep-order first actionable).",
            file=sys.stderr,
        )
        return 1

    addressed, exhausted, patterns_tried = _read_markers(workdir)
    clean, issues_stages = _verdict_stages(verdict)
    resolved = addressed | clean | exhausted

    unmet = [p for p in prereqs.get(determined, []) if p not in resolved]
    correct = _first_actionable(stage_order, issues_stages, prereqs, resolved)

    if not unmet:
        # determined_stage's prereqs are resolved; non-skipping.
        tried = patterns_tried.get(determined, [])
        # find the determined stage's full patterns list from the contract
        stage_patterns = []
        for s in contract.get("stages", []):
            if str(s.get("id")) == determined:
                stage_patterns = [str(p) for p in s.get("patterns", [])]
                break
        remaining = [p for p in stage_patterns if p not in tried]
        if tried and remaining:
            print(
                f"PASS: determined_stage={determined} (prereqs resolved). "
                f"Tried [{', '.join(tried)}] — RE-VERIFY their signals are gone in current code. "
                f"Remaining [{', '.join(remaining)}] — check their Use When."
            )
        elif tried and not remaining:
            print(
                f"PASS: determined_stage={determined} (prereqs resolved). "
                f"All patterns tried [{', '.join(tried)}] — RE-VERIFY each signal is gone; if all gone, mark clean."
            )
        else:
            print(f"PASS: determined_stage={determined} (prereqs resolved).")
        if correct and correct != determined:
            print(
                f"NOTE: dep-order first actionable is {correct}, but "
                f"{determined} is also non-skipping (its prereqs are resolved). "
                "Proceeding with your pick is allowed; consider {correct} first."
            )
        return 0

    # Skip detected.
    suggestion = correct if correct else "(no actionable stage found; address a prereq first)"
    print(
        f"FAIL: determined_stage={determined} skips unmet prerequisite(s): "
        f"{', '.join(unmet)}. The dep-order first actionable stage is "
        f"{suggestion}. Re-analyze by stage and pick {suggestion} instead, "
        "then re-run this check.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
