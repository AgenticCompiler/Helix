from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import cast

WORKFLOW_SCHEMA_VERSION = 1

PHASE_BASELINE = "baseline"
PHASE_AWAITING_ROUND_START = "awaiting_round_start"
PHASE_ROUND_ACTIVE = "round_active"
_PHASES = {PHASE_BASELINE, PHASE_AWAITING_ROUND_START, PHASE_ROUND_ACTIVE}

ROUND_DIR_PATTERN = re.compile(r"opt-round-(\d+)$")

ROUND_STRATEGIES = (
    "exploration",
    "structural_change",
    "focused_tuning",
    "stabilization",
    "plateau_review",
)
# Order encodes increasing required evidence depth, shallowest first.
ANALYSIS_POLICIES = (
    "pattern_entry",
    "profile_required",
    "ir_required",
    "compiler_source_required",
)
UPDATED_BY_VALUES = ("start-round", "set-current-round-state")
_ANALYSIS_POLICY_ORDER = {
    name: index for index, name in enumerate(ANALYSIS_POLICIES)
}
_WARNING_WORTHY_STRATEGY_TRANSITIONS = {
    ("structural_change", "exploration"): (
        "Returning from structural_change to exploration is unusual; confirm the previous rewrite direction is no longer justified."
    ),
    ("focused_tuning", "exploration"): (
        "Returning from focused_tuning to exploration is unusual; confirm the round really needs a broader search again."
    ),
    ("plateau_review", "focused_tuning"): (
        "Leaving plateau_review for focused_tuning is unusual; confirm the plateau conclusion has been resolved."
    ),
    ("plateau_review", "structural_change"): (
        "Leaving plateau_review for structural_change is unusual; confirm a new structural hypothesis is now evidence-backed."
    ),
}
_UNUSUAL_STRATEGY_POLICY_COMBINATIONS = {
    ("exploration", "compiler_source_required"): (
        "The combination exploration + compiler_source_required is unusual; compiler-source depth is often too deep for an exploratory round."
    ),
    ("structural_change", "pattern_entry"): (
        "The combination structural_change + pattern_entry is unusual; structural rewrites often need deeper evidence than pattern_entry alone."
    ),
    ("plateau_review", "pattern_entry"): (
        "The combination plateau_review + pattern_entry is unusual; plateau review usually follows deeper evidence."
    ),
}


def load_state(state_path: Path) -> dict[str, object]:
    try:
        raw_payload = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed workflow state JSON: {exc}") from exc
    if not isinstance(raw_payload, dict):
        raise ValueError("workflow state must be a JSON object")
    payload = cast(dict[str, object], raw_payload)
    _validate_state(payload)
    return payload


def bootstrap_state(
    state_path: Path,
    *,
    run_id: str,
    source_operator: str,
    baseline_reused: bool,
) -> None:
    payload: dict[str, object] = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "run_id": run_id,
        "phase": PHASE_AWAITING_ROUND_START if baseline_reused else PHASE_BASELINE,
        "source_operator": source_operator,
        "current_round": None,
        "baseline": {
            "status": "passed" if baseline_reused else "pending",
            "submitted_at": None,
        },
        "rounds": {},
    }
    _atomic_write_json(state_path, payload)


def mark_baseline_passed(state_path: Path) -> None:
    payload = load_state(state_path)
    baseline = _require_baseline_dict(payload)
    baseline["status"] = "passed"
    baseline["submitted_at"] = _utc_now()
    payload["phase"] = PHASE_AWAITING_ROUND_START
    payload["current_round"] = None
    _atomic_write_json(state_path, payload)


def start_round(
    state_path: Path,
    round_dir: str,
    *,
    round_strategy: str,
    analysis_policy: str,
    reason: str,
) -> dict[str, object]:
    normalized_round_strategy = _validate_round_strategy(round_strategy)
    normalized_analysis_policy = _validate_analysis_policy(analysis_policy)
    normalized_reason = _validate_reason(reason)
    warnings = _build_strategy_state_warnings(
        round_strategy=normalized_round_strategy,
        analysis_policy=normalized_analysis_policy,
    )

    payload = load_state(state_path)
    round_number = _parse_round_number(round_dir)
    round_key = str(round_number)
    rounds = _require_rounds_dict(payload)
    baseline = _require_baseline_dict(payload)

    if baseline.get("status") != "passed":
        raise ValueError("cannot start a round before baseline.status=passed")

    if (
        payload["phase"] == PHASE_ROUND_ACTIVE
        and payload["current_round"] == round_number
        and round_key in rounds
    ):
        active_round = rounds[round_key]
        if isinstance(active_round, dict) and cast(dict[str, object], active_round).get("status") == "active":
            active_round_dict = cast(dict[str, object], active_round)
            existing_state = _get_optional_strategy_state(active_round_dict)
            if existing_state is None:
                raise ValueError(
                    f"active round {round_dir} is missing strategy_state; use set-current-round-state to initialize it"
                )
            if (
                existing_state["round_strategy"] != normalized_round_strategy
                or existing_state["analysis_policy"] != normalized_analysis_policy
                or existing_state["reason"] != normalized_reason
            ):
                raise ValueError(
                    f"cannot reinitialize active round {round_dir} with different strategy state"
                )
            return _start_round_result(
                round_dir=round_dir,
                round_strategy=normalized_round_strategy,
                analysis_policy=normalized_analysis_policy,
                reason=normalized_reason,
                warnings=warnings,
            )

    if payload["phase"] != PHASE_AWAITING_ROUND_START:
        raise ValueError(
            f"cannot start {round_dir} while workflow phase is {payload['phase']}"
        )

    existing = rounds.get(round_key)
    if isinstance(existing, dict):
        existing_round = cast(dict[str, object], existing)
        if existing_round.get("status") == "passed":
            raise ValueError(f"cannot reopen completed round {round_dir}")

    payload["phase"] = PHASE_ROUND_ACTIVE
    payload["current_round"] = round_number
    rounds[round_key] = {
        "status": "active",
        "round_dir": round_dir,
        "started_at": _utc_now(),
        "ended_at": None,
        "strategy_state": _build_strategy_state(
            round_strategy=normalized_round_strategy,
            analysis_policy=normalized_analysis_policy,
            reason=normalized_reason,
            updated_by="start-round",
        ),
    }
    _atomic_write_json(state_path, payload)
    result_warnings = list(warnings)
    try:
        _append_state_update_block(
            _attempts_path_for_round(state_path, round_dir),
            source="start-round",
            round_strategy=normalized_round_strategy,
            analysis_policy=normalized_analysis_policy,
            reason=normalized_reason,
            warnings=result_warnings,
        )
    except OSError as exc:
        result_warnings.append(
            f"attempts.md history mirror could not be updated: {exc}. Workflow state remains authoritative."
        )
    return _start_round_result(
        round_dir=round_dir,
        round_strategy=normalized_round_strategy,
        analysis_policy=normalized_analysis_policy,
        reason=normalized_reason,
        warnings=result_warnings,
    )


def set_current_round_state(
    state_path: Path,
    *,
    round_strategy: str | None = None,
    analysis_policy: str | None = None,
    reason: str,
) -> dict[str, object]:
    normalized_reason = _validate_reason(reason)
    if round_strategy is None and analysis_policy is None:
        raise ValueError(
            "set-current-round-state requires --round-strategy and/or --analysis-policy"
        )

    payload = load_state(state_path)
    if payload.get("phase") != PHASE_ROUND_ACTIVE or not isinstance(payload.get("current_round"), int):
        raise ValueError("no optimize round is currently active")

    current_round = cast(int, payload["current_round"])
    round_key = str(current_round)
    rounds = _require_rounds_dict(payload)
    round_entry_obj = rounds.get(round_key)
    if not isinstance(round_entry_obj, dict):
        raise ValueError(f"missing workflow state entry for opt-round-{current_round}")
    round_entry = cast(dict[str, object], round_entry_obj)
    if round_entry.get("status") != "active":
        raise ValueError(f"cannot update non-active round opt-round-{current_round}")

    existing_state = _get_optional_strategy_state(round_entry)
    if existing_state is None and (round_strategy is None or analysis_policy is None):
        raise ValueError(
            "legacy active round is missing strategy_state; provide both --round-strategy and --analysis-policy"
        )

    previous_round_strategy = (
        existing_state["round_strategy"]
        if existing_state is not None
        else _validate_round_strategy(cast(str, round_strategy))
    )
    previous_analysis_policy = (
        existing_state["analysis_policy"]
        if existing_state is not None
        else _validate_analysis_policy(cast(str, analysis_policy))
    )
    next_round_strategy = (
        _validate_round_strategy(round_strategy)
        if round_strategy is not None
        else previous_round_strategy
    )
    next_analysis_policy = (
        _validate_analysis_policy(analysis_policy)
        if analysis_policy is not None
        else previous_analysis_policy
    )

    if existing_state is not None:
        if (
            previous_round_strategy == next_round_strategy
            and previous_analysis_policy == next_analysis_policy
        ):
            raise ValueError("state update would be a no-op")
        if _ANALYSIS_POLICY_ORDER[next_analysis_policy] < _ANALYSIS_POLICY_ORDER[previous_analysis_policy]:
            raise ValueError(
                "analysis_policy cannot become shallower within the same round"
            )

    warnings = _build_transition_warnings(
        previous_round_strategy=previous_round_strategy,
        next_round_strategy=next_round_strategy,
        next_analysis_policy=next_analysis_policy,
    )
    round_entry["strategy_state"] = _build_strategy_state(
        round_strategy=next_round_strategy,
        analysis_policy=next_analysis_policy,
        reason=normalized_reason,
        updated_by="set-current-round-state",
    )
    _atomic_write_json(state_path, payload)

    round_dir = round_entry.get("round_dir")
    if not isinstance(round_dir, str) or not round_dir:
        raise ValueError(f"workflow state round {round_key} is missing round_dir")
    result_warnings = list(warnings)
    try:
        _append_state_update_block(
            _attempts_path_for_round(state_path, round_dir),
            source="set-current-round-state",
            round_strategy=next_round_strategy,
            analysis_policy=next_analysis_policy,
            reason=normalized_reason,
            previous_round_strategy=previous_round_strategy if existing_state is not None else "<unset>",
            previous_analysis_policy=previous_analysis_policy if existing_state is not None else "<unset>",
            warnings=result_warnings,
        )
    except OSError as exc:
        result_warnings.append(
            f"attempts.md history mirror could not be updated: {exc}. Workflow state remains authoritative."
        )
    return _set_current_round_state_result(
        round_dir=round_dir,
        previous_round_strategy=previous_round_strategy if existing_state is not None else None,
        next_round_strategy=next_round_strategy,
        previous_analysis_policy=previous_analysis_policy if existing_state is not None else None,
        next_analysis_policy=next_analysis_policy,
        reason=normalized_reason,
        warnings=result_warnings,
    )


def complete_round(
    state_path: Path,
    round_dir: str,
    *,
    current_round_arg: int | None = None,
) -> None:
    payload = load_state(state_path)
    round_number = _parse_round_number(round_dir)
    round_key = str(round_number)
    rounds = _require_rounds_dict(payload)

    if payload["phase"] != PHASE_ROUND_ACTIVE:
        raise ValueError(
            f"cannot complete {round_dir} while workflow phase is {payload['phase']}"
        )
    if payload.get("current_round") != round_number:
        raise ValueError(
            f"workflow state current_round={payload.get('current_round')} does not match {round_dir}"
        )
    if current_round_arg is not None and current_round_arg != round_number:
        raise ValueError(
            f"--current-round={current_round_arg} does not match workflow state round {round_number}"
        )

    round_entry_obj = rounds.get(round_key)
    if not isinstance(round_entry_obj, dict):
        raise ValueError(f"missing workflow state entry for {round_dir}")
    round_entry = cast(dict[str, object], round_entry_obj)
    if round_entry.get("status") != "active":
        raise ValueError(f"cannot complete non-active round {round_dir}")

    round_entry["status"] = "passed"
    round_entry["ended_at"] = _utc_now()
    payload["phase"] = PHASE_AWAITING_ROUND_START
    payload["current_round"] = None
    _atomic_write_json(state_path, payload)


def render_phase_summary(state_path: Path) -> str:
    payload = load_state(state_path)
    baseline = _require_baseline_dict(payload)
    reused = baseline.get("status") == "passed" and baseline.get("submitted_at") is None
    baseline_source = "pending"
    if baseline.get("status") == "passed":
        baseline_source = "reused" if reused else "freshly passed in this run"
    lines = [f"Current phase: {payload['phase']}"]
    current_round = payload.get("current_round")
    lines.append(
        f"Current round: {current_round}" if current_round is not None else "Current round: none"
    )
    if payload["phase"] == PHASE_ROUND_ACTIVE and isinstance(current_round, int):
        round_entry_obj = _require_rounds_dict(payload).get(str(current_round))
        if isinstance(round_entry_obj, dict):
            strategy_state = _get_optional_strategy_state(cast(dict[str, object], round_entry_obj))
            if strategy_state is None:
                lines.append("Current round strategy state: missing")
            else:
                lines.append(
                    f"Current round strategy: {strategy_state['round_strategy']}"
                )
                lines.append(
                    f"Required analysis depth: {strategy_state['analysis_policy']}"
                )
                lines.append(
                    f"Current round reason: {strategy_state['reason']}"
                )
    lines.append(f"Baseline source: {baseline_source}")
    return "\n".join(lines)


def write_round_timings_archive(state_path: Path, archive_path: Path) -> bool:
    payload = load_state(state_path)
    rows: list[dict[str, object]] = []
    for round_key, round_state in sorted(
        _require_rounds_dict(payload).items(),
        key=lambda item: int(item[0]),
    ):
        if not isinstance(round_state, dict):
            raise ValueError(f"workflow state round {round_key} must be an object")
        round_state_dict = cast(dict[str, object], round_state)
        if round_state_dict.get("status") != "passed":
            continue
        started_at = round_state_dict.get("started_at")
        ended_at = round_state_dict.get("ended_at")
        if not isinstance(started_at, str) or not started_at:
            raise ValueError(f"completed round {round_key} is missing started_at")
        if not isinstance(ended_at, str) or not ended_at:
            raise ValueError(f"completed round {round_key} is missing ended_at")
        rows.append(
            {
                "round": int(round_key),
                "started_at": started_at,
                "ended_at": ended_at,
            }
        )
    if not rows:
        return False
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text(
        json.dumps(rows, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return True


def _validate_state(payload: dict[str, object]) -> None:
    schema_version = payload.get("schema_version")
    if schema_version != WORKFLOW_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported workflow state schema_version: {schema_version!r}"
        )

    phase = payload.get("phase")
    if phase not in _PHASES:
        raise ValueError(f"unknown workflow state phase: {phase!r}")

    baseline = _require_baseline_dict(payload)
    baseline_status = baseline.get("status")
    if baseline_status not in {"pending", "passed"}:
        raise ValueError(f"unknown baseline status: {baseline_status!r}")
    submitted_at = baseline.get("submitted_at")
    if submitted_at is not None and not isinstance(submitted_at, str):
        raise ValueError("baseline.submitted_at must be a string or null")

    rounds = _require_rounds_dict(payload)
    current_round = payload.get("current_round")
    if phase == PHASE_ROUND_ACTIVE:
        if not isinstance(current_round, int):
            raise ValueError("phase=round_active requires a non-null integer current_round")
        current_entry_obj = rounds.get(str(current_round))
        if not isinstance(current_entry_obj, dict):
            raise ValueError(f"phase=round_active requires rounds[{current_round}]")
        current_entry = cast(dict[str, object], current_entry_obj)
        if current_entry.get("status") != "active":
            raise ValueError(f"phase=round_active requires active state for round {current_round}")
        if not isinstance(current_entry.get("started_at"), str) or not current_entry.get("started_at"):
            raise ValueError(f"active round {current_round} must have started_at")
    else:
        if current_round is not None:
            raise ValueError(f"phase={phase} requires current_round=null")

    for round_key, round_state in rounds.items():
        if not isinstance(round_state, dict):
            raise ValueError(f"workflow state round {round_key} must be an object")
        round_state_dict = cast(dict[str, object], round_state)
        status = round_state_dict.get("status")
        if status not in {"active", "passed"}:
            raise ValueError(f"unknown round status for {round_key}: {status!r}")
        round_dir = round_state_dict.get("round_dir")
        if not isinstance(round_dir, str) or not round_dir:
            raise ValueError(f"workflow state round {round_key} is missing round_dir")
        if _parse_round_number(round_dir) != int(round_key):
            raise ValueError(
                f"workflow state round key {round_key} does not match round_dir {round_dir}"
            )
        started_at = round_state_dict.get("started_at")
        if not isinstance(started_at, str) or not started_at:
            raise ValueError(f"workflow state round {round_key} is missing started_at")
        ended_at = round_state_dict.get("ended_at")
        if ended_at is not None and not isinstance(ended_at, str):
            raise ValueError(f"workflow state round {round_key} ended_at must be a string or null")
        if status == "passed" and ended_at is None:
            raise ValueError(f"completed round {round_key} is missing ended_at")
        strategy_state = round_state_dict.get("strategy_state")
        if strategy_state is not None:
            _validate_strategy_state(strategy_state, round_key=round_key)


def _validate_strategy_state(strategy_state: object, *, round_key: str) -> None:
    if not isinstance(strategy_state, dict):
        raise ValueError(f"workflow state round {round_key} strategy_state must be an object")
    strategy_state_dict = cast(dict[str, object], strategy_state)
    round_strategy = strategy_state_dict.get("round_strategy")
    if round_strategy not in ROUND_STRATEGIES:
        raise ValueError(
            f"workflow state round {round_key} has unknown round_strategy: {round_strategy!r}"
        )
    analysis_policy = strategy_state_dict.get("analysis_policy")
    if analysis_policy not in ANALYSIS_POLICIES:
        raise ValueError(
            f"workflow state round {round_key} has unknown analysis_policy: {analysis_policy!r}"
        )
    reason = strategy_state_dict.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"workflow state round {round_key} strategy_state.reason must be a non-empty string")
    updated_at = strategy_state_dict.get("updated_at")
    if not isinstance(updated_at, str) or not updated_at:
        raise ValueError(f"workflow state round {round_key} strategy_state.updated_at must be a non-empty string")
    updated_by = strategy_state_dict.get("updated_by")
    if updated_by not in UPDATED_BY_VALUES:
        raise ValueError(
            f"workflow state round {round_key} strategy_state.updated_by must be one of {UPDATED_BY_VALUES}"
        )


def _require_baseline_dict(payload: dict[str, object]) -> dict[str, object]:
    baseline = payload.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("workflow state baseline must be an object")
    return cast(dict[str, object], baseline)


def _require_rounds_dict(payload: dict[str, object]) -> dict[str, object]:
    rounds = payload.get("rounds")
    if not isinstance(rounds, dict):
        raise ValueError("workflow state rounds must be an object")
    return cast(dict[str, object], rounds)


def _get_optional_strategy_state(round_entry: dict[str, object]) -> dict[str, str] | None:
    strategy_state = round_entry.get("strategy_state")
    if strategy_state is None:
        return None
    if not isinstance(strategy_state, dict):
        raise ValueError("workflow round strategy_state must be an object")
    strategy_state_dict = cast(dict[str, object], strategy_state)
    round_strategy = strategy_state_dict.get("round_strategy")
    analysis_policy = strategy_state_dict.get("analysis_policy")
    reason = strategy_state_dict.get("reason")
    if not isinstance(round_strategy, str) or round_strategy not in ROUND_STRATEGIES:
        raise ValueError(f"workflow round strategy_state.round_strategy is invalid: {round_strategy!r}")
    if not isinstance(analysis_policy, str) or analysis_policy not in ANALYSIS_POLICIES:
        raise ValueError(f"workflow round strategy_state.analysis_policy is invalid: {analysis_policy!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("workflow round strategy_state.reason must be a non-empty string")
    return {
        "round_strategy": round_strategy,
        "analysis_policy": analysis_policy,
        "reason": reason.strip(),
    }


def _validate_round_strategy(round_strategy: str) -> str:
    normalized = round_strategy.strip()
    if normalized not in ROUND_STRATEGIES:
        raise ValueError(f"unknown round_strategy: {round_strategy!r}")
    return normalized


def _validate_analysis_policy(analysis_policy: str) -> str:
    normalized = analysis_policy.strip()
    if normalized not in ANALYSIS_POLICIES:
        raise ValueError(f"unknown analysis_policy: {analysis_policy!r}")
    return normalized


def _validate_reason(reason: str) -> str:
    normalized = reason.strip()
    if not normalized:
        raise ValueError("reason is required")
    return normalized


def _build_strategy_state(
    *,
    round_strategy: str,
    analysis_policy: str,
    reason: str,
    updated_by: str,
) -> dict[str, object]:
    return {
        "round_strategy": round_strategy,
        "analysis_policy": analysis_policy,
        "reason": reason,
        "updated_at": _utc_now(),
        "updated_by": updated_by,
    }


def _build_strategy_state_warnings(
    *,
    round_strategy: str,
    analysis_policy: str,
) -> list[str]:
    warning = _UNUSUAL_STRATEGY_POLICY_COMBINATIONS.get(
        (round_strategy, analysis_policy)
    )
    return [warning] if warning is not None else []


def _build_transition_warnings(
    *,
    previous_round_strategy: str,
    next_round_strategy: str,
    next_analysis_policy: str,
) -> list[str]:
    warnings: list[str] = []
    transition_warning = _WARNING_WORTHY_STRATEGY_TRANSITIONS.get(
        (previous_round_strategy, next_round_strategy)
    )
    if transition_warning is not None:
        warnings.append(transition_warning)
    warnings.extend(
        _build_strategy_state_warnings(
            round_strategy=next_round_strategy,
            analysis_policy=next_analysis_policy,
        )
    )
    return warnings


def _append_state_update_block(
    attempts_path: Path,
    *,
    source: str,
    round_strategy: str,
    analysis_policy: str,
    reason: str,
    warnings: list[str],
    previous_round_strategy: str | None = None,
    previous_analysis_policy: str | None = None,
) -> None:
    lines = [f"## State Update {_utc_now()}", f"- Source: {source}"]
    if previous_round_strategy is None and previous_analysis_policy is None:
        lines.extend(
            [
                f"- Round strategy: {round_strategy}",
                f"- Analysis policy: {analysis_policy}",
            ]
        )
    else:
        lines.extend(
            [
                f"- Round strategy: {previous_round_strategy} -> {round_strategy}",
                f"- Analysis policy: {previous_analysis_policy} -> {analysis_policy}",
            ]
        )
    lines.append(f"- Reason: {reason}")
    for warning in warnings:
        lines.append(f"- Warning: {warning}")
    block = "\n".join(lines) + "\n"
    attempts_path.parent.mkdir(parents=True, exist_ok=True)
    existing = attempts_path.read_text(encoding="utf-8") if attempts_path.exists() else ""
    separator = ""
    if existing:
        separator = "\n" if existing.endswith("\n") else "\n\n"
    attempts_path.write_text(existing + separator + block, encoding="utf-8")


def _attempts_path_for_round(state_path: Path, round_dir: str) -> Path:
    return state_path.parent.parent / round_dir / "attempts.md"


def _start_round_result(
    *,
    round_dir: str,
    round_strategy: str,
    analysis_policy: str,
    reason: str,
    warnings: list[str],
) -> dict[str, object]:
    result: dict[str, object] = {
        "round": round_dir,
        "round_strategy": round_strategy,
        "analysis_policy": analysis_policy,
        "reason": reason,
    }
    if warnings:
        result["warnings"] = warnings
    return result


def _set_current_round_state_result(
    *,
    round_dir: str,
    previous_round_strategy: str | None,
    next_round_strategy: str,
    previous_analysis_policy: str | None,
    next_analysis_policy: str,
    reason: str,
    warnings: list[str],
) -> dict[str, object]:
    result: dict[str, object] = {
        "round": round_dir,
        "round_strategy": next_round_strategy,
        "analysis_policy": next_analysis_policy,
        "reason": reason,
    }
    if previous_round_strategy is not None:
        result["previous_round_strategy"] = previous_round_strategy
    if previous_analysis_policy is not None:
        result["previous_analysis_policy"] = previous_analysis_policy
    if warnings:
        result["warnings"] = warnings
    return result


def _parse_round_number(round_dir: str) -> int:
    match = ROUND_DIR_PATTERN.fullmatch(round_dir)
    if match is None:
        raise ValueError(f"invalid round directory name: {round_dir!r}")
    round_number = int(match.group(1), 10)
    if round_number < 1:
        raise ValueError(f"round number must be >= 1: {round_dir!r}")
    return round_number


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, separators=(",", ":"))
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        temp_path = Path(temp_name)
        if temp_path.exists():
            temp_path.unlink()
