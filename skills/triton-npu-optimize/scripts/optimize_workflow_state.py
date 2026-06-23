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


def load_state(state_path: Path) -> dict[str, object]:
    try:
        raw_payload = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed workflow state JSON at {state_path}: {exc}") from exc
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


def start_round(state_path: Path, round_dir: str) -> None:
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
            return

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
    }
    _atomic_write_json(state_path, payload)


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
    lines.append(f"Baseline source: {baseline_source}")
    lines.append(f"Workflow state path: {state_path.as_posix()}")
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


def _parse_round_number(round_dir: str) -> int:
    match = ROUND_DIR_PATTERN.fullmatch(round_dir)
    if match is None:
        raise ValueError(f"invalid round directory name: {round_dir!r}")
    round_number = int(match.group(1), 10)
    if round_number < 1:
        raise ValueError(f"round number must be >= 1: {round_dir!r}")
    return round_number


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _atomic_write_json(state_path: Path, payload: dict[str, object]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=state_path.parent,
        prefix=f"{state_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=True, separators=(",", ":"))
        handle.write("\n")
    try:
        os.replace(temp_path, state_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
