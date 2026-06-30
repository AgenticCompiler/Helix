#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil


PLUGIN_AGENT_NAME = "triton-agent-optimize"
WORKFLOW_SCHEMA_VERSION = 1
PHASE_BASELINE = "baseline"
PHASE_AWAITING_ROUND_START = "awaiting_round_start"
PHASE_ROUND_ACTIVE = "round_active"
_VALID_PHASES = {PHASE_BASELINE, PHASE_AWAITING_ROUND_START, PHASE_ROUND_ACTIVE}


@dataclass(frozen=True)
class BootstrapResult:
    additional_context: str | None = None


def bootstrap_runtime_state(workspace: Path) -> BootstrapResult:
    runtime_dir = workspace / ".triton-agent"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    state_path = runtime_dir / "state.json"
    if state_path.exists():
        return validate_existing_state(state_path)

    return BootstrapResult(
        _workflow_repair_guidance(
            "Optimize runtime state is not initialized for this session."
        )
    )


def validate_existing_state(state_path: Path) -> BootstrapResult:
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return BootstrapResult(None)
    except json.JSONDecodeError as exc:
        return BootstrapResult(
            _workflow_repair_guidance(
                f"Existing optimize workflow state is malformed: {exc}."
            )
        )
    if not isinstance(payload, dict):
        return BootstrapResult(
            _workflow_repair_guidance(
                "Existing optimize workflow state must be a JSON object."
            )
        )
    try:
        _validate_state_payload(payload)
    except ValueError as exc:
        return BootstrapResult(
            _workflow_repair_guidance(
                f"Existing optimize workflow state is invalid: {exc}."
            )
        )
    return BootstrapResult(None)


def cleanup_runtime_tree(runtime_dir: Path) -> None:
    if runtime_dir.name != ".triton-agent":
        return
    if runtime_dir.is_symlink() or runtime_dir.is_file():
        runtime_dir.unlink()
        return
    if runtime_dir.is_dir():
        shutil.rmtree(runtime_dir)


def should_manage_payload(payload: dict[str, object]) -> bool:
    agent_type = payload.get("agent_type")
    if not isinstance(agent_type, str) or not agent_type:
        return False
    return agent_type == PLUGIN_AGENT_NAME or agent_type.endswith(f":{PLUGIN_AGENT_NAME}")


def resolve_workspace(payload: dict[str, object]) -> Path | None:
    cwd = payload.get("cwd")
    if isinstance(cwd, str) and cwd:
        return Path(cwd).expanduser().resolve()
    return None


def missing_state_denial_reason(workspace: Path) -> str | None:
    runtime_dir = workspace / ".triton-agent"
    state_path = runtime_dir / "state.json"
    if not runtime_dir.exists():
        return _edit_blocked_workflow_guidance(
            "Optimize workflow state is not initialized for this session."
        )
    if not state_path.exists():
        return _edit_blocked_workflow_guidance(
            "Optimize workflow state is missing for this session."
        )
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _edit_blocked_workflow_guidance(
            f"Optimize workflow state is malformed: {exc}."
        )
    if not isinstance(payload, dict):
        return _edit_blocked_workflow_guidance(
            "Optimize workflow state is invalid."
        )
    try:
        _validate_state_payload(payload)
    except ValueError as exc:
        return _edit_blocked_workflow_guidance(
            f"Optimize workflow state is invalid: {exc}."
        )
    return None


def _workflow_repair_guidance(problem: str) -> str:
    return (
        f"{problem} "
        "Use `ascend-npu-optimize-state` `submit-baseline` to repair session state, "
        "then use `start-round` to reopen the intended `opt-round-N/` before "
        "continuing round edits, same-round state updates, or round submission."
    )


def _edit_blocked_workflow_guidance(problem: str) -> str:
    return (
        "Built-in edit tool blocked by optimize workflow policy. "
        + _workflow_repair_guidance(problem)
    )


def _validate_state_payload(payload: dict[str, object]) -> None:
    if payload.get("schema_version") != WORKFLOW_SCHEMA_VERSION:
        raise ValueError("unsupported workflow state schema_version")
    phase = payload.get("phase")
    if phase not in _VALID_PHASES:
        raise ValueError("unknown workflow state phase")
    source_operator = payload.get("source_operator")
    if not isinstance(source_operator, str) or not source_operator.strip():
        raise ValueError("workflow state source_operator must be a non-empty string")
    baseline = payload.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("workflow state baseline must be an object")
    if baseline.get("status") not in {"pending", "passed"}:
        raise ValueError("workflow state baseline.status must be pending or passed")
    current_round = payload.get("current_round")
    if phase != PHASE_ROUND_ACTIVE and current_round is not None:
        raise ValueError("non-active workflow phases require current_round=null")
    rounds = payload.get("rounds")
    if not isinstance(rounds, dict):
        raise ValueError("workflow state rounds must be an object")

__all__ = [
    "BootstrapResult",
    "PLUGIN_AGENT_NAME",
    "bootstrap_runtime_state",
    "cleanup_runtime_tree",
    "missing_state_denial_reason",
    "resolve_workspace",
    "should_manage_payload",
    "validate_existing_state",
]
