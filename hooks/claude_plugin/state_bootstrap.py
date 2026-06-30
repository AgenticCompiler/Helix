#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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

    source_operator = infer_source_operator(workspace)
    if source_operator is None:
        return BootstrapResult(
            "Created `.triton-agent/`, but could not infer the source operator for workflow-state bootstrap. "
            "Inspect the workspace and re-run the optimize agent after baseline artifacts are available."
        )

    phase = infer_phase_from_workspace(workspace)
    payload = build_minimal_state_payload(
        phase=phase,
        source_operator=source_operator,
        baseline_reused=phase == PHASE_AWAITING_ROUND_START,
    )
    state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return BootstrapResult(
        f"Initialized `.triton-agent/state.json` with recovered phase `{phase}`."
    )


def validate_existing_state(state_path: Path) -> BootstrapResult:
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return BootstrapResult(None)
    except json.JSONDecodeError as exc:
        return BootstrapResult(
            f"Existing `.triton-agent/state.json` is malformed: {exc}. "
            "Remove it and restart the optimize agent so the plugin can rebuild workflow state."
        )
    if not isinstance(payload, dict):
        return BootstrapResult(
            "Existing `.triton-agent/state.json` must be a JSON object. "
            "Remove it and restart the optimize agent so the plugin can rebuild workflow state."
        )
    try:
        _validate_state_payload(payload)
    except ValueError as exc:
        return BootstrapResult(
            f"Existing `.triton-agent/state.json` is invalid: {exc}. "
            "Remove it and restart the optimize agent so the plugin can rebuild workflow state."
        )
    return BootstrapResult(None)


def infer_phase_from_workspace(workspace: Path) -> str:
    baseline_state_path = workspace / "baseline" / "state.json"
    if baseline_state_path.is_file() and baseline_looks_established(baseline_state_path):
        return PHASE_AWAITING_ROUND_START
    return PHASE_BASELINE


def baseline_looks_established(state_path: Path) -> bool:
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    if not bool(payload.get("baseline_established")):
        return False
    correctness_status = payload.get("correctness_status")
    benchmark_status = payload.get("benchmark_status")
    if correctness_status is not None and correctness_status != "passed":
        return False
    if benchmark_status is not None and benchmark_status != "passed":
        return False
    return True


def infer_source_operator(workspace: Path) -> str | None:
    baseline_state_path = workspace / "baseline" / "state.json"
    if baseline_state_path.is_file():
        try:
            payload = json.loads(baseline_state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            source_operator = payload.get("source_operator")
            if isinstance(source_operator, str) and source_operator.strip():
                return source_operator.strip()

    candidates = [
        path.name
        for path in sorted(workspace.iterdir())
        if path.is_file()
        and path.suffix == ".py"
        and not path.name.startswith(("test_", "bench_", "opt_", "differential_test_"))
    ]
    if not candidates:
        return None
    return candidates[0]


def build_minimal_state_payload(
    *,
    phase: str,
    source_operator: str,
    baseline_reused: bool,
) -> dict[str, object]:
    return {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "run_id": f"plugin-{_utc_now().replace(':', '').replace('-', '')}",
        "phase": phase,
        "source_operator": source_operator,
        "current_round": None,
        "baseline": {
            "status": "passed" if baseline_reused else "pending",
            "submitted_at": None,
        },
        "rounds": {},
    }


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
    return agent_type == PLUGIN_AGENT_NAME


def resolve_workspace(payload: dict[str, object]) -> Path | None:
    cwd = payload.get("cwd")
    if isinstance(cwd, str) and cwd:
        return Path(cwd).expanduser().resolve()
    return None


def missing_state_denial_reason(workspace: Path) -> str | None:
    runtime_dir = workspace / ".triton-agent"
    state_path = runtime_dir / "state.json"
    if not runtime_dir.exists():
        return (
            "Built-in edit tool blocked by optimize workflow policy. "
            "The plugin-managed `.triton-agent/` runtime directory is missing. "
            "Restart the optimize agent so the plugin can rebuild workflow state."
        )
    if not state_path.exists():
        return (
            "Built-in edit tool blocked by optimize workflow policy. "
            "The plugin-managed `.triton-agent/state.json` file is missing. "
            "The plugin only auto-recovers baseline or awaiting_round_start state; inspect durable baseline artifacts and restart the optimize agent if recovery is needed."
        )
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return (
            "Built-in edit tool blocked by optimize workflow policy. "
            f"The plugin-managed `.triton-agent/state.json` file is malformed: {exc}. "
            "Remove it and restart the optimize agent so the plugin can rebuild workflow state."
        )
    if not isinstance(payload, dict):
        return (
            "Built-in edit tool blocked by optimize workflow policy. "
            "The plugin-managed `.triton-agent/state.json` file is invalid. "
            "Remove it and restart the optimize agent so the plugin can rebuild workflow state."
        )
    try:
        _validate_state_payload(payload)
    except ValueError as exc:
        return (
            "Built-in edit tool blocked by optimize workflow policy. "
            f"The plugin-managed `.triton-agent/state.json` file is invalid: {exc}. "
            "Remove it and restart the optimize agent so the plugin can rebuild workflow state."
        )
    return None


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
