from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from helix.models import AgentRequest, AgentResult


TRACE_PATH_ENV = "HELIX_OTEL_TRACE_PATH"
TRACE_RUN_ID_ENV = "HELIX_OTEL_RUN_ID"
TRACE_WORKSPACE_ROOT_ENV = "HELIX_WORKSPACE_ROOT"
_RUN_ID_COLLISION_COUNTS: Counter[str] = Counter()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_trace_run_id(prefix: str = "") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = f"{prefix}-{ts}" if prefix else ts
    _RUN_ID_COLLISION_COUNTS[base] += 1
    count = _RUN_ID_COLLISION_COUNTS[base]
    return base if count == 1 else f"{base}-{count}"


def append_trace_event(trace_path: Path | str | None, event: Mapping[str, Any]) -> None:
    if trace_path is None:
        return
    path = Path(trace_path)
    payload = dict(event)
    payload.setdefault("timestamp", utc_timestamp())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def trace_path_from_request(request: AgentRequest) -> Path | None:
    if request.extra_env is None:
        return None
    raw_path = request.extra_env.get(TRACE_PATH_ENV)
    if not raw_path:
        return None
    return Path(raw_path)


def tool_trace_path(run_dir: Path) -> Path:
    return run_dir / "tool-traces.jsonl"


def trace_summary_path(trace_path: Path) -> Path:
    name = trace_path.name
    if name.startswith("trace-") and name.endswith(".jsonl"):
        return trace_path.with_name(name[:-6] + ".summary.json")
    return trace_path.parent / "summary.json"


def build_trace_env(
    existing: dict[str, str] | None,
    *,
    trace_path: Path,
    run_id: str,
    workspace_root: Path,
) -> dict[str, str]:
    env = dict(existing or {})
    env[TRACE_PATH_ENV] = str(trace_path)
    env[TRACE_RUN_ID_ENV] = run_id
    env[TRACE_WORKSPACE_ROOT_ENV] = str(workspace_root)
    return env


def build_tool_trace_env(
    existing: dict[str, str] | None,
    *,
    workdir: Path,
    run_id: str | None = None,
    run_id_prefix: str = "",
) -> tuple[dict[str, str], Path, str]:
    resolved_run_id = run_id or new_trace_run_id(prefix=run_id_prefix)
    run_dir = workdir / "helix-logs" / resolved_run_id
    trace_path = tool_trace_path(run_dir)
    return (
        build_trace_env(
            existing,
            trace_path=trace_path,
            run_id=resolved_run_id,
            workspace_root=workdir,
        ),
        trace_path,
        resolved_run_id,
    )


def build_code_agent_event(
    *,
    request: AgentRequest,
    command: list[str],
    start_time: str,
    end_time: str,
    duration_ms: int,
    result: AgentResult | None,
    exception: BaseException | None = None,
) -> dict[str, Any]:
    if exception is not None:
        status = "error"
        return_code = None
    elif result is None:
        status = "unknown"
        return_code = None
    else:
        status = "stalled" if result.stalled else ("ok" if result.return_code == 0 else "error")
        return_code = result.return_code
    event: dict[str, Any] = {
        "schema_version": 1,
        "type": "agent_invocation",
        "phase": "end",
        "start_time": start_time,
        "end_time": end_time,
        "duration_ms": duration_ms,
        "status": status,
        "summary": summarize_agent_command(command, request.prompt),
        "return_code": return_code,
        "command_kind": request.command_kind.value,
        "agent": request.agent_name,
        "source": "runner",
        "confidence": "high",
    }
    if result is not None and result.session_id:
        event["session_id"] = result.session_id
    if exception is not None:
        event["error"] = str(exception)
    return event


def summarize_agent_command(command: list[str], prompt: str) -> str:
    summarized: list[str] = []
    prompt_replaced = False
    for token in command:
        if token == prompt and not prompt_replaced:
            summarized.append("<prompt>")
            prompt_replaced = True
        else:
            summarized.append(token)
    return " ".join(summarized)
