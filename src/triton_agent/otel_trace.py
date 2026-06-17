from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, cast

from triton_agent.models import AgentRequest, AgentResult


TRACE_PATH_ENV = "TRITON_AGENT_OTEL_TRACE_PATH"
TRACE_RUN_ID_ENV = "TRITON_AGENT_OTEL_RUN_ID"
TRACE_WORKSPACE_ROOT_ENV = "TRITON_AGENT_WORKSPACE_ROOT"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_trace_run_id(prefix: str = "") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    return f"{prefix}-{ts}" if prefix else ts


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
    run_dir = workdir / "triton-agent-logs" / resolved_run_id
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


def write_tool_trace_summary(
    *,
    trace_path: Path,
    command_kind: str,
    show_output_path: Path | None = None,
) -> list[str]:
    warnings: list[str] = []
    summary_path = trace_summary_path(trace_path)
    try:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.touch(exist_ok=True)
        events = _read_trace_events(trace_path)
        summary = build_tool_trace_summary(
            events,
            trace_path=trace_path,
            command_kind=command_kind,
            show_output_path=show_output_path,
        )
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        warnings.append(f"Failed to write tool trace summary under {trace_path.parent}: {exc}")
    except ValueError as exc:
        warnings.append(f"Failed to parse tool trace at {trace_path}: {exc}")
    return warnings


def build_tool_trace_summary(
    events: list[dict[str, Any]],
    *,
    trace_path: Path,
    command_kind: str,
    show_output_path: Path | None = None,
) -> dict[str, Any]:
    event_counts = Counter(str(event.get("type") or "unknown") for event in events)
    capabilities = _tool_trace_capabilities(events)
    evidence_gaps = _tool_trace_evidence_gaps(capabilities)
    paths = {
        "trace": trace_path.as_posix(),
        "summary": trace_summary_path(trace_path).as_posix(),
    }
    if show_output_path is not None:
        paths["show_output"] = show_output_path.as_posix()
    time_ms_by_category = _time_ms_by_category(events)
    return {
        "run_id": trace_path.parent.name,
        "command_kind": command_kind,
        "tool_trace_enabled": bool(events),
        "tool_trace_capability": _tool_trace_capability_label(capabilities),
        "tool_trace_source": _detect_trace_source(events),
        "capabilities": capabilities,
        "paths": paths,
        "event_counts": {
            "total": len(events),
            "agent_invocation": event_counts.get("agent_invocation", 0),
            "tool_call": event_counts.get("tool_call", 0),
            "file_access": event_counts.get("file_access", 0),
            "command": event_counts.get("command", 0),
            "edit": event_counts.get("edit", 0),
        },
        "time_ms_by_category": time_ms_by_category,
        "top_slow_operations": _build_top_slow_operations(events, limit=10),
        "duration_quality": _build_duration_quality(events),
        "evidence_gaps": evidence_gaps,
    }


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


def _read_trace_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
        if isinstance(payload, dict):
            events.append(cast(dict[str, Any], payload))
    return events


def _tool_trace_capabilities(events: list[dict[str, Any]]) -> dict[str, bool]:
    tool_events = [event for event in events if event.get("type") == "tool_call"]
    command_events = [event for event in events if event.get("type") == "command"]
    file_events = [event for event in events if event.get("type") == "file_access"]
    edit_events = [event for event in events if event.get("type") == "edit"]
    invocation_events = [event for event in events if event.get("type") == "agent_invocation"]
    completion_events = [
        event
        for event in tool_events + command_events
        if event.get("phase") == "end" or event.get("return_code") is not None
    ]
    pre_tool_events = [
        event
        for event in tool_events + command_events + file_events + edit_events
        if event.get("phase") in {"start", "instant"} or event.get("type") != "tool_call"
    ]
    return {
        "agent_invocation": bool(invocation_events),
        "pre_tool_events": bool(pre_tool_events),
        "tool_completion_events": bool(completion_events),
        "command_events": bool(command_events),
        "file_access_events": bool(file_events),
        "edit_events": bool(edit_events),
    }


def _tool_trace_capability_label(capabilities: Mapping[str, bool]) -> str:
    if capabilities.get("tool_completion_events"):
        return "tool_completion_events"
    if capabilities.get("pre_tool_events"):
        return "pre_tool_events"
    if capabilities.get("agent_invocation"):
        return "agent_invocation_only"
    return "disabled"


def _tool_trace_evidence_gaps(capabilities: Mapping[str, bool]) -> list[str]:
    gaps: list[str] = []
    if not capabilities.get("agent_invocation"):
        gaps.append("No agent invocation event was available; agent runtime attribution may be missing.")
    if not capabilities.get("pre_tool_events"):
        gaps.append("No pre-tool events were available; command, file, and edit attribution is limited.")
    if not capabilities.get("tool_completion_events"):
        gaps.append("No tool completion events were available; per-tool duration and return-code evidence is incomplete.")
    if not capabilities.get("edit_events"):
        gaps.append("No edit events were available; edit attribution is incomplete.")
    return gaps


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


def _detect_trace_source(events: list[dict[str, Any]]) -> str:
    """Detect the primary trace source from event metadata."""
    if not events:
        return "unknown"
    sources = [event.get("source", "unknown") for event in events]
    for source in ("codex_native_json", "codex_posttooluse", "hook_clock_join", "show_output_parser", "codex_hook", "opencode_hook", "runner"):
        if source in sources:
            return source
    return "unknown"


def _time_ms_by_category(events: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate duration_ms by command_kind or type category."""
    result: dict[str, int] = {}
    for event in events:
        duration = event.get("duration_ms")
        if not isinstance(duration, (int, float)) or duration <= 0:
            continue
        category = str(event.get("command_kind") or event.get("type") or "unknown")
        result[category] = result.get(category, 0) + int(duration)
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


def _build_duration_quality(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute duration coverage quality metrics."""
    with_duration = sum(1 for e in events if isinstance(e.get("duration_ms"), (int, float)) and e.get("duration_ms", 0) > 0)
    without_duration = len(events) - with_duration
    sources = Counter(
        str(e.get("duration_source", "unknown")) for e in events if e.get("duration_ms")
    )
    total = max(len(events), 1)
    coverage_pct = round(with_duration / total * 100, 1) if total > 0 else 0.0
    return {
        "events_with_duration": with_duration,
        "events_without_duration": without_duration,
        "duration_sources": dict(sources),
        "duration_coverage_pct": coverage_pct,
    }


def _build_top_slow_operations(events: list[dict[str, Any]], *, limit: int = 10) -> list[dict[str, Any]]:
    """Return top slow operations sorted by duration_ms descending."""
    def _duration(event: dict[str, Any]) -> int:
        d = event.get("duration_ms")
        return int(d) if isinstance(d, (int, float)) else 0

    sorted_events = sorted(events, key=_duration, reverse=True)
    results: list[dict[str, Any]] = []
    for event in sorted_events[:limit]:
        d = _duration(event)
        if d == 0:
            continue
        results.append({
            "type": str(event.get("type", "unknown")),
            "tool": str(event.get("tool", "")),
            "command_kind": str(event.get("command_kind", "")),
            "summary": str(event.get("command") or event.get("summary") or ""),
            "duration_ms": d,
            "status": str(event.get("status", "")),
            "return_code": event.get("return_code"),
        })
    return results
