from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any, cast

from triton_agent.trace.core import trace_summary_path


def analyze_trace(*, trace_path: Path) -> list[str]:
    """Parse an otel trace file and write summary.json alongside it."""
    warnings: list[str] = []
    output_dir = trace_path.parent
    summary_json_path = trace_summary_path(trace_path)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path.touch(exist_ok=True)
        events = _read_trace_events(trace_path)
        summary = build_summary(events, trace_path=trace_path)
        summary_json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        warnings.append(f"Failed to write trace analysis summary under {output_dir}: {exc}")
    except ValueError as exc:
        warnings.append(f"Failed to parse trace at {trace_path}: {exc}")
    return warnings


def build_summary(
    events: list[dict[str, Any]],
    *,
    trace_path: Path,
) -> dict[str, Any]:
    file_events = [event for event in events if event.get("type") == "file_access"]
    command_events = [event for event in events if event.get("type") == "command"]
    tool_events = [event for event in events if event.get("type") == "tool_call"]
    invocation_events = [
        event
        for event in events
        if event.get("type") == "agent_invocation"
        or (event.get("type") == "tool_call" and event.get("tool") == "code_agent")
    ]
    edit_events = [event for event in events if event.get("type") == "edit"]

    file_paths = [path for event in file_events if (path := _event_path(event)) is not None]
    file_counts: Counter[str] = Counter(file_paths)
    command_kind_counts = Counter(_text(event.get("command_kind"), "unknown") for event in command_events)
    tool_counts = Counter(_text(event.get("tool"), "unknown") for event in tool_events)

    skill_script_reads = [
        path
        for path in (_event_path(event) for event in file_events)
        if path is not None and _is_skill_script_path(path)
    ]
    repeated_file_reads = {
        path: count
        for path, count in sorted(file_counts.items())
        if count > 1
    }
    repeated_failed_commands = _repeated_failed_commands(command_events)
    full_msprof_commands = [
        _command_text(event)
        for event in command_events
        if _looks_like_full_msprof_benchmark(event)
    ]

    duration_by_category: defaultdict[str, int] = defaultdict(int)
    for event in command_events:
        category = _text(event.get("command_kind"), "unknown")
        duration_by_category[category] += _duration_ms(event)
    for event in tool_events:
        if event.get("tool") == "code_agent":
            duration_by_category["code_agent"] += _duration_ms(event)
    for event in invocation_events:
        duration_by_category["agent_invocation"] += _duration_ms(event)

    command_failures: list[dict[str, Any]] = [
        {
            "command": _command_text(event),
            "command_kind": _text(event.get("command_kind"), "unknown"),
            "return_code": event.get("return_code"),
            "stderr_digest": _text(event.get("stderr_digest"), ""),
        }
        for event in command_events
        if _is_failed_command(event)
    ]

    findings = _build_findings(
        skill_script_read_count=len(skill_script_reads),
        repeated_file_reads=repeated_file_reads,
        repeated_failed_commands=repeated_failed_commands,
        full_msprof_count=len(full_msprof_commands),
        command_failures=command_failures,
        command_events=command_events,
    )

    capabilities = _build_capabilities(
        invocation_events=invocation_events,
        tool_events=tool_events,
        command_events=command_events,
        file_events=file_events,
        edit_events=edit_events,
    )
    evidence_gaps = _build_evidence_gaps(capabilities)

    return {
        "tool_trace_enabled": bool(events),
        "tool_trace_capability": _capability_label(capabilities),
        "tool_trace_source": _detect_trace_source(events),
        "capabilities": capabilities,
        "paths": {
            "trace": trace_path.as_posix(),
            "summary_json": trace_summary_path(trace_path).as_posix(),
        },
        "event_counts": {
            "total": len(events),
            "agent_invocation": len(invocation_events),
            "tool_call": len(tool_events),
            "file_access": len(file_events),
            "command": len(command_events),
            "edit": len(edit_events),
        },
        "tool_call_counts": dict(sorted(tool_counts.items())),
        "file_access": {
            "skill_md_reads": _count_path_suffix(file_events, "SKILL.md"),
            "reference_reads": sum(1 for event in file_events if "/references/" in (_event_path(event) or "")),
            "skill_script_reads": len(skill_script_reads),
            "skill_script_read_paths": dict(sorted(Counter(skill_script_reads).items())),
            "repeated_file_reads": repeated_file_reads,
        },
        "commands": {
            "by_kind": dict(sorted(command_kind_counts.items())),
            "failures": command_failures,
            "repeated_failed_commands": repeated_failed_commands,
            "full_msprof_benchmark_commands": dict(sorted(Counter(full_msprof_commands).items())),
        },
        "time_ms_by_category": dict(sorted(duration_by_category.items())),
        "top_slow_operations": _build_top_slow_operations(command_events + tool_events, limit=10),
        "duration_quality": _build_duration_quality(events),
        "findings": findings,
        "evidence_gaps": evidence_gaps,
    }


# ---------------------------------------------------------------------------
# Trace event parsers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


def _build_findings(
    *,
    skill_script_read_count: int,
    repeated_file_reads: Mapping[str, int],
    repeated_failed_commands: list[dict[str, Any]],
    full_msprof_count: int,
    command_failures: list[dict[str, Any]],
    command_events: list[dict[str, Any]],
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if skill_script_read_count:
        severity = "fail" if skill_script_read_count > 3 else "warn"
        findings.append(
            {
                "id": "staged_skill_script_reads",
                "severity": severity,
                "detail": (
                    f"Detected {skill_script_read_count} reads of staged skill implementation scripts. "
                    "These reads should be limited to helper debugging, patching, or verification."
                ),
            }
        )
    if repeated_file_reads:
        repeated_total = sum(count - 1 for count in repeated_file_reads.values())
        findings.append(
            {
                "id": "repeated_file_reads",
                "severity": "warn",
                "detail": f"Detected {repeated_total} repeated reads across {len(repeated_file_reads)} files.",
            }
        )
    if command_failures:
        findings.append(
            {
                "id": "failed_commands",
                "severity": "warn",
                "detail": f"Detected {len(command_failures)} failed command events.",
            }
        )
    if repeated_failed_commands:
        findings.append(
            {
                "id": "repeated_failed_commands",
                "severity": "warn",
                "detail": f"{len(repeated_failed_commands)} command(s) repeated after identical earlier failure.",
            }
        )
    if full_msprof_count:
        findings.append(
            {
                "id": "full_msprof_benchmark",
                "severity": "warn",
                "detail": (
                    f"Detected {full_msprof_count} full msprof benchmark(s) with instrumentation enabled. "
                    "Full msprof profiling adds run-time overhead that distorts wall-time measurements."
                ),
            }
        )
    return findings


# ---------------------------------------------------------------------------
# Capabilities & evidence gaps
# ---------------------------------------------------------------------------


def _build_capabilities(
    *,
    invocation_events: list[dict[str, Any]],
    tool_events: list[dict[str, Any]],
    command_events: list[dict[str, Any]],
    file_events: list[dict[str, Any]],
    edit_events: list[dict[str, Any]],
) -> dict[str, bool]:
    return {
        "agent_invocation": bool(invocation_events),
        "pre_tool_events": any(
            _text(event.get("phase"), "") == "start"
            for event in tool_events
        ),
        "tool_completion_events": any(
            _text(event.get("phase"), "") in {"end", "instant"}
            for event in tool_events
        ),
        "file_access_events": bool(file_events),
        "command_events": bool(command_events),
        "edit_events": bool(edit_events),
    }


def _build_evidence_gaps(capabilities: dict[str, bool]) -> list[str]:
    gaps: list[str] = []
    if not capabilities.get("agent_invocation"):
        gaps.append("No agent invocation events detected — agent lifecycle not traced.")
    if not capabilities.get("pre_tool_events"):
        gaps.append("No pre-tool events detected — prompt contents are not captured.")
    if not capabilities.get("tool_completion_events"):
        gaps.append("No tool completion events detected — tool outcomes are not captured.")
    return gaps


def _capability_label(capabilities: dict[str, bool]) -> str:
    if capabilities.get("tool_completion_events"):
        return "tool_completion_events"
    if capabilities.get("pre_tool_events"):
        return "pre_tool_events"
    if capabilities.get("agent_invocation"):
        return "agent_invocation"
    if any(capabilities.values()):
        return "partial"
    return "disabled"


def _detect_trace_source(events: list[dict[str, Any]]) -> str:
    for event in events:
        source = _text(event.get("source"), "")
        if source:
            return source
        if event.get("type") == "agent_invocation":
            return "agent_invocation"
    if any(event.get("type") == "command" for event in events):
        return "command_events"
    return "unknown"


# ---------------------------------------------------------------------------
# Duration quality
# ---------------------------------------------------------------------------


def _build_duration_quality(events: list[dict[str, Any]]) -> dict[str, Any]:
    all_tool_events = [e for e in events if e.get("type") == "tool_call"]
    failed_events = [e for e in events if e.get("type") == "tool_call" and _is_tool_failed(e)]
    missing_durations = [
        e for e in events
        if e.get("type") in ("tool_call", "command", "agent_invocation") and not e.get("duration_ms")
    ]
    inconsistent_durations = 0
    for e in all_tool_events:
        dur = _int_value(e.get("duration_ms"))
        llm_dur = _int_value(e.get("llm_duration_ms"))
        if dur and llm_dur and llm_dur > dur:
            inconsistent_durations += 1
    return {
        "tool_events": len(all_tool_events),
        "tool_events_with_duration": sum(1 for e in all_tool_events if e.get("duration_ms")),
        "failed_tool_events": len(failed_events),
        "failed_tool_events_with_duration": sum(1 for e in failed_events if e.get("duration_ms")),
        "events_missing_duration": len(missing_durations),
        "inconsistent_llm_duration_count": inconsistent_durations,
    }


def _build_top_slow_operations(
    operations: list[dict[str, Any]],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    sorted_ops = sorted(operations, key=lambda op: _int_value(op.get("duration_ms")), reverse=True)
    top = sorted_ops[:limit]
    return [
        {
            "type": str(op.get("type", "-")),
            "name": _operation_name(op),
            "duration_ms": _int_value(op.get("duration_ms")),
            "status": _operation_status(op),
        }
        for op in top
    ]


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------


def _event_path(event: dict[str, Any]) -> str | None:
    path = event.get("path")
    if isinstance(path, str) and path:
        return path
    return None


def _is_skill_script_path(path: str) -> bool:
    normalized = _normalize_path(path)
    prefixes = (
        ".codex/skills/",
        ".opencode/skills/",
        "skills/",
    )
    return normalized.startswith(prefixes) and "/scripts/" in normalized and path.endswith(".py")


def _normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _command_text(event: dict[str, Any]) -> str:
    return _text(event.get("command"), _text(event.get("summary"), ""))


def _is_failed_command(event: dict[str, Any]) -> bool:
    rc = event.get("return_code")
    if rc is not None:
        try:
            return int(rc) != 0
        except (TypeError, ValueError):
            return False
    return bool(event.get("error"))


def _is_tool_failed(event: dict[str, Any]) -> bool:
    return bool(event.get("error") or event.get("failure_reason"))


def _looks_like_full_msprof_benchmark(event: dict[str, Any]) -> bool:
    command = _command_text(event).lower()
    kind = _text(event.get("command_kind"), "").lower()
    return "msprof" in command and ("bench" in command or "benchmark" in kind or kind == "remote_bench")


def _repeated_failed_commands(command_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failed: list[tuple[str, dict[str, Any]]] = []
    for event in command_events:
        if _is_failed_command(event):
            failed.append((_command_text(event), event))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for cmd_text, event in failed:
        grouped.setdefault(cmd_text, []).append(event)
    return [
        {"command": cmd_text, "count": len(events), "return_code": events[-1].get("return_code")}
        for cmd_text, events in grouped.items()
        if len(events) > 1
    ]


def _count_path_suffix(file_events: list[dict[str, Any]], suffix: str) -> int:
    return sum(1 for event in file_events if (_event_path(event) or "").endswith(suffix))


def _operation_name(op: dict[str, Any]) -> str:
    cmd = op.get("command") or op.get("command_kind") or op.get("summary", "")
    tool = op.get("tool", "")
    return _text(cmd or tool, "")


def _operation_status(op: dict[str, Any]) -> str:
    if op.get("error") or op.get("failure_reason"):
        return "failed"
    if op.get("return_code") is not None:
        try:
            return "failed" if int(op["return_code"]) != 0 else "ok"
        except (TypeError, ValueError):
            pass
    return "-"


# ---------------------------------------------------------------------------
# Generic value helpers
# ---------------------------------------------------------------------------


def _text(value: Any, default: str) -> str:
    if isinstance(value, str):
        return value
    return default


def _int_value(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _duration_ms(event: dict[str, Any]) -> int:
    return _int_value(event.get("duration_ms"))
