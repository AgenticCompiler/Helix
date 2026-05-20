from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
import json
from pathlib import Path
from typing import Any, cast

from triton_agent.optimize.archive import ArchiveState


def write_agent_audit(*, workdir: Path, archive: ArchiveState) -> list[str]:
    warnings: list[str] = []
    trace_path = archive.otel_trace_path
    summary_path = archive.otel_summary_path
    audit_path = archive.agent_audit_path

    try:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.touch(exist_ok=True)
        events = _read_trace_events(trace_path)
        summary = build_summary(
            events,
            workdir=workdir,
            trace_path=trace_path,
            show_output_path=workdir / "triton-agent-logs" / "optimize.show-output.log",
            agent_sessions_path=archive.agent_sessions_path,
        )
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        audit_path.write_text(render_audit_markdown(summary), encoding="utf-8")
    except OSError as exc:
        warnings.append(f"Failed to write optimize agent audit under {archive.otel_run_dir}: {exc}")
    except ValueError as exc:
        warnings.append(f"Failed to parse optimize agent trace at {trace_path}: {exc}")
    return warnings


def build_summary(
    events: list[dict[str, Any]],
    *,
    workdir: Path,
    trace_path: Path,
    show_output_path: Path,
    agent_sessions_path: Path,
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

    artifact_presence = {
        "trace": trace_path.is_file(),
        "show_output": show_output_path.is_file(),
        "agent_sessions": agent_sessions_path.is_file(),
        "baseline": (workdir / "baseline").is_dir(),
        "round_count": sum(1 for path in workdir.glob("opt-round-*") if path.is_dir()),
    }
    worker_timeline = _build_worker_timeline(workdir)

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
    evidence_gaps.extend(_round_evidence_gaps(worker_timeline))

    return {
        "run_id": trace_path.parent.name,
        "tool_trace_enabled": bool(events),
        "tool_trace_capability": _capability_label(capabilities),
        "tool_trace_source": _detect_trace_source(events),
        "capabilities": capabilities,
        "paths": {
            "trace": trace_path.as_posix(),
            "summary": (trace_path.parent / "summary.json").as_posix(),
            "agent_audit": (trace_path.parent / "agent-audit.md").as_posix(),
            "show_output": show_output_path.as_posix(),
            "agent_sessions": agent_sessions_path.as_posix(),
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
        "artifacts": artifact_presence,
        "worker_timeline": worker_timeline,
        "findings": findings,
        "evidence_gaps": evidence_gaps,
    }


def render_audit_markdown(summary: Mapping[str, Any]) -> str:
    event_counts = _mapping(summary.get("event_counts"))
    file_access = _mapping(summary.get("file_access"))
    commands = _mapping(summary.get("commands"))
    artifacts = _mapping(summary.get("artifacts"))
    findings = list(summary.get("findings") or [])
    time_ms_by_category = _mapping(summary.get("time_ms_by_category"))
    capabilities = _mapping(summary.get("capabilities"))
    evidence_gaps = [str(item) for item in list(summary.get("evidence_gaps") or [])]
    worker_timeline: list[Mapping[str, Any]] = []
    raw_worker_timeline = summary.get("worker_timeline")
    if isinstance(raw_worker_timeline, list):
        for raw_item in cast(list[Any], raw_worker_timeline):
            if isinstance(raw_item, Mapping):
                worker_timeline.append(cast(Mapping[str, Any], raw_item))

    rows = [
        (
            "Skill script reads",
            str(file_access.get("skill_script_reads", 0)),
            "-",
            _finding_for(findings, "staged_skill_script_reads"),
        ),
        (
            "Repeated file reads",
            str(sum(_int_value(value) - 1 for value in _mapping(file_access.get("repeated_file_reads")).values())),
            "-",
            _finding_for(findings, "repeated_file_reads"),
        ),
        (
            "Failed commands",
            str(len(commands.get("failures") or [])),
            "-",
            _finding_for(findings, "failed_commands"),
        ),
        (
            "Full msprof benchmark",
            str(_sum_int_values(_mapping(commands.get("full_msprof_benchmark_commands")))),
            _format_ms(_int_value(time_ms_by_category.get("benchmark")) + _int_value(time_ms_by_category.get("remote_bench"))),
            _finding_for(findings, "full_msprof_benchmark"),
        ),
    ]

    lines = [
        "# Agent Execution Audit Report",
        "",
        "## Overview",
        "",
        "| Category | Count | Time | Finding |",
        "| --- | ---: | ---: | --- |",
    ]
    for category, count, duration, finding in rows:
        lines.append(f"| {category} | {count} | {duration} | {finding or 'No issue detected'} |")

    lines.extend(
        [
            "",
            "## Evidence Sources",
            "",
            f"- OTEL trace: {'present' if artifacts.get('trace') else 'missing'}",
            f"- show-output log: {'present' if artifacts.get('show_output') else 'missing'}",
            f"- agent sessions: {'present' if artifacts.get('agent_sessions') else 'missing'}",
            f"- baseline directory: {'present' if artifacts.get('baseline') else 'missing'}",
            f"- opt-round directories: {artifacts.get('round_count', 0)}",
            f"- tool trace enabled: {summary.get('tool_trace_enabled', False)}",
            f"- tool trace capability: {summary.get('tool_trace_capability', 'disabled')}",
            f"- tool trace source: {summary.get('tool_trace_source', 'unknown')}",
            "",
            "## Capabilities",
            "",
            f"- agent invocation: {_yes_no(capabilities.get('agent_invocation'))}",
            f"- pre-tool events: {_yes_no(capabilities.get('pre_tool_events'))}",
            f"- tool completion events: {_yes_no(capabilities.get('tool_completion_events'))}",
            f"- command events: {_yes_no(capabilities.get('command_events'))}",
            f"- file access events: {_yes_no(capabilities.get('file_access_events'))}",
            f"- edit events: {_yes_no(capabilities.get('edit_events'))}",
            "",
            "## Event Counts",
            "",
            f"- total events: {event_counts.get('total', 0)}",
            f"- agent invocations: {event_counts.get('agent_invocation', 0)}",
            f"- tool calls: {event_counts.get('tool_call', 0)}",
            f"- file accesses: {event_counts.get('file_access', 0)}",
            f"- commands: {event_counts.get('command', 0)}",
            f"- edits: {event_counts.get('edit', 0)}",
            "",
            "## Main Time Costs",
            "",
        ]
    )
    if time_ms_by_category:
        for category, duration_ms in sorted(
            time_ms_by_category.items(),
            key=lambda item: _int_value(item[1]),
            reverse=True,
        ):
            lines.append(f"- {category}: {_format_ms(_int_value(duration_ms))}")
    else:
        lines.append("- No duration evidence was available in the trace.")

    top_slow = list(summary.get("top_slow_operations") or [])
    if top_slow:
        lines.extend(["", "## Top Slow Operations", ""])
        lines.append("| # | Type | Command/Tool | Duration | Status |")
        lines.append("| --- | --- | --- | ---: | --- |")
        for i, op in enumerate(top_slow, start=1):
            op_type = str(op.get("type", ""))
            cmd = str(op.get("command_kind") or op.get("tool") or op.get("summary", "")[:50])
            dur = _format_ms(int(op.get("duration_ms", 0)))
            status = str(op.get("status", ""))
            lines.append(f"| {i} | {op_type} | {cmd} | {dur} | {status} |")

    duration_quality = _mapping(summary.get("duration_quality"))
    if duration_quality:
        lines.extend(["", "## Duration Quality", ""])
        lines.append(f"- events with duration: {duration_quality.get('events_with_duration', 0)}")
        lines.append(f"- events without duration: {duration_quality.get('events_without_duration', 0)}")
        lines.append(f"- duration coverage: {duration_quality.get('duration_coverage_pct', 0)}%")
        ds = duration_quality.get("duration_sources", {})
        if ds:
            lines.append("- duration sources:")
            for src, count in sorted(ds.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  - {src}: {count}")

    lines.extend(["", "## Redundant Operations", ""])
    redundant_lines = _render_redundant_lines(summary)
    lines.extend(redundant_lines or ["- No redundant operation was detected from available trace evidence."])

    lines.extend(["", "## Worker Timeline", ""])
    timeline_lines = _render_worker_timeline(worker_timeline)
    lines.extend(timeline_lines or ["- No opt-round timeline artifacts were available."])

    lines.extend(["", "## Recommendations", ""])
    recommendation_lines = _render_recommendation_lines(summary)
    lines.extend(recommendation_lines or ["- Keep using structured trace evidence for future optimize runs."])
    lines.extend(["", "## Evidence Gaps", ""])
    lines.extend([f"- {gap}" for gap in evidence_gaps] or ["- No evidence gap was detected from available trace metadata."])
    lines.append("")
    return "\n".join(lines)


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
                "severity": "fail",
                "detail": f"Detected {len(repeated_failed_commands)} repeated failed command loops.",
            }
        )
    if full_msprof_count > 1:
        unstable_context = any(_is_failed_command(event) for event in command_events)
        severity = "warn" if unstable_context else "info"
        findings.append(
            {
                "id": "full_msprof_benchmark",
                "severity": severity,
                "detail": f"Detected {full_msprof_count} full msprof benchmark command events.",
            }
        )
    return findings


def _build_capabilities(
    *,
    invocation_events: list[dict[str, Any]],
    tool_events: list[dict[str, Any]],
    command_events: list[dict[str, Any]],
    file_events: list[dict[str, Any]],
    edit_events: list[dict[str, Any]],
) -> dict[str, bool]:
    completion_events = [
        event
        for event in tool_events + command_events
        if event.get("phase") == "end" or event.get("return_code") is not None
    ]
    pre_tool_events = [
        event
        for event in tool_events + command_events + file_events + edit_events
        if not (event.get("type") == "tool_call" and event.get("tool") == "code_agent")
    ]
    return {
        "agent_invocation": bool(invocation_events),
        "pre_tool_events": bool(pre_tool_events),
        "tool_completion_events": bool(completion_events),
        "command_events": bool(command_events),
        "file_access_events": bool(file_events),
        "edit_events": bool(edit_events),
    }


def _capability_label(capabilities: Mapping[str, bool]) -> str:
    if capabilities.get("tool_completion_events"):
        return "tool_completion_events"
    if capabilities.get("pre_tool_events"):
        return "pre_tool_events"
    if capabilities.get("agent_invocation"):
        return "agent_invocation_only"
    return "disabled"


def _build_evidence_gaps(capabilities: Mapping[str, bool]) -> list[str]:
    gaps: list[str] = []
    if not capabilities.get("agent_invocation"):
        gaps.append("No agent invocation event was available; agent runtime attribution may be missing.")
    if not capabilities.get("pre_tool_events"):
        gaps.append("No pre-tool events were available; command, file, and edit attribution is limited.")
    if not capabilities.get("tool_completion_events"):
        gaps.append("No tool completion events were available; per-tool duration and return-code evidence is incomplete.")
    if not capabilities.get("edit_events"):
        gaps.append("No edit events were available; edit-test-fail loop analysis may be incomplete.")
    return gaps


def _build_worker_timeline(workdir: Path) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for round_dir in sorted(_iter_round_dirs(workdir), key=lambda path: _round_sort_key(path.name)):
        state = _read_json_object(round_dir / "round-state.json")
        summary_text = _read_text_excerpt(round_dir / "summary.md", limit=500)
        attempts_text = _read_text_excerpt(round_dir / "attempts.md", limit=500)
        timeline.append(
            {
                "round": round_dir.name,
                "parent_round": _text(state.get("parent_round"), "") if state else "",
                "hypothesis": _text(state.get("hypothesis"), "") if state else "",
                "correctness_status": _text(state.get("correctness_status"), "unknown") if state else "unknown",
                "benchmark_status": _text(state.get("benchmark_status"), "unknown") if state else "unknown",
                "round_disposition": _text(state.get("round_disposition"), "unknown") if state else "unknown",
                "evidence_sources": _string_list(state.get("evidence_sources")) if state else [],
                "has_round_state": bool(state),
                "has_summary": (round_dir / "summary.md").is_file(),
                "has_attempts": (round_dir / "attempts.md").is_file(),
                "summary_excerpt": summary_text,
                "attempts_excerpt": attempts_text,
            }
        )
    return timeline


def _round_evidence_gaps(worker_timeline: list[dict[str, Any]]) -> list[str]:
    gaps: list[str] = []
    for item in worker_timeline:
        round_name = _text(item.get("round"), "unknown")
        if not item.get("has_round_state"):
            gaps.append(f"{round_name} is missing round-state.json; structured round status is unavailable.")
        if not item.get("has_summary"):
            gaps.append(f"{round_name} is missing summary.md; round conclusion evidence is unavailable.")
        if not item.get("has_attempts"):
            gaps.append(f"{round_name} is missing attempts.md; command and reasoning timeline is incomplete.")
        if not item.get("evidence_sources"):
            gaps.append(f"{round_name} does not list evidence_sources in round-state.json.")
    return gaps


def _render_worker_timeline(worker_timeline: list[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    for item in worker_timeline:
        round_name = _text(item.get("round"), "unknown")
        hypothesis = _text(item.get("hypothesis"), "")
        evidence_sources = ", ".join(_string_list(item.get("evidence_sources"))) or "none"
        status = (
            f"correctness={_text(item.get('correctness_status'), 'unknown')}, "
            f"benchmark={_text(item.get('benchmark_status'), 'unknown')}, "
            f"disposition={_text(item.get('round_disposition'), 'unknown')}"
        )
        if hypothesis:
            lines.append(f"- {round_name}: {status}; evidence={evidence_sources}; hypothesis={hypothesis}")
        else:
            lines.append(f"- {round_name}: {status}; evidence={evidence_sources}")
    return lines


def _iter_round_dirs(workdir: Path) -> list[Path]:
    return [
        path
        for path in workdir.glob("opt-round-*")
        if path.is_dir() and _round_sort_key(path.name)[0] >= 0
    ]


def _round_sort_key(name: str) -> tuple[int, str]:
    prefix = "opt-round-"
    if not name.startswith(prefix):
        return (-1, name)
    suffix = name[len(prefix) :]
    if not suffix.isdigit():
        return (-1, name)
    return (int(suffix), name)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return cast(dict[str, Any], data)


def _read_text_excerpt(path: Path, *, limit: int) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n<truncated>"


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        typed_value = cast(list[Any] | tuple[Any, ...], value)
        return [str(item) for item in typed_value if item is not None]
    return []


def _render_redundant_lines(summary: Mapping[str, Any]) -> list[str]:
    file_access = _mapping(summary.get("file_access"))
    commands = _mapping(summary.get("commands"))
    lines: list[str] = []
    for path, count in _mapping(file_access.get("skill_script_read_paths")).items():
        lines.append(
            f"- Read `{path}` {count} time(s). Verify that each read was tied to helper debugging, patching, or verification."
        )
    for path, count in _mapping(file_access.get("repeated_file_reads")).items():
        lines.append(f"- Repeatedly read `{path}` {count} time(s).")
    raw_repeated_failed = commands.get("repeated_failed_commands")
    repeated_failed_items: list[Any] = cast(list[Any], raw_repeated_failed) if isinstance(raw_repeated_failed, list) else []
    for item in repeated_failed_items:
        if not isinstance(item, Mapping):
            continue
        typed_item = cast(Mapping[str, Any], item)
        lines.append(
            f"- Repeated failed command `{typed_item.get('command', '')}` {typed_item.get('count', 0)} time(s)."
        )
    for command, count in _mapping(commands.get("full_msprof_benchmark_commands")).items():
        if _int_value(count) > 1:
            lines.append(f"- Repeated full msprof benchmark `{command}` {count} time(s).")
    return lines


def _render_recommendation_lines(summary: Mapping[str, Any]) -> list[str]:
    file_access = _mapping(summary.get("file_access"))
    commands = _mapping(summary.get("commands"))
    lines: list[str] = []
    if _int_value(file_access.get("skill_script_reads")):
        lines.append("- Avoid reading staged skill implementation scripts unless debugging, patching, or verifying the helper.")
    if _mapping(file_access.get("repeated_file_reads")):
        lines.append("- Reuse the previously read skill or reference content instead of reopening the same file repeatedly.")
    if commands.get("repeated_failed_commands"):
        lines.append("- After two equivalent command failures, inspect the semantic cause before rerunning the same command.")
    if _sum_int_values(_mapping(commands.get("full_msprof_benchmark_commands"))) > 1:
        lines.append("- Stabilize correctness, baseline metadata, and benchmark metadata before repeating full msprof runs.")
    return lines


def _repeated_failed_commands(command_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repeated: list[dict[str, Any]] = []
    previous_key: tuple[str, str] | None = None
    streak = 0
    for event in command_events:
        if not _is_failed_command(event):
            previous_key = None
            streak = 0
            continue
        key = (_command_text(event), _text(event.get("stderr_digest"), ""))
        if key == previous_key:
            streak += 1
        else:
            if previous_key is not None and streak > 1:
                repeated.append({"command": previous_key[0], "stderr_digest": previous_key[1], "count": streak})
            previous_key = key
            streak = 1
    if previous_key is not None and streak > 1:
        repeated.append({"command": previous_key[0], "stderr_digest": previous_key[1], "count": streak})
    return repeated


def _event_path(event: Mapping[str, Any]) -> str | None:
    raw = event.get("path")
    if not isinstance(raw, str) or not raw:
        return None
    return _normalize_path(raw)


def _normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _is_skill_script_path(path: str) -> bool:
    normalized = _normalize_path(path)
    prefixes = (
        ".codex/skills/",
        ".opencode/skills/",
        "skills/",
    )
    return normalized.startswith(prefixes) and "/scripts/" in normalized


def _looks_like_full_msprof_benchmark(event: Mapping[str, Any]) -> bool:
    command = _command_text(event).lower()
    kind = _text(event.get("command_kind"), "").lower()
    return "msprof" in command and ("bench" in command or "benchmark" in kind or kind == "remote_bench")


def _is_failed_command(event: Mapping[str, Any]) -> bool:
    return_code = event.get("return_code")
    if isinstance(return_code, int):
        return return_code != 0
    status = event.get("status")
    return isinstance(status, str) and status.lower() in {"error", "failed", "blocked"}


def _command_text(event: Mapping[str, Any]) -> str:
    return _text(event.get("command"), _text(event.get("summary"), ""))


def _duration_ms(event: Mapping[str, Any]) -> int:
    duration = event.get("duration_ms")
    if isinstance(duration, int):
        return max(duration, 0)
    if isinstance(duration, float):
        return max(int(duration), 0)
    return 0


def _count_path_suffix(events: Iterable[Mapping[str, Any]], suffix: str) -> int:
    return sum(1 for event in events if (_event_path(event) or "").endswith(suffix))


def _format_ms(duration_ms: int) -> str:
    if duration_ms <= 0:
        return "-"
    seconds = duration_ms // 1000
    if seconds < 60:
        return f"{seconds}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{remaining_seconds:02d}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h{remaining_minutes:02d}m"


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in cast(Mapping[Any, Any], value).items()}
    return {}


def _finding_for(findings: list[Any], finding_id: str) -> str:
    for finding in findings:
        if not isinstance(finding, Mapping):
            continue
        typed_finding = cast(Mapping[str, Any], finding)
        if typed_finding.get("id") == finding_id:
            return str(typed_finding.get("detail") or "")
    return ""


def _text(value: Any, default: str) -> str:
    if value is None:
        return default
    return str(value)


def _int_value(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _sum_int_values(values: Mapping[str, Any]) -> int:
    return sum(_int_value(value) for value in values.values())


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _detect_trace_source(events: list[dict[str, Any]]) -> str:
    """Detect the primary trace source from event metadata."""
    if not events:
        return "unknown"
    sources = [event.get("source", "unknown") for event in events]
    for source in ("codex_native_json", "codex_posttooluse", "hook_clock_join", "show_output_parser", "codex_hook", "runner"):
        if source in sources:
            return source
    return "unknown"


def _build_duration_quality(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute duration coverage quality metrics."""
    with_duration = sum(1 for e in events if isinstance(e.get("duration_ms"), (int, float)) and e.get("duration_ms", 0) > 0)
    without_duration = len(events) - with_duration
    sources: Counter[str] = Counter(
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
    results = []
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
