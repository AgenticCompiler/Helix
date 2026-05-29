from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO

from triton_agent.models import AgentRequest, AgentResult
from triton_agent.otel_trace import TRACE_RUN_ID_ENV


def show_output_log_path(request: AgentRequest) -> Path:
    run_id = (request.extra_env or {}).get(TRACE_RUN_ID_ENV)
    if run_id:
        return request.workdir / "triton-agent-logs" / run_id / "show-output.log"
    return request.workdir / "triton-agent-logs" / f"{request.command_kind.value}.show-output.log"


@contextmanager
def open_show_output_log(request: AgentRequest) -> Iterator[TextIO | None]:
    if request.interact or not request.show_output:
        yield None
        return
    path = show_output_log_path(request)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        yield stream


def write_show_output_attempt_start(
    stream: TextIO | None,
    *,
    request: AgentRequest,
    attempt_number: int,
) -> None:
    if stream is None:
        return
    timestamp = _timestamp()
    stream.write(
        f"\n===== triton-agent show-output start "
        f"time={timestamp} command={request.command_kind.value} agent={request.agent_name} "
        f"attempt={attempt_number} =====\n"
    )
    stream.flush()


def write_show_output_attempt_result(
    stream: TextIO | None,
    *,
    result: AgentResult,
) -> None:
    if stream is None:
        return
    if result.stdout:
        stream.write(result.stdout)
        if not result.stdout.endswith("\n"):
            stream.write("\n")
    timestamp = _timestamp()
    counts = _show_output_counts(result.stdout)
    stream.write(
        f"===== triton-agent show-output end "
        f"time={timestamp} return_code={result.return_code} stalled={result.stalled} "
        f"session_id={result.session_id or 'unknown'} "
        f"events={counts['events']} tools={counts['tools']} errors={counts['errors']} =====\n"
    )
    stream.flush()


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _show_output_counts(stdout: str) -> dict[str, int]:
    events = 0
    tools = 0
    errors = 0
    for line in stdout.splitlines():
        if line.startswith("["):
            events += 1
        if line.startswith("[tool:start]"):
            tools += 1
        if line.startswith("[tool:end]") and " error " in line:
            errors += 1
    return {"events": events, "tools": tools, "errors": errors}

