from __future__ import annotations

import json
from pathlib import Path

from triton_agent.otel_trace import utc_timestamp


def write_agent_exit_log(
    *,
    workdir: Path,
    run_id: str,
    label: str,
    return_code: int,
    stderr: str,
    stalled: bool,
    session_id: str | None,
    duration_ms: int,
) -> None:
    log_dir = workdir / "triton-agent-logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"agent-exit-{label}.json"
    payload = {
        "ended_at": utc_timestamp(),
        "duration_ms": duration_ms,
        "return_code": return_code,
        "stalled": stalled,
        "session_id": session_id,
        "stderr": stderr,
    }
    log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
