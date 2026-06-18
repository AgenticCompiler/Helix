from __future__ import annotations

import json
from pathlib import Path


def write_agent_exit_log(
    *,
    workdir: Path,
    run_id: str,
    label: str,
    return_code: int,
    stderr: str,
    stalled: bool,
    session_id: str | None,
) -> None:
    log_dir = workdir / "triton-agent-logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"agent-exit-{label}.json"
    payload = {
        "return_code": return_code,
        "stalled": stalled,
        "session_id": session_id,
        "stderr": stderr,
    }
    log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
