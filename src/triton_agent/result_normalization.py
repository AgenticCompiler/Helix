from __future__ import annotations

from collections.abc import Mapping

from triton_agent.models import AgentResult

RunSkillPayload = Mapping[str, object]


def normalize_agent_result(result: AgentResult | RunSkillPayload) -> AgentResult:
    if isinstance(result, AgentResult):
        return result
    payload = result
    required_keys = ("return_code", "stdout", "stderr")
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        raise ValueError(
            "Run skill result payload is missing required keys: "
            + ", ".join(sorted(missing_keys))
        )
    session_id = payload.get("session_id")
    return AgentResult(
        return_code=int(str(payload["return_code"])),
        stdout=str(payload["stdout"]),
        stderr=str(payload["stderr"]),
        stalled=bool(payload.get("stalled", False)),
        session_id=None if session_id is None else str(session_id),
    )
