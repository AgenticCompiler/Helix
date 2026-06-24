from __future__ import annotations

from typing import TypedDict


class ResultPayload(TypedDict):
    return_code: int
    stdout: str
    stderr: str
    stalled: bool
    session_id: str | None


def make_result(
    *,
    return_code: int,
    stdout: str,
    stderr: str,
    stalled: bool = False,
    session_id: str | None = None,
) -> ResultPayload:
    return {
        "return_code": return_code,
        "stdout": stdout,
        "stderr": stderr,
        "stalled": stalled,
        "session_id": session_id,
    }
