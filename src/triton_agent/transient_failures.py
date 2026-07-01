from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from triton_agent.models import AgentResult

TRANSIENT_AGENT_FAILURE_PATTERNS = (
    "429 too many requests",
    "exceeded retry limit",
    "rate limit",
)

OPTIMIZE_WORKER_RETRY_DELAYS_SECONDS = (2, 4, 8)


def contains_transient_agent_failure_text(text: str) -> bool:
    normalized = text.casefold()
    return any(pattern in normalized for pattern in TRANSIENT_AGENT_FAILURE_PATTERNS)


def is_transient_agent_failure(result: "AgentResult") -> bool:
    if result.stalled or result.return_code == 130:
        return False
    if result.retryable_failure:
        return True
    combined = f"{result.stdout}\n{result.stderr}".lower()
    return contains_transient_agent_failure_text(combined)


def is_optimize_worker_retryable(result: "AgentResult") -> bool:
    """Whether optimize should retry a failed worker batch after backend retries are done."""
    if result.succeeded:
        return False
    if result.stalled or result.return_code == 130:
        return False
    return not is_transient_agent_failure(result)
