from __future__ import annotations

TRANSIENT_AGENT_FAILURE_PATTERNS = (
    "429 too many requests",
    "exceeded retry limit",
    "rate limit",
)


def contains_transient_agent_failure_text(text: str) -> bool:
    return any(pattern in text for pattern in TRANSIENT_AGENT_FAILURE_PATTERNS)
