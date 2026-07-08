from __future__ import annotations

from collections.abc import Mapping

_OPTIMIZE_MIN_SPEEDUP_ENV = "TRITON_AGENT_OPTIMIZE_MIN_SPEEDUP"


def optimize_min_speedup_env_name() -> str:
    return _OPTIMIZE_MIN_SPEEDUP_ENV


def merge_optimize_session_env(
    extra_env: Mapping[str, str] | None,
    *,
    min_speedup: float | None,
) -> dict[str, str] | None:
    merged = dict(extra_env or {})
    if min_speedup is not None:
        merged[optimize_min_speedup_env_name()] = format(min_speedup, "g")
    return merged or None
