from __future__ import annotations

import os
from collections.abc import Mapping

from env_registry import ASCEND_RT_VISIBLE_DEVICES, TRITON_AGENT_DEBUG
_DEBUG_PREFIX = "[TRITON_AGENT_DEBUG]"


def debug_enabled(env: Mapping[str, str] | None = None) -> bool:
    current_env = os.environ if env is None else env
    raw = current_env.get(TRITON_AGENT_DEBUG, "")
    return raw.strip().lower() in {"true", "1"}


def maybe_print_visible_devices(env: Mapping[str, str] | None = None) -> None:
    current_env = os.environ if env is None else env
    if not debug_enabled(current_env):
        return
    value = current_env.get(ASCEND_RT_VISIBLE_DEVICES, "<unset>")
    print(f"{_DEBUG_PREFIX} {ASCEND_RT_VISIBLE_DEVICES}={value}")
