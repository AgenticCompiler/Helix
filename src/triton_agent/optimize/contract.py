from __future__ import annotations

import json

from triton_agent.resources import skills_root


_CONTRACT_PATH = (
    skills_root()
    / "triton-npu-optimize-check"
    / "references"
    / "contract.json"
)
_CONTRACT_DATA = json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))

BASELINE_STATE_FIELDS = {
    str(field_name): str(description)
    for field_name, description in _CONTRACT_DATA["baseline_state_fields"].items()
}
BASELINE_STATE_REQUIRED_FIELDS = tuple(BASELINE_STATE_FIELDS)
ROUND_STATE_REQUIRED_FIELDS = tuple(_CONTRACT_DATA["round_state_required_fields"])


def baseline_state_contract_lines() -> tuple[str, ...]:
    lines = ["Write `baseline/state.json` with these required fields:"]
    lines.extend(
        f"`{field_name}`: {description}"
        for field_name, description in BASELINE_STATE_FIELDS.items()
    )
    lines.append(
        "Set `baseline_established` to `true` only after `correctness_status` is `passed` and `benchmark_status` is `passed`."
    )
    return tuple(lines)
