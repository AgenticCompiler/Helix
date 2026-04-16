from __future__ import annotations

import json
from pathlib import Path


_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "skills"
    / "triton-npu-optimize-check"
    / "references"
    / "contract.json"
)
_CONTRACT_DATA = json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))

BASELINE_STATE_REQUIRED_FIELDS = tuple(_CONTRACT_DATA["baseline_state_required_fields"])
ROUND_STATE_REQUIRED_FIELDS = tuple(_CONTRACT_DATA["round_state_required_fields"])
_BASELINE_STATE_FIELD_DESCRIPTIONS = tuple(
    (str(field_name), str(description))
    for field_name, description in _CONTRACT_DATA["baseline_state_field_descriptions"]
)


def baseline_state_contract_lines() -> tuple[str, ...]:
    lines = ["Write `baseline/state.json` with these required fields:"]
    lines.extend(
        f"`{field_name}`: {description}"
        for field_name, description in _BASELINE_STATE_FIELD_DESCRIPTIONS
    )
    lines.append(
        "Set `baseline_established` to `true` only after `correctness_status` is `passed` and `benchmark_status` is `passed`."
    )
    return tuple(lines)
