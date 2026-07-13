from __future__ import annotations

import json

from helix.paths import skills_root


_BASELINE_CONTRACT_PATH = (
    skills_root()
    / "common"
    / "ascend-npu-optimize-state"
    / "references"
    / "baseline-contract.json"
)
_ROUND_CONTRACT_PATH = (
    skills_root()
    / "common"
    / "ascend-npu-optimize-state"
    / "references"
    / "round-contract.json"
)
_BASELINE_CONTRACT_DATA = json.loads(_BASELINE_CONTRACT_PATH.read_text(encoding="utf-8"))
_ROUND_CONTRACT_DATA = json.loads(_ROUND_CONTRACT_PATH.read_text(encoding="utf-8"))

BASELINE_STATE_FIELDS = {
    str(field_name): str(description)
    for field_name, description in _BASELINE_CONTRACT_DATA["baseline_state_fields"].items()
}
BASELINE_STATE_REQUIRED_FIELDS = tuple(BASELINE_STATE_FIELDS)
ROUND_STATE_REQUIRED_FIELDS = tuple(_ROUND_CONTRACT_DATA["round_state_required_fields"])
ROUND_STATE_OPTIONAL_FIELDS = {
    str(field_name): str(description)
    for field_name, description in _ROUND_CONTRACT_DATA["round_state_optional_fields"].items()
}


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
