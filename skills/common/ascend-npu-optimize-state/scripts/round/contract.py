from __future__ import annotations

import json
from pathlib import Path


_SKILL_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = _SKILL_ROOT / "references" / "round-contract.json"
CONTRACT_DATA = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
ROUND_STATUS_ENUMS = CONTRACT_DATA["status_enums"]
ROUND_CORRECTNESS_STATUS_VALUES = tuple(ROUND_STATUS_ENUMS["correctness_status"])
ROUND_BENCHMARK_STATUS_VALUES = tuple(ROUND_STATUS_ENUMS["benchmark_status"])
ROUND_STATE_REQUIRED_FIELDS = tuple(CONTRACT_DATA["round_state_required_fields"])
ROUND_STATE_OPTIONAL_FIELDS = {
    str(field_name): str(description)
    for field_name, description in CONTRACT_DATA["round_state_optional_fields"].items()
}
