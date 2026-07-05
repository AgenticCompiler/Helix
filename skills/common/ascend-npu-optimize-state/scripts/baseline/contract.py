from __future__ import annotations

import json
from pathlib import Path


CONTRACT_PATH = Path(__file__).resolve().parents[2] / "references" / "baseline-contract.json"
CONTRACT_DATA = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
BASELINE_STATE_FIELDS = {
    str(field_name): str(description)
    for field_name, description in CONTRACT_DATA["baseline_state_fields"].items()
}
BASELINE_STATE_REQUIRED_FIELDS = tuple(BASELINE_STATE_FIELDS)
