import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.optimize.contract import (
    BASELINE_STATE_REQUIRED_FIELDS,
    ROUND_STATE_REQUIRED_FIELDS,
    baseline_state_contract_lines,
)


class OptimizeContractTests(unittest.TestCase):
    def test_baseline_contract_uses_single_field_map(self) -> None:
        contract_path = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-optimize-submit-baseline"
            / "references"
            / "contract.json"
        )
        data = json.loads(contract_path.read_text(encoding="utf-8"))

        self.assertIn("baseline_state_fields", data)
        self.assertNotIn("round_state_required_fields", data)
        self.assertNotIn("baseline_state_required_fields", data)
        self.assertNotIn("baseline_state_field_descriptions", data)

        field_map = data["baseline_state_fields"]
        for key in BASELINE_STATE_REQUIRED_FIELDS:
            self.assertIn(key, field_map)
            self.assertIsInstance(field_map[key], str)

    def test_round_contract_uses_described_field_maps_without_baseline_duplication(
        self,
    ) -> None:
        contract_path = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-optimize-submit-round"
            / "references"
            / "contract.json"
        )
        data = json.loads(contract_path.read_text(encoding="utf-8"))

        self.assertIn("round_state_required_fields", data)
        self.assertNotIn("baseline_state_fields", data)
        self.assertNotIn("baseline_state_required_fields", data)

        field_map = data["round_state_required_fields"]
        for key in ROUND_STATE_REQUIRED_FIELDS:
            self.assertIn(key, field_map)
            self.assertIsInstance(field_map[key], str)

    def test_baseline_contract_lines_match_required_fields(self) -> None:
        lines = baseline_state_contract_lines()
        for key in BASELINE_STATE_REQUIRED_FIELDS:
            self.assertTrue(
                any(key in line for line in lines),
                f"Missing required field {key} in baseline contract lines",
            )

    def test_baseline_contract_lines_are_nonempty(self) -> None:
        lines = baseline_state_contract_lines()
        self.assertTrue(all(line.strip() for line in lines))


if __name__ == "__main__":
    unittest.main()
