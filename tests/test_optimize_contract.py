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
            / "triton-npu-optimize-submit-baseline"
            / "references"
            / "contract.json"
        )
        data = json.loads(contract_path.read_text(encoding="utf-8"))

        self.assertIn("baseline_state_fields", data)
        self.assertNotIn("round_state_required_fields", data)
        self.assertNotIn("baseline_state_required_fields", data)
        self.assertNotIn("baseline_state_field_descriptions", data)

        field_map = data["baseline_state_fields"]
        self.assertEqual(tuple(field_map.keys()), BASELINE_STATE_REQUIRED_FIELDS)

    def test_baseline_state_contract_lines_render_from_field_map(self) -> None:
        lines = baseline_state_contract_lines()

        self.assertEqual(lines[0], "Write `baseline/state.json` with these required fields:")
        self.assertIn(
            "`baseline_kind`: record whether the canonical baseline is the original operator or a minimally repaired prepared baseline.",
            lines,
        )
        self.assertIn(
            "`baseline_established`: set this to `true` only after `correctness_status` is `passed`, `benchmark_status` is `passed`, and the canonical baseline artifacts are written.",
            lines,
        )

    def test_round_contract_uses_described_field_maps_without_baseline_duplication(self) -> None:
        contract_path = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "triton-npu-optimize-submit-round"
            / "references"
            / "contract.json"
        )
        data = json.loads(contract_path.read_text(encoding="utf-8"))

        self.assertNotIn("baseline_state_fields", data)
        self.assertIn("round_state_required_fields", data)
        self.assertIn("round_state_optional_fields", data)
        self.assertIsInstance(data["round_state_required_fields"], dict)
        self.assertIsInstance(data["round_state_optional_fields"], dict)
        self.assertEqual(tuple(data["round_state_required_fields"].keys()), ROUND_STATE_REQUIRED_FIELDS)
        self.assertIn("comparison_target", data["round_state_required_fields"])
        self.assertIn("perf_analysis_path", data["round_state_optional_fields"])
        self.assertNotIn("analysis_comparison_sources", data["round_state_optional_fields"])
        self.assertNotIn("validated_candidate", data["round_state_optional_fields"])


if __name__ == "__main__":
    unittest.main()
