import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.skill_loader import load_skill_script_module


def load_workflow_state_module():
    return load_skill_script_module("ascend-npu-optimize-state", "state_manage/workflow")


class OptimizeWorkflowStateTests(unittest.TestCase):
    def test_start_round_module_is_exposed_from_new_skill(self) -> None:
        module = load_skill_script_module(
            "ascend-npu-optimize-state",
            "state_manage/start_round",
        )
        self.assertTrue(hasattr(module, "build_parser"))
        self.assertTrue(hasattr(module, "main"))

    def test_submit_baseline_module_is_exposed_from_state_manage(self) -> None:
        module = load_skill_script_module(
            "ascend-npu-optimize-state",
            "state_manage/submit_baseline",
        )
        self.assertTrue(hasattr(module, "build_parser"))
        self.assertTrue(hasattr(module, "main"))

    def test_submit_round_module_is_exposed_from_state_manage(self) -> None:
        module = load_skill_script_module(
            "ascend-npu-optimize-state",
            "state_manage/submit_round",
        )
        self.assertTrue(hasattr(module, "build_parser"))
        self.assertTrue(hasattr(module, "main"))

    def test_bootstrap_state_writes_expected_hook_gated_baseline_payload(self) -> None:
        module = load_workflow_state_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / ".triton-agent" / "state.json"
            state_path.parent.mkdir()

            module.bootstrap_state(
                state_path,
                run_id="optimize-20260622-123456-abcdef",
                source_operator="kernel.py",
                baseline_reused=False,
            )
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["phase"], "baseline")
            self.assertEqual(payload["baseline"], {"status": "pending", "submitted_at": None})

            module.bootstrap_state(
                state_path,
                run_id="optimize-20260622-123456-abcdef",
                source_operator="kernel.py",
                baseline_reused=True,
            )
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["phase"], "awaiting_round_start")
            self.assertEqual(payload["baseline"], {"status": "passed", "submitted_at": None})

    def test_start_round_is_idempotent_for_same_active_round(self) -> None:
        module = load_workflow_state_module()
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / ".triton-agent" / "state.json"
            state_path.parent.mkdir()
            module.bootstrap_state(
                state_path,
                run_id="optimize-20260622-123456-abcdef",
                source_operator="kernel.py",
                baseline_reused=True,
            )

            module.start_round(state_path, "opt-round-1")
            first_text = state_path.read_text(encoding="utf-8")
            module.start_round(state_path, "opt-round-1")
            second_text = state_path.read_text(encoding="utf-8")

        self.assertEqual(first_text, second_text)

    def test_complete_round_records_end_time_and_resets_phase(self) -> None:
        module = load_workflow_state_module()
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / ".triton-agent" / "state.json"
            state_path.parent.mkdir()
            module.bootstrap_state(
                state_path,
                run_id="optimize-20260622-123456-abcdef",
                source_operator="kernel.py",
                baseline_reused=True,
            )
            module.start_round(state_path, "opt-round-1")
            module.complete_round(state_path, "opt-round-1", current_round_arg=1)
            payload = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["phase"], "awaiting_round_start")
        self.assertIsNone(payload["current_round"])
        self.assertEqual(payload["rounds"]["1"]["status"], "passed")
        self.assertIsNotNone(payload["rounds"]["1"]["ended_at"])

    def test_write_round_timings_archive_only_includes_passed_rounds(self) -> None:
        module = load_workflow_state_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / ".triton-agent" / "state.json"
            archive_path = root / "triton-agent-logs" / "optimize-20260622" / "round-timings.json"
            state_path.parent.mkdir()
            module.bootstrap_state(
                state_path,
                run_id="optimize-20260622-123456-abcdef",
                source_operator="kernel.py",
                baseline_reused=True,
            )
            module.start_round(state_path, "opt-round-1")
            module.complete_round(state_path, "opt-round-1", current_round_arg=1)
            module.start_round(state_path, "opt-round-2")

            wrote = module.write_round_timings_archive(state_path, archive_path)
            payload = json.loads(archive_path.read_text(encoding="utf-8"))

        self.assertTrue(wrote)
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["round"], 1)
        self.assertIn("started_at", payload[0])
        self.assertIn("ended_at", payload[0])
        self.assertNotIn("run_id", payload[0])
        self.assertNotIn("round_dir", payload[0])

    def test_load_state_rejects_unknown_schema_version(self) -> None:
        module = load_workflow_state_module()
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / ".triton-agent" / "state.json"
            state_path.parent.mkdir()
            state_path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported workflow state schema_version"):
                module.load_state(state_path)

    def test_render_phase_summary_omits_workflow_state_path(self) -> None:
        module = load_workflow_state_module()
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / ".triton-agent" / "state.json"
            state_path.parent.mkdir()
            module.bootstrap_state(
                state_path,
                run_id="optimize-20260623-123456-abcdef",
                source_operator="kernel.py",
                baseline_reused=True,
            )
            module.start_round(state_path, "opt-round-2")

            summary = module.render_phase_summary(state_path)

        self.assertIn("Current phase: round_active", summary)
        self.assertIn("Current round: 2", summary)
        self.assertIn("Baseline source: reused", summary)
        self.assertNotIn("Workflow state path:", summary)
        self.assertNotIn(".triton-agent/state.json", summary)

    def test_load_state_malformed_json_does_not_leak_workflow_state_path(self) -> None:
        module = load_workflow_state_module()
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / ".triton-agent" / "state.json"
            state_path.parent.mkdir()
            state_path.write_text("{", encoding="utf-8")

            with self.assertRaises(ValueError) as raised:
                module.load_state(state_path)

        self.assertIn("malformed workflow state JSON", str(raised.exception))
        self.assertNotIn(".triton-agent/state.json", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
