import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.optimize.contract import ROUND_STATE_OPTIONAL_FIELDS
from triton_agent.optimize.stages import Stage
from triton_agent.optimize.workflow_state import (
    get_stages_addressed_from_state,
    record_stage_addressed_in_workflow_state,
)
from triton_agent.skills.loader import load_skill_script_module


_WORKFLOW = load_skill_script_module("ascend-npu-optimize-state", "state_manage/state_machine")
_ROUND_CONTRACT = load_skill_script_module("ascend-npu-optimize-state", "round/check")


def _bootstrap(state_path: Path) -> None:
    _WORKFLOW.bootstrap_state(
        state_path,
        run_id="run-1",
        baseline_reused=True,
    )


class WorkflowStateStageTrackingTests(unittest.TestCase):
    def test_bootstrap_initializes_empty_stages_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / ".triton-agent" / "state.json"
            _bootstrap(state_path)
            self.assertEqual(_WORKFLOW.get_stages_addressed(state_path), [])

    def test_record_stage_appends_and_dedups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / ".triton-agent" / "state.json"
            _bootstrap(state_path)
            _WORKFLOW.record_stage_addressed(state_path, "algorithmic")
            _WORKFLOW.record_stage_addressed(state_path, "algorithmic")  # dedup
            _WORKFLOW.record_stage_addressed(state_path, "memory_access")
            self.assertEqual(
                _WORKFLOW.get_stages_addressed(state_path),
                ["algorithmic", "memory_access"],
            )

    def test_record_stage_rejects_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / ".triton-agent" / "state.json"
            _bootstrap(state_path)
            with self.assertRaises(ValueError):
                _WORKFLOW.record_stage_addressed(state_path, "")

    def test_cli_bridge_returns_typed_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / ".triton-agent" / "state.json"
            _bootstrap(state_path)
            record_stage_addressed_in_workflow_state(state_path, Stage.ALGORITHMIC)
            record_stage_addressed_in_workflow_state(state_path, Stage.MEMORY_ACCESS)
            addressed = get_stages_addressed_from_state(state_path)
            self.assertEqual(addressed, [Stage.ALGORITHMIC, Stage.MEMORY_ACCESS])

    def test_cli_bridge_drops_invalid_stage_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / ".triton-agent" / "state.json"
            _bootstrap(state_path)
            _WORKFLOW.record_stage_addressed(state_path, "algorithmic")
            _WORKFLOW.record_stage_addressed(state_path, "totally_made_up")
            addressed = get_stages_addressed_from_state(state_path)
            self.assertEqual(addressed, [Stage.ALGORITHMIC])

    def test_cli_bridge_handles_missing_state(self) -> None:
        self.assertEqual(get_stages_addressed_from_state(None), [])
        self.assertEqual(
            get_stages_addressed_from_state(Path("/nonexistent/state.json")), []
        )


class RoundStateStageFieldTests(unittest.TestCase):
    def test_stage_is_in_optional_fields_contract(self) -> None:
        self.assertIn("stage", ROUND_STATE_OPTIONAL_FIELDS)

    def test_load_round_state_reads_stage(self) -> None:
        minimal_round_state = {
            "round": "opt-round-1",
            "parent_round": "round-0",
            "hypothesis": "fuse per-launch loop",
            "evidence_sources": ["benchmark"],
            "correctness_status": "passed",
            "benchmark_status": "passed",
            "perf_artifact": "opt_op_perf.txt",
            "comparison_target": "../baseline/op_perf.txt",
            "effective_metric_source": "kernel",
            "summary_path": "summary.md",
            "opt_note_updated": True,
            "stage": "algorithmic",
        }
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp)
            (round_dir / "round-state.json").write_text(
                json.dumps(minimal_round_state), encoding="utf-8"
            )
            state = _ROUND_CONTRACT.load_round_state(round_dir)
        self.assertEqual(state.stage, "algorithmic")

    def test_load_round_state_stage_defaults_none_when_absent(self) -> None:
        minimal_round_state = {
            "round": "opt-round-1",
            "parent_round": "round-0",
            "hypothesis": "tune tiles",
            "evidence_sources": ["benchmark"],
            "correctness_status": "passed",
            "benchmark_status": "passed",
            "perf_artifact": "opt_op_perf.txt",
            "comparison_target": "../baseline/op_perf.txt",
            "effective_metric_source": "kernel",
            "summary_path": "summary.md",
            "opt_note_updated": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp)
            (round_dir / "round-state.json").write_text(
                json.dumps(minimal_round_state), encoding="utf-8"
            )
            state = _ROUND_CONTRACT.load_round_state(round_dir)
        self.assertIsNone(state.stage)


if __name__ == "__main__":
    unittest.main()
