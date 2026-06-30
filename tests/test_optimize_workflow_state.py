import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.skill_loader import load_skill_script_module


def load_state_machine_module():
    return load_skill_script_module("ascend-npu-optimize-state", "state_manage/state_machine")


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

    def test_set_current_round_state_module_is_exposed_from_state_manage(self) -> None:
        module = load_skill_script_module(
            "ascend-npu-optimize-state",
            "state_manage/set_current_round_state",
        )
        self.assertTrue(hasattr(module, "build_parser"))
        self.assertTrue(hasattr(module, "main"))

    def test_bootstrap_state_writes_expected_hook_gated_baseline_payload(self) -> None:
        module = load_state_machine_module()
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

    def test_bootstrap_state_writes_pretty_printed_json(self) -> None:
        module = load_state_machine_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / ".triton-agent" / "state.json"
            state_path.parent.mkdir()

            module.bootstrap_state(
                state_path,
                run_id="optimize-20260630-123456-abcdef",
                source_operator="kernel.py",
                baseline_reused=False,
            )
            text = state_path.read_text(encoding="utf-8")

        self.assertTrue(text.startswith("{\n"))
        self.assertIn('\n  "schema_version": 1,\n', text)
        self.assertTrue(text.endswith("}\n"))

    def test_start_round_is_idempotent_for_same_active_round(self) -> None:
        module = load_state_machine_module()
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / ".triton-agent" / "state.json"
            state_path.parent.mkdir()
            module.bootstrap_state(
                state_path,
                run_id="optimize-20260622-123456-abcdef",
                source_operator="kernel.py",
                baseline_reused=True,
            )

            module.start_round(
                state_path,
                "opt-round-1",
                round_strategy="exploration",
                analysis_policy="pattern_entry",
                reason="Need to narrow the first promising direction.",
            )
            first_text = state_path.read_text(encoding="utf-8")
            module.start_round(
                state_path,
                "opt-round-1",
                round_strategy="exploration",
                analysis_policy="pattern_entry",
                reason="Need to narrow the first promising direction.",
            )
            second_text = state_path.read_text(encoding="utf-8")

        self.assertEqual(first_text, second_text)

    def test_start_round_records_strategy_state(self) -> None:
        module = load_state_machine_module()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            state_path = workspace / ".triton-agent" / "state.json"
            round_dir = workspace / "opt-round-1"
            attempts_path = round_dir / "attempts.md"
            state_path.parent.mkdir()
            round_dir.mkdir()
            module.bootstrap_state(
                state_path,
                run_id="optimize-20260627-123456-abcdef",
                source_operator="kernel.py",
                baseline_reused=True,
            )

            module.start_round(
                state_path,
                round_dir.name,
                round_strategy="exploration",
                analysis_policy="pattern_entry",
                reason="Need to narrow the first promising direction.",
            )
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            attempts_exists = attempts_path.is_file()

        strategy_state = payload["rounds"]["1"]["strategy_state"]
        self.assertEqual(strategy_state["round_strategy"], "exploration")
        self.assertEqual(strategy_state["analysis_policy"], "pattern_entry")
        self.assertEqual(strategy_state["updated_by"], "start-round")
        self.assertTrue(attempts_exists)

    def test_start_round_keeps_state_when_attempts_mirror_write_fails(self) -> None:
        module = load_state_machine_module()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            state_path = workspace / ".triton-agent" / "state.json"
            round_dir = workspace / "opt-round-1"
            state_path.parent.mkdir()
            round_dir.mkdir()
            module.bootstrap_state(
                state_path,
                run_id="optimize-20260627-123456-abcdef",
                source_operator="kernel.py",
                baseline_reused=True,
            )

            original_append = cast(Any, module)._append_state_update_block
            try:
                def _raise_append(*args: object, **kwargs: object) -> None:
                    del args, kwargs
                    raise OSError("disk full")

                setattr(module, "_append_state_update_block", _raise_append)
                result = module.start_round(
                    state_path,
                    round_dir.name,
                    round_strategy="exploration",
                    analysis_policy="pattern_entry",
                    reason="Need to narrow the first promising direction.",
                )
            finally:
                setattr(module, "_append_state_update_block", original_append)

            payload = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["rounds"]["1"]["strategy_state"]["round_strategy"], "exploration")
        self.assertIn("attempts.md history mirror could not be updated", result["warnings"][0])

    def test_set_current_round_state_rejects_noop_and_policy_rollback(self) -> None:
        module = load_state_machine_module()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            state_path = workspace / ".triton-agent" / "state.json"
            round_dir = workspace / "opt-round-2"
            state_path.parent.mkdir()
            round_dir.mkdir()
            module.bootstrap_state(
                state_path,
                run_id="optimize-20260627-123456-abcdef",
                source_operator="kernel.py",
                baseline_reused=True,
            )
            module.start_round(
                state_path,
                round_dir.name,
                round_strategy="structural_change",
                analysis_policy="profile_required",
                reason="Need profiler-backed structural evidence first.",
            )

            with self.assertRaisesRegex(ValueError, "state update would be a no-op"):
                module.set_current_round_state(
                    state_path,
                    round_strategy="structural_change",
                    analysis_policy="profile_required",
                    reason="same state",
                )

            with self.assertRaisesRegex(ValueError, "analysis_policy cannot become shallower"):
                module.set_current_round_state(
                    state_path,
                    round_strategy="focused_tuning",
                    analysis_policy="pattern_entry",
                    reason="rollback should fail",
                )

    def test_set_current_round_state_initializes_missing_legacy_strategy_state(self) -> None:
        module = load_state_machine_module()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            state_path = workspace / ".triton-agent" / "state.json"
            round_dir = workspace / "opt-round-4"
            state_path.parent.mkdir()
            round_dir.mkdir()
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "optimize-20260627-123456-abcdef",
                        "phase": "round_active",
                        "source_operator": "kernel.py",
                        "current_round": 4,
                        "baseline": {"status": "passed", "submitted_at": "2026-06-27T12:34:56Z"},
                        "rounds": {
                            "4": {
                                "status": "active",
                                "round_dir": "opt-round-4",
                                "started_at": "2026-06-27T12:40:00Z",
                                "ended_at": None,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            module.set_current_round_state(
                state_path,
                round_strategy="stabilization",
                analysis_policy="ir_required",
                reason="Legacy active round needs explicit repair state.",
            )
            payload = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["rounds"]["4"]["strategy_state"]["round_strategy"], "stabilization")

    def test_state_updates_append_structured_attempts_log_blocks(self) -> None:
        module = load_state_machine_module()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            state_path = workspace / ".triton-agent" / "state.json"
            round_dir = workspace / "opt-round-3"
            attempts_path = round_dir / "attempts.md"
            state_path.parent.mkdir()
            round_dir.mkdir()
            module.bootstrap_state(
                state_path,
                run_id="optimize-20260627-123456-abcdef",
                source_operator="kernel.py",
                baseline_reused=True,
            )
            module.start_round(
                state_path,
                round_dir.name,
                round_strategy="exploration",
                analysis_policy="pattern_entry",
                reason="Start from pattern triage.",
            )
            module.set_current_round_state(
                state_path,
                round_strategy="structural_change",
                analysis_policy="profile_required",
                reason="Profiler evidence is now required before the main rewrite.",
            )

            attempts_text = attempts_path.read_text(encoding="utf-8")

        self.assertIn("## State Update", attempts_text)
        self.assertIn("Source: start-round", attempts_text)
        self.assertIn("Source: set-current-round-state", attempts_text)
        self.assertIn("Round strategy: exploration -> structural_change", attempts_text)

    def test_complete_round_records_end_time_and_resets_phase(self) -> None:
        module = load_state_machine_module()
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / ".triton-agent" / "state.json"
            state_path.parent.mkdir()
            module.bootstrap_state(
                state_path,
                run_id="optimize-20260622-123456-abcdef",
                source_operator="kernel.py",
                baseline_reused=True,
            )
            module.start_round(
                state_path,
                "opt-round-1",
                round_strategy="exploration",
                analysis_policy="pattern_entry",
                reason="Need to narrow the first promising direction.",
            )
            module.complete_round(state_path, "opt-round-1", current_round_arg=1)
            payload = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["phase"], "awaiting_round_start")
        self.assertIsNone(payload["current_round"])
        self.assertEqual(payload["rounds"]["1"]["status"], "passed")
        self.assertIsNotNone(payload["rounds"]["1"]["ended_at"])

    def test_write_round_timings_archive_only_includes_passed_rounds(self) -> None:
        module = load_state_machine_module()
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
            module.start_round(
                state_path,
                "opt-round-1",
                round_strategy="exploration",
                analysis_policy="pattern_entry",
                reason="Need to narrow the first promising direction.",
            )
            module.complete_round(state_path, "opt-round-1", current_round_arg=1)
            module.start_round(
                state_path,
                "opt-round-2",
                round_strategy="focused_tuning",
                analysis_policy="profile_required",
                reason="Round 1 narrowed the next tuning target.",
            )

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
        module = load_state_machine_module()
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / ".triton-agent" / "state.json"
            state_path.parent.mkdir()
            state_path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported workflow state schema_version"):
                module.load_state(state_path)

    def test_render_phase_summary_omits_workflow_state_path(self) -> None:
        module = load_state_machine_module()
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / ".triton-agent" / "state.json"
            state_path.parent.mkdir()
            module.bootstrap_state(
                state_path,
                run_id="optimize-20260623-123456-abcdef",
                source_operator="kernel.py",
                baseline_reused=True,
            )
            module.start_round(
                state_path,
                "opt-round-2",
                round_strategy="focused_tuning",
                analysis_policy="profile_required",
                reason="Round 1 narrowed the next tuning target.",
            )

            summary = module.render_phase_summary(state_path)

        self.assertIn("Current phase: round_active", summary)
        self.assertIn("Current round: 2", summary)
        self.assertIn("Baseline source: reused", summary)
        self.assertIn("Current round strategy: focused_tuning", summary)
        self.assertIn("Required analysis depth: profile_required", summary)
        self.assertIn("Current round reason: Round 1 narrowed the next tuning target.", summary)
        self.assertNotIn("Workflow state path:", summary)
        self.assertNotIn(".triton-agent/state.json", summary)

    def test_load_state_malformed_json_does_not_leak_workflow_state_path(self) -> None:
        module = load_state_machine_module()
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
