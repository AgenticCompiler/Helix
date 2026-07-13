import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import helix.optimize.workflow_state as workflow_state_module
from helix.skills.loader import load_skill_script_module


def load_state_machine_module():
    return load_skill_script_module("ascend-npu-optimize-state", "state_manage/state_machine")


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        cast(dict[str, object], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_resumable_optimize_workspace(workspace: Path, operator: Path) -> None:
    baseline_dir = workspace / "baseline"
    baseline_dir.mkdir()
    (baseline_dir / "state.json").write_text(
        json.dumps(
            {
                "baseline_kind": "original",
                "source_operator": f"../{operator.name}",
                "baseline_operator": "opt_kernel.py",
                "test_file": f"../differential_test_{operator.stem}.py",
                "test_mode": "differential",
                "bench_file": f"../bench_{operator.stem}.py",
                "bench_mode": "torch-npu-profiler",
                "perf_artifact": "perf.txt",
                "correctness_status": "passed",
                "benchmark_status": "passed",
                "baseline_established": True,
            }
        ),
        encoding="utf-8",
    )
    (baseline_dir / "perf.txt").write_text("latency-a: 1.0\n", encoding="utf-8")
    (baseline_dir / "opt_kernel.py").write_text("print('baseline')\n", encoding="utf-8")
    (workspace / "opt-note.md").write_text("history\n", encoding="utf-8")
    (workspace / "differential_test_{}".format(operator.stem + ".py")).write_text(
        "# test-mode: differential\nprint('test')\n",
        encoding="utf-8",
    )
    (workspace / "bench_{}".format(operator.stem + ".py")).write_text(
        "# bench-mode: torch-npu-profiler\n# kernel: k\nprint('bench')\n",
        encoding="utf-8",
    )
    (workspace / "opt-round-1").mkdir()


class OptimizeWorkflowStateTests(unittest.TestCase):
    def test_prepare_or_restore_workflow_state_reuses_existing_valid_state(self) -> None:
        module = load_state_machine_module()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            state_path = workspace / ".helix" / "state.json"
            state_path.parent.mkdir()
            module.bootstrap_state(
                state_path,
                run_id="existing-run",
                baseline_reused=True,
            )

            result = workflow_state_module.prepare_or_restore_optimize_workflow_state(
                operator,
                workspace,
                state_path=state_path,
                run_id="new-run",
            )
            payload = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertEqual(result.mode, "reused-existing-state")
        self.assertEqual(payload["run_id"], "existing-run")
        self.assertEqual(payload["phase"], "awaiting_round_start")
        self.assertNotIn("source_operator", payload)

    def test_prepare_or_restore_workflow_state_rebuilds_from_resumable_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            _write_resumable_optimize_workspace(workspace, operator)
            state_path = workspace / ".helix" / "state.json"
            state_path.parent.mkdir()

            result = workflow_state_module.prepare_or_restore_optimize_workflow_state(
                operator,
                workspace,
                state_path=state_path,
                run_id="resume-run",
            )
            payload = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertEqual(result.mode, "rebuilt-from-durable-artifacts")
        self.assertEqual(payload["run_id"], "resume-run")
        self.assertEqual(payload["phase"], "awaiting_round_start")
        self.assertEqual(payload["baseline"], {"status": "passed", "submitted_at": None})
        self.assertIsNone(payload["current_round"])
        self.assertNotIn("source_operator", payload)

    def test_prepare_or_restore_workflow_state_bootstraps_fresh_baseline_when_workspace_is_new(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            state_path = workspace / ".helix" / "state.json"
            state_path.parent.mkdir()

            result = workflow_state_module.prepare_or_restore_optimize_workflow_state(
                operator,
                workspace,
                state_path=state_path,
                run_id="fresh-run",
            )
            payload = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertEqual(result.mode, "bootstrapped-fresh-baseline")
        self.assertEqual(payload["run_id"], "fresh-run")
        self.assertEqual(payload["phase"], "baseline")
        self.assertEqual(payload["baseline"], {"status": "pending", "submitted_at": None})
        self.assertNotIn("source_operator", payload)

    def test_prepare_or_restore_workflow_state_rejects_invalid_existing_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            state_path = workspace / ".helix" / "state.json"
            state_path.parent.mkdir()
            state_path.write_text("{", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "malformed workflow state JSON"):
                workflow_state_module.prepare_or_restore_optimize_workflow_state(
                    operator,
                    workspace,
                    state_path=state_path,
                    run_id="broken-run",
                )

    def test_prepare_or_restore_without_hint_rejects_optimize_markers_when_baseline_state_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            baseline_dir = workspace / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text("{", encoding="utf-8")
            (workspace / "opt-note.md").write_text("history\n", encoding="utf-8")
            (workspace / "opt-round-1").mkdir()
            state_path = workspace / ".helix" / "state.json"
            state_path.parent.mkdir()

            with self.assertRaisesRegex(
                ValueError,
                "cannot determine source operator from baseline/state.json",
            ):
                workflow_state_module.prepare_or_restore_optimize_workflow_state(
                    None,
                    workspace,
                    state_path=state_path,
                    run_id="broken-baseline-run",
                )

            self.assertFalse(state_path.exists())

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
            state_path = root / ".helix" / "state.json"
            state_path.parent.mkdir()

            module.bootstrap_state(
                state_path,
                run_id="optimize-20260622-123456-abcdef",
                baseline_reused=False,
            )
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["phase"], "baseline")
            self.assertEqual(payload["baseline"], {"status": "pending", "submitted_at": None})
            self.assertNotIn("source_operator", payload)

            module.bootstrap_state(
                state_path,
                run_id="optimize-20260622-123456-abcdef",
                baseline_reused=True,
            )
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["phase"], "awaiting_round_start")
            self.assertEqual(payload["baseline"], {"status": "passed", "submitted_at": None})
            self.assertNotIn("source_operator", payload)

    def test_bootstrap_state_writes_pretty_printed_json(self) -> None:
        module = load_state_machine_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / ".helix" / "state.json"
            state_path.parent.mkdir()

            module.bootstrap_state(
                state_path,
                run_id="optimize-20260630-123456-abcdef",
                baseline_reused=False,
            )
            text = state_path.read_text(encoding="utf-8")

        self.assertTrue(text.startswith("{\n"))
        self.assertIn('\n  "schema_version": 1,\n', text)
        self.assertTrue(text.endswith("}\n"))

    def test_start_round_is_idempotent_for_same_active_round(self) -> None:
        module = load_state_machine_module()
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / ".helix" / "state.json"
            state_path.parent.mkdir()
            module.bootstrap_state(
                state_path,
                run_id="optimize-20260622-123456-abcdef",
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
            state_path = workspace / ".helix" / "state.json"
            round_dir = workspace / "opt-round-1"
            attempts_path = round_dir / "attempts.md"
            timing_path = workspace / ".helix" / "round-timings" / "opt-round-1.jsonl"
            state_path.parent.mkdir()
            round_dir.mkdir()
            module.bootstrap_state(
                state_path,
                run_id="optimize-20260627-123456-abcdef",
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
            timing_events = _load_jsonl(timing_path)

        strategy_state = payload["rounds"]["1"]["strategy_state"]
        self.assertEqual(strategy_state["round_strategy"], "exploration")
        self.assertEqual(strategy_state["analysis_policy"], "pattern_entry")
        self.assertEqual(strategy_state["updated_by"], "start-round")
        self.assertTrue(attempts_exists)
        self.assertNotIn("started_at", payload["rounds"]["1"])
        self.assertNotIn("ended_at", payload["rounds"]["1"])
        self.assertEqual([event["event"] for event in timing_events], ["round_start"])
        self.assertEqual(timing_events[0]["round"], "opt-round-1")
        self.assertEqual(timing_events[0]["run_id"], "optimize-20260627-123456-abcdef")

    def test_start_round_keeps_state_when_attempts_mirror_write_fails(self) -> None:
        module = load_state_machine_module()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            state_path = workspace / ".helix" / "state.json"
            round_dir = workspace / "opt-round-1"
            state_path.parent.mkdir()
            round_dir.mkdir()
            module.bootstrap_state(
                state_path,
                run_id="optimize-20260627-123456-abcdef",
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
            state_path = workspace / ".helix" / "state.json"
            round_dir = workspace / "opt-round-2"
            state_path.parent.mkdir()
            round_dir.mkdir()
            module.bootstrap_state(
                state_path,
                run_id="optimize-20260627-123456-abcdef",
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
            state_path = workspace / ".helix" / "state.json"
            round_dir = workspace / "opt-round-4"
            state_path.parent.mkdir()
            round_dir.mkdir()
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "optimize-20260627-123456-abcdef",
                        "phase": "round_active",
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
            state_path = workspace / ".helix" / "state.json"
            round_dir = workspace / "opt-round-3"
            attempts_path = round_dir / "attempts.md"
            state_path.parent.mkdir()
            round_dir.mkdir()
            module.bootstrap_state(
                state_path,
                run_id="optimize-20260627-123456-abcdef",
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
            workspace = Path(tmp)
            state_path = workspace / ".helix" / "state.json"
            timing_path = workspace / ".helix" / "round-timings" / "opt-round-1.jsonl"
            state_path.parent.mkdir()
            module.bootstrap_state(
                state_path,
                run_id="optimize-20260622-123456-abcdef",
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
            timing_events = _load_jsonl(timing_path)

        self.assertEqual(payload["phase"], "awaiting_round_start")
        self.assertIsNone(payload["current_round"])
        self.assertEqual(payload["rounds"]["1"]["status"], "passed")
        self.assertNotIn("started_at", payload["rounds"]["1"])
        self.assertNotIn("ended_at", payload["rounds"]["1"])
        self.assertEqual(
            [event["event"] for event in timing_events],
            ["round_start", "round_end"],
        )
        self.assertEqual(timing_events[1]["round"], "opt-round-1")
        self.assertEqual(timing_events[1]["run_id"], "optimize-20260622-123456-abcdef")

    def test_complete_round_can_close_rejected_terminal_round_as_failed(self) -> None:
        module = load_state_machine_module()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            state_path = workspace / ".helix" / "state.json"
            state_path.parent.mkdir()
            module.bootstrap_state(
                state_path,
                run_id="optimize-20260709-123456-abcdef",
                baseline_reused=True,
            )
            module.start_round(
                state_path,
                "opt-round-1",
                round_strategy="exploration",
                analysis_policy="pattern_entry",
                reason="The first attempt hit a correctness blocker.",
            )

            module.complete_round(
                state_path,
                "opt-round-1",
                current_round_arg=1,
                final_status="failed",
            )
            payload = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["phase"], "awaiting_round_start")
        self.assertIsNone(payload["current_round"])
        self.assertEqual(payload["rounds"]["1"]["status"], "failed")

    def test_load_state_accepts_legacy_round_timestamps(self) -> None:
        module = load_state_machine_module()
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / ".helix" / "state.json"
            state_path.parent.mkdir()
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "optimize-20260706-123456-abcdef",
                        "phase": "awaiting_round_start",
                        "current_round": None,
                        "baseline": {"status": "passed", "submitted_at": "2026-07-06T12:34:56Z"},
                        "rounds": {
                            "1": {
                                "status": "passed",
                                "round_dir": "opt-round-1",
                                "started_at": "2026-07-06T12:40:00Z",
                                "ended_at": "2026-07-06T12:55:00Z",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            payload = module.load_state(state_path)

        self.assertEqual(payload["phase"], "awaiting_round_start")
        self.assertEqual(payload["rounds"]["1"]["status"], "passed")

    def test_load_state_rejects_unknown_schema_version(self) -> None:
        module = load_state_machine_module()
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / ".helix" / "state.json"
            state_path.parent.mkdir()
            state_path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported workflow state schema_version"):
                module.load_state(state_path)

    def test_render_phase_summary_omits_workflow_state_path(self) -> None:
        module = load_state_machine_module()
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / ".helix" / "state.json"
            state_path.parent.mkdir()
            module.bootstrap_state(
                state_path,
                run_id="optimize-20260623-123456-abcdef",
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
        self.assertNotIn(".helix/state.json", summary)

    def test_load_state_malformed_json_does_not_leak_workflow_state_path(self) -> None:
        module = load_state_machine_module()
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / ".helix" / "state.json"
            state_path.parent.mkdir()
            state_path.write_text("{", encoding="utf-8")

            with self.assertRaises(ValueError) as raised:
                module.load_state(state_path)

        self.assertIn("malformed workflow state JSON", str(raised.exception))
        self.assertNotIn(".helix/state.json", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
